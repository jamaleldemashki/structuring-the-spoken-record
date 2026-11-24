"""Mention extraction with GLiNER. Every mention keeps its exact character offsets."""

import json
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from spoken_record.adapters.base import sha256, stable_id
from spoken_record.models import TYPES
from spoken_record.extraction_models import ExtractionCandidate, ExtractionConfig, ExtractedMention
from spoken_record.ingestion import workspace_path
from spoken_record.passage_models import ModelWindow, Passage

TOKEN_PATTERN = re.compile(r"\w+(?:[-_]\w+)*|\S")


@dataclass(frozen=True)
class GLiNERWhitespaceBudget:
    """The same whitespace word splitter GLiNER 0.2.21 uses with words_splitter_type=whitespace."""

    revision: str
    name: str = ""
    verified_for_model: bool = True

    def __post_init__(self):
        object.__setattr__(self, "name", f"gliner-0.2.21-whitespace@{self.revision}")

    def spans(self, text: str) -> list[tuple[int, int]]:
        return [(m.start(), m.end()) for m in TOKEN_PATTERN.finditer(text)]

    def fits(self, text: str, max_tokens: int) -> bool:
        return len(self.spans(text)) <= max_tokens


def load_configuration(root: Path, model_name: str) -> tuple[ExtractionConfig, dict, Path, Path]:
    model_path = workspace_path(root, model_name)
    model_config = ExtractionConfig.model_validate_json(model_path.read_bytes())
    labels_path = workspace_path(root, model_config.labels_config)
    labels = json.loads(labels_path.read_bytes())
    if labels.get("schema_version") != 2 or set(labels.get("labels", {})) != set(TYPES):
        raise ValueError("entity labels must exactly match the frozen type inventory")
    descriptions = [labels["labels"][label] for label in TYPES]
    if any(not isinstance(x, str) or not x.strip() for x in descriptions) or len(set(descriptions)) != len(descriptions):
        raise ValueError("model label descriptions must be unique nonempty strings")
    threshold = labels.get("provisional_threshold")
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool) or not 0 <= threshold <= 1:
        raise ValueError("invalid acceptance threshold")
    if model_config.candidate_floor > threshold:
        raise ValueError("candidate floor must not exceed acceptance threshold")
    return model_config, labels, model_path, labels_path


def load_gliner(config: ExtractionConfig):
    """Load the pinned model revision from the local cache only."""
    # In this GLiNER version local_files_only reaches the checkpoint but not the nested
    # encoder and tokenizer. So we put all of Hugging Face in offline mode and use the
    # cache from the smoke test.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from gliner import GLiNER

    model = GLiNER.from_pretrained(config.repo_id, revision=config.revision, local_files_only=True)
    model.to(config.device)
    model.eval()
    return model


def verify_runtime(model, config: ExtractionConfig, budget: GLiNERWhitespaceBudget) -> None:
    if int(model.config.max_len) != config.model_max_words:
        raise ValueError("runtime checkpoint maximum differs from frozen configuration")
    splitter = model.data_processor.words_splitter
    if splitter.splitter.__class__.__name__ != "WhitespaceTokenSplitter":
        raise ValueError("runtime checkpoint uses a different word splitter")
    encoder_max = int(model.model.token_rep_layer.bert_layer.model.config.max_position_embeddings)
    if encoder_max != config.encoder_max_tokens:
        raise ValueError("runtime encoder maximum differs from frozen configuration")
    probes = ["Thomas Aquinas, free-will.", " Pāṇini\r\nGod's work—today! "]
    for text in probes:
        actual = [(start, end) for _, start, end in splitter(text)]
        if actual != budget.spans(text):
            raise ValueError("runtime GLiNER splitter differs from extraction budget")


def verify_encoder_batch(model, texts: list[str], labels: list[str], config: ExtractionConfig,
                         budget: GLiNERWhitespaceBudget) -> int:
    """Check that the label prompt and every word of the text survive the encoder tokenizer."""
    model_input, raw = model.prepare_model_inputs(texts, labels)
    expected_words = [len(budget.spans(text)) for text in texts]
    raw_words = [int(value) for value in raw["seq_length"]]
    if raw_words != expected_words:
        raise ValueError("GLiNER preprocessing changed the checkpoint word count")
    words_mask = model_input["words_mask"]
    observed_words = [int(row.max().item()) for row in words_mask]
    if observed_words != expected_words:
        raise ValueError("encoder tokenization truncated one or more GLiNER words")
    encoder_lengths = [int(value) for value in model_input["attention_mask"].sum(dim=1).tolist()]
    if any(length > config.encoder_max_tokens for length in encoder_lengths):
        raise ValueError("label prompt plus text exceeds the encoder token limit")
    return max(encoder_lengths, default=0)


def extraction_fingerprint(config: ExtractionConfig, labels: dict) -> str:
    payload = {"algorithm": "gliner-extraction-1", "config": config.model_dump(mode="json"),
               "labels": labels["labels"], "threshold": labels["provisional_threshold"]}
    return sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode())


def raw_candidate(run_id: str, window: ModelWindow, result: dict, label_map: dict[str, str]) -> dict:
    local_start, local_end = result.get("start"), result.get("end")
    label, score = result.get("label"), result.get("score")
    if (not isinstance(local_start, int) or isinstance(local_start, bool) or
            not isinstance(local_end, int) or isinstance(local_end, bool) or
            local_start < 0 or local_end <= local_start or local_end > window.char_end - window.char_start):
        raise ValueError("model returned an invalid local span")
    if label not in label_map:
        raise ValueError("model returned an unknown label")
    score = float(score)
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("model returned an invalid score")
    start, end = window.char_start + local_start, window.char_start + local_end
    return {"created_by_run_id": run_id,
            "candidate_id": stable_id("cand", window.window_id, str(start), str(end), label_map[label]),
            "window_id": window.window_id, "episode_id": window.episode_id,
            "transcript_version_id": window.transcript_version_id, "char_start": start, "char_end": end,
            "text": result.get("text"), "type": label_map[label], "model_label": label, "score": score}


def resolve_candidates(raw: list[dict], threshold: float, fingerprint: str,
                       passages: list[Passage]) -> tuple[list[ExtractionCandidate], list[ExtractedMention]]:
    """Drop duplicates from overlapping windows, then keep non-overlapping spans greedily."""
    type_order = {label: i for i, label in enumerate(TYPES)}
    by_transcript = defaultdict(list)
    for row in raw:
        by_transcript[row["transcript_version_id"]].append(row)
    passage_map = defaultdict(list)
    for passage in passages:
        passage_map[passage.transcript_version_id].append(passage)
    decisions, mentions = {}, []
    for transcript_id, rows in sorted(by_transcript.items()):
        eligible = defaultdict(list)
        for row in rows:
            if row["score"] < threshold:
                decisions[row["candidate_id"]] = ("below_threshold", None)
            else:
                eligible[(row["char_start"], row["char_end"], row["type"])].append(row)
        representatives = []
        for key, group in eligible.items():
            group.sort(key=lambda r: (-r["score"], r["window_id"], r["candidate_id"]))
            representatives.append((group[0], group))
        representatives.sort(key=lambda item: (-item[0]["score"], -(item[0]["char_end"] - item[0]["char_start"]),
                                                 type_order[item[0]["type"]], item[0]["char_start"], item[0]["char_end"],
                                                 item[0]["candidate_id"]))
        accepted = []
        for winner, group in representatives:
            if any(winner["char_start"] < end and winner["char_end"] > start for start, end in accepted):
                for row in group:
                    decisions[row["candidate_id"]] = ("overlap", None)
                continue
            mention_id = stable_id("m", transcript_id, fingerprint, str(winner["char_start"]),
                                   str(winner["char_end"]), winner["type"])
            intersecting = [p.passage_id for p in passage_map[transcript_id]
                            if p.char_start < winner["char_end"] and p.char_end > winner["char_start"]]
            if not intersecting:
                raise ValueError("accepted mention has no passage evidence")
            mentions.append(ExtractedMention(
                created_by_run_id=winner["created_by_run_id"], mention_id=mention_id,
                episode_id=winner["episode_id"], transcript_version_id=transcript_id,
                char_start=winner["char_start"], char_end=winner["char_end"], text=winner["text"],
                type=winner["type"], score=winner["score"],
                source_window_ids=sorted({r["window_id"] for r in group}), passage_ids=intersecting,
                extraction_fingerprint=fingerprint))
            accepted.append((winner["char_start"], winner["char_end"]))
            decisions[winner["candidate_id"]] = ("accepted", mention_id)
            for row in group[1:]:
                decisions[row["candidate_id"]] = ("duplicate", mention_id)
    candidates = [ExtractionCandidate.model_validate({**row, "decision": decisions[row["candidate_id"]][0],
                  "selected_mention_id": decisions[row["candidate_id"]][1]}) for row in raw]
    return sorted(candidates, key=lambda c: (c.transcript_version_id, c.window_id, c.char_start, c.char_end, c.type)), \
        sorted(mentions, key=lambda m: (m.transcript_version_id, m.char_start, m.char_end, m.type))


def predict_episode(model, bundle, windows: list[ModelWindow], descriptions: list[str],
                    label_map: dict[str, str], config: ExtractionConfig,
                    budget: GLiNERWhitespaceBudget, run_id: str) -> tuple[list[dict], int]:
    """GLiNER over the windows of one episode. Returns the raw candidates above the floor and
    the largest encoder input seen."""
    raw, encoder_max = [], 0
    for offset in range(0, len(windows), config.batch_size):
        batch = windows[offset:offset + config.batch_size]
        texts = [bundle.transcript.text[w.char_start:w.char_end] for w in batch]
        encoder_max = max(encoder_max, verify_encoder_batch(model, texts, descriptions, config, budget))
        predicted = model.batch_predict_entities(texts, descriptions, flat_ner=False,
                                                 threshold=config.candidate_floor, multi_label=False)
        if len(predicted) != len(batch):
            raise ValueError("model returned the wrong batch length")
        for window, results, text in zip(batch, predicted, texts):
            for result in results:
                row = raw_candidate(run_id, window, result, label_map)
                if row["text"] != text[result["start"]:result["end"]] or \
                        row["text"] != bundle.transcript.text[row["char_start"]:row["char_end"]]:
                    raise ValueError("model entity text does not reproduce the source")
                raw.append(row)
    return raw, encoder_max
