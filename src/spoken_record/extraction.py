"""Mention extraction with GLiNER. Every mention keeps its exact character offsets."""

import importlib.metadata
import json
import math
import os
import platform
import re
import resource
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from spoken_record import __version__
from spoken_record.adapters.base import sha256, stable_id
from spoken_record.evaluation import TYPES, Mention, Prediction, load_sample, score_records
from spoken_record.extraction_models import ExtractionCandidate, ExtractionConfig, ExtractedMention
from spoken_record.ingestion import dump_json, dump_jsonl, read_jsonl, workspace_path
from spoken_record.passage_models import ModelWindow, Passage, SegmentationConfig
from spoken_record.segmentation import load_sources, validate_run as validate_upstream
from spoken_record.windows import build_windows, require_model_ready

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


def predictions_for_packet(packet: Path, mentions: list[ExtractedMention], outcomes: list[dict]) -> list[Prediction]:
    _, samples = load_sample(packet)
    status = {o["episode_id"]: o["status"] for o in outcomes}
    result = []
    for passage_id, sample in samples.items():
        rows = [m for m in mentions if m.episode_id == sample["episode_id"] and
                m.char_start >= sample["char_start"] and m.char_end <= sample["char_end"]]
        result.append(Prediction(
            passage_id=passage_id, text_sha256=sample["text_sha256"],
            status="complete" if status.get(sample["episode_id"]) == "processed" else "failed",
            mentions=[Mention(start=m.char_start - sample["char_start"], end=m.char_end - sample["char_start"],
                              text=m.text, type=m.type) for m in rows]))
    return result


def extract_run(root: Path, input_name: str, model_name: str, packet_name: str | None = None,
                run_id: str | None = None, model_loader=None) -> Path:
    """Run GLiNER and write the mentions with their exact offsets."""
    root = root.resolve()
    started, now = perf_counter(), datetime.now(timezone.utc)
    run_id = run_id or f"extract-{now:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]{0,95}", run_id):
        raise ValueError("invalid run ID")
    run_root, parent = root / "data/prototype1/runs", workspace_path(root, input_name)
    if parent.parent != run_root or parent.name == run_id:
        raise ValueError("input must be a sibling passage snapshot")
    output = run_root / run_id
    if output.exists():
        raise FileExistsError(output)
    parent_manifest = validate_upstream(parent)
    if parent_manifest["stage"] != "segment":
        raise ValueError("extraction requires a passage snapshot")
    parent_digest = sha256((parent / "manifest.json").read_bytes())
    source = parent.parent / parent_manifest["parent_run_id"]
    bundles = load_sources(source)
    passages = [Passage.model_validate(r) for r in read_jsonl(parent / "passages.jsonl")]
    by_transcript = defaultdict(list)
    for passage in passages:
        by_transcript[passage.transcript_version_id].append(passage)
    config, labels, config_path, labels_path = load_configuration(root, model_name)
    config_bytes, labels_bytes = config_path.read_bytes(), labels_path.read_bytes()
    segmentation = SegmentationConfig.model_validate_json((parent / "config.json").read_bytes()).model_copy(update={
        "window_max_tokens": config.window_max_tokens, "window_overlap_tokens": config.window_overlap_tokens})
    budget = GLiNERWhitespaceBudget(config.revision)
    model = (model_loader or load_gliner)(config)
    verify_runtime(model, config, budget)
    descriptions = [labels["labels"][label] for label in TYPES]
    label_map = {description: canonical for canonical, description in labels["labels"].items()}
    windows = [w for bundle in bundles for w in build_windows(bundle, by_transcript[bundle.transcript.transcript_version_id],
                                                               segmentation, run_id, budget)]
    require_model_ready(windows)
    # The model's word limit and the encoder's subword limit are two different things.
    # If a longer label list makes the prompt too big, fail here instead of letting the
    # tokenizer truncate it.
    for offset in range(0, len(windows), config.batch_size):
        batch = windows[offset:offset + config.batch_size]
        texts = [next(b for b in bundles if b.episode.episode_id == w.episode_id).transcript.text[w.char_start:w.char_end]
                 for w in batch]
        verify_encoder_batch(model, texts, descriptions, config, budget)
    raw, outcomes = [], []
    for bundle in bundles:
        episode_start = perf_counter()
        episode_windows = [w for w in windows if w.episode_id == bundle.episode.episode_id]
        episode_raw, episode_encoder_max = [], 0
        try:
            for offset in range(0, len(episode_windows), config.batch_size):
                batch = episode_windows[offset:offset + config.batch_size]
                texts = [bundle.transcript.text[w.char_start:w.char_end] for w in batch]
                episode_encoder_max = max(episode_encoder_max,
                    verify_encoder_batch(model, texts, descriptions, config, budget))
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
                        episode_raw.append(row)
            raw.extend(episode_raw)
            outcomes.append({"episode_id": bundle.episode.episode_id, "source_path": bundle.episode.source_path,
                             "status": "processed", "reason": None, "windows": len(episode_windows),
                             "encoder_token_max_observed": episode_encoder_max,
                             "seconds": round(perf_counter() - episode_start, 4)})
        except Exception as exc:
            outcomes.append({"episode_id": bundle.episode.episode_id, "source_path": bundle.episode.source_path,
                             "status": "failed", "reason": f"{type(exc).__name__}: inference failed",
                             "windows": len(episode_windows), "encoder_token_max_observed": episode_encoder_max,
                             "seconds": round(perf_counter() - episode_start, 4)})
    fingerprint = extraction_fingerprint(config, labels)
    candidates, mentions = resolve_candidates(raw, float(labels["provisional_threshold"]), fingerprint, passages)
    for outcome in outcomes:
        outcome["candidates"] = sum(c.episode_id == outcome["episode_id"] for c in candidates)
        outcome["mentions"] = sum(m.episode_id == outcome["episode_id"] for m in mentions)
    packet, predictions, evaluation = None, [], None
    support = {str(config_path.relative_to(root)): sha256(config_bytes),
               str(labels_path.relative_to(root)): sha256(labels_bytes)}
    if packet_name:
        packet = workspace_path(root, packet_name)
        predictions = predictions_for_packet(packet, mentions, outcomes)
        support[str((packet / "manifest.json").relative_to(root))] = sha256((packet / "manifest.json").read_bytes())
        support[str((packet / "sample.jsonl").relative_to(root))] = sha256((packet / "sample.jsonl").read_bytes())
        gold_path = packet / "annotations.jsonl"
        support[str(gold_path.relative_to(root))] = sha256(gold_path.read_bytes())
        if all(p.status == "complete" for p in predictions):
            evaluation = score_records(packet, read_jsonl(gold_path), [p.model_dump(mode="json") for p in predictions],
                                       json.loads((packet / "manifest.json").read_bytes())["split"])
            evaluation["status"] = "complete"
        else:
            evaluation = {"schema_version": 1, "status": "not_scored", "reason": "failed sampled episode"}
    validate_upstream(parent)
    if sha256((parent / "manifest.json").read_bytes()) != parent_digest or config_path.read_bytes() != config_bytes or \
            labels_path.read_bytes() != labels_bytes:
        raise ValueError("parent or extraction configuration changed during inference")
    for name, digest in support.items():
        if sha256(workspace_path(root, name).read_bytes()) != digest:
            raise ValueError("evaluation input changed during inference")
    output.mkdir(parents=True, exist_ok=False)
    dump_jsonl(output / "model_windows.jsonl", windows)
    dump_jsonl(output / "candidates.jsonl", candidates)
    dump_jsonl(output / "mentions.jsonl", mentions)
    dump_jsonl(output / "outcomes.jsonl", outcomes)
    dump_json(output / "config.json", config.model_dump(mode="json"))
    dump_json(output / "labels.json", labels)
    names = ["model_windows.jsonl", "candidates.jsonl", "mentions.jsonl", "outcomes.jsonl", "config.json", "labels.json"]
    if packet:
        dump_jsonl(output / "predictions.jsonl", predictions)
        evaluation["gold_sha256"] = support[str((packet / "annotations.jsonl").relative_to(root))]
        evaluation["predictions_sha256"] = sha256((output / "predictions.jsonl").read_bytes())
        dump_json(output / "evaluation.json", evaluation)
        names += ["predictions.jsonl", "evaluation.json"]
    counts = {"selected": len(bundles), "processed": sum(o["status"] == "processed" for o in outcomes),
              "failed": sum(o["status"] == "failed" for o in outcomes), "windows": len(windows),
              "candidates": len(candidates), "accepted_mentions": len(mentions),
              "below_threshold": sum(c.decision == "below_threshold" for c in candidates),
              "duplicates": sum(c.decision == "duplicate" for c in candidates),
              "overlaps": sum(c.decision == "overlap" for c in candidates)}
    code_root = Path(__file__).parent
    manifest = {"schema_version": 2, "stage": "extract", "run_id": run_id,
        "status": "failed" if counts["failed"] else "complete", "package_version": __version__,
        "started_utc": now.isoformat(), "elapsed_seconds": round(perf_counter() - started, 4),
        "python": platform.python_version(), "machine": platform.machine(),
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "dependencies": {n: importlib.metadata.version(n) for n in ("gliner", "torch", "transformers", "pydantic")},
        "parent_run_id": parent.name, "parent_manifest_sha256": parent_digest,
        "support_sha256": support, "evaluation_packet": packet_name,
        "extraction_fingerprint": fingerprint, "counts": counts,
        "code_sha256": {str(p.relative_to(code_root)): sha256(p.read_bytes()) for p in sorted(code_root.rglob("*.py"))},
        "artifact_sha256": {name: sha256((output / name).read_bytes()) for name in names},
        "model": {"framework": config.framework, "version": config.framework_version,
                  "repo_id": config.repo_id, "revision": config.revision, "device": config.device,
                  "max_words": config.model_max_words, "encoder_max_tokens": config.encoder_max_tokens,
                  "encoder_token_max_observed": max((o["encoder_token_max_observed"] for o in outcomes), default=0),
                  "splitter": config.words_splitter},
        "thresholds": {"candidate_floor": config.candidate_floor,
                       "acceptance": labels["provisional_threshold"]}}
    dump_json(output / "manifest.json", manifest)
    validate_run(output)
    return output


def validate_run(output: Path) -> dict:
    output = output.resolve()
    manifest = json.loads((output / "manifest.json").read_bytes())
    if manifest.get("schema_version") != 2 or manifest.get("stage") != "extract":
        raise ValueError("unsupported extraction snapshot")
    parent = output.parent / manifest["parent_run_id"]
    if sha256((parent / "manifest.json").read_bytes()) != manifest["parent_manifest_sha256"]:
        raise ValueError("parent passage manifest changed")
    parent_manifest = validate_upstream(parent)
    source, root = parent.parent / parent_manifest["parent_run_id"], output.parents[3]
    bundles = load_sources(source)
    passages = [Passage.model_validate(r) for r in read_jsonl(parent / "passages.jsonl")]
    config = ExtractionConfig.model_validate_json((output / "config.json").read_bytes())
    labels = json.loads((output / "labels.json").read_bytes())
    model_names = [name for name in manifest["support_sha256"] if Path(name).name.startswith("model")
                   and Path(name).suffix == ".json"]
    if len(model_names) != 1:
        raise ValueError("manifest must identify one model configuration")
    original_config, original_labels, _, _ = load_configuration(root, model_names[0])
    if config != original_config or labels != original_labels:
        raise ValueError("snapshotted extraction configuration differs from input")
    expected_names = {"model_windows.jsonl", "candidates.jsonl", "mentions.jsonl", "outcomes.jsonl", "config.json", "labels.json"}
    if manifest.get("evaluation_packet"):
        expected_names |= {"predictions.jsonl", "evaluation.json"}
    if set(manifest["artifact_sha256"]) != expected_names:
        raise ValueError("unexpected extraction artifact set")
    for name, digest in manifest["artifact_sha256"].items():
        if sha256((output / name).read_bytes()) != digest:
            raise ValueError(f"artifact checksum mismatch: {name}")
    for name, digest in manifest["support_sha256"].items():
        if sha256(workspace_path(root, name).read_bytes()) != digest:
            raise ValueError(f"extraction support changed: {name}")
    by_transcript = defaultdict(list)
    for passage in passages:
        by_transcript[passage.transcript_version_id].append(passage)
    parent_segmentation = SegmentationConfig.model_validate_json((parent / "config.json").read_bytes()).model_copy(update={
        "window_max_tokens": config.window_max_tokens, "window_overlap_tokens": config.window_overlap_tokens})
    budget = GLiNERWhitespaceBudget(config.revision)
    windows = [ModelWindow.model_validate(r) for r in read_jsonl(output / "model_windows.jsonl")]
    expected_windows = [w for b in bundles for w in build_windows(b, by_transcript[b.transcript.transcript_version_id],
                                                                  parent_segmentation, manifest["run_id"], budget)]
    if windows != expected_windows:
        raise ValueError("model windows differ from the checkpoint-bound budget")
    require_model_ready(windows)
    transcripts = {b.transcript.transcript_version_id: b for b in bundles}
    window_map = {w.window_id: w for w in windows}
    candidates = [ExtractionCandidate.model_validate(r) for r in read_jsonl(output / "candidates.jsonl")]
    raw = []
    for candidate in candidates:
        if candidate.created_by_run_id != manifest["run_id"] or candidate.window_id not in window_map:
            raise ValueError("candidate has invalid provenance or window")
        window, bundle = window_map[candidate.window_id], transcripts[candidate.transcript_version_id]
        if candidate.episode_id != window.episode_id or not (window.char_start <= candidate.char_start < candidate.char_end <= window.char_end):
            raise ValueError("candidate exceeds its source window")
        if bundle.transcript.text[candidate.char_start:candidate.char_end] != candidate.text:
            raise ValueError("candidate does not reproduce source text")
        raw.append(candidate.model_dump(exclude={"decision", "selected_mention_id"}, mode="json"))
    expected_candidates, expected_mentions = resolve_candidates(raw, float(labels["provisional_threshold"]),
                                                                 manifest["extraction_fingerprint"], passages)
    mentions = [ExtractedMention.model_validate(r) for r in read_jsonl(output / "mentions.jsonl")]
    if candidates != expected_candidates or mentions != expected_mentions:
        raise ValueError("candidate decisions or accepted mentions are not reproducible")
    outcomes = read_jsonl(output / "outcomes.jsonl")
    if len(outcomes) != len(bundles) or {o["episode_id"] for o in outcomes} != {b.episode.episode_id for b in bundles}:
        raise ValueError("outcomes do not cover selected episodes")
    for outcome in outcomes:
        if outcome["status"] not in {"processed", "failed"}:
            raise ValueError("invalid extraction outcome")
        if not isinstance(outcome.get("encoder_token_max_observed"), int) or not 0 <= outcome["encoder_token_max_observed"] <= config.encoder_max_tokens:
            raise ValueError("invalid observed encoder length")
        if outcome["candidates"] != sum(c.episode_id == outcome["episode_id"] for c in candidates) or \
                outcome["mentions"] != sum(m.episode_id == outcome["episode_id"] for m in mentions):
            raise ValueError("outcome counts differ from extraction records")
    counts = {"selected": len(bundles), "processed": sum(o["status"] == "processed" for o in outcomes),
              "failed": sum(o["status"] == "failed" for o in outcomes), "windows": len(windows),
              "candidates": len(candidates), "accepted_mentions": len(mentions),
              "below_threshold": sum(c.decision == "below_threshold" for c in candidates),
              "duplicates": sum(c.decision == "duplicate" for c in candidates),
              "overlaps": sum(c.decision == "overlap" for c in candidates)}
    if counts != manifest["counts"] or manifest["status"] != ("failed" if counts["failed"] else "complete"):
        raise ValueError("extraction manifest counts or status differ")
    if manifest["model"].get("encoder_max_tokens") != config.encoder_max_tokens or \
            manifest["model"].get("encoder_token_max_observed") != max(
                (o["encoder_token_max_observed"] for o in outcomes), default=0):
        raise ValueError("manifest encoder budget differs from outcomes")
    if manifest.get("evaluation_packet"):
        packet = workspace_path(root, manifest["evaluation_packet"])
        expected_predictions = predictions_for_packet(packet, mentions, outcomes)
        predictions = [Prediction.model_validate(r) for r in read_jsonl(output / "predictions.jsonl")]
        if predictions != expected_predictions:
            raise ValueError("evaluation predictions differ from accepted mentions")
        evaluation = json.loads((output / "evaluation.json").read_bytes())
        if all(p.status == "complete" for p in predictions):
            expected_evaluation = score_records(packet, read_jsonl(packet / "annotations.jsonl"),
                [p.model_dump(mode="json") for p in predictions], json.loads((packet / "manifest.json").read_bytes())["split"])
            expected_evaluation.update({"status": "complete",
                "gold_sha256": sha256((packet / "annotations.jsonl").read_bytes()),
                "predictions_sha256": sha256((output / "predictions.jsonl").read_bytes())})
        else:
            expected_evaluation = {"schema_version": 1, "status": "not_scored", "reason": "failed sampled episode",
                "gold_sha256": sha256((packet / "annotations.jsonl").read_bytes()),
                "predictions_sha256": sha256((output / "predictions.jsonl").read_bytes())}
        if evaluation != expected_evaluation:
            raise ValueError("saved evaluation is not reproducible")
    return manifest
