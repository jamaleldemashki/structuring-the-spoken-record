import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_ingestion import transcript, workspace
from spoken_record.adapters.base import sha256
from spoken_record.evaluation import TYPES
from spoken_record.extraction import (GLiNERWhitespaceBudget, TOKEN_PATTERN, extract_run,
                                      resolve_candidates, validate_run, verify_encoder_batch)
from spoken_record.ingestion import dump_json, ingest, read_jsonl
from spoken_record.passage_models import Passage, SegmentationConfig
from spoken_record.segmentation import segment_run


class WhitespaceTokenSplitter:
    def __call__(self, text):
        for match in TOKEN_PATTERN.finditer(text):
            yield match.group(), match.start(), match.end()


class FakeWordsSplitter:
    def __init__(self):
        self.splitter = WhitespaceTokenSplitter()

    def __call__(self, text):
        yield from self.splitter(text)


class FakeModel:
    def __init__(self, fail=False):
        self.config = SimpleNamespace(max_len=384)
        self.data_processor = SimpleNamespace(words_splitter=FakeWordsSplitter())
        self.model = SimpleNamespace(token_rep_layer=SimpleNamespace(bert_layer=SimpleNamespace(
            model=SimpleNamespace(config=SimpleNamespace(max_position_embeddings=512)))))
        self.fail = fail

    def prepare_model_inputs(self, texts, labels):
        tokens = [list(TOKEN_PATTERN.finditer(text)) for text in texts]
        lengths = [len(row) + 2 * len(labels) + 2 for row in tokens]
        attention = FakeMatrix([[1] * length for length in lengths])
        words = FakeMatrix([list(range(1, len(row) + 1)) for row in tokens])
        return {"attention_mask": attention, "words_mask": words}, {"seq_length": [len(row) for row in tokens]}

    def batch_predict_entities(self, texts, labels, **kwargs):
        if self.fail:
            raise RuntimeError("synthetic failure")
        rows = []
        for text in texts:
            found = []
            for needle, label, confidence in (("Pāṇini", "person", .9), ("free will", "philosophical concept", .8),
                                               ("words", "philosophical concept", .3)):
                start = text.find(needle)
                while start >= 0:
                    found.append({"start": start, "end": start + len(needle), "text": needle,
                                  "label": label, "score": confidence})
                    start = text.find(needle, start + len(needle))
            rows.append(found)
        return rows


class FakeNumber:
    def __init__(self, value): self.value = value
    def item(self): return self.value


class FakeVector(list):
    def max(self): return FakeNumber(max(self, default=0))
    def tolist(self): return list(self)


class FakeMatrix(list):
    def __init__(self, rows): super().__init__(FakeVector(row) for row in rows)
    def sum(self, dim):
        assert dim == 1
        return FakeVector([sum(row) for row in self])


def write_rows(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def configured_workspace(workspace):
    root, name = workspace
    data = transcript(["Pāṇini studies free will and words. " * 45])
    (root / name).write_bytes(data)
    pilot = json.loads((root / "pilot.json").read_bytes())
    pilot["episodes"][0]["source_sha256"] = sha256(data)
    dump_json(root / "pilot.json", pilot)
    if (root / "audit.jsonl").exists():  # Corpora without an accepted audit rely on the pilot checksums.
        audit = read_jsonl(root / "audit.jsonl")
        audit[0]["source_sha256"] = sha256(data)
        write_rows(root / "audit.jsonl", audit)
    source = ingest(root, "config.json", "pilot.json", "source")
    segmentation = SegmentationConfig(passage_min_words=12, passage_target_words=18, passage_max_words=24,
                                      window_max_tokens=20, window_overlap_tokens=5)
    dump_json(root / "segmentation.json", segmentation.model_dump())
    passages = segment_run(root, str(source.relative_to(root)), "segmentation.json", run_id="passages")
    labels = {"schema_version": 2, "status": "test", "decoding": "flat",
              "labels": {"Person": "person", "Work": "written work", "School": "school",
                         "Concept": "philosophical concept", "Place": "location", "Period": "period",
                         "Language": "language", "Group": "group"},
              "provisional_threshold": .5, "changelog": []}
    dump_json(root / "labels.json", labels)
    model = {"schema_version": 1, "task": "mention_detection", "framework": "gliner",
             "framework_version": "0.2.21", "repo_id": "synthetic/model", "revision": "a" * 40,
             "revision_resolved_at": "2026-09-13", "device": "cpu", "labels_config": "labels.json",
             "words_splitter": "whitespace", "model_max_words": 384, "window_max_tokens": 20,
             "encoder_max_tokens": 512,
             "window_overlap_tokens": 5, "candidate_floor": .25, "batch_size": 3,
             "smoke_test": {"status": "synthetic"}, "notes": []}
    dump_json(root / "model.test-calibrated.json", model)
    return root, name, data, passages


def test_checkpoint_windows_offsets_candidates_and_validation(workspace):
    root, name, source_bytes, parent = configured_workspace(workspace)
    parent_hashes = {p.name: sha256(p.read_bytes()) for p in parent.iterdir()}
    output = extract_run(root, str(parent.relative_to(root)), "model.test-calibrated.json", run_id="entities",
                         model_loader=lambda _: FakeModel())
    manifest = validate_run(output)
    assert manifest["status"] == "complete"
    assert manifest["counts"]["processed"] == 1
    assert manifest["counts"]["accepted_mentions"] > 0
    assert manifest["counts"]["below_threshold"] > 0
    assert manifest["counts"]["duplicates"] > 0
    assert all(w["budget_verified_for_model"] for w in read_jsonl(output / "model_windows.jsonl"))
    transcript_text = read_jsonl(root / "data/prototype1/runs/source/transcripts.jsonl")[0]["text"]
    assert all(transcript_text[m["char_start"]:m["char_end"]] == m["text"]
               for m in read_jsonl(output / "mentions.jsonl"))
    assert parent_hashes == {p.name: sha256(p.read_bytes()) for p in parent.iterdir()}
    assert (root / name).read_bytes() == source_bytes
    relative = output.relative_to(root)
    previous = Path.cwd()
    try:
        __import__("os").chdir(root)
        assert validate_run(relative)["status"] == "complete"
    finally:
        __import__("os").chdir(previous)
    with pytest.raises(FileExistsError):
        extract_run(root, str(parent.relative_to(root)), "model.test-calibrated.json", run_id="entities",
                    model_loader=lambda _: FakeModel())


def test_episode_failure_is_explicit_and_not_empty_success(workspace):
    root, _, _, parent = configured_workspace(workspace)
    output = extract_run(root, str(parent.relative_to(root)), "model.test-calibrated.json", run_id="failed-entities",
                         model_loader=lambda _: FakeModel(fail=True))
    manifest = validate_run(output)
    assert manifest["status"] == "failed"
    assert manifest["counts"]["processed"] == 0 and manifest["counts"]["failed"] == 1
    assert read_jsonl(output / "mentions.jsonl") == []
    assert read_jsonl(output / "outcomes.jsonl")[0]["reason"] == "RuntimeError: inference failed"


def test_flat_resolution_tiebreaks_and_exact_duplicate_windows():
    passage = Passage(created_by_run_id="p", passage_id="p1", episode_id="e", transcript_version_id="t",
        index=0, char_start=0, char_end=20, text="abcdefghijklmnopqrst", word_count=1,
        timing_precision="unavailable", source_segment_ids=[], method="token_fallback",
        segmentation_fingerprint="x")
    base = {"created_by_run_id": "r", "episode_id": "e", "transcript_version_id": "t",
            "model_label": "person"}
    raw = [
        {**base, "candidate_id": "c1", "window_id": "w1", "char_start": 2, "char_end": 6,
         "text": "cdef", "type": "Person", "score": .8},
        {**base, "candidate_id": "c2", "window_id": "w2", "char_start": 2, "char_end": 6,
         "text": "cdef", "type": "Person", "score": .7},
        {**base, "candidate_id": "c3", "window_id": "w1", "char_start": 3, "char_end": 7,
         "text": "defg", "type": "Person", "score": .75},
        {**base, "candidate_id": "c4", "window_id": "w1", "char_start": 10, "char_end": 12,
         "text": "kl", "type": "Person", "score": .4},
    ]
    candidates, mentions = resolve_candidates(raw, .5, "fingerprint", [passage])
    decisions = {c.candidate_id: c.decision for c in candidates}
    assert decisions == {"c1": "accepted", "c2": "duplicate", "c3": "overlap", "c4": "below_threshold"}
    assert len(mentions) == 1 and mentions[0].source_window_ids == ["w1", "w2"]


def test_configuration_inventory_is_eight_types():
    root = Path(__file__).resolve().parents[1]
    labels = json.loads((root / "config/entity_labels.development-calibrated.json").read_bytes())
    model = json.loads((root / "config/model.development-calibrated.json").read_bytes())
    assert set(labels["labels"]) == set(TYPES)
    assert model["window_max_tokens"] <= model["model_max_words"]
    assert model["candidate_floor"] <= labels["provisional_threshold"]


def test_encoder_prompt_budget_is_checked_before_inference():
    model = FakeModel()
    budget = GLiNERWhitespaceBudget("a" * 40)
    config = SimpleNamespace(encoder_max_tokens=25)
    with pytest.raises(ValueError, match="encoder token limit"):
        verify_encoder_batch(model, [" ".join(["word"] * 10)], list(TYPES), config, budget)
