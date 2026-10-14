import json
import subprocess
import sys
from pathlib import Path

import pytest

from test_ingestion import workspace, transcript
from spoken_record.adapters.base import sha256
from spoken_record.evaluation import TYPES, prepare_packet, score
from spoken_record.ingestion import dump_json, dump_jsonl, ingest, read_jsonl
from spoken_record.passage_models import SegmentationConfig
from spoken_record.segmentation import segment_run


def write_rows(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


@pytest.fixture
def scoring_packet(tmp_path):
    text = "😀 Pāṇini studies free will."
    digest = sha256(text.encode())
    dump_jsonl(tmp_path / "sample.jsonl", [{"passage_id": "p1", "episode_id": "e1", "text": text,
               "text_sha256": digest, "split": "development"}])
    dump_json(tmp_path / "manifest.json", {"schema_version": 1, "stage": "annotation_packet", "split": "development",
              "types": list(TYPES), "passage_count": 1, "episode_count": 1,
              "sample_sha256": sha256((tmp_path / "sample.jsonl").read_bytes())})
    gold = [{"passage_id": "p1", "text_sha256": digest, "status": "reviewed", "reviewer": "synthetic-test",
             "exhaustive": True, "mentions": [{"start": 2, "end": 8, "text": "Pāṇini", "type": "Person"},
                                                 {"start": 17, "end": 26, "text": "free will", "type": "Concept"}]}]
    predictions = [{"passage_id": "p1", "text_sha256": digest, "status": "complete", "mentions": gold[0]["mentions"].copy()}]
    return tmp_path, gold, predictions


def evaluate(packet, gold, predictions, split="development"):
    write_rows(packet / "gold.jsonl", gold)
    write_rows(packet / "predictions.jsonl", predictions)
    return score(packet, packet / "gold.jsonl", packet / "predictions.jsonl", split)


def test_exact_unicode_and_concept_metrics(scoring_packet):
    packet, gold, predictions = scoring_packet
    result = evaluate(packet, gold, predictions)
    assert result["micro"]["f1"] == 1
    assert result["per_type"]["Concept"]["tp"] == 1
    assert result["per_type"]["Place"]["f1"] is None
    assert result["macro_types"] == ["Person", "Concept"]


def test_wrong_type_and_boundary_each_count_fp_fn(scoring_packet):
    packet, gold, predictions = scoring_packet
    predictions[0]["mentions"] = [{"start": 2, "end": 8, "text": "Pāṇini", "type": "Work"},
                                    {"start": 17, "end": 21, "text": "free", "type": "Concept"}]
    result = evaluate(packet, gold, predictions)
    assert [result["micro"][k] for k in ("tp", "fp", "fn")] == [0, 2, 2]
    assert result["per_type"]["Work"]["fp"] == 1
    assert result["macro_f1_gold_supported_types"] == 0


@pytest.mark.parametrize("change", ["unreviewed", "nonexhaustive", "reviewer", "uncertain", "missing", "duplicate", "failed", "hash", "slice", "overlap", "label"])
def test_reject_incomplete_or_invalid_evaluation(scoring_packet, change):
    packet, gold, predictions = scoring_packet
    if change == "unreviewed": gold[0]["status"] = "unreviewed"
    if change == "nonexhaustive": gold[0]["exhaustive"] = False
    if change == "reviewer": gold[0]["reviewer"] = " "
    if change == "uncertain": gold[0]["unresolved_notes"] = ["Ambiguous type"]
    if change == "missing": predictions.clear()
    if change == "duplicate": predictions[0]["mentions"].append(predictions[0]["mentions"][0])
    if change == "failed": predictions[0]["status"] = "failed"
    if change == "hash": predictions[0]["text_sha256"] = "stale"
    if change == "slice": predictions[0]["mentions"] = [{"start": 3, "end": 9, "text": "Pāṇini", "type": "Person"}]
    if change == "overlap": predictions[0]["mentions"].append({"start": 2, "end": 8, "text": "Pāṇini", "type": "Work"})
    if change == "label": predictions[0]["mentions"] = [{"start": 2, "end": 8, "text": "Pāṇini", "type": "Organization"}]
    with pytest.raises(ValueError): evaluate(packet, gold, predictions)


def test_reviewed_empty_is_valid_but_not_perfect(scoring_packet):
    packet, gold, predictions = scoring_packet
    gold[0]["mentions"] = []
    predictions[0]["mentions"] = []
    result = evaluate(packet, gold, predictions)
    assert result["micro"]["f1"] is None
    assert result["macro_f1_gold_supported_types"] is None


def test_split_and_sample_corruption_rejected(scoring_packet):
    packet, gold, predictions = scoring_packet
    with pytest.raises(ValueError, match="split"): evaluate(packet, gold, predictions, "test")
    with (packet / "sample.jsonl").open("a") as f: f.write("\n")
    with pytest.raises(ValueError, match="checksum"): evaluate(packet, gold, predictions)


def test_prepare_real_contract_and_cli_without_mutating_source(workspace):
    root, name = workspace
    data = transcript(["Pāṇini studies words. " * 18])
    (root / name).write_bytes(data)
    pilot = json.loads((root / "pilot.json").read_bytes())
    pilot["split"] = "development"
    pilot["episodes"][0]["source_sha256"] = sha256(data)
    dump_json(root / "pilot.json", pilot)
    audit = read_jsonl(root / "audit.jsonl")
    audit[0]["source_sha256"] = sha256(data)
    write_rows(root / "audit.jsonl", audit)
    source = ingest(root, "config.json", "pilot.json", "source")
    config = SegmentationConfig(passage_min_words=3, passage_target_words=6, passage_max_words=9,
                                 window_max_tokens=8, window_overlap_tokens=2)
    dump_json(root / "seg.json", config.model_dump())
    parent = segment_run(root, str(source.relative_to(root)), "seg.json", run_id="passages")
    hashes = {p.name: sha256(p.read_bytes()) for p in parent.iterdir()}
    packet = prepare_packet(root, str(parent.relative_to(root)), "pilot.json", "review")
    assert len(read_jsonl(packet / "sample.jsonl")) == 2
    assert all(r["status"] == "unreviewed" for r in read_jsonl(packet / "annotations.jsonl"))
    assert hashes == {p.name: sha256(p.read_bytes()) for p in parent.iterdir()}
    assert (root / name).read_bytes() == data
    with pytest.raises(FileExistsError): prepare_packet(root, str(parent.relative_to(root)), "pilot.json", "review")
    result = subprocess.run([sys.executable, "-m", "spoken_record.cli", "prepare-annotations", "--root", str(root),
        "--input-run", str(parent.relative_to(root)), "--pilot", "pilot.json", "--packet-id", "cli-review"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "unreviewed"


def test_frozen_pilot_split_and_initial_labels():
    root = Path(__file__).resolve().parents[1]
    dev, test = [json.loads((root / f"config/pilots/evaluation-{s}.json").read_bytes())["episodes"]
                 for s in ("development", "test")]
    assert len(dev) == len(test) == 12
    assert len({e["content_era"] for e in dev}) == len({e["content_era"] for e in test}) == 10
    assert not {e["source_path"] for e in dev} & {e["source_path"] for e in test}
    smoke = json.loads((root / "config/pilots/development-three.json").read_bytes())["episodes"]
    assert {e["source_path"] for e in smoke} <= {e["source_path"] for e in dev}
    assert set(json.loads((root / "config/entity_labels.development-calibrated.json").read_bytes())["labels"]) == set(TYPES)
