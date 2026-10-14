import json
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from spoken_record.adapters.base import sha256, stable_id
from spoken_record.adapters.hop_whisper import HoPWhisperAdapter, MetadataIndex
from spoken_record.adapters.plain_text import PlainTextAdapter
from spoken_record.ingestion import dump_json, ingest, read_jsonl, validate_run
from spoken_record.models import SourceBundle

PROFILE = {
    "schema_version": 1, "adapter": "hop_whisper", "transcripts_dir": "HoP/01_whisper",
    "known_shows": ["HoP", "HAP", "HPI", "HPC", "GPW"],
    "series_to_group": {"India": "HPI", "Africana": "HAP", "China": "HPC"},
    "main_series": ["Medieval", "Classical Greek"],
}


def transcript(parts=None):
    parts = parts if parts is not None else [" Pāṇini studied language.\r\n", "Thomas discussed café life. 😀"]
    return json.dumps({"text": "".join(parts), "language": "en", "segments": [
        {"id": i, "text": text, "start": i * 3.0, "end": (i + 1) * 3.0,
         "avg_logprob": -0.12, "no_speech_prob": 0.01}
        for i, text in enumerate(parts)
    ]}, ensure_ascii=False).encode()


def parse(data=None, *, corpus="hop", run="run-a", records=None):
    adapter = HoPWhisperAdapter(MetadataIndex(records or [], PROFILE))
    return adapter.ingest(data or transcript(), corpus_id=corpus, source_key="HoP001.json",
                          source_path="HoP/01_whisper/HoP001.json", run_id=run)


def test_unicode_offsets_roundtrip_and_namespace():
    bundle = parse()
    assert SourceBundle.model_validate_json(bundle.model_dump_json()) == bundle
    for segment in bundle.segments:
        assert bundle.transcript.text[segment.char_start:segment.char_end] == segment.text
    assert "\r\n" in bundle.transcript.text
    assert bundle.episode.episode_id == parse(run="another-run").episode.episode_id
    assert bundle.transcript.transcript_version_id == parse(run="another-run").transcript.transcript_version_id
    assert bundle.episode.episode_id != parse(corpus="other-podcast").episode.episode_id
    changed = parse(transcript(["Edited source."]))
    assert bundle.episode.episode_id == changed.episode.episode_id
    assert bundle.transcript.transcript_version_id != changed.transcript.transcript_version_id
    assert stable_id("ep", "ab", "c") != stable_id("ep", "a", "bc")


def test_source_alignment_is_not_guessed():
    data = json.loads(transcript())
    data["text"] = data["text"].strip()
    with pytest.raises(ValueError, match="source_alignment_error"):
        parse(json.dumps(data).encode())


@pytest.mark.parametrize("change", ["backward", "reversed", "nan", "partial_time"])
def test_bad_timestamps_rejected(change):
    data = json.loads(transcript())
    if change == "backward":
        data["segments"][0]["start"] = 2
        data["segments"][1]["start"] = 1
    elif change == "reversed":
        data["segments"][1]["end"] = 2
    elif change == "nan":
        data["segments"][0]["start"] = float("nan")
    else:
        del data["segments"][0]["start"]
    with pytest.raises(ValidationError):
        parse(json.dumps(data).encode())


def test_missing_metadata_is_valid_at_adapter_boundary():
    episode = parse().episode
    assert episode.metadata_status == "missing"
    assert episode.source_url is None
    assert episode.facets["content_era"] is None


def test_exact_and_numbered_metadata_with_namespaces_and_suffix():
    records = [
        {"filename": "HoP001.json", "title": "Exact", "series": "Medieval"},
        {"nav_title": "46. Main", "series": "Classical Greek"},
        {"nav_title": "46. India", "series": "India"},
        {"nav_title": "300. Main", "series": "Medieval"},
        {"nav_title": "300b. Bonus", "series": "Medieval"},
        {"nav_title": "249. Main", "series": "Medieval"},
    ]
    index = MetadataIndex(records, PROFILE)
    assert index.match("HoP001.json")[2] == "tier1_filename"
    assert index.match("08HPI46-Title.json")[0]["series"] == "India"
    assert index.match("OfhhekHoP_046_Title.json")[0]["series"] == "Classical Greek"
    assert index.match("HoP300b-Title.json")[0]["nav_title"].startswith("300b")
    assert index.match("HoP249Title.json")[0]["nav_title"].startswith("249")
    assert index.match("GPW46-Title.json")[1] == "missing"
    assert MetadataIndex(records + [records[0]], PROFILE).match("HoP001.json")[1] == "ambiguous"
    assert MetadataIndex(records + [records[2]], PROFILE).match("HPI46.json")[1] == "ambiguous"


def test_plain_text_uses_same_contract_without_invented_times():
    data = "Pāṇini\r\n\r\nDiscussed language. 😀".encode()
    bundle = PlainTextAdapter().ingest(data, corpus_id="another", source_key="episode.txt",
                                      source_path="episode.txt", run_id="r")
    assert bundle.transcript.text.encode() == data
    assert bundle.segments == []
    assert bundle.episode.language == "und"
    assert bundle.episode.audio_url is None


@pytest.fixture
def workspace(tmp_path):
    source = "HoP/01_whisper/HoP001.json"
    (tmp_path / source).parent.mkdir(parents=True)
    (tmp_path / source).write_bytes(transcript())
    digest = sha256(transcript())
    dump_json(tmp_path / "profile.json", PROFILE)
    dump_json(tmp_path / "config.json", {"schema_version": 1, "corpus_id": "hop", "profile": "profile.json",
              "metadata": "meta.jsonl", "audit": "audit.jsonl", "quarantine": "quarantine.jsonl"})
    dump_json(tmp_path / "pilot.json", {"schema_version": 1, "episodes": [{"source_path": source, "source_sha256": digest}]})
    (tmp_path / "meta.jsonl").write_text(json.dumps({"filename": "HoP001.json", "title": "Synthetic", "series": "Medieval"}) + "\n")
    (tmp_path / "audit.jsonl").write_text(json.dumps({"source_path": source, "source_sha256": digest,
        "episode_id": "ep_legacy", "join_tier": "tier1_filename", "content_era": "Medieval"}) + "\n")
    (tmp_path / "quarantine.jsonl").write_text("")
    return tmp_path, source


def test_snapshot_integrity_determinism_and_no_overwrite(workspace):
    root, source = workspace
    original = (root / source).read_bytes()
    first = ingest(root, "config.json", "pilot.json", "first")
    second = ingest(root, "config.json", "pilot.json", "second")
    assert (root / source).read_bytes() == original
    assert validate_run(first)["counts"]["processed"] == 1
    for filename in ("episodes.jsonl", "transcripts.jsonl", "segments.jsonl"):
        a, b = read_jsonl(first / filename), read_jsonl(second / filename)
        for record in a + b:
            record.pop("created_by_run_id")
        assert a == b
    with pytest.raises(FileExistsError):
        ingest(root, "config.json", "pilot.json", "first")


def test_stale_source_is_reported_without_ingesting(workspace):
    root, source = workspace
    (root / source).write_bytes(transcript(["Changed."]))
    output = ingest(root, "config.json", "pilot.json", "stale")
    manifest = validate_run(output)
    assert manifest["status"] == "failed"
    assert manifest["counts"]["processed"] == 0
    assert "stale_source" in read_jsonl(output / "outcomes.jsonl")[0]["reason"]


def test_quarantine_is_not_read(workspace):
    root, source = workspace
    (root / "audit.jsonl").write_text("")
    (root / "quarantine.jsonl").write_text(json.dumps({"source_path": source, "reason_code": "short"}) + "\n")
    (root / source).unlink()  # A quarantined file need not even exist for exclusion to work.
    output = ingest(root, "config.json", "pilot.json", "quarantine")
    manifest = validate_run(output)
    assert manifest["counts"]["excluded"] == 1
    assert manifest["inputs"] == []


def test_tampering_and_orphan_references_are_detected(workspace):
    root, _ = workspace
    output = ingest(root, "config.json", "pilot.json", "tamper")
    segments = read_jsonl(output / "segments.jsonl")
    segments[0]["transcript_version_id"] = "wrong-id"
    (output / "segments.jsonl").write_text("\n".join(json.dumps(x) for x in segments) + "\n")
    with pytest.raises(ValueError, match="checksum"):
        validate_run(output)
    manifest = json.loads((output / "manifest.json").read_bytes())
    manifest["artifact_sha256"]["segments.jsonl"] = sha256((output / "segments.jsonl").read_bytes())
    dump_json(output / "manifest.json", manifest)
    with pytest.raises(ValueError, match="orphan"):
        validate_run(output)


def test_output_symlinks_and_unsafe_run_ids_are_rejected(workspace):
    root, _ = workspace
    with pytest.raises(ValueError, match="run ID"):
        ingest(root, "config.json", "pilot.json", "../../HoP")
    (root / "data").mkdir()
    (root / "data/prototype1").symlink_to(root / "HoP", target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        ingest(root, "config.json", "pilot.json", "unsafe")


def test_cli_runs_in_fresh_process_and_returns_failure_for_stale_input(workspace):
    root, source = workspace
    command = [sys.executable, "-m", "spoken_record.cli", "ingest", "--root", str(root),
               "--config", "config.json", "--pilot", "pilot.json"]
    result = subprocess.run(command + ["--run-id", "cli-ok"], cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["counts"]["processed"] == 1
    (root / source).write_bytes(transcript(["Changed."]))
    result = subprocess.run(command + ["--run-id", "cli-failed"], cwd=root, capture_output=True, text=True)
    assert result.returncode == 1, result.stderr
    assert json.loads(result.stdout)["counts"]["failed"] == 1


def test_duplicate_pilot_and_stale_metadata(workspace):
    root, _ = workspace
    pilot = json.loads((root / "pilot.json").read_bytes())
    pilot["episodes"].append(pilot["episodes"][0])
    dump_json(root / "duplicate.json", pilot)
    with pytest.raises(ValueError, match="unique"):
        ingest(root, "config.json", "duplicate.json", "dup")
    (root / "meta.jsonl").write_text(json.dumps({"filename": "HoP001.json", "series": "India"}) + "\n")
    output = ingest(root, "config.json", "pilot.json", "changed-meta")
    assert validate_run(output)["counts"]["failed"] == 1
    assert "stale_metadata" in read_jsonl(output / "outcomes.jsonl")[0]["reason"]
