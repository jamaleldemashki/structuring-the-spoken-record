"""Ingestion: reads only the transcripts listed in the pilot and writes them as a new snapshot."""

import importlib.metadata
import json
import platform
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from pydantic import ValidationError

from spoken_record import __version__
from spoken_record.adapters.base import sha256
from spoken_record.adapters.hop_whisper import HoPWhisperAdapter, MetadataIndex
from spoken_record.adapters.whisper_json import FileMetadata, WhisperJsonAdapter
from spoken_record.models import Episode, SourceBundle, SourceSegment, TranscriptVersion


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def workspace_path(root: Path, name: str) -> Path:
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("paths must be relative to the workspace without '..'")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("input path escapes workspace")
    return path


def index_rows(rows: list[dict]) -> dict[str, dict]:
    result = {}
    for row in rows:
        key = row["source_path"]
        if key in result:
            raise ValueError(f"duplicate audit source_path: {key}")
        result[key] = row
    return result


def dump_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def dump_jsonl(path: Path, rows: list) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            data = row.model_dump(mode="json") if hasattr(row, "model_dump") else row
            handle.write(json.dumps(data, ensure_ascii=False, sort_keys=True) + "\n")


ADAPTERS = ("hop_whisper", "whisper_json")


def make_adapter(profile: dict, metadata_rows: list[dict]):
    """Each podcast has a profile that names its transcript format and folder."""
    if profile.get("adapter") == "hop_whisper":
        return HoPWhisperAdapter(MetadataIndex(metadata_rows, profile))
    if profile.get("adapter") == "whisper_json":
        return WhisperJsonAdapter(FileMetadata(metadata_rows))
    raise ValueError(f"unknown transcript adapter {profile.get('adapter')!r}; supported: {', '.join(ADAPTERS)}")


def ingest(root: Path, config_name: str, pilot_name: str, run_id: str | None = None) -> Path:
    """Ingest a pilot (a list of transcripts with checksums) into a new snapshot.

    The config names the corpus, its profile (transcript format and folder) and its metadata.
    A corpus audit and a quarantine list are optional. With an audit, only audited transcripts
    are read and their metadata join has to agree with it. Without one, the pilot checksums
    alone decide what gets read.
    """
    root = root.resolve()
    started = perf_counter()
    timestamp = datetime.now(timezone.utc)
    run_id = run_id or f"ingest-{timestamp:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]{0,95}", run_id):
        raise ValueError("run ID must contain only letters, numbers and dashes")
    config_path = workspace_path(root, config_name)
    pilot_path = workspace_path(root, pilot_name)
    config = json.loads(config_path.read_bytes())
    pilot = json.loads(pilot_path.read_bytes())
    profile_path = workspace_path(root, config["profile"])
    profile = json.loads(profile_path.read_bytes())
    if any(item.get("schema_version") != 1 for item in (config, pilot, profile)):
        raise ValueError("unsupported configuration schema version")
    if profile.get("adapter") not in ADAPTERS:
        raise ValueError(f"unknown transcript adapter {profile.get('adapter')!r}; supported: {', '.join(ADAPTERS)}")
    if not isinstance(config.get("corpus_id"), str) or not config["corpus_id"].strip():
        raise ValueError("corpus_id must be a nonempty namespace")
    optional = {key: workspace_path(root, config[key]) if config.get(key) else None
                for key in ("metadata", "audit", "quarantine")}
    metadata_path, audit_path, quarantine_path = optional["metadata"], optional["audit"], optional["quarantine"]
    require_metadata = config.get("require_metadata", True)
    if require_metadata and metadata_path is None:
        raise ValueError("metadata is required unless require_metadata is false")
    audited = index_rows(read_jsonl(audit_path)) if audit_path else None
    quarantined = index_rows(read_jsonl(quarantine_path)) if quarantine_path else {}
    if audited is not None and set(audited) & set(quarantined):
        raise ValueError("audit and quarantine overlap")
    entries = pilot.get("episodes", [])
    if not entries or len({e["source_path"] for e in entries}) != len(entries):
        raise ValueError("pilot must contain nonempty unique source paths")
    source_root = workspace_path(root, profile["transcripts_dir"])
    selected = []
    for entry in entries:
        path = workspace_path(root, entry["source_path"])
        if not path.is_relative_to(source_root) or path.suffix != ".json":
            raise ValueError("selected transcript must be a JSON file in the configured source directory")
        if not re.fullmatch(r"[0-9a-f]{64}", entry.get("source_sha256", "")):
            raise ValueError("each pilot entry requires a frozen SHA-256")
        selected.append((entry, path))
    if len({path for _, path in selected}) != len(selected):
        raise ValueError("multiple pilot entries resolve to the same source")

    # The output folder is fixed, so no option can make us write into the source data.
    run_root = root / "data" / "prototype1" / "runs"
    for path in (run_root, *run_root.parents):
        if path == root:
            break
        if path.is_symlink():
            raise ValueError("output ancestors must not be symlinks")
    output = run_root / run_id
    if source_root.is_relative_to(run_root) or run_root.is_relative_to(source_root):
        raise ValueError("source and output directories must not overlap")
    adapter = make_adapter(profile, read_jsonl(metadata_path) if metadata_path else [])
    support = [p for p in (config_path, pilot_path, profile_path, metadata_path, audit_path, quarantine_path) if p]
    support_hashes = {str(p.relative_to(root)): sha256(p.read_bytes()) for p in support}
    code_root = Path(__file__).parent
    code_hashes = {str(p.relative_to(code_root)): sha256(p.read_bytes()) for p in sorted(code_root.rglob("*.py"))}
    bundles, outcomes, inputs = [], [], []
    for entry, path in selected:
        name = entry["source_path"]
        if name in quarantined:
            outcomes.append({"source_path": name, "status": "excluded", "reason": quarantined[name]["reason_code"]})
            continue
        if audited is not None and name not in audited:
            outcomes.append({"source_path": name, "status": "failed", "reason": "not_in_accepted_audit"})
            continue
        try:
            data = path.read_bytes()
            digest = sha256(data)
            inputs.append({"source_path": name, "sha256": digest})
            if digest != entry["source_sha256"] or (audited is not None and digest != audited[name]["source_sha256"]):
                raise ValueError("stale_source: pilot/audit checksum differs from source")
            bundle = adapter.ingest(
                data, corpus_id=config["corpus_id"], source_key=path.relative_to(source_root).as_posix(),
                source_path=name, run_id=run_id,
            )
            if require_metadata and bundle.episode.metadata_status != "matched":
                raise ValueError("metadata_join_unresolved: accepted pilot requires a unique match")
            if audited is not None:
                audit = audited[name]
                if adapter.audit_differs(bundle, audit):
                    raise ValueError("stale_metadata: join differs from accepted audit")
                bundle.episode.legacy_audit_id = audit.get("episode_id")
            bundles.append(bundle)
            outcomes.append({"source_path": name, "status": "processed", "reason": None})
        except (ValueError, OSError, KeyError, TypeError) as exc:
            # Pydantic errors contain the input, i.e. transcript text. Keep it out of the report.
            reason = "invalid_source_record" if isinstance(exc, ValidationError) else str(exc)
            outcomes.append({"source_path": name, "status": "failed", "reason": reason})
    # Hash the sources again after reading, to catch files that changed in the meantime.
    for item in inputs:
        if sha256(workspace_path(root, item["source_path"]).read_bytes()) != item["sha256"]:
            raise ValueError("source changed during ingestion; no snapshot published")
    for path in support:
        if sha256(path.read_bytes()) != support_hashes[str(path.relative_to(root))]:
            raise ValueError("configuration/audit changed during ingestion")
    if len({b.episode.episode_id for b in bundles}) != len(bundles):
        raise ValueError("episode identity collision")

    output.mkdir(parents=True, exist_ok=False)
    artifacts = {
        "episodes.jsonl": [b.episode for b in bundles],
        "transcripts.jsonl": [b.transcript for b in bundles],
        "segments.jsonl": [s for b in bundles for s in b.segments],
        "outcomes.jsonl": outcomes,
    }
    for filename, rows in artifacts.items():
        dump_jsonl(output / filename, rows)
    counts = {status: sum(o["status"] == status for o in outcomes) for status in ("processed", "excluded", "failed")}
    counts.update(selected=len(entries), segments=sum(len(b.segments) for b in bundles),
                  characters=sum(len(b.transcript.text) for b in bundles),
                  words=sum(len(b.transcript.text.split()) for b in bundles))
    manifest = {
        "schema_version": 1, "stage": "ingest", "run_id": run_id,
        "status": "failed" if counts["failed"] else "complete",
        "package_version": __version__, "started_utc": timestamp.isoformat(),
        "elapsed_seconds": round(perf_counter() - started, 4),
        "python": platform.python_version(), "machine": platform.machine(),
        "dependencies": {name: importlib.metadata.version(name) for name in ("pydantic", "pydantic-core")},
        "inputs": inputs, "support_sha256": support_hashes, "code_sha256": code_hashes,
        "counts": counts, "source_hashes_verified_after_read": True,
        "artifact_sha256": {name: sha256((output / name).read_bytes()) for name in artifacts},
        "model": None,
    }
    # The manifest is written last. A folder without one is an unfinished run.
    dump_json(output / "manifest.json", manifest)
    validate_run(output)
    return output


def validate_run(output: Path) -> dict:
    manifest = json.loads((output / "manifest.json").read_bytes())
    if manifest.get("schema_version") != 1 or manifest.get("stage") != "ingest":
        raise ValueError("unsupported run schema or stage")
    names = {"episodes.jsonl", "transcripts.jsonl", "segments.jsonl", "outcomes.jsonl"}
    if set(manifest["artifact_sha256"]) != names:
        raise ValueError("unexpected artifact set")
    for name, digest in manifest["artifact_sha256"].items():
        if sha256((output / name).read_bytes()) != digest:
            raise ValueError(f"artifact checksum mismatch: {name}")
    episodes = [Episode.model_validate(x) for x in read_jsonl(output / "episodes.jsonl")]
    transcripts = [TranscriptVersion.model_validate(x) for x in read_jsonl(output / "transcripts.jsonl")]
    segments = [SourceSegment.model_validate(x) for x in read_jsonl(output / "segments.jsonl")]
    for rows, key in ((episodes, "episode_id"), (transcripts, "transcript_version_id"), (segments, "segment_id")):
        if len({getattr(row, key) for row in rows}) != len(rows):
            raise ValueError("duplicate record IDs")
        if any(row.created_by_run_id != manifest["run_id"] for row in rows):
            raise ValueError("record run ID differs from manifest")
    by_episode = {t.episode_id: t for t in transcripts}
    if len(by_episode) != len(transcripts) or set(by_episode) != {e.episode_id for e in episodes}:
        raise ValueError("orphan or multiple transcripts")
    by_transcript = defaultdict(list)
    for segment in segments:
        by_transcript[segment.transcript_version_id].append(segment)
    if set(by_transcript) - {t.transcript_version_id for t in transcripts}:
        raise ValueError("orphan segments")
    for episode in episodes:
        transcript = by_episode[episode.episode_id]
        SourceBundle(episode=episode, transcript=transcript, segments=by_transcript[transcript.transcript_version_id])
    outcomes = read_jsonl(output / "outcomes.jsonl")
    if len({o["source_path"] for o in outcomes}) != len(outcomes):
        raise ValueError("duplicate outcomes")
    if any(o["status"] not in {"processed", "excluded", "failed"} for o in outcomes):
        raise ValueError("unknown outcome status")
    processed_paths = {o["source_path"] for o in outcomes if o["status"] == "processed"}
    if processed_paths != {e.source_path for e in episodes}:
        raise ValueError("outcomes do not match episodes")
    expected = {status: sum(o["status"] == status for o in outcomes) for status in ("processed", "excluded", "failed")}
    expected.update(selected=len(outcomes), segments=len(segments), characters=sum(len(t.text) for t in transcripts),
                    words=sum(len(t.text.split()) for t in transcripts))
    if expected != manifest["counts"]:
        raise ValueError("manifest counts do not match artifacts")
    if manifest["status"] != ("failed" if expected["failed"] else "complete"):
        raise ValueError("manifest status does not match outcomes")
    return manifest
