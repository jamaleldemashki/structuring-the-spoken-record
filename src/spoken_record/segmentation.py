"""Segmentation stage (passages and windows), and the `validate` dispatch for every run type.
The ingestion snapshot itself is never changed.
"""

import importlib.metadata
import json
import platform
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from spoken_record import __version__
from spoken_record.adapters.base import sha256, stable_id
from spoken_record.ingestion import dump_json, dump_jsonl, read_jsonl, validate_run as validate_ingest, workspace_path
from spoken_record.models import Episode, SourceBundle, SourceSegment, TranscriptVersion
from spoken_record.passage_models import ModelWindow, Passage, ReviewedExclusion, SegmentationConfig
from spoken_record.passages import build_passages, evidence_segments, time_bounds
from spoken_record.windows import WhitespaceBudget, build_windows


def load_sources(path: Path) -> list[SourceBundle]:
    manifest = validate_ingest(path)
    if manifest["status"] != "complete" or not manifest["counts"]["processed"]:
        raise ValueError("segmentation requires a complete, nonempty ingestion snapshot")
    transcripts = {x["episode_id"]: TranscriptVersion.model_validate(x) for x in read_jsonl(path / "transcripts.jsonl")}
    segments = defaultdict(list)
    for row in read_jsonl(path / "segments.jsonl"):
        segment = SourceSegment.model_validate(row)
        segments[segment.transcript_version_id].append(segment)
    return [SourceBundle(episode=Episode.model_validate(e), transcript=transcripts[e["episode_id"]],
                         segments=segments[transcripts[e["episode_id"]].transcript_version_id])
            for e in read_jsonl(path / "episodes.jsonl")]


def validate_records(bundles: list[SourceBundle], passages: list[Passage], windows: list[ModelWindow],
                     exclusions: list[ReviewedExclusion], run_id: str, config: SegmentationConfig) -> None:
    transcripts = {b.transcript.transcript_version_id: b for b in bundles}
    by_passage, by_window = defaultdict(list), defaultdict(list)
    for records, key in ((passages, "passage_id"), (windows, "window_id"), (exclusions, "exclusion_id")):
        if len({getattr(r, key) for r in records}) != len(records):
            raise ValueError("duplicate derived record IDs")
        if any(r.created_by_run_id != run_id or r.transcript_version_id not in transcripts for r in records):
            raise ValueError("derived record has invalid run or transcript reference")
    for passage in passages:
        bundle = transcripts[passage.transcript_version_id]
        if passage.episode_id != bundle.episode.episode_id:
            raise ValueError("passage refers to wrong episode")
        text = bundle.transcript.text
        if text[passage.char_start:passage.char_end] != passage.text:
            raise ValueError("passage does not reproduce source text")
        expected_segments = evidence_segments(bundle.segments, passage.char_start, passage.char_end)
        if passage.source_segment_ids != [s.segment_id for s in expected_segments]:
            raise ValueError("passage source segment references are incorrect")
        if (passage.start_s, passage.end_s) != time_bounds(expected_segments):
            raise ValueError("passage timestamps do not match source segments")
        if passage.passage_id != stable_id("p", passage.transcript_version_id, passage.segmentation_fingerprint,
                                           str(passage.char_start), str(passage.char_end)):
            raise ValueError("passage ID does not match version and range")
        by_passage[passage.transcript_version_id].append(passage)
    for bundle in bundles:
        records = by_passage[bundle.transcript.transcript_version_id]
        cursor = 0
        for index, passage in enumerate(records):
            if passage.index != index or passage.char_start != cursor:
                raise ValueError("passage coverage has a gap, overlap or wrong index")
            cursor = passage.char_end
        if cursor != len(bundle.transcript.text):
            raise ValueError("passages do not cover the full transcript")
    for window in windows:
        by_window[window.transcript_version_id].append(window)
    # These windows only use a rough whitespace budget. Extraction builds and checks its
    # own model windows instead of reusing these.
    for bundle in bundles:
        tv = bundle.transcript.transcript_version_id
        expected = build_windows(bundle, by_passage[tv], config, run_id)
        if by_window[tv] != expected:
            raise ValueError("windows fail exact budget, coverage, overlap or passage-reference checks")
    previous_exclusion_end = defaultdict(int)
    for exclusion in sorted(exclusions, key=lambda e: (e.transcript_version_id, e.char_start)):
        text = transcripts[exclusion.transcript_version_id].transcript.text
        if exclusion.char_end > len(text) or exclusion.char_start < previous_exclusion_end[exclusion.transcript_version_id]:
            raise ValueError("exclusion exceeds transcript or overlaps another exclusion")
        if sha256(text[exclusion.char_start:exclusion.char_end].encode()) != exclusion.source_text_sha256:
            raise ValueError("reviewed exclusion no longer matches its evidence")
        if exclusion.exclusion_id != stable_id("exc", exclusion.transcript_version_id,
                str(exclusion.char_start), str(exclusion.char_end), exclusion.source_text_sha256):
            raise ValueError("exclusion identity does not match evidence")
        previous_exclusion_end[exclusion.transcript_version_id] = exclusion.char_end


def counts_for(bundles: list[SourceBundle], passages: list[Passage], windows: list[ModelWindow],
               exclusions: list[ReviewedExclusion]) -> dict:
    return {"selected": len(bundles), "processed": len(bundles), "excluded": 0, "failed": 0,
            "passages": len(passages), "windows": len(windows), "credit_spans": len(exclusions),
            "characters": sum(len(p.text) for p in passages), "words": sum(p.word_count for p in passages),
            "untimed_passages": sum(p.start_s is None for p in passages),
            "methods": dict(Counter(p.method for p in passages))}


def reading_view(bundles: list[SourceBundle], passages: list[Passage], exclusions: list[ReviewedExclusion]) -> str:
    lines = ["# Passage reading view", "", "Exact source slices are shown in text blocks. Times are approximate source-segment bounds.",
             "Credits remain in the text; reviewed exclusions are marked separately. This is not an Obsidian export.", ""]
    for bundle in bundles:
        lines.extend([f"## {bundle.episode.title}", "", f"Source: `{bundle.episode.source_path}`", ""])
        for p in passages:
            if p.transcript_version_id != bundle.transcript.transcript_version_id:
                continue
            time = "timestamp unavailable" if p.start_s is None else f"{p.start_s:.2f}–{p.end_s:.2f} seconds"
            lines.extend([f"### Passage {p.index + 1} · {p.word_count} words · {time}", "",
                          f"ID: `{p.passage_id}` · offsets `[{p.char_start}, {p.char_end})` · {p.method}", ""])
            matches = [e for e in exclusions if e.transcript_version_id == p.transcript_version_id
                       and e.char_start < p.char_end and e.char_end > p.char_start]
            if matches:
                lines.extend(["Contains a reviewed production-credit span; only that span is flagged, not the whole passage.", ""])
            # The text can contain backticks, so the fence must be longer than any run of them.
            fence = "`" * max(3, 1 + max((len(m[0]) for m in re.finditer(r"`+", p.text)), default=0))
            lines.extend([fence + "text", p.text, fence, ""])
    return "\n".join(lines) + "\n"


def segment_run(root: Path, input_name: str, config_name: str, exclusions_name: str | None = None,
                run_id: str | None = None) -> Path:
    """Build passages and provisional windows from an ingestion run."""
    root = root.resolve()
    started = perf_counter()
    now = datetime.now(timezone.utc)
    run_id = run_id or f"segment-{now:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]{0,95}", run_id):
        raise ValueError("invalid run ID")
    run_root = root / "data/prototype1/runs"
    for path in (run_root, *run_root.parents):
        if path == root:
            break
        if path.is_symlink():
            raise ValueError("output ancestors must not be symlinks")
    parent = workspace_path(root, input_name)
    if parent.parent != run_root or parent.name == run_id:
        raise ValueError("input must be a sibling ingestion snapshot; output must be a new run")
    output = run_root / run_id
    if output.exists():
        raise FileExistsError(output)
    parent_digest = sha256((parent / "manifest.json").read_bytes())
    bundles = load_sources(parent)
    config_path = workspace_path(root, config_name)
    config_bytes = config_path.read_bytes()
    config = SegmentationConfig.model_validate_json(config_bytes)
    exclusions = []
    exclusions_digest = None
    if exclusions_name:
        exclusion_path = workspace_path(root, exclusions_name)
        exclusion_bytes = exclusion_path.read_bytes()
        exclusions_digest = sha256(exclusion_bytes)
        for line in exclusion_bytes.decode().splitlines():
            if line.strip():
                # The copied exclusions get this run's ID, the review fields stay as they were.
                row = json.loads(line)
                row["created_by_run_id"] = run_id
                exclusions.append(ReviewedExclusion.model_validate(row))
    passages, windows = [], []
    for bundle in bundles:
        current = build_passages(bundle, config, run_id)
        passages.extend(current)
        windows.extend(build_windows(bundle, current, config, run_id, WhitespaceBudget()))
    validate_records(bundles, passages, windows, exclusions, run_id, config)
    # Validate the whole parent run again (not just its manifest) before writing.
    validate_ingest(parent)
    if sha256((parent / "manifest.json").read_bytes()) != parent_digest or config_path.read_bytes() != config_bytes:
        raise ValueError("upstream snapshot/configuration changed during segmentation")
    if exclusions_name and sha256(exclusion_path.read_bytes()) != exclusions_digest:
        raise ValueError("reviewed exclusions changed during segmentation")
    output.mkdir(parents=True, exist_ok=False)
    dump_jsonl(output / "passages.jsonl", passages)
    dump_jsonl(output / "windows.jsonl", windows)
    dump_jsonl(output / "exclusions.jsonl", exclusions)
    dump_json(output / "config.json", config.model_dump())
    (output / "passages.md").write_text(reading_view(bundles, passages, exclusions), encoding="utf-8")
    names = ("passages.jsonl", "windows.jsonl", "exclusions.jsonl", "config.json", "passages.md")
    code_root = Path(__file__).parent
    manifest = {"schema_version": 1, "stage": "segment", "run_id": run_id, "status": "complete",
                "package_version": __version__, "started_utc": now.isoformat(),
                "elapsed_seconds": round(perf_counter() - started, 4), "python": platform.python_version(),
                "dependencies": {n: importlib.metadata.version(n) for n in ("spacy", "pydantic")},
                "parent_run_id": parent.name, "parent_manifest_sha256": parent_digest,
                "input_config_sha256": sha256(config_bytes), "input_exclusions_sha256": exclusions_digest,
                "code_sha256": {str(p.relative_to(code_root)): sha256(p.read_bytes()) for p in sorted(code_root.rglob("*.py"))},
                "counts": counts_for(bundles, passages, windows, exclusions),
                "artifact_sha256": {n: sha256((output / n).read_bytes()) for n in names},
                "model": None, "window_budget": "provisional whitespace tokens; not validated for GLiNER"}
    dump_json(output / "manifest.json", manifest)
    validate_run(output)
    return output


def validate_run(output: Path) -> dict:
    manifest = json.loads((output / "manifest.json").read_bytes())
    if manifest.get("stage") == "ingest":
        return validate_ingest(output)
    if manifest.get("stage") == "extract":
        from spoken_record.extraction import validate_run as validate_extraction
        return validate_extraction(output)
    if manifest.get("stage") == "resolve":
        from spoken_record.resolution import validate_run as validate_resolution
        return validate_resolution(output)
    if manifest.get("stage") == "analyze_resolution":
        from spoken_record.context_resolution import validate_run as validate_context_resolution
        return validate_context_resolution(output)
    if manifest.get("stage") == "refine_resolution":
        from spoken_record.identity_refinement import validate_run as validate_identity_refinement
        return validate_identity_refinement(output)
    if manifest.get("stage") == "entity_linking":
        from spoken_record.entity_linking import validate_run as validate_entity_linking
        return validate_entity_linking(output)
    if manifest.get("stage") == "coreference":
        from spoken_record.coreference import validate_run as validate_coreference
        return validate_coreference(output)
    if manifest.get("stage") == "relations":
        from spoken_record.relations import validate_run as validate_relations
        return validate_relations(output)
    if manifest.get("stage") == "relation_verification":
        from spoken_record.relation_verification import validate_run as validate_verification
        return validate_verification(output)
    if manifest.get("stage") == "passage_discovery":
        from spoken_record.discovery import validate_run as validate_discovery
        return validate_discovery(output)
    if manifest.get("stage") == "reference_evaluation_packet":
        from spoken_record.reference_evaluation import validate_reference_packet
        return validate_reference_packet(output)
    if manifest.get("stage") == "export_obsidian":
        from spoken_record.vault import validate_run as validate_vault
        return validate_vault(output)
    if manifest.get("stage") == "entity_enrichment":
        from spoken_record.enrichment import validate_run as validate_enrichment
        return validate_enrichment(output)
    if manifest.get("stage") == "export_web":
        from spoken_record.web_export import validate_run as validate_web
        return validate_web(output)
    if manifest.get("schema_version") != 1 or manifest.get("stage") != "segment" or manifest.get("status") != "complete":
        raise ValueError("unsupported or incomplete segmentation snapshot")
    parent_id = manifest["parent_run_id"]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]{0,95}", parent_id) or parent_id == output.name:
        raise ValueError("invalid parent snapshot reference")
    parent = output.parent / parent_id
    if sha256((parent / "manifest.json").read_bytes()) != manifest["parent_manifest_sha256"]:
        raise ValueError("parent snapshot manifest changed")
    bundles = load_sources(parent)
    names = {"passages.jsonl", "windows.jsonl", "exclusions.jsonl", "config.json", "passages.md"}
    if set(manifest["artifact_sha256"]) != names:
        raise ValueError("unexpected segmentation artifact set")
    for name in names:
        if sha256((output / name).read_bytes()) != manifest["artifact_sha256"][name]:
            raise ValueError(f"artifact checksum mismatch: {name}")
    config = SegmentationConfig.model_validate_json((output / "config.json").read_bytes())
    passages = [Passage.model_validate(r) for r in read_jsonl(output / "passages.jsonl")]
    windows = [ModelWindow.model_validate(r) for r in read_jsonl(output / "windows.jsonl")]
    exclusions = [ReviewedExclusion.model_validate(r) for r in read_jsonl(output / "exclusions.jsonl")]
    validate_records(bundles, passages, windows, exclusions, manifest["run_id"], config)
    if manifest["counts"] != counts_for(bundles, passages, windows, exclusions):
        raise ValueError("segmentation counts differ from records")
    return manifest
