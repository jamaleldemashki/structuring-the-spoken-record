"""Pronoun coreference with FCoref. Offsets point into the original transcript, and a
pronoun is only linked to an entity when the match is clear.
"""

import importlib.metadata
import json
import platform
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Callable
from uuid import uuid4

import spacy

from spoken_record import __version__
from spoken_record.backends import CoreferenceBackend
from spoken_record.adapters.base import sha256, stable_id
from spoken_record.coreference_models import (CoreferenceConfig, CoreferenceLink,
                                              CoreferenceOutcome, CoreferenceSpan,
                                              ModelCoreferenceCluster)
from spoken_record.extraction_models import ExtractedMention
from spoken_record.ingestion import dump_json, dump_jsonl, read_jsonl, workspace_path
from spoken_record.models import TranscriptVersion
from spoken_record.passage_models import Passage
from spoken_record.resolution import validate_run as validate_resolution
from spoken_record.resolution_models import CorpusEntity, MentionAssignment


def intersecting_passages(passage_rows: list[Passage], start: int, end: int) -> list[str]:
    return [passage.passage_id for passage in passage_rows
            if passage.char_start < end and passage.char_end > start]


def span_matches_mention(span: CoreferenceSpan, mention: ExtractedMention) -> bool:
    return ((span.char_start <= mention.char_start and span.char_end >= mention.char_end) or
            (mention.char_start <= span.char_start and mention.char_end >= span.char_end))


def capitalized_name(surface: str) -> bool:
    return any(token[:1].isupper() for token in re.findall(r"[^\W_]+", surface, re.UNICODE))


def project_clusters(run_id: str, episode_id: str, transcript: TranscriptVersion,
                     raw_clusters: list[list[tuple[int, int]]], passages: list[Passage],
                     mentions: list[ExtractedMention], assignments: list[MentionAssignment],
                     config: CoreferenceConfig, fingerprint: str):
    passage_rows = sorted((passage for passage in passages if passage.episode_id == episode_id),
                          key=lambda passage: passage.index)
    person_mentions = [mention for mention in mentions
                       if mention.episode_id == episode_id and mention.type == config.anchor_type]
    assignment_by_mention = {row.mention_id: row for row in assignments}
    forms = set(config.reference_forms)
    clusters, links = [], []
    occupied_references = set()
    for raw_spans in raw_clusters:
        spans = [CoreferenceSpan(char_start=start, char_end=end, text=transcript.text[start:end])
                 for start, end in raw_spans]
        if len(spans) < 2:
            continue
        cluster_id = stable_id("fc", transcript.transcript_version_id,
                               json.dumps(raw_spans, separators=(",", ":")), fingerprint)
        cluster = ModelCoreferenceCluster(created_by_run_id=run_id, cluster_id=cluster_id,
            episode_id=episode_id, transcript_version_id=transcript.transcript_version_id,
            spans=spans, model_fingerprint=fingerprint)
        clusters.append(cluster)
        reference_spans = [span for span in spans if span.text.casefold().strip() in forms]
        anchor_spans = [span for span in spans if span.text.casefold().strip() not in forms]
        anchors = sorted({mention.mention_id for span in anchor_spans for mention in person_mentions
                          if span_matches_mention(span, mention) and
                          (not config.require_capitalized_anchor or capitalized_name(mention.text))})
        episode_entities = sorted({assignment_by_mention[mid].episode_entity_id for mid in anchors})
        corpus_entities = sorted({assignment_by_mention[mid].corpus_entity_id for mid in anchors})
        for span in reference_spans:
            reference_key = (span.char_start, span.char_end)
            if reference_key in occupied_references:
                raise ValueError("one reference span occurs in multiple model clusters")
            occupied_references.add(reference_key)
            passage_ids = intersecting_passages(passage_rows, span.char_start, span.char_end)
            if not passage_ids:
                raise ValueError("coreference span has no evidence passage")
            if len(corpus_entities) == 1:
                decision, reason, corpus_entity_id = "linked", "single_corpus_entity", corpus_entities[0]
            elif not corpus_entities:
                decision, reason, corpus_entity_id = "abstain", "no_named_person_anchor", None
            else:
                decision, reason, corpus_entity_id = "abstain", "ambiguous_person_anchors", None
            links.append(CoreferenceLink(created_by_run_id=run_id,
                reference_id=stable_id("cr", transcript.transcript_version_id,
                                       str(span.char_start), str(span.char_end), fingerprint),
                episode_id=episode_id, transcript_version_id=transcript.transcript_version_id,
                cluster_id=cluster_id, char_start=span.char_start, char_end=span.char_end,
                text=span.text, reference_kind="third_person_pronoun", passage_ids=passage_ids,
                anchor_mention_ids=anchors, episode_entity_ids=episode_entities,
                corpus_entity_id=corpus_entity_id, decision=decision, reason=reason))
    return (tuple(sorted(clusters, key=lambda row: row.cluster_id)),
            tuple(sorted(links, key=lambda row: (row.episode_id, row.char_start, row.char_end))))


def load_chain(parent: Path):
    manifest = validate_resolution(parent)
    extraction = parent.parent / manifest.get("extraction_run_id", manifest["parent_run_id"])
    extraction_manifest = json.loads((extraction / "manifest.json").read_bytes())
    passage_run = extraction.parent / extraction_manifest["parent_run_id"]
    passage_manifest = json.loads((passage_run / "manifest.json").read_bytes())
    source_run = passage_run.parent / passage_manifest["parent_run_id"]
    return (manifest,
        [TranscriptVersion.model_validate(row) for row in read_jsonl(source_run / "transcripts.jsonl")],
        [Passage.model_validate(row) for row in read_jsonl(passage_run / "passages.jsonl")],
        [ExtractedMention.model_validate(row) for row in read_jsonl(extraction / "mentions.jsonl")],
        [MentionAssignment.model_validate(row) for row in read_jsonl(parent / "assignments.jsonl")],
        [CorpusEntity.model_validate(row) for row in read_jsonl(parent / "corpus_entities.jsonl")])


def report(links: tuple[CoreferenceLink, ...], entities: list[CorpusEntity]) -> str:
    entity_by_id = {entity.corpus_entity_id: entity for entity in entities}
    lines = ["# Coreference probe review", "",
             "> [!warning] Model-assisted provisional output",
             "> Links require evaluation before they can be treated as reliable evidence.", ""]
    linked = [row for row in links if row.decision == "linked"]
    lines.extend([f"## Linked references ({len(linked)})", "",
                  "| Reference | Target | Episode | Offset |",
                  "|---|---|---|---:|"])
    for row in linked:
        lines.append(f"| {row.text} | {entity_by_id[row.corpus_entity_id].canonical_label} | "
                     f"{row.episode_id} | {row.char_start} |")
    abstained = [row for row in links if row.decision == "abstain"]
    lines.extend(["", f"## Abstentions ({len(abstained)})", "",
                  "| Reference | Episode | Offset | Reason |",
                  "|---|---|---:|---|"])
    for row in abstained:
        lines.append(f"| {row.text} | {row.episode_id} | {row.char_start} | {row.reason} |")
    return "\n".join(lines) + "\n"


def summary_of(clusters, links, outcomes):
    return {"selected": len(outcomes), "processed": sum(row.status == "processed" for row in outcomes),
        "failed": sum(row.status == "failed" for row in outcomes), "clusters": len(clusters),
        "candidate_references": len(links), "linked_references": sum(row.decision == "linked" for row in links),
        "abstained_references": sum(row.decision == "abstain" for row in links),
        "by_reference_form": dict(sorted(Counter(row.text.casefold() for row in links).items())),
        "by_abstention_reason": dict(sorted(Counter(row.reason for row in links if row.decision == "abstain").items()))}


def load_model(config: CoreferenceConfig) -> CoreferenceBackend:
    """Import and load the model only for inference. Validation doesn't need it."""
    from fastcoref import FCoref
    from huggingface_hub import snapshot_download

    local_model = snapshot_download(config.repo_id, revision=config.revision,
                                    local_files_only=config.offline_only)
    return FCoref(model_name_or_path=local_model, device=config.device,
                  nlp=spacy.blank("en"), enable_progress_bar=False)


def coreference_run(root: Path, input_name: str, config_name: str, run_id: str | None = None,
                    *, model_loader: Callable[[CoreferenceConfig], CoreferenceBackend] | None = None) -> Path:
    """Run FCoref per episode and link pronouns to CorpusEntities where the match is clear."""
    root, started, now = root.resolve(), perf_counter(), datetime.now(timezone.utc)
    run_id = run_id or f"coref-{now:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]{0,95}", run_id):
        raise ValueError("invalid run ID")
    parent = workspace_path(root, input_name)
    if parent.parent != root / "data/prototype1/runs":
        raise ValueError("input must be a resolution snapshot")
    output = parent.parent / run_id
    if output.exists():
        raise FileExistsError(output)
    parent_digest = sha256((parent / "manifest.json").read_bytes())
    config_path = workspace_path(root, config_name)
    config_bytes = config_path.read_bytes()
    config = CoreferenceConfig.model_validate_json(config_bytes)
    _, transcripts, passages, mentions, assignments, entities = load_chain(parent)
    model = (model_loader or load_model)(config)
    fingerprint = sha256(config_bytes)
    all_clusters, all_links, outcomes = [], [], []
    for transcript in sorted(transcripts, key=lambda row: row.episode_id):
        episode_started = perf_counter()
        try:
            prediction = model.predict(texts=[transcript.text],
                max_tokens_in_batch=config.max_tokens_in_batch)[0]
            raw = prediction.get_clusters(as_strings=False)
            clusters, links = project_clusters(run_id, transcript.episode_id, transcript, raw,
                                                passages, mentions, assignments, config, fingerprint)
            all_clusters.extend(clusters)
            all_links.extend(links)
            outcomes.append(CoreferenceOutcome(created_by_run_id=run_id,
                episode_id=transcript.episode_id, transcript_version_id=transcript.transcript_version_id,
                status="processed", clusters=len(clusters), candidate_references=len(links),
                linked_references=sum(row.decision == "linked" for row in links),
                abstained_references=sum(row.decision == "abstain" for row in links),
                elapsed_seconds=round(perf_counter() - episode_started, 4)))
            if hasattr(prediction, "release_logits"):
                prediction.release_logits()
        except Exception as exc:
            outcomes.append(CoreferenceOutcome(created_by_run_id=run_id,
                episode_id=transcript.episode_id, transcript_version_id=transcript.transcript_version_id,
                status="failed", clusters=0, candidate_references=0, linked_references=0,
                abstained_references=0, elapsed_seconds=round(perf_counter() - episode_started, 4),
                error=f"{type(exc).__name__}: {exc}"))
    clusters = tuple(sorted(all_clusters, key=lambda row: row.cluster_id))
    links = tuple(sorted(all_links, key=lambda row: (row.episode_id, row.char_start, row.char_end)))
    outcomes = tuple(sorted(outcomes, key=lambda row: row.episode_id))
    if sha256((parent / "manifest.json").read_bytes()) != parent_digest or config_path.read_bytes() != config_bytes:
        raise ValueError("parent or coreference configuration changed")
    output.mkdir(parents=True, exist_ok=False)
    dump_jsonl(output / "model_clusters.jsonl", list(clusters))
    dump_jsonl(output / "coreference_links.jsonl", list(links))
    dump_jsonl(output / "outcomes.jsonl", list(outcomes))
    dump_json(output / "config.json", config.model_dump(mode="json"))
    summary = summary_of(clusters, links, outcomes)
    dump_json(output / "summary.json", summary)
    dump_json(output / "evaluation.json", {"schema_version": 1, "status": "not_evaluated",
        "reason": "no reviewed pronoun-coreference packet exists yet"})
    (output / "review.md").write_text(report(links, entities), encoding="utf-8")
    names = ("model_clusters.jsonl", "coreference_links.jsonl", "outcomes.jsonl", "config.json",
             "summary.json", "evaluation.json", "review.md")
    code_root = Path(__file__).parent
    manifest = {"schema_version": 1, "stage": "coreference", "run_id": run_id, "status": "complete",
        "package_version": __version__, "started_utc": now.isoformat(),
        "elapsed_seconds": round(perf_counter() - started, 4), "python": platform.python_version(),
        "dependencies": {name: importlib.metadata.version(name)
                         for name in ("fastcoref", "torch", "transformers", "spacy", "pydantic")},
        "parent_run_id": parent.name, "parent_manifest_sha256": parent_digest,
        "input_config": config_name, "input_config_sha256": sha256(config_bytes),
        "model": {"repo_id": config.repo_id, "revision": config.revision,
                  "architecture": config.architecture, "device": config.device},
        "counts": summary,
        "code_sha256": {str(path.relative_to(code_root)): sha256(path.read_bytes())
                        for path in sorted(code_root.rglob("*.py"))},
        "artifact_sha256": {name: sha256((output / name).read_bytes()) for name in names}}
    dump_json(output / "manifest.json", manifest)
    validate_run(output)
    return output


def validate_run(output: Path) -> dict:
    output = output.resolve()
    manifest = json.loads((output / "manifest.json").read_bytes())
    if manifest.get("schema_version") != 1 or manifest.get("stage") != "coreference" \
            or manifest.get("status") != "complete":
        raise ValueError("unsupported coreference snapshot")
    root = output.parents[3]
    parent = output.parent / manifest["parent_run_id"]
    if sha256((parent / "manifest.json").read_bytes()) != manifest["parent_manifest_sha256"]:
        raise ValueError("parent resolution manifest changed")
    validate_resolution(parent)
    config_path = workspace_path(root, manifest["input_config"])
    if sha256(config_path.read_bytes()) != manifest["input_config_sha256"]:
        raise ValueError("coreference configuration changed")
    expected_names = {"model_clusters.jsonl", "coreference_links.jsonl", "outcomes.jsonl", "config.json",
                      "summary.json", "evaluation.json", "review.md"}
    if set(manifest["artifact_sha256"]) != expected_names:
        raise ValueError("unexpected coreference artifact set")
    for name, digest in manifest["artifact_sha256"].items():
        if sha256((output / name).read_bytes()) != digest:
            raise ValueError(f"coreference artifact changed: {name}")
    config = CoreferenceConfig.model_validate_json((output / "config.json").read_bytes())
    _, transcripts, passages, mentions, assignments, entities = load_chain(parent)
    transcript_by_id = {row.transcript_version_id: row for row in transcripts}
    passage_ids = {row.passage_id for row in passages}
    mention_ids = {row.mention_id for row in mentions}
    entity_ids = {row.corpus_entity_id for row in entities}
    clusters = tuple(ModelCoreferenceCluster.model_validate(row)
                     for row in read_jsonl(output / "model_clusters.jsonl"))
    links = tuple(CoreferenceLink.model_validate(row) for row in read_jsonl(output / "coreference_links.jsonl"))
    outcomes = tuple(CoreferenceOutcome.model_validate(row) for row in read_jsonl(output / "outcomes.jsonl"))
    cluster_ids = {row.cluster_id for row in clusters}
    if len(cluster_ids) != len(clusters):
        raise ValueError("duplicate coreference cluster ID")
    for cluster in clusters:
        transcript = transcript_by_id[cluster.transcript_version_id]
        if transcript.episode_id != cluster.episode_id:
            raise ValueError("coreference cluster episode differs from transcript")
        for span in cluster.spans:
            if transcript.text[span.char_start:span.char_end] != span.text:
                raise ValueError("coreference cluster span does not reproduce transcript")
    expected_clusters, expected_links = [], []
    for transcript in sorted(transcripts, key=lambda row: row.episode_id):
        episode_clusters = sorted((row for row in clusters if row.episode_id == transcript.episode_id),
                                  key=lambda row: row.cluster_id)
        raw = [[(span.char_start, span.char_end) for span in row.spans] for row in episode_clusters]
        rebuilt_clusters, rebuilt_links = project_clusters(manifest["run_id"], transcript.episode_id,
            transcript, raw, passages, mentions, assignments, config, manifest["input_config_sha256"])
        expected_clusters.extend(rebuilt_clusters)
        expected_links.extend(rebuilt_links)
    if tuple(sorted(expected_clusters, key=lambda row: row.cluster_id)) != clusters:
        raise ValueError("coreference clusters are not reproducible from saved spans")
    if tuple(sorted(expected_links, key=lambda row: (row.episode_id, row.char_start, row.char_end))) != links:
        raise ValueError("coreference entity projection is not reproducible")
    seen_references = set()
    for link in links:
        transcript = transcript_by_id[link.transcript_version_id]
        key = (link.transcript_version_id, link.char_start, link.char_end)
        if key in seen_references:
            raise ValueError("duplicate coreference reference")
        seen_references.add(key)
        if transcript.text[link.char_start:link.char_end] != link.text:
            raise ValueError("coreference link span does not reproduce transcript")
        if link.cluster_id not in cluster_ids or not set(link.passage_ids) <= passage_ids:
            raise ValueError("coreference link has an unknown cluster or passage")
        if not set(link.anchor_mention_ids) <= mention_ids:
            raise ValueError("coreference link has an unknown anchor mention")
        if link.corpus_entity_id is not None and link.corpus_entity_id not in entity_ids:
            raise ValueError("coreference link has an unknown corpus entity")
        if link.text.casefold().strip() not in set(config.reference_forms):
            raise ValueError("coreference link is outside the configured reference forms")
    if len({row.episode_id for row in outcomes}) != len(outcomes) or \
            {row.episode_id for row in outcomes} != {row.episode_id for row in transcripts}:
        raise ValueError("coreference outcomes do not cover each selected episode exactly once")
    summary = summary_of(clusters, links, outcomes)
    if summary != manifest["counts"] or summary != json.loads((output / "summary.json").read_bytes()):
        raise ValueError("coreference counts differ")
    if report(links, entities) != (output / "review.md").read_text(encoding="utf-8"):
        raise ValueError("coreference review is not reproducible")
    return manifest
