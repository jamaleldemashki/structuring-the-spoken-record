"""Review packets and scoring for identity merges and pronoun coreference."""

import importlib.metadata
import json
import platform
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from spoken_record import __version__
from spoken_record.adapters.base import sha256
from spoken_record.coreference import validate_run as validate_coreference
from spoken_record.coreference_models import CoreferenceLink
from spoken_record.extraction_models import ExtractedMention
from spoken_record.identity_models import IdentityCandidateDecision
from spoken_record.ingestion import dump_json, dump_jsonl, read_jsonl, workspace_path
from spoken_record.models import Episode
from spoken_record.passage_models import Passage
from spoken_record.reference_evaluation_models import (
    CoreferenceEvaluationAnnotation,
    CoreferenceEvaluationSample,
    CoreferenceTarget,
    EvidencePassage,
    IdentityEvaluationAnnotation,
    IdentityEvaluationSample,
    ReferenceEvaluationConfig,
)
from spoken_record.resolution_models import CorpusEntity, MentionAssignment
from spoken_record.segmentation import validate_run as validate_stage


def stable_rank(salt: str, value: str) -> str:
    """Pseudo-random rank from a hash, so the global random state is not touched."""
    return sha256(f"{salt}\0{value}".encode())


def stratified_sample(rows: list[CoreferenceLink], maximum: int, salt: str) -> list[CoreferenceLink]:
    """Round-robin over hash-ordered groups, so every form, reason and episode shows up."""
    groups = defaultdict(list)
    for row in rows:
        groups[(row.episode_id, row.reason, row.text.casefold().strip())].append(row)
    for key, values in groups.items():
        values.sort(key=lambda row: (stable_rank(salt, row.reference_id), row.reference_id))
    keys = sorted(groups, key=lambda key: stable_rank(salt, "|".join(key)))
    selected = []
    while len(selected) < maximum:
        advanced = False
        for key in keys:
            if groups[key] and len(selected) < maximum:
                selected.append(groups[key].pop(0))
                advanced = True
        if not advanced:
            break
    return selected


def select_coreference_links(links: list[CoreferenceLink], config: ReferenceEvaluationConfig) -> list[CoreferenceLink]:
    """Mix linked and abstained pronoun decisions within the packet limit."""
    excluded = set(config.excluded_reference_forms)
    kinds = set(config.included_reference_kinds)
    eligible = [row for row in links if row.reference_kind in kinds and row.text.casefold().strip() not in excluded]
    linked = [row for row in eligible if row.decision == "linked"]
    abstained = [row for row in eligible if row.decision == "abstain"]
    linked_quota = min(len(linked), config.coreference_max_examples // 2)
    abstain_quota = min(len(abstained), config.coreference_max_examples - linked_quota)
    if linked_quota + abstain_quota < config.coreference_max_examples:
        linked_quota += min(len(linked) - linked_quota,
                            config.coreference_max_examples - linked_quota - abstain_quota)
    if linked_quota + abstain_quota < config.coreference_max_examples:
        abstain_quota += min(len(abstained) - abstain_quota,
                             config.coreference_max_examples - linked_quota - abstain_quota)
    selected = stratified_sample(linked, linked_quota, config.sampling_salt + "|linked")
    selected += stratified_sample(abstained, abstain_quota, config.sampling_salt + "|abstain")
    return sorted(selected, key=lambda row: (row.episode_id, row.char_start, row.char_end, row.reference_id))


def evidence_record(passage: Passage, episode: Episode) -> EvidencePassage:
    """Copy the exact passage text and metadata that the reviewer will see."""
    return EvidencePassage(passage_id=passage.passage_id, episode_id=passage.episode_id,
        episode_title=episode.title, passage_index=passage.index, char_start=passage.char_start,
        char_end=passage.char_end, start_s=passage.start_s, end_s=passage.end_s, text=passage.text,
        text_sha256=sha256(passage.text.encode()))


def select_identity_evidence(candidate: IdentityCandidateDecision, passages: dict[str, Passage],
                             maximum: int, salt: str) -> list[Passage]:
    """Prefer different episodes first, then fill the remaining slots in a fixed order."""
    available = [passages[value] for value in candidate.evidence_passage_ids]
    by_episode = defaultdict(list)
    for passage in available:
        by_episode[passage.episode_id].append(passage)
    chosen = []
    for episode_id in sorted(by_episode):
        rows = sorted(by_episode[episode_id],
                      key=lambda row: stable_rank(salt, candidate.candidate_id + "|" + row.passage_id))
        if rows and len(chosen) < maximum:
            chosen.append(rows[0])
    remaining = sorted((row for row in available if row not in chosen),
                       key=lambda row: stable_rank(salt, candidate.candidate_id + "|" + row.passage_id))
    chosen.extend(remaining[:maximum - len(chosen)])
    return sorted(chosen, key=lambda row: (row.episode_id, row.index, row.passage_id))


def build_samples(config: ReferenceEvaluationConfig, candidates: list[IdentityCandidateDecision],
                  links: list[CoreferenceLink], episodes: list[Episode], passages: list[Passage],
                  mentions: list[ExtractedMention], assignments: list[MentionAssignment],
                  entities: list[CorpusEntity]) -> tuple[list[IdentityEvaluationSample],
                                                        list[CoreferenceEvaluationSample]]:
    """Build the review samples from the saved identity and coreference decisions."""
    episode_map = {row.episode_id: row for row in episodes}
    passage_map = {row.passage_id: row for row in passages}
    mention_map = {row.mention_id: row for row in mentions}
    assignment_map = {row.mention_id: row for row in assignments}
    entity_map = {row.corpus_entity_id: row for row in entities}
    passages_by_episode = defaultdict(list)
    for passage in passages:
        passages_by_episode[passage.episode_id].append(passage)
    for values in passages_by_episode.values():
        values.sort(key=lambda passage: passage.index)
    identity_samples = []
    for row in candidates:
        if row.type not in config.identity_types:
            continue
        evidence = [evidence_record(value, episode_map[value.episode_id]) for value in
                    select_identity_evidence(row, passage_map, config.identity_max_evidence_passages,
                                             config.sampling_salt)]
        identity_samples.append(IdentityEvaluationSample(candidate_id=row.candidate_id, type=row.type,
            left_corpus_entity_id=row.left_corpus_entity_id, right_corpus_entity_id=row.right_corpus_entity_id,
            left_label=row.left_label, right_label=row.right_label, episode_ids=row.episode_ids,
            model_decision=row.decision, model_reason=row.reason, retrieval_methods=row.retrieval_methods,
            surface_similarity=row.surface_similarity, lexical_context_similarity=row.lexical_context_similarity,
            neighbor_similarity=row.neighbor_similarity, combined_score=row.combined_score, evidence=evidence))
    coreference_samples = []
    for row in select_coreference_links(links, config):
        anchor_mentions = [mention_map[value] for value in row.anchor_mention_ids]
        # One passage before and after gives enough context to find the antecedent.
        reference_indices = {passage_map[value].index for value in row.passage_ids}
        context_indices = {index + delta for index in reference_indices for delta in (-1, 0, 1)}
        context_passages = [passage for passage in passages_by_episode[row.episode_id]
                            if passage.index in context_indices]
        context_ids = {passage.passage_id for passage in context_passages}
        context_mentions = [mention for mention in mentions if mention.episode_id == row.episode_id and
                            mention.type == "Person" and set(mention.passage_ids) & context_ids]
        target_sources = defaultdict(set)
        anchor_counts = defaultdict(int)
        for mention in anchor_mentions:
            entity_id = assignment_map[mention.mention_id].corpus_entity_id
            target_sources[entity_id].add("model_anchor")
            anchor_counts[entity_id] += 1
        for mention in context_mentions:
            target_sources[assignment_map[mention.mention_id].corpus_entity_id].add("local_context")
        targets = [CoreferenceTarget(corpus_entity_id=entity_id,
            canonical_label=entity_map[entity_id].canonical_label,
            observed_surfaces=entity_map[entity_id].observed_surfaces,
            sources=sorted(sources), model_anchor_mentions=anchor_counts[entity_id])
            for entity_id, sources in target_sources.items()]
        evidence = [evidence_record(value, episode_map[row.episode_id]) for value in context_passages]
        label = entity_map[row.corpus_entity_id].canonical_label if row.corpus_entity_id else None
        coreference_samples.append(CoreferenceEvaluationSample(reference_id=row.reference_id,
            episode_id=row.episode_id, episode_title=episode_map[row.episode_id].title,
            transcript_version_id=row.transcript_version_id, char_start=row.char_start, char_end=row.char_end,
            text=row.text, reference_kind=row.reference_kind, model_decision=row.decision,
            model_reason=row.reason, predicted_corpus_entity_id=row.corpus_entity_id, predicted_label=label,
            candidate_targets=sorted(targets, key=lambda target: (target.canonical_label.casefold(),
                                                                  target.corpus_entity_id)),
            evidence=evidence))
    return (sorted(identity_samples, key=lambda row: row.candidate_id),
            sorted(coreference_samples, key=lambda row: (row.episode_id, row.char_start, row.reference_id)))


def load_parent_chain(root: Path, coreference_run: Path):
    """Load the identity records and upstream evidence that a coreference run used."""
    coreference_manifest = validate_coreference(coreference_run)
    identity_run = coreference_run.parent / coreference_manifest["parent_run_id"]
    identity_manifest = validate_stage(identity_run)
    if identity_manifest.get("stage") != "refine_resolution":
        raise ValueError("reference evaluation requires a refine-resolution parent")
    extraction = identity_run.parent / identity_manifest["extraction_run_id"]
    extraction_manifest = json.loads((extraction / "manifest.json").read_bytes())
    passage_run = extraction.parent / extraction_manifest["parent_run_id"]
    passage_manifest = json.loads((passage_run / "manifest.json").read_bytes())
    source_run = passage_run.parent / passage_manifest["parent_run_id"]
    return (coreference_manifest, identity_run, identity_manifest,
        [IdentityCandidateDecision.model_validate(row) for row in read_jsonl(identity_run / "identity_candidates.jsonl")],
        [CoreferenceLink.model_validate(row) for row in read_jsonl(coreference_run / "coreference_links.jsonl")],
        [Episode.model_validate(row) for row in read_jsonl(source_run / "episodes.jsonl")],
        [Passage.model_validate(row) for row in read_jsonl(passage_run / "passages.jsonl")],
        [ExtractedMention.model_validate(row) for row in read_jsonl(extraction / "mentions.jsonl")],
        [MentionAssignment.model_validate(row) for row in read_jsonl(identity_run / "assignments.jsonl")],
        [CorpusEntity.model_validate(row) for row in read_jsonl(identity_run / "corpus_entities.jsonl")])


def review_markdown(identity: list[IdentityEvaluationSample],
                    coreference: list[CoreferenceEvaluationSample]) -> str:
    """Markdown version for the reviewer, with model decisions hidden. The JSONL stays the real record."""
    lines = ["# Identity and third-person coreference review", "",
        "Edit `identity_annotations.jsonl` and `coreference_annotations.jsonl` after checking the evidence.",
        "Model decisions, scores, and predicted targets are intentionally hidden here to reduce review bias.",
        "For identity, use `same`, `different`, or `uncertain`.",
        "For coreference, use `link` plus a listed CorpusEntity ID, `no_link` when no listed Person is the referent, "
        "`ambiguous` when the evidence is insufficient, or `out_of_scope` for a malformed/non-reference item.",
        "Speaker attribution and first-person references are outside this packet.", "",
        "## Identity candidates", ""]
    for index, row in enumerate(identity, 1):
        lines += [f"### I{index}. {row.left_label} ↔ {row.right_label}", "",
            f"Candidate: `{row.candidate_id}`", "",
            f"IDs: `{row.left_corpus_entity_id}` ↔ `{row.right_corpus_entity_id}`", ""]
        for evidence in row.evidence:
            stamp = "untimed" if evidence.start_s is None else f"{evidence.start_s:.1f}–{evidence.end_s:.1f}s"
            lines += [f"#### {evidence.episode_title} · passage {evidence.passage_index + 1} · {stamp}", "",
                      *["> " + value for value in evidence.text.splitlines()], ""]
    lines += ["## Third-person coreference", ""]
    for index, row in enumerate(coreference, 1):
        lines += [f"### C{index}. {row.text!r}", "", f"Reference: `{row.reference_id}`", "",
            f"Transcript offset: `[{row.char_start}, {row.char_end})`", "", "Candidate targets:", ""]
        if row.candidate_targets:
            for candidate in row.candidate_targets:
                lines += [f"- `{candidate.corpus_entity_id}` — {candidate.canonical_label}"]
        else:
            lines += ["- none"]
        lines.append("")
        for evidence in row.evidence:
            role = "reference passage" if evidence.char_start <= row.char_start < evidence.char_end else "context"
            rendered = evidence.text
            if role == "reference passage":
                # Bold the reviewed span in the Markdown only. The stored text is not changed.
                start, end = row.char_start - evidence.char_start, row.char_end - evidence.char_start
                rendered = rendered[:start] + "**" + rendered[start:end] + "**" + rendered[end:]
            lines += [f"#### {evidence.episode_title} · passage {evidence.passage_index + 1} · {role}", "",
                      *["> " + value for value in rendered.splitlines()], ""]
    return "\n".join(lines).rstrip() + "\n"


def packet_counts(identity: list[IdentityEvaluationSample], coreference: list[CoreferenceEvaluationSample]) -> dict:
    """Count what is in the sample. Says nothing about accuracy."""
    return {"identity_candidates": len(identity),
        "identity_model_decisions": {value: sum(row.model_decision == value for row in identity)
                                     for value in ("merge", "separate", "abstain")},
        "coreference_examples": len(coreference),
        "coreference_model_decisions": {value: sum(row.model_decision == value for row in coreference)
                                        for value in ("linked", "abstain")},
        "coreference_forms": dict(sorted({form: sum(row.text.casefold().strip() == form for row in coreference)
                                          for form in {row.text.casefold().strip() for row in coreference}}.items()))}


def prepare_reference_packet(root: Path, input_name: str, config_name: str, packet_id: str) -> Path:
    """Create a new, unreviewed packet. The model runs are not changed."""
    root, started, now = root.resolve(), perf_counter(), datetime.now(timezone.utc)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]{0,95}", packet_id):
        raise ValueError("invalid reference evaluation packet ID")
    parent = workspace_path(root, input_name)
    config_path = workspace_path(root, config_name)
    config_bytes = config_path.read_bytes()
    config = ReferenceEvaluationConfig.model_validate_json(config_bytes)
    (coreference_manifest, identity_run, identity_manifest, candidates, links, episodes, passages,
     mentions, assignments, entities) = load_parent_chain(root, parent)
    parent_digest = sha256((parent / "manifest.json").read_bytes())
    identity_digest = sha256((identity_run / "manifest.json").read_bytes())
    identity, coreference = build_samples(config, candidates, links, episodes, passages, mentions,
                                          assignments, entities)
    if not identity or not coreference:
        raise ValueError("reference evaluation packet cannot be empty")
    output = root / "data/prototype1/evaluation" / packet_id
    if output.exists():
        raise FileExistsError(output)
    # Validate again right before writing, in case the inputs changed meanwhile.
    validate_coreference(parent)
    if sha256((parent / "manifest.json").read_bytes()) != parent_digest or \
            sha256((identity_run / "manifest.json").read_bytes()) != identity_digest or \
            config_path.read_bytes() != config_bytes:
        raise ValueError("parent run or reference evaluation configuration changed")
    output.mkdir(parents=True, exist_ok=False)
    dump_jsonl(output / "identity_sample.jsonl", identity)
    dump_jsonl(output / "identity_annotations.jsonl",
               [IdentityEvaluationAnnotation(candidate_id=row.candidate_id) for row in identity])
    dump_jsonl(output / "coreference_sample.jsonl", coreference)
    dump_jsonl(output / "coreference_annotations.jsonl",
               [CoreferenceEvaluationAnnotation(reference_id=row.reference_id) for row in coreference])
    dump_json(output / "config.json", config.model_dump(mode="json"))
    (output / "review.md").write_text(review_markdown(identity, coreference), encoding="utf-8")
    immutable = ("identity_sample.jsonl", "coreference_sample.jsonl", "config.json", "review.md")
    manifest = {"schema_version": 1, "stage": "reference_evaluation_packet", "packet_id": packet_id,
        "status": "unreviewed", "package_version": __version__, "created_utc": now.isoformat(),
        "elapsed_seconds": round(perf_counter() - started, 4), "python": platform.python_version(),
        "dependencies": {"pydantic": importlib.metadata.version("pydantic")},
        "parent_run": input_name, "parent_manifest_sha256": parent_digest,
        "identity_run": str(identity_run.relative_to(root)), "identity_manifest_sha256": identity_digest,
        "input_config": config_name, "input_config_sha256": sha256(config_bytes),
        "scope": "Person identity candidates and third-person singular coreference only; no speaker attribution",
        "counts": packet_counts(identity, coreference),
        "artifact_sha256": {name: sha256((output / name).read_bytes()) for name in immutable}}
    dump_json(output / "manifest.json", manifest)
    validate_reference_packet(output)
    return output


def load_packet(packet: Path):
    """Validate the packet and return its samples and the editable annotations."""
    identity = [IdentityEvaluationSample.model_validate(row) for row in read_jsonl(packet / "identity_sample.jsonl")]
    coreference = [CoreferenceEvaluationSample.model_validate(row) for row in read_jsonl(packet / "coreference_sample.jsonl")]
    identity_annotations = [IdentityEvaluationAnnotation.model_validate(row)
                            for row in read_jsonl(packet / "identity_annotations.jsonl")]
    coreference_annotations = [CoreferenceEvaluationAnnotation.model_validate(row)
                               for row in read_jsonl(packet / "coreference_annotations.jsonl")]
    if {row.candidate_id for row in identity} != {row.candidate_id for row in identity_annotations} or \
            len(identity) != len(identity_annotations):
        raise ValueError("identity annotations do not cover the sample exactly once")
    if {row.reference_id for row in coreference} != {row.reference_id for row in coreference_annotations} or \
            len(coreference) != len(coreference_annotations):
        raise ValueError("coreference annotations do not cover the sample exactly once")
    return identity, coreference, identity_annotations, coreference_annotations


def validate_reference_packet(output: Path) -> dict:
    """Rebuild the samples and validate the human annotations."""
    output = output.resolve()
    manifest = json.loads((output / "manifest.json").read_bytes())
    if manifest.get("schema_version") != 1 or manifest.get("stage") != "reference_evaluation_packet":
        raise ValueError("unsupported reference evaluation packet")
    root = output.parents[3]
    parent = workspace_path(root, manifest["parent_run"])
    identity_run = workspace_path(root, manifest["identity_run"])
    config_path = workspace_path(root, manifest["input_config"])
    if sha256((parent / "manifest.json").read_bytes()) != manifest["parent_manifest_sha256"] or \
            sha256((identity_run / "manifest.json").read_bytes()) != manifest["identity_manifest_sha256"]:
        raise ValueError("reference evaluation parent changed")
    if sha256(config_path.read_bytes()) != manifest["input_config_sha256"]:
        raise ValueError("reference evaluation configuration changed")
    if set(manifest["artifact_sha256"]) != {"identity_sample.jsonl", "coreference_sample.jsonl",
                                                    "config.json", "review.md"}:
        raise ValueError("unexpected immutable reference evaluation artifacts")
    for name, digest in manifest["artifact_sha256"].items():
        if sha256((output / name).read_bytes()) != digest:
            raise ValueError(f"reference evaluation artifact changed: {name}")
    config = ReferenceEvaluationConfig.model_validate_json((output / "config.json").read_bytes())
    if config != ReferenceEvaluationConfig.model_validate_json(config_path.read_bytes()):
        raise ValueError("reference evaluation config snapshot differs")
    (_, rebuilt_identity_run, _, candidates, links, episodes, passages, mentions,
     assignments, entities) = load_parent_chain(root, parent)
    if rebuilt_identity_run != identity_run:
        raise ValueError("reference evaluation identity parent differs")
    expected_identity, expected_coreference = build_samples(config, candidates, links, episodes, passages,
                                                             mentions, assignments, entities)
    identity, coreference, identity_annotations, coreference_annotations = load_packet(output)
    if identity != expected_identity or coreference != expected_coreference:
        raise ValueError("reference evaluation samples are not reproducible")
    if review_markdown(identity, coreference) != (output / "review.md").read_text(encoding="utf-8"):
        raise ValueError("reference evaluation review view is not reproducible")
    if packet_counts(identity, coreference) != manifest["counts"]:
        raise ValueError("reference evaluation counts differ")
    sample_by_id = {row.reference_id: row for row in coreference}
    for annotation in coreference_annotations:
        available = {row.corpus_entity_id for row in sample_by_id[annotation.reference_id].candidate_targets}
        if annotation.gold_corpus_entity_id is not None and annotation.gold_corpus_entity_id not in available:
            raise ValueError("coreference annotation target is not available in its review sample")
    return manifest


def binary_metrics(tp: int, fp: int, fn: int) -> dict:
    """Precision, recall and F1. Undefined values are null."""
    return {"tp": tp, "fp": fp, "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None}


def score_identity(samples: list[IdentityEvaluationSample],
                   annotations: list[IdentityEvaluationAnnotation]) -> dict:
    """Score merges. Gold pairs marked uncertain are left out of the binary metrics."""
    gold = {row.candidate_id: row for row in annotations}
    evaluated = [row for row in samples if gold[row.candidate_id].gold_decision != "uncertain"]
    tp = sum(row.model_decision == "merge" and gold[row.candidate_id].gold_decision == "same" for row in evaluated)
    fp = sum(row.model_decision == "merge" and gold[row.candidate_id].gold_decision == "different" for row in evaluated)
    fn = sum(row.model_decision != "merge" and gold[row.candidate_id].gold_decision == "same" for row in evaluated)
    tn = sum(row.model_decision != "merge" and gold[row.candidate_id].gold_decision == "different" for row in evaluated)
    uncertain = [row for row in samples if gold[row.candidate_id].gold_decision == "uncertain"]
    return {"reviewed_pairs": len(samples), "binary_pairs": len(evaluated),
        "gold_uncertain": len(uncertain), "merge": binary_metrics(tp, fp, fn),
        "true_separations": tn, "false_merges": fp, "false_splits": fn,
        "model_abstentions": sum(row.model_decision == "abstain" for row in samples),
        "abstentions_on_uncertain_gold": sum(row.model_decision == "abstain" for row in uncertain),
        "candidate_retrieval_recall": None,
        "candidate_retrieval_note": "not measurable from a packet containing only retrieved candidates"}


def score_coreference_subset(samples: list[CoreferenceEvaluationSample],
                             annotations: dict[str, CoreferenceEvaluationAnnotation]) -> dict:
    """Score exact link targets and whether abstaining was the right call."""
    samples = [row for row in samples if annotations[row.reference_id].gold_decision != "out_of_scope"]
    correct = lambda row: (row.model_decision == "linked" and
        annotations[row.reference_id].gold_decision == "link" and
        row.predicted_corpus_entity_id == annotations[row.reference_id].gold_corpus_entity_id)
    tp = sum(correct(row) for row in samples)
    fp = sum(row.model_decision == "linked" and not correct(row) for row in samples)
    fn = sum(annotations[row.reference_id].gold_decision == "link" and not correct(row) for row in samples)
    abstained = [row for row in samples if row.model_decision == "abstain"]
    appropriate = sum(annotations[row.reference_id].gold_decision in {"no_link", "ambiguous"}
                      for row in abstained)
    attempted_linkable = [row for row in samples if row.model_decision == "linked" and
                          annotations[row.reference_id].gold_decision == "link"]
    return {"examples": len(samples), "link": binary_metrics(tp, fp, fn),
        "coverage": sum(row.model_decision == "linked" for row in samples) / len(samples) if samples else None,
        "target_accuracy_on_attempted_linkable":
            sum(correct(row) for row in attempted_linkable) / len(attempted_linkable) if attempted_linkable else None,
        "model_abstentions": len(abstained), "appropriate_abstentions": appropriate,
        "abstention_precision": appropriate / len(abstained) if abstained else None}


def score_reference_packet(packet: Path) -> dict:
    """Score a fully reviewed packet (identity and coreference metrics)."""
    manifest = validate_reference_packet(packet)
    identity, coreference, identity_annotations, coreference_annotations = load_packet(packet)
    if any(row.status != "reviewed" for row in identity_annotations + coreference_annotations):
        raise ValueError("reference evaluation scoring requires every annotation to be reviewed")
    coref_gold = {row.reference_id: row for row in coreference_annotations}
    forms = sorted({row.text.casefold().strip() for row in coreference})
    episodes = sorted({row.episode_id for row in coreference})
    return {"schema_version": 1, "metric": "identity-coreference-review-v1",
        "packet_id": manifest["packet_id"], "identity": score_identity(identity, identity_annotations),
        "coreference": score_coreference_subset(coreference, coref_gold),
        "coreference_by_form": {form: score_coreference_subset(
            [row for row in coreference if row.text.casefold().strip() == form], coref_gold) for form in forms},
        "coreference_by_episode": {episode: score_coreference_subset(
            [row for row in coreference if row.episode_id == episode], coref_gold) for episode in episodes},
        "scope": "third-person singular references only; speaker attribution excluded",
        "identity_annotations_sha256": sha256((packet / "identity_annotations.jsonl").read_bytes()),
        "coreference_annotations_sha256": sha256((packet / "coreference_annotations.jsonl").read_bytes())}
