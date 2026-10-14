"""Relation extraction: co-occurrence edges per passage and zero-shot relations with
GLiREL, all tied to transcript evidence.
"""

import importlib.metadata
import itertools
import json
import platform
import re
import os
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Callable
from uuid import uuid4

import spacy

from spoken_record import __version__
from spoken_record.backends import RelationBackend
from spoken_record.adapters.base import sha256, stable_id
from spoken_record.coreference import validate_run as validate_coreference
from spoken_record.coreference_models import CoreferenceLink
from spoken_record.extraction_models import ExtractedMention
from spoken_record.ingestion import dump_json, dump_jsonl, read_jsonl, workspace_path
from spoken_record.passage_models import Passage
from spoken_record.relation_models import (CooccurrenceEdge, RelationAssertion, RelationCandidate,
                                           RelationEvidence, RelationExtractionConfig,
                                           RelationOutcome, RelationLabel, RelationContext)
from spoken_record.relation_contexts import GLiRELBudget, antecedent_contexts
from spoken_record.resolution import validate_run as validate_resolution
from spoken_record.resolution_models import CorpusEntity, MentionAssignment


@dataclass(frozen=True)
class RelationEndpoint:
    """A span the model sees, tied to one corpus entity and its source record."""

    source_id: str
    kind: str
    corpus_entity_id: str
    type: str
    text: str
    char_start: int
    char_end: int


def relation_input(parent: Path) -> Path:
    """The resolution run under an optional coreference run."""
    manifest = json.loads((parent / "manifest.json").read_bytes())
    if manifest.get("stage") == "coreference":
        validate_coreference(parent)
        resolution = parent.parent / manifest["parent_run_id"]
        validate_resolution(resolution)
        return resolution
    validate_resolution(parent)
    return parent


def load_chain(parent: Path):
    """Load the relation inputs, plus linked pronouns when the parent is a coreference run."""
    parent_manifest = json.loads((parent / "manifest.json").read_bytes())
    resolution = relation_input(parent)
    manifest = json.loads((resolution / "manifest.json").read_bytes())
    extraction = resolution.parent / manifest.get("extraction_run_id", manifest["parent_run_id"])
    extraction_manifest = json.loads((extraction / "manifest.json").read_bytes())
    passage_run = extraction.parent / extraction_manifest["parent_run_id"]
    coreference_links = []
    if parent_manifest.get("stage") == "coreference":
        coreference_links = [CoreferenceLink.model_validate(row)
                             for row in read_jsonl(parent / "coreference_links.jsonl")]
    return (
        [Passage.model_validate(row) for row in read_jsonl(passage_run / "passages.jsonl")],
        [ExtractedMention.model_validate(row) for row in read_jsonl(extraction / "mentions.jsonl")],
        [MentionAssignment.model_validate(row) for row in read_jsonl(resolution / "assignments.jsonl")],
        [CorpusEntity.model_validate(row) for row in read_jsonl(resolution / "corpus_entities.jsonl")],
        coreference_links,
    )


def passage_mentions(passage: Passage, mentions: list[ExtractedMention]) -> list[ExtractedMention]:
    """Mentions that lie completely inside one passage."""
    return sorted((mention for mention in mentions if mention.episode_id == passage.episode_id
                   and passage.char_start <= mention.char_start
                   and mention.char_end <= passage.char_end),
                  key=lambda mention: (mention.char_start, mention.char_end, mention.mention_id))


def passage_endpoints(passage: Passage, mentions: list[ExtractedMention],
                      assignment_by_mention: dict[str, MentionAssignment],
                      coreference_links: list[CoreferenceLink]) -> list[RelationEndpoint]:
    """Named mentions plus pronouns that were resolved to exactly one entity.

    If a pronoun has the same span as a named mention, the named one wins. That avoids
    duplicate endpoints for GLiREL and keeps the stronger provenance.
    """
    endpoints = [RelationEndpoint(source_id=mention.mention_id, kind="mention",
        corpus_entity_id=assignment_by_mention[mention.mention_id].corpus_entity_id,
        type=mention.type, text=mention.text, char_start=mention.char_start,
        char_end=mention.char_end) for mention in passage_mentions(passage, mentions)]
    occupied = {(row.char_start, row.char_end) for row in endpoints}
    for link in coreference_links:
        if link.decision != "linked" or link.episode_id != passage.episode_id \
                or not (passage.char_start <= link.char_start
                        and link.char_end <= passage.char_end):
            continue
        if (link.char_start, link.char_end) in occupied:
            continue
        endpoints.append(RelationEndpoint(source_id=link.reference_id,
            kind="coreference_reference", corpus_entity_id=link.corpus_entity_id,
            type="Person", text=link.text, char_start=link.char_start,
            char_end=link.char_end))
        occupied.add((link.char_start, link.char_end))
    return sorted(endpoints, key=lambda row: (row.char_start, row.char_end, row.kind, row.source_id))


def align_mentions(passage: Passage, mentions: list[ExtractedMention], nlp):
    """Convert character spans to GLiREL token spans (end inclusive)."""
    doc = nlp.make_doc(passage.text)
    aligned, skipped = [], []
    for mention in passage_mentions(passage, mentions):
        local_start = mention.char_start - passage.char_start
        local_end = mention.char_end - passage.char_start
        span = doc.char_span(local_start, local_end, alignment_mode="strict")
        if span is None or span.text != mention.text:
            skipped.append(mention.mention_id)
            continue
        # GLiREL takes inclusive end indices as input but returns exclusive ones.
        aligned.append((mention, span.start, span.end))
    tokens = [token.text for token in doc]
    ner = [[start, end - 1, mention.type, mention.text] for mention, start, end in aligned]
    by_token_span = {(start, end): mention for mention, start, end in aligned}
    return tokens, ner, by_token_span, skipped


def sentence_contexts(passage: Passage, endpoints: list[RelationEndpoint], nlp,
                      context_unit: str = "sentence_within_passage"):
    """Build the GLiREL inputs. Pronouns can optionally see the previous sentence.

    Only endpoints in the focal sentence are candidates. The previous sentence just gives the
    model the antecedent, it doesn't create pairs across sentences.
    """
    doc = nlp(passage.text)
    contexts, aligned_ids, span_by_id = [], set(), {}
    for endpoint in endpoints:
        local_start = endpoint.char_start - passage.char_start
        local_end = endpoint.char_end - passage.char_start
        span = doc.char_span(local_start, local_end, alignment_mode="strict")
        if span is not None and span.text == endpoint.text:
            aligned_ids.add(endpoint.source_id)
            span_by_id[endpoint.source_id] = span
    sentences = list(doc.sents)
    for sentence_index, sentence in enumerate(sentences):
        focal = [endpoint for endpoint in endpoints
                 if endpoint.source_id in span_by_id
                 and sentence.start <= span_by_id[endpoint.source_id].start
                 and span_by_id[endpoint.source_id].end <= sentence.end]
        if len(focal) < 2:
            continue
        has_reference = any(endpoint.kind == "coreference_reference" for endpoint in focal)
        first_sentence = sentences[sentence_index - 1] if has_reference and sentence_index > 0 \
            and context_unit == "coreference_sentence_with_previous" else sentence
        context_start_token, context_end_token = first_sentence.start, sentence.end
        context_start_char, context_end_char = first_sentence.start_char, sentence.end_char
        by_token_span = {}
        ner = []
        for endpoint in focal:
            span = span_by_id[endpoint.source_id]
            start, end = span.start - context_start_token, span.end - context_start_token
            ner.append([start, end - 1, endpoint.type, endpoint.text])
            by_token_span[(start, end)] = endpoint
        if len(ner) >= 2:
            contexts.append((
                [token.text for token in doc[context_start_token:context_end_token]],
                ner, by_token_span,
                passage.char_start + context_start_char,
                passage.char_start + context_end_char,
            ))
    skipped = sorted(endpoint.source_id for endpoint in endpoints
                     if endpoint.source_id not in aligned_ids)
    return contexts, aligned_ids, skipped


def prediction_span(position, by_token_span):
    """Map GLiREL's output span (end exclusive), with a fallback for the inclusive form."""
    key = tuple(int(value) for value in position)
    if key in by_token_span:
        return by_token_span[key]
    # Older GLiREL versions return inclusive ends, like the input.
    return by_token_span.get((key[0], key[1] + 1))


def grounded_endpoint(value: ExtractedMention | RelationEndpoint,
                      assignment_by_mention: dict[str, MentionAssignment]) -> RelationEndpoint:
    """Bring old named-mention inputs and coreference endpoints into one form."""
    if isinstance(value, RelationEndpoint):
        return value
    assignment = assignment_by_mention[value.mention_id]
    return RelationEndpoint(source_id=value.mention_id, kind="mention",
        corpus_entity_id=assignment.corpus_entity_id, type=value.type, text=value.text,
        char_start=value.char_start, char_end=value.char_end)


def relation_decision(head: RelationEndpoint, tail: RelationEndpoint, score: float,
                      label: RelationLabel, threshold: float) -> tuple[str, str]:
    """Apply the acceptance rules to grounded evidence, independent of the model."""
    if head.corpus_entity_id == tail.corpus_entity_id:
        return "self_relation", "both mentions resolve to the same CorpusEntity"
    if head.type not in label.allowed_head or tail.type not in label.allowed_tail:
        return "type_rejected", "head or tail type violates the controlled predicate"
    if score < threshold:
        return "below_threshold", "score is below the configured acceptance threshold"
    return "accepted", "grounded entity types and score satisfy the configured rule"


def project_predictions(run_id: str, passage: Passage, predictions: list[dict],
                        by_token_span: dict[tuple[int, int], ExtractedMention],
                        assignment_by_mention: dict[str, MentionAssignment],
                        config: RelationExtractionConfig, fingerprint: str,
                        context_char_start: int | None = None,
                        context_char_end: int | None = None):
    """Map model output to mentions and apply the type, self-link and score checks."""
    label_by_prompt = {label.prompt_label: label for label in config.labels}
    context_char_start = passage.char_start if context_char_start is None else context_char_start
    context_char_end = passage.char_end if context_char_end is None else context_char_end
    rows = []
    for prediction in predictions:
        prompt = prediction.get("label")
        label = label_by_prompt.get(prompt)
        if label is None:
            raise ValueError(f"GLiREL returned an unconfigured relation label: {prompt!r}")
        head = prediction_span(prediction.get("head_pos", []), by_token_span)
        tail = prediction_span(prediction.get("tail_pos", []), by_token_span)
        if head is None or tail is None:
            raise ValueError("GLiREL returned a relation span that cannot be grounded to a mention")
        head = grounded_endpoint(head, assignment_by_mention)
        tail = grounded_endpoint(tail, assignment_by_mention)
        score = float(prediction["score"])
        decision, reason = relation_decision(head, tail, score, label, config.acceptance_threshold)
        rows.append(RelationCandidate(
            created_by_run_id=run_id,
            candidate_id=stable_id("rel", passage.passage_id, str(context_char_start),
                                   head.source_id, tail.source_id,
                                   label.canonical_predicate, fingerprint),
            episode_id=passage.episode_id, passage_id=passage.passage_id,
            context_char_start=context_char_start, context_char_end=context_char_end,
            head_mention_id=head.source_id, tail_mention_id=tail.source_id,
            head_endpoint_kind=head.kind, tail_endpoint_kind=tail.kind,
            head_corpus_entity_id=head.corpus_entity_id,
            tail_corpus_entity_id=tail.corpus_entity_id,
            head_text=head.text, tail_text=tail.text,
            head_char_start=head.char_start, head_char_end=head.char_end,
            tail_char_start=tail.char_start, tail_char_end=tail.char_end,
            prompt_label=label.prompt_label, canonical_predicate=label.canonical_predicate,
            score=score, decision=decision, reason=reason))
    return tuple(sorted(rows, key=lambda row: row.candidate_id))


def build_assertions(run_id: str, candidates: list[RelationCandidate]) -> tuple[RelationAssertion, ...]:
    """Group accepted predictions per entity pair, keeping every piece of evidence."""
    grouped = defaultdict(list)
    for row in candidates:
        if row.decision == "accepted":
            grouped[(row.head_corpus_entity_id, row.canonical_predicate,
                     row.tail_corpus_entity_id)].append(row)
    assertions = []
    for (subject, predicate, object_id), rows in sorted(grouped.items()):
        evidence = [RelationEvidence(passage_id=row.passage_id,
                    head_mention_id=row.head_mention_id, tail_mention_id=row.tail_mention_id,
                    head_endpoint_kind=row.head_endpoint_kind,
                    tail_endpoint_kind=row.tail_endpoint_kind,
                    score=row.score) for row in sorted(rows, key=lambda item: item.candidate_id)]
        assertions.append(RelationAssertion(created_by_run_id=run_id,
            assertion_id=stable_id("ra", subject, predicate, object_id),
            subject_corpus_entity_id=subject, predicate=predicate,
            object_corpus_entity_id=object_id, evidence=evidence,
            confidence=max(item.score for item in evidence)))
    return tuple(assertions)


def abstain_direction_conflicts(candidates: tuple[RelationCandidate, ...],
                                config: RelationExtractionConfig) -> tuple[RelationCandidate, ...]:
    """Abstain when one context predicts the same asymmetric predicate in both directions."""
    asymmetric = {label.canonical_predicate for label in config.labels if label.asymmetric}
    directions = defaultdict(set)
    for row in candidates:
        if row.decision != "accepted" or row.canonical_predicate not in asymmetric:
            continue
        endpoints = tuple(sorted((row.head_corpus_entity_id, row.tail_corpus_entity_id)))
        key = (row.passage_id, row.context_char_start, row.context_char_end,
               row.canonical_predicate, endpoints)
        directions[key].add((row.head_corpus_entity_id, row.tail_corpus_entity_id))
    conflicts = {key for key, values in directions.items() if len(values) > 1}
    revised = []
    for row in candidates:
        endpoints = tuple(sorted((row.head_corpus_entity_id, row.tail_corpus_entity_id)))
        key = (row.passage_id, row.context_char_start, row.context_char_end,
               row.canonical_predicate, endpoints)
        if row.decision == "accepted" and key in conflicts:
            row = row.model_copy(update={"decision": "direction_conflict",
                "reason": "the same context predicts this asymmetric predicate in both directions"})
        revised.append(row)
    return tuple(revised)


def build_cooccurrence(run_id: str, passages: list[Passage], mentions: list[ExtractedMention],
                       assignments: list[MentionAssignment]) -> tuple[CooccurrenceEdge, ...]:
    """Undirected co-occurrence pairs with the passages they come from."""
    assignment_by_mention = {row.mention_id: row for row in assignments}
    evidence = defaultdict(set)
    for passage in passages:
        entity_ids = sorted({assignment_by_mention[mention.mention_id].corpus_entity_id
                             for mention in passage_mentions(passage, mentions)})
        for left, right in itertools.combinations(entity_ids, 2):
            evidence[(left, right)].add(passage.passage_id)
    return tuple(CooccurrenceEdge(created_by_run_id=run_id,
        edge_id=stable_id("co", left, right), left_corpus_entity_id=left,
        right_corpus_entity_id=right, passage_ids=sorted(passage_ids),
        passage_count=len(passage_ids)) for (left, right), passage_ids in sorted(evidence.items()))


def summary_of(passages, candidates, assertions, cooccurrence, outcomes,
               coreference_links: list[CoreferenceLink] | None = None):
    """Counts for the manifest and the report."""
    summary = {
        "passages": len(passages),
        "processed": sum(row.status == "processed" for row in outcomes),
        "skipped": sum(row.status == "skipped" for row in outcomes),
        "failed": sum(row.status == "failed" for row in outcomes),
        "aligned_mentions": sum(row.aligned_mentions for row in outcomes),
        "unaligned_mentions": sum(row.input_mentions - row.aligned_mentions for row in outcomes),
        "relation_candidates": len(candidates),
        "accepted_candidates": sum(row.decision == "accepted" for row in candidates),
        "assertions": len(assertions),
        "cooccurrence_edges": len(cooccurrence),
        "by_candidate_decision": dict(sorted(Counter(row.decision for row in candidates).items())),
        "by_predicate": dict(sorted(Counter(row.predicate for row in assertions).items())),
    }
    if coreference_links is not None:
        summary["linked_coreference_references"] = len(coreference_links)
        summary["candidates_with_coreference"] = sum(
            row.head_endpoint_kind == "coreference_reference"
            or row.tail_endpoint_kind == "coreference_reference" for row in candidates)
        summary["assertions_with_coreference"] = sum(
            any(item.head_endpoint_kind == "coreference_reference"
                or item.tail_endpoint_kind == "coreference_reference" for item in row.evidence)
            for row in assertions)
    return summary


def report(assertions: list[RelationAssertion], entities: list[CorpusEntity], summary: dict) -> str:
    """Markdown report of the provisional relations, for a quick look at the evidence."""
    entity_by_id = {entity.corpus_entity_id: entity for entity in entities}
    grounding_text = ("> Every typed relation is grounded to named mentions or resolved third-person references and passages, but has not been human-reviewed."
                      if "assertions_with_coreference" in summary else
                      "> Every typed relation is grounded to existing mentions and passages, but has not been human-reviewed.")
    lines = ["# GLiREL relation review", "",
             "> [!warning] Provisional zero-shot output",
             grounding_text, "",
             "## Summary", "",
             f"- Accepted mention-level predictions: {summary['accepted_candidates']}",
             f"- Aggregated entity-level assertions: {summary['assertions']}",
             f"- Structural passage co-occurrence edges: {summary['cooccurrence_edges']}", "",
             "## Typed assertions", ""]
    if "assertions_with_coreference" in summary:
        lines.insert(10, f"- Assertions using a resolved coreference endpoint: "
                        f"{summary['assertions_with_coreference']}")
    if not assertions:
        lines.append("No candidate exceeded the configured acceptance threshold.")
    for row in sorted(assertions, key=lambda item: (-item.confidence, item.assertion_id)):
        subject = entity_by_id[row.subject_corpus_entity_id].canonical_label
        object_label = entity_by_id[row.object_corpus_entity_id].canonical_label
        passages = ", ".join(sorted({item.passage_id for item in row.evidence}))
        lines.extend([f"### {subject} — {row.predicate} — {object_label}", "",
                      f"- Confidence: {row.confidence:.3f}",
                      f"- Evidence: `{passages}`", ""])
    return "\n".join(lines).rstrip() + "\n"


def load_model(config: RelationExtractionConfig) -> RelationBackend:
    """Load GLiREL and its encoder (both pinned) fully offline."""
    from glirel import GLiREL
    from huggingface_hub import snapshot_download

    relation_path = Path(snapshot_download(config.repo_id, revision=config.revision,
                                           local_files_only=config.offline_only))
    encoder_path = Path(snapshot_download(config.encoder_repo_id, revision=config.encoder_revision,
                                          local_files_only=config.offline_only))
    # GLiREL's config names its encoder by Hub repo name. We write a temporary config that
    # points at the local copy, so it doesn't go online.
    with tempfile.TemporaryDirectory(prefix="spoken-record-glirel-") as temporary:
        model_path = Path(temporary)
        os.symlink(relation_path / "pytorch_model.bin", model_path / "pytorch_model.bin")
        model_config = json.loads((relation_path / "glirel_config.json").read_bytes())
        model_config["model_name"] = str(encoder_path)
        (model_path / "glirel_config.json").write_text(
            json.dumps(model_config, ensure_ascii=False), encoding="utf-8")
        model = GLiREL.from_pretrained(model_path, local_files_only=True)
    model.to(config.device)
    model.eval()
    return model


def context_transcripts(parent: Path):
    """Read the transcripts behind the selected identity run."""
    from spoken_record.coreference import load_chain as load_coreference_chain
    return load_coreference_chain(relation_input(parent))[1]


def context_summary(records):
    """Context coverage and budget omissions, reported apart from the relations."""
    return {"contexts": len(records),
        "ready_contexts": sum(row.status == "ready" for row in records),
        "budget_skipped_contexts": sum(row.status == "budget_exceeded" for row in records),
        "contexts_with_antecedent": sum(bool(row.included_anchor_ids) for row in records),
        "contexts_with_omitted_antecedent": sum(bool(set(row.selected_anchor_ids) - set(row.included_anchor_ids))
                                               or bool(row.missing_anchor_reference_ids) for row in records),
        "cross_passage_contexts": sum(len(row.evidence_passage_ids) > 1 for row in records)}


def relation_run(root: Path, input_name: str, config_name: str, run_id: str | None = None,
                 *, model_loader: Callable[[RelationExtractionConfig], RelationBackend] | None = None,
                 budget_factory=None) -> Path:
    """Run GLiREL on named mentions and resolved pronouns and write the results."""
    root, started, now = root.resolve(), perf_counter(), datetime.now(timezone.utc)
    run_id = run_id or f"relations-{now:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]{0,95}", run_id):
        raise ValueError("invalid run ID")
    parent = workspace_path(root, input_name)
    if parent.parent != root / "data/prototype1/runs":
        raise ValueError("input must be a resolution or coreference snapshot")
    output = parent.parent / run_id
    if output.exists():
        raise FileExistsError(output)
    parent_digest = sha256((parent / "manifest.json").read_bytes())
    config_path = workspace_path(root, config_name)
    config_bytes = config_path.read_bytes()
    config = RelationExtractionConfig.model_validate_json(config_bytes)
    passages, mentions, assignments, entities, available_coreference = load_chain(parent)
    # A coreference run can have zero links. Check the stage name, not the row count.
    if config.include_linked_coreference and json.loads((parent / "manifest.json").read_bytes())["stage"] != "coreference":
        raise ValueError("coreference-aware relations require a coreference snapshot as input")
    coreference_links = ([row for row in available_coreference if row.decision == "linked"]
                         if config.include_linked_coreference else [])
    assignment_by_mention = {row.mention_id: row for row in assignments}
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")

    # Import lazily, so validation works without the relation model packages installed.
    model = (model_loader or load_model)(config)
    bounded = config.context_unit == "coreference_antecedent_window"
    budget = (budget_factory or GLiRELBudget)(model, config) if bounded else None
    documents = {row.transcript_version_id: nlp(row.text) for row in context_transcripts(parent)} if bounded else {}
    fingerprint = sha256(config_bytes)
    prompts = [label.prompt_label for label in config.labels]
    all_candidates, outcomes, context_records = [], [], []
    for passage in sorted(passages, key=lambda row: (row.episode_id, row.index)):
        passage_started = perf_counter()
        endpoints = passage_endpoints(passage, mentions, assignment_by_mention, coreference_links)
        if bounded:
            contexts, aligned_ids, skipped, records = antecedent_contexts(run_id, passage,
                endpoints, documents[passage.transcript_version_id], mentions, coreference_links,
                assignment_by_mention, passages, config, budget)
            context_records.extend(records)
        else:
            contexts, aligned_ids, skipped = sentence_contexts(
                passage, endpoints, nlp, config.context_unit)
        if not contexts:
            outcomes.append(RelationOutcome(created_by_run_id=run_id,
                episode_id=passage.episode_id, passage_id=passage.passage_id, status="skipped",
                input_mentions=len(endpoints), aligned_mentions=len(aligned_ids), predictions=0, accepted=0,
                elapsed_seconds=round(perf_counter() - passage_started, 4),
                reason="no model-ready context; inspect context budgets and endpoint alignment"))
            continue
        try:
            rows = []
            for tokens, ner, by_token_span, context_start, context_end in contexts:
                predictions = model.predict_relations(tokens, prompts, threshold=config.candidate_floor,
                                                      ner=ner, top_k=config.top_k)
                projected = project_predictions(run_id, passage, predictions, by_token_span,
                    assignment_by_mention, config, fingerprint, context_start, context_end)
                rows.extend(abstain_direction_conflicts(projected, config))
            all_candidates.extend(rows)
            outcomes.append(RelationOutcome(created_by_run_id=run_id,
                episode_id=passage.episode_id, passage_id=passage.passage_id, status="processed",
                input_mentions=len(endpoints), aligned_mentions=len(aligned_ids), predictions=len(rows),
                accepted=sum(row.decision == "accepted" for row in rows),
                elapsed_seconds=round(perf_counter() - passage_started, 4),
                reason=(f"{len(skipped)} mention(s) failed strict token alignment" if skipped else None)))
        except Exception as exc:
            outcomes.append(RelationOutcome(created_by_run_id=run_id,
                episode_id=passage.episode_id, passage_id=passage.passage_id, status="failed",
                input_mentions=len(endpoints), aligned_mentions=len(aligned_ids), predictions=0, accepted=0,
                elapsed_seconds=round(perf_counter() - passage_started, 4),
                error=f"{type(exc).__name__}: {exc}"))
    candidates = tuple(sorted(all_candidates, key=lambda row: row.candidate_id))
    assertions = build_assertions(run_id, list(candidates))
    cooccurrence = build_cooccurrence(run_id, passages, mentions, assignments)
    outcomes = tuple(sorted(outcomes, key=lambda row: (row.episode_id, row.passage_id)))
    if sha256((parent / "manifest.json").read_bytes()) != parent_digest or config_path.read_bytes() != config_bytes:
        raise ValueError("parent or relation configuration changed")
    output.mkdir(parents=True, exist_ok=False)
    dump_jsonl(output / "relation_candidates.jsonl", list(candidates))
    dump_jsonl(output / "relation_assertions.jsonl", list(assertions))
    dump_jsonl(output / "cooccurrence_edges.jsonl", list(cooccurrence))
    dump_jsonl(output / "outcomes.jsonl", list(outcomes))
    dump_json(output / "config.json", config.model_dump(mode="json"))
    summary = summary_of(passages, candidates, assertions, cooccurrence, outcomes,
                         coreference_links if config.include_linked_coreference else None)
    if bounded:
        dump_jsonl(output / "relation_contexts.jsonl", context_records)
        summary.update(context_summary(context_records))
    dump_json(output / "summary.json", summary)
    (output / "review.md").write_text(report(list(assertions), entities, summary), encoding="utf-8")
    names = ("relation_candidates.jsonl", "relation_assertions.jsonl", "cooccurrence_edges.jsonl",
             "outcomes.jsonl", "config.json", "summary.json", "review.md")
    if bounded:
        names += ("relation_contexts.jsonl",)
    code_root = Path(__file__).parent
    manifest = {"schema_version": 1, "stage": "relations", "run_id": run_id, "status": "complete",
        "package_version": __version__, "started_utc": now.isoformat(),
        "elapsed_seconds": round(perf_counter() - started, 4), "python": platform.python_version(),
        "dependencies": {name: importlib.metadata.version(name)
                         for name in ("glirel", "torch", "transformers", "spacy", "pydantic")},
        "parent_run_id": parent.name, "parent_manifest_sha256": parent_digest,
        "resolution_run_id": relation_input(parent).name,
        "input_config": config_name, "input_config_sha256": fingerprint,
        "model": {"repo_id": config.repo_id, "revision": config.revision,
                  "encoder_repo_id": config.encoder_repo_id,
                  "encoder_revision": config.encoder_revision,
                  "framework": config.framework, "device": config.device},
        "counts": summary,
        "code_sha256": {str(path.relative_to(code_root)): sha256(path.read_bytes())
                        for path in sorted(code_root.rglob("*.py"))},
        "artifact_sha256": {name: sha256((output / name).read_bytes()) for name in names}}
    dump_json(output / "manifest.json", manifest)
    validate_run(output)
    return output


def validate_run(output: Path) -> dict:
    """Validate the relation provenance and the parts that don't need the model."""
    output = output.resolve()
    manifest = json.loads((output / "manifest.json").read_bytes())
    if manifest.get("schema_version") != 1 or manifest.get("stage") != "relations" \
            or manifest.get("status") != "complete":
        raise ValueError("unsupported relation snapshot")
    root = output.parents[3]
    parent = output.parent / manifest["parent_run_id"]
    if sha256((parent / "manifest.json").read_bytes()) != manifest["parent_manifest_sha256"]:
        raise ValueError("parent manifest changed")
    passages, mentions, assignments, entities, available_coreference = load_chain(parent)
    config_path = workspace_path(root, manifest["input_config"])
    if sha256(config_path.read_bytes()) != manifest["input_config_sha256"]:
        raise ValueError("relation configuration changed")
    config = RelationExtractionConfig.model_validate_json((output / "config.json").read_bytes())
    coreference_links = ([row for row in available_coreference if row.decision == "linked"]
                         if config.include_linked_coreference else [])
    expected_names = {"relation_candidates.jsonl", "relation_assertions.jsonl", "cooccurrence_edges.jsonl",
                      "outcomes.jsonl", "config.json", "summary.json", "review.md"}
    bounded = config.context_unit == "coreference_antecedent_window"
    if bounded:
        expected_names.add("relation_contexts.jsonl")
    if set(manifest["artifact_sha256"]) != expected_names:
        raise ValueError("unexpected relation artifact set")
    for name, digest in manifest["artifact_sha256"].items():
        if sha256((output / name).read_bytes()) != digest:
            raise ValueError(f"relation artifact changed: {name}")
    candidates = [RelationCandidate.model_validate(row)
                  for row in read_jsonl(output / "relation_candidates.jsonl")]
    assertions = [RelationAssertion.model_validate(row)
                  for row in read_jsonl(output / "relation_assertions.jsonl")]
    cooccurrence = [CooccurrenceEdge.model_validate(row)
                    for row in read_jsonl(output / "cooccurrence_edges.jsonl")]
    outcomes = [RelationOutcome.model_validate(row) for row in read_jsonl(output / "outcomes.jsonl")]
    mention_by_id = {row.mention_id: row for row in mentions}
    assignment_by_mention = {row.mention_id: row for row in assignments}
    reference_by_id = {row.reference_id: row for row in coreference_links}
    entity_ids = {row.corpus_entity_id for row in entities}
    passage_ids = {row.passage_id for row in passages}
    label_by_prompt = {label.prompt_label: label for label in config.labels}
    if len({row.candidate_id for row in candidates}) != len(candidates):
        raise ValueError("duplicate relation candidate ID")
    for row in candidates:
        head = (mention_by_id.get(row.head_mention_id) if row.head_endpoint_kind == "mention"
                else reference_by_id.get(row.head_mention_id))
        tail = (mention_by_id.get(row.tail_mention_id) if row.tail_endpoint_kind == "mention"
                else reference_by_id.get(row.tail_mention_id))
        if head is None or tail is None:
            raise ValueError("relation candidate references an unknown endpoint")
        label = label_by_prompt[row.prompt_label]
        if row.passage_id not in passage_ids or row.head_corpus_entity_id not in entity_ids \
                or row.tail_corpus_entity_id not in entity_ids:
            raise ValueError("relation candidate references an unknown record")
        if (head.char_start, head.char_end, head.text) != \
                (row.head_char_start, row.head_char_end, row.head_text) or \
                (tail.char_start, tail.char_end, tail.text) != \
                (row.tail_char_start, row.tail_char_end, row.tail_text):
            raise ValueError("relation candidate no longer reproduces its mentions")
        expected_head_entity = (assignment_by_mention[head.mention_id].corpus_entity_id
                                if row.head_endpoint_kind == "mention" else head.corpus_entity_id)
        expected_tail_entity = (assignment_by_mention[tail.mention_id].corpus_entity_id
                                if row.tail_endpoint_kind == "mention" else tail.corpus_entity_id)
        if (row.head_corpus_entity_id, row.tail_corpus_entity_id) != \
                (expected_head_entity, expected_tail_entity):
            raise ValueError("relation candidate endpoint identity changed")
        if row.canonical_predicate != label.canonical_predicate:
            raise ValueError("relation candidate predicate differs from configuration")
    rebuilt_assertions = list(build_assertions(manifest["run_id"], candidates))
    rebuilt_cooccurrence = list(build_cooccurrence(manifest["run_id"], passages, mentions, assignments))
    if rebuilt_assertions != assertions or rebuilt_cooccurrence != cooccurrence:
        raise ValueError("relation aggregation is not reproducible")
    if len(outcomes) != len(passages) or {row.passage_id for row in outcomes} != passage_ids:
        raise ValueError("relation outcomes do not cover every passage exactly once")
    summary = summary_of(passages, candidates, assertions, cooccurrence, outcomes,
                         coreference_links if config.include_linked_coreference else None)
    if bounded:
        contexts = [RelationContext.model_validate(row) for row in read_jsonl(output / "relation_contexts.jsonl")]
        validate_context_records(contexts, context_transcripts(parent), passages, mentions,
                                 assignments, coreference_links, candidates, config, manifest["run_id"])
        summary.update(context_summary(contexts))
    if summary != manifest["counts"] or summary != json.loads((output / "summary.json").read_bytes()):
        raise ValueError("relation summary differs")
    if report(assertions, entities, summary) != (output / "review.md").read_text(encoding="utf-8"):
        raise ValueError("relation review is not reproducible")
    return manifest


def validate_context_records(contexts, transcripts, passages, mentions, assignments,
                             references, candidates, config, run_id):
    """Check context provenance and budgets offline (no tokenizer download, no inference)."""
    transcript_by_id = {row.transcript_version_id: row for row in transcripts}
    passage_by_id = {row.passage_id: row for row in passages}
    assignment_by_id = {row.mention_id: row for row in assignments}
    mention_by_id = {row.mention_id: row for row in mentions}
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    seen, ready = set(), {}
    for row in contexts:
        if row.context_id in seen or row.created_by_run_id != run_id:
            raise ValueError("duplicate context or invalid context provenance")
        seen.add(row.context_id)
        passage = passage_by_id[row.passage_id]
        transcript = transcript_by_id[row.transcript_version_id]
        if transcript.episode_id != row.episode_id or passage.transcript_version_id != row.transcript_version_id:
            raise ValueError("context crosses transcript versions")
        if not passage.char_start <= row.focal_start < row.focal_end <= passage.char_end:
            raise ValueError("context focal sentence leaves its evidence passage")
        if row.char_end > len(transcript.text):
            raise ValueError("context exceeds transcript")
        text = transcript.text[row.char_start:row.char_end]
        if sha256(text.encode("utf-8")) != row.text_sha256 or len(nlp.make_doc(text)) != row.token_count:
            raise ValueError("context no longer reproduces transcript tokens")
        if len(list(nlp(text).sents)) != row.sentence_count or row.sentence_count > config.max_context_sentences:
            raise ValueError("context sentence budget differs")
        expected_passages = [p.passage_id for p in passages
            if p.transcript_version_id == row.transcript_version_id
            and p.char_start < row.char_end and p.char_end > row.char_start]
        if row.evidence_passage_ids != expected_passages:
            raise ValueError("context evidence passages differ")
        endpoints = {ep.source_id: ep for ep in passage_endpoints(passage, mentions, assignment_by_id, references)}
        for endpoint_id in row.endpoint_ids:
            endpoint = endpoints[endpoint_id]
            if not row.focal_start <= endpoint.char_start < endpoint.char_end <= row.focal_end:
                raise ValueError("context endpoint leaves the focal sentence")
        anchors = set()
        for reference in references:
            if reference.reference_id in row.endpoint_ids:
                anchors.update(mid for mid in reference.anchor_mention_ids if mid in mention_by_id
                    and mention_by_id[mid].transcript_version_id == row.transcript_version_id
                    and mention_by_id[mid].char_end <= reference.char_start
                    and assignment_by_id[mid].corpus_entity_id == reference.corpus_entity_id)
        if not set(row.selected_anchor_ids) <= anchors:
            raise ValueError("context anchor is not supported by resolved coreference")
        included = sorted(mid for mid in row.selected_anchor_ids
                          if row.char_start <= mention_by_id[mid].char_start
                          and mention_by_id[mid].char_end <= row.char_end)
        if included != row.included_anchor_ids:
            raise ValueError("context included anchors differ")
        fits = row.token_count <= config.max_context_tokens and row.encoder_token_count <= config.encoder_max_tokens
        if (row.status == "ready") != fits:
            raise ValueError("context budget status differs")
        if fits:
            ready[(row.passage_id, row.char_start, row.char_end)] = row
    for candidate in candidates:
        context = ready.get((candidate.passage_id, candidate.context_char_start, candidate.context_char_end))
        if context is None or not {candidate.head_mention_id, candidate.tail_mention_id} <= set(context.endpoint_ids):
            raise ValueError("relation candidate lacks a model-ready evidence context")
