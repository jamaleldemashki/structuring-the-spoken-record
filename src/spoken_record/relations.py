"""Relation extraction: co-occurrence edges per passage and zero-shot relations with
GLiREL, all tied to transcript evidence.
"""

import itertools
import json
import os
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import spacy

from spoken_record.backends import RelationBackend
from spoken_record.adapters.base import stable_id
from spoken_record.coreference_models import CoreferenceLink
from spoken_record.extraction_models import ExtractedMention
from spoken_record.passage_models import Passage
from spoken_record.relation_models import (CooccurrenceEdge, RelationAssertion, RelationCandidate,
                                           RelationEvidence, RelationExtractionConfig,
                                           RelationOutcome, RelationLabel)
from spoken_record.relation_contexts import antecedent_contexts
from spoken_record.resolution_models import MentionAssignment

# GLiREL's own value for returning every label.
ALL_LABELS = -1


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
    """Map model output to mentions and apply the type, self-link and score checks.

    Several wordings of one predicate give one candidate per pair and direction, the one with
    the best score.
    """
    label_by_prompt = config.label_by_prompt()
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
            prompt_label=prompt, canonical_predicate=label.canonical_predicate,
            score=score, decision=decision, reason=reason))
    best = {}
    for row in rows:
        if row.candidate_id not in best or row.score > best[row.candidate_id].score:
            best[row.candidate_id] = row
    return tuple(sorted(best.values(), key=lambda row: row.candidate_id))


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


def choose_proposals(candidates: tuple[RelationCandidate, ...],
                     config: RelationExtractionConfig) -> tuple[RelationCandidate, ...]:
    """Per context and pair of entities, keep the strongest predicates as proposals.

    A predicate counts with its best score in either direction. It stays proposed in every
    direction that fits its types, since the NLI check, not GLiREL, decides the direction. A
    predicate without direction is proposed once, in its best direction.
    """
    symmetric = {label.canonical_predicate for label in config.labels if not label.asymmetric}
    strength = defaultdict(dict)
    for row in candidates:
        if row.decision != "accepted":
            continue
        pair = (row.passage_id, row.context_char_start,
                *sorted((row.head_corpus_entity_id, row.tail_corpus_entity_id)))
        strength[pair][row.canonical_predicate] = max(row.score, strength[pair].get(row.canonical_predicate, 0))
    chosen = {pair: set(sorted(scores, key=lambda p: (-scores[p], p))[:config.proposals_per_pair])
              for pair, scores in strength.items()}
    best_direction = {}
    for row in candidates:
        if row.decision == "accepted" and row.canonical_predicate in symmetric:
            key = (row.passage_id, row.context_char_start, row.canonical_predicate,
                   *sorted((row.head_corpus_entity_id, row.tail_corpus_entity_id)))
            if key not in best_direction or row.score > best_direction[key].score:
                best_direction[key] = row
    revised = []
    for row in candidates:
        pair = (row.passage_id, row.context_char_start,
                *sorted((row.head_corpus_entity_id, row.tail_corpus_entity_id)))
        if row.decision == "accepted":
            if row.canonical_predicate not in chosen.get(pair, ()):
                row = row.model_copy(update={"decision": "not_proposed",
                    "reason": "stronger predicates were proposed for this pair"})
            elif row.canonical_predicate in symmetric and best_direction[
                    (row.passage_id, row.context_char_start, row.canonical_predicate, *pair[2:])] is not row:
                row = row.model_copy(update={"decision": "not_proposed",
                    "reason": "a predicate without direction is proposed once per pair"})
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


def compute_relations(run_id: str, passages: list[Passage], mentions: list[ExtractedMention],
                      assignments: list[MentionAssignment], available_coreference: list[CoreferenceLink],
                      transcripts, config: RelationExtractionConfig, fingerprint: str, model, budget):
    """GLiREL over every passage, then grouping into assertions and co-occurrence edges.

    ``model`` and ``budget`` can be wrappers that only load GLiREL when a context is not cached.
    """
    coreference_links = ([row for row in available_coreference if row.decision == "linked"]
                         if config.include_linked_coreference else [])
    assignment_by_mention = {row.mention_id: row for row in assignments}
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")

    bounded = config.context_unit == "coreference_antecedent_window"
    documents = {row.transcript_version_id: nlp(row.text) for row in transcripts} if bounded else {}
    prompts = config.prompts()
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
                                                      ner=ner, top_k=ALL_LABELS)
                projected = project_predictions(run_id, passage, predictions, by_token_span,
                    assignment_by_mention, config, fingerprint, context_start, context_end)
                rows.extend(choose_proposals(projected, config))
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
    return candidates, assertions, cooccurrence, outcomes, context_records, coreference_links
