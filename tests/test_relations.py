"""Tests for mapping GLiREL output to entities and grouping relations."""

import pytest
import spacy

from spoken_record.coreference_models import CoreferenceLink
from spoken_record.extraction_models import ExtractedMention
from spoken_record.passage_models import Passage
from spoken_record.relation_models import RelationExtractionConfig
from spoken_record.relations import (build_assertions, build_cooccurrence, choose_proposals, passage_endpoints,
                                     project_predictions, sentence_contexts)
from spoken_record.resolution_models import MentionAssignment


def relation_config() -> RelationExtractionConfig:
    """Small predicate list, without loading the model."""
    return RelationExtractionConfig.model_validate({
        "schema_version": 1, "framework": "glirel", "framework_version": "1.2.1",
        "repo_id": "example/model", "revision": "a" * 40, "device": "cpu",
        "encoder_repo_id": "example/encoder", "encoder_revision": "b" * 40,
        "offline_only": True, "tokenizer": "spacy-blank-en-v1",
        "context_unit": "sentence_within_passage",
        "candidate_floor": 0.1, "acceptance_threshold": 0.5,
        "labels": [{"prompt_label": "student of", "canonical_predicate": "student_of",
                    "asymmetric": True,
                    "allowed_head": ["Person"], "allowed_tail": ["Person"]},
                   {"prompt_label": "author of", "canonical_predicate": "authored",
                    "asymmetric": True,
                    "allowed_head": ["Person"], "allowed_tail": ["Work"]}],
    })


def records():
    """One passage with two named mentions and their resolved entities."""
    text = "Aquinas studied under Albert."
    passage = Passage(created_by_run_id="m2", passage_id="p1", episode_id="ep1",
        transcript_version_id="tv1", index=0, char_start=0, char_end=len(text), text=text,
        word_count=4, start_s=0, end_s=2, timing_precision="source_segment",
        source_segment_ids=["s1"], method="sentence_packing", segmentation_fingerprint="seg")
    aquinas = ExtractedMention(created_by_run_id="m4", mention_id="m1", episode_id="ep1",
        transcript_version_id="tv1", char_start=0, char_end=7, text="Aquinas", type="Person",
        score=.9, source_window_ids=["w1"], passage_ids=["p1"], extraction_fingerprint="ext")
    albert = ExtractedMention(created_by_run_id="m4", mention_id="m2", episode_id="ep1",
        transcript_version_id="tv1", char_start=22, char_end=28, text="Albert", type="Person",
        score=.9, source_window_ids=["w1"], passage_ids=["p1"], extraction_fingerprint="ext")
    assignments = [
        MentionAssignment(created_by_run_id="m5", mention_id="m1", episode_entity_id="ee1",
                          corpus_entity_id="ce-aquinas"),
        MentionAssignment(created_by_run_id="m5", mention_id="m2", episode_entity_id="ee2",
                          corpus_entity_id="ce-albert"),
    ]
    return passage, [aquinas, albert], assignments


def context():
    """The model input for the only sentence of the passage."""
    passage, mentions, assignments = records()
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    endpoints = passage_endpoints(passage, mentions, {row.mention_id: row for row in assignments}, [])
    contexts, _, skipped = sentence_contexts(passage, endpoints, nlp)
    [(tokens, ner, by_span, _, _)] = contexts
    return tokens, ner, by_span, skipped


def test_glirel_prediction_is_grounded_and_aggregated():
    """Exclusive token spans from the model map to mentions, entities and one assertion."""
    passage, mentions, assignments = records()
    tokens, ner, by_span, skipped = context()
    assert tokens == ["Aquinas", "studied", "under", "Albert", "."]
    assert ner == [[0, 0, "Person", "Aquinas"], [3, 3, "Person", "Albert"]]
    assert skipped == []
    predictions = [{"head_pos": [0, 1], "tail_pos": [3, 4], "head_text": ["Aquinas"],
                    "tail_text": ["Albert"], "label": "student of", "score": .91}]
    assignment_by_mention = {row.mention_id: row for row in assignments}
    candidates = project_predictions("m6", passage, predictions, by_span,
                                     assignment_by_mention, relation_config(), "fingerprint")
    assert len(candidates) == 1
    assert candidates[0].decision == "accepted"
    assertions = build_assertions("m6", list(candidates))
    assert [(row.subject_corpus_entity_id, row.predicate, row.object_corpus_entity_id)
            for row in assertions] == [("ce-aquinas", "student_of", "ce-albert")]
    assert assertions[0].confidence == .91


def test_low_score_is_retained_but_not_asserted():
    """Weak output above the floor is kept for inspection, but not accepted."""
    passage, mentions, assignments = records()
    _, _, by_span, _ = context()
    predictions = [{"head_pos": [0, 1], "tail_pos": [3, 4], "label": "student of", "score": .2}]
    candidates = project_predictions("m6", passage, predictions, by_span,
        {row.mention_id: row for row in assignments}, relation_config(), "fingerprint")
    assert candidates[0].decision == "below_threshold"
    assert build_assertions("m6", list(candidates)) == ()


def test_cooccurrence_is_undirected_and_passage_grounded():
    """Co-occurrence edges use sorted entity IDs and keep the passage evidence."""
    passage, mentions, assignments = records()
    edges = build_cooccurrence("m6", [passage], mentions, assignments)
    assert len(edges) == 1
    assert (edges[0].left_corpus_entity_id, edges[0].right_corpus_entity_id) == \
           ("ce-albert", "ce-aquinas")
    assert edges[0].passage_ids == ["p1"]


def test_resolved_pronoun_becomes_canonical_relation_endpoint():
    """A linked `He` span maps a GLiREL relation to its Person CorpusEntity."""
    text = "Thomas Aquinas was born in Beirut. He later wrote the Book of Reason."
    passage = Passage(created_by_run_id="m2", passage_id="p-coref", episode_id="ep1",
        transcript_version_id="tv1", index=0, char_start=0, char_end=len(text), text=text,
        word_count=len(text.split()), start_s=0, end_s=4, timing_precision="source_segment",
        source_segment_ids=["s1"], method="sentence_packing", segmentation_fingerprint="seg")
    thomas_start = text.index("Thomas Aquinas")
    book_start = text.index("Book of Reason")
    he_start = text.index("He")
    thomas = ExtractedMention(created_by_run_id="m4", mention_id="m-thomas", episode_id="ep1",
        transcript_version_id="tv1", char_start=thomas_start, char_end=thomas_start + 14,
        text="Thomas Aquinas", type="Person", score=.9, source_window_ids=["w1"],
        passage_ids=["p-coref"], extraction_fingerprint="ext")
    book = ExtractedMention(created_by_run_id="m4", mention_id="m-book", episode_id="ep1",
        transcript_version_id="tv1", char_start=book_start, char_end=book_start + 14,
        text="Book of Reason", type="Work", score=.9, source_window_ids=["w1"],
        passage_ids=["p-coref"], extraction_fingerprint="ext")
    assignments = [
        MentionAssignment(created_by_run_id="m5", mention_id="m-thomas",
                          episode_entity_id="ee-thomas", corpus_entity_id="ce-thomas"),
        MentionAssignment(created_by_run_id="m5", mention_id="m-book",
                          episode_entity_id="ee-book", corpus_entity_id="ce-book"),
    ]
    reference = CoreferenceLink(created_by_run_id="coref", reference_id="cr-he",
        episode_id="ep1", transcript_version_id="tv1", cluster_id="fc-1",
        char_start=he_start, char_end=he_start + 2, text="He",
        reference_kind="third_person_pronoun", passage_ids=["p-coref"],
        anchor_mention_ids=["m-thomas"], episode_entity_ids=["ee-thomas"],
        corpus_entity_id="ce-thomas", decision="linked", reason="single_corpus_entity")
    assignment_by_mention = {row.mention_id: row for row in assignments}
    endpoints = passage_endpoints(passage, [thomas, book], assignment_by_mention, [reference])
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    contexts, _, _ = sentence_contexts(
        passage, endpoints, nlp, "coreference_sentence_with_previous")
    assert len(contexts) == 1
    tokens, _, by_span, context_start, context_end = contexts[0]
    assert "Thomas" in tokens and "He" in tokens and "Book" in tokens
    span_by_id = {endpoint.source_id: span for span, endpoint in by_span.items()}
    prediction = {"head_pos": list(span_by_id["cr-he"]),
                  "tail_pos": list(span_by_id["m-book"]),
                  "label": "author of", "score": .92}
    candidates = project_predictions("m6", passage, [prediction], by_span,
        assignment_by_mention, relation_config(), "fingerprint", context_start, context_end)
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.head_endpoint_kind == "coreference_reference"
    assert candidate.head_corpus_entity_id == "ce-thomas"
    assert candidate.tail_corpus_entity_id == "ce-book"
    assertion = build_assertions("m6", list(candidates))[0]
    assert (assertion.subject_corpus_entity_id, assertion.predicate,
            assertion.object_corpus_entity_id) == ("ce-thomas", "authored", "ce-book")
    assert assertion.evidence[0].head_endpoint_kind == "coreference_reference"


def proposal_config(**changes) -> RelationExtractionConfig:
    """Predicates with wordings, a predicate without direction and a low threshold."""
    data = relation_config().model_dump()
    data.update({"proposals_per_pair": 1, "acceptance_threshold": 0.2,
                 "labels": [{"prompt_label": "student of", "wordings": ["taught by"], "canonical_predicate": "student_of",
                             "asymmetric": True, "allowed_head": ["Person"], "allowed_tail": ["Person"]},
                            {"prompt_label": "agreed with", "canonical_predicate": "agreed_with",
                             "asymmetric": False, "allowed_head": ["Person"], "allowed_tail": ["Person"]}]})
    data.update(changes)
    return RelationExtractionConfig.model_validate(data)


def predict(config, predictions):
    passage, mentions, assignments = records()
    _, _, by_span, _ = context()
    return project_predictions("m6", passage, predictions, by_span,
                               {row.mention_id: row for row in assignments}, config, "fingerprint")


def test_wordings_of_one_predicate_give_one_candidate_with_the_best_score():
    config = proposal_config()
    assert config.prompts() == ["student of", "taught by", "agreed with"]
    rows = predict(config, [{"head_pos": [0, 1], "tail_pos": [3, 4], "label": "student of", "score": .3},
                            {"head_pos": [0, 1], "tail_pos": [3, 4], "label": "taught by", "score": .6}])
    assert [(row.canonical_predicate, row.prompt_label, row.score) for row in rows] == [("student_of", "taught by", .6)]


def test_the_strongest_predicate_is_proposed_in_both_directions():
    """GLiREL ranks the kind of relation, the NLI check decides the direction."""
    rows = choose_proposals(predict(proposal_config(), [
        {"head_pos": [0, 1], "tail_pos": [3, 4], "label": "student of", "score": .4},
        {"head_pos": [3, 4], "tail_pos": [0, 1], "label": "student of", "score": .35},
        {"head_pos": [0, 1], "tail_pos": [3, 4], "label": "agreed with", "score": .3}]), proposal_config())
    kept = sorted((row.head_corpus_entity_id, row.canonical_predicate) for row in rows if row.decision == "accepted")
    assert kept == [("ce-albert", "student_of"), ("ce-aquinas", "student_of")]
    assert [row.decision for row in rows if row.canonical_predicate == "agreed_with"] == ["not_proposed"]


def test_a_predicate_without_direction_is_proposed_once():
    config = proposal_config(proposals_per_pair=2)
    rows = choose_proposals(predict(config, [
        {"head_pos": [0, 1], "tail_pos": [3, 4], "label": "agreed with", "score": .4},
        {"head_pos": [3, 4], "tail_pos": [0, 1], "label": "agreed with", "score": .5}]), config)
    assert [(row.head_corpus_entity_id, row.decision) for row in sorted(rows, key=lambda r: r.head_corpus_entity_id)] == \
        [("ce-albert", "accepted"), ("ce-aquinas", "not_proposed")]
