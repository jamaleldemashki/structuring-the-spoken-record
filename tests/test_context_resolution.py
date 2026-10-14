import pytest
from pydantic import ValidationError

from spoken_record.context_models import ContextResolutionConfig
from spoken_record.context_resolution import build_candidates, surface_relation
from spoken_record.extraction_models import ExtractedMention
from spoken_record.passage_models import Passage
from spoken_record.resolution_models import CorpusEntity, MentionAssignment


def entity(entity_id, label, keys, entity_type="Person", episode_id="e1"):
    return CorpusEntity(created_by_run_id="resolve", corpus_entity_id=entity_id, corpus_id="hop",
        type=entity_type, canonical_label=label, normalized_keys=keys, observed_surfaces=[label],
        reviewed_aliases=[], episode_entity_ids=[f"ee-{entity_id}"], episode_ids=[episode_id],
        mention_count=1, episode_count=1, resolution_method="episode_local",
        resolution_status="provisional", registry_key=f"episode:{entity_id}")


def passage(episode_id, passage_id, text):
    return Passage(created_by_run_id="passages", passage_id=passage_id, episode_id=episode_id,
        transcript_version_id=f"t-{episode_id}", index=0, char_start=0, char_end=len(text), text=text,
        word_count=len(text.split()), timing_precision="unavailable", source_segment_ids=[],
        method="token_fallback", segmentation_fingerprint="segment-v1")


def mention(entity_row, passage_row, mention_id):
    start = passage_row.text.index(entity_row.canonical_label)
    return ExtractedMention(created_by_run_id="extract", mention_id=mention_id,
        episode_id=passage_row.episode_id, transcript_version_id=passage_row.transcript_version_id,
        char_start=start, char_end=start + len(entity_row.canonical_label), text=entity_row.canonical_label,
        type=entity_row.type, score=.9, source_window_ids=[f"w-{mention_id}"],
        passage_ids=[passage_row.passage_id], extraction_fingerprint="extract-v1")


def config(**overrides):
    values = dict(candidate_types=["Person", "Place"],
        suggestion_methods=["exact_key", "person_name_containment"], block_conflicting_numbers=True,
        fuzzy_similarity_floor=.88,
        suggestion_threshold=.58, minimum_context_similarity=.05,
        surface_weight=.45, lexical_context_weight=.4, neighbor_weight=.15)
    values.update(overrides)
    return ContextResolutionConfig(**values)


def test_surface_retrieval_handles_exact_suffix_and_fuzzy_forms():
    aquinas = entity("ce-a", "Aquinas", ["aquinas"])
    thomas = entity("ce-t", "Thomas Aquinas", ["thomas aquinas"], episode_id="e2")
    aristotle = entity("ce-r", "Aristotle", ["aristotle"])
    aristotel = entity("ce-s", "Aristotel", ["aristotel"], episode_id="e2")
    assert surface_relation(aquinas, thomas, .88) == (("person_name_containment",), .9)
    methods, score = surface_relation(aristotle, aristotel, .88)
    assert methods == ("fuzzy_surface",) and score >= .88
    century_13 = entity("ce-13", "13th century", ["13th century"], entity_type="Period")
    century_15 = entity("ce-15", "15th century", ["15th century"], entity_type="Period")
    assert surface_relation(century_13, century_15, .88) == ((), 0.0)


def test_candidate_analysis_uses_context_and_keeps_types_separate():
    aquinas = entity("ce-a", "Aquinas", ["aquinas"], episode_id="e1")
    thomas = entity("ce-t", "Thomas Aquinas", ["thomas aquinas"], episode_id="e2")
    paris = entity("ce-p", "Paris", ["paris"], entity_type="Place", episode_id="e2")
    passages = [
        passage("e1", "p1", "Aquinas discusses essence and being"),
        passage("e2", "p2", "Thomas Aquinas discusses essence and being"),
        passage("e2", "p3", "Paris is a city in France"),
    ]
    mentions = [mention(aquinas, passages[0], "m1"), mention(thomas, passages[1], "m2"),
                mention(paris, passages[2], "m3")]
    assignments = [MentionAssignment(created_by_run_id="resolve", mention_id=m.mention_id,
        episode_entity_id=f"ee-{row.corpus_entity_id}", corpus_entity_id=row.corpus_entity_id)
        for m, row in zip(mentions, (aquinas, thomas, paris))]
    candidates = build_candidates("context", [aquinas, thomas, paris], mentions, assignments,
                                  passages, config())
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.retrieval_methods == ["person_name_containment"]
    assert candidate.lexical_context_similarity == pytest.approx(1)
    assert candidate.decision == "suggest_merge"
    assert candidate.evidence_passage_ids == ["p1", "p2"]


def test_context_weights_must_sum_to_one():
    with pytest.raises(ValidationError, match="weights must sum to one"):
        config(surface_weight=.5)


def test_fuzzy_candidate_abstains_even_with_matching_context():
    left = entity("ce-l", "Aristotle", ["aristotle"], episode_id="e1")
    right = entity("ce-r", "Aristotel", ["aristotel"], episode_id="e2")
    passages = [passage("e1", "p1", "Aristotle discusses causes and substance"),
                passage("e2", "p2", "Aristotel discusses causes and substance")]
    mentions = [mention(left, passages[0], "m1"), mention(right, passages[1], "m2")]
    assignments = [MentionAssignment(created_by_run_id="resolve", mention_id=m.mention_id,
        episode_entity_id=f"ee-{row.corpus_entity_id}", corpus_entity_id=row.corpus_entity_id)
        for m, row in zip(mentions, (left, right))]
    candidates = build_candidates("context", [left, right], mentions, assignments, passages,
                                  config(suggestion_threshold=.1))
    assert candidates[0].retrieval_methods == ["fuzzy_surface"]
    assert candidates[0].decision == "abstain"
    assert "not allowed" in candidates[0].reason
