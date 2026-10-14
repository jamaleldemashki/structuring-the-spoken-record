import pytest

from spoken_record.entity_linking import (apply_decisions, build_contexts, make_decisions,
                                            score_candidates, type_compatibility)
from spoken_record.extraction_models import ExtractedMention
from spoken_record.linking_models import EntityLinkingConfig
from spoken_record.passage_models import Passage
from spoken_record.resolution_models import CorpusEntity, MentionAssignment


def config(**updates):
    values = dict(entity_types=["Person"], embedding_repo_id="test/model",
        embedding_revision="a" * 40, name_weight=.35, type_weight=.25,
        context_weight=.40, linked_threshold=.68, minimum_margin=.08)
    values.update(updates)
    return EntityLinkingConfig(**values)


def entity(entity_id="ce-aquinas", label="Thomas Aquinas"):
    return CorpusEntity(created_by_run_id="resolve", corpus_entity_id=entity_id,
        corpus_id="hop", type="Person", canonical_label=label,
        normalized_keys=[label.casefold()], observed_surfaces=[label, "Aquinas"],
        reviewed_aliases=["Aquinas"], episode_entity_ids=[f"ee-{entity_id}"],
        episode_ids=["ep-1"], mention_count=1, episode_count=1,
        resolution_method="reviewed_alias", resolution_status="reviewed",
        registry_key=f"reviewed:{entity_id}")


@pytest.mark.parametrize("vectors", [[], [[]], [[float("nan")]], [[1.0], [0.0]]])
def test_malformed_encoder_response_is_rejected(vectors):
    """Missing, empty, non-finite or extra vectors must raise instead of changing the ranking."""
    class BadEncoder:
        """Breaks the text-encoder contract on purpose."""

        def encode(self, texts):
            """Returns the broken response being tested."""
            return vectors

    item = entity()
    with pytest.raises(ValueError, match="encoder must return"):
        score_candidates("run", [item], {}, {item.corpus_entity_id: "context"}, config(), BadEncoder())


class FixedEncoder:
    """One context vector, then two candidate vectors."""

    def encode(self, texts):
        assert len(texts) == 3
        return [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]


def test_contextual_ranking_links_clear_candidate_and_enriches_entity():
    local = entity()
    raw = {local.corpus_entity_id: [
        {"qid": "Q9438", "label": "Thomas Aquinas", "description": "Italian philosopher and theologian",
         "aliases": ["Aquinas"], "instance_of_qids": ["Q5"], "retrieval_rank": 1},
        {"qid": "Q1", "label": "Aquinas", "description": "album", "aliases": [],
         "instance_of_qids": [], "retrieval_rank": 2},
    ]}
    candidates = score_candidates("link", [local], raw, {local.corpus_entity_id: "medieval philosopher"},
                                  config(), FixedEncoder())
    assert candidates[0].qid == "Q9438"
    assert candidates[0].type_compatibility == 1
    assert candidates[0].context_similarity == 1
    assert candidates[1].type_compatibility == 0
    decisions = make_decisions("link", [local], list(candidates), config())
    assert decisions[0].decision == "linked" and decisions[0].selected_qid == "Q9438"
    enriched = apply_decisions("link", [local], list(decisions))[0]
    assert (enriched.wikidata_qid, enriched.link_status) == ("Q9438", "linked")


def test_missing_and_close_candidates_abstain_explicitly():
    local = entity()
    assert make_decisions("link", [local], [], config())[0].decision == "nil"
    raw = {local.corpus_entity_id: [
        {"qid": "Q1", "label": "Thomas Aquinas", "description": "philosopher", "aliases": [],
         "instance_of_qids": ["Q5"], "retrieval_rank": 1},
        {"qid": "Q2", "label": "Thomas Aquinas", "description": "philosopher", "aliases": [],
         "instance_of_qids": ["Q5"], "retrieval_rank": 2},
    ]}
    candidates = score_candidates("link", [local], raw, {local.corpus_entity_id: "context"},
                                  config(), FixedEncoder())
    # The second vector is orthogonal, so set a margin that can't be reached.
    decision = make_decisions("link", [local], list(candidates),
                              config(minimum_margin=1))[0]
    assert decision.decision == "ambiguous" and decision.selected_qid is None


def test_passage_context_is_bounded_and_keeps_evidence_ids():
    local = entity()
    text = "Thomas Aquinas wrote about reason. " * 20
    passage = Passage(created_by_run_id="segment", passage_id="p-1", episode_id="ep-1",
        transcript_version_id="tv-1", index=0, char_start=0, char_end=len(text), text=text,
        word_count=len(text.split()), start_s=0, end_s=10, timing_precision="source_segment",
        source_segment_ids=["s-1"], method="sentence_packing", segmentation_fingerprint="seg")
    mention = ExtractedMention(created_by_run_id="extract", mention_id="m-1", episode_id="ep-1",
        transcript_version_id="tv-1", char_start=0, char_end=14, text="Thomas Aquinas",
        type="Person", score=.9, source_window_ids=["w-1"], passage_ids=["p-1"],
        extraction_fingerprint="extract")
    assignment = MentionAssignment(created_by_run_id="resolve", mention_id="m-1",
        episode_entity_id="ee-1", corpus_entity_id=local.corpus_entity_id)
    contexts, evidence = build_contexts([local], [mention], [assignment], [passage], 200)
    assert evidence[local.corpus_entity_id] == ["p-1"]
    assert len(contexts[local.corpus_entity_id]) <= 280
    assert "Podcast evidence" in contexts[local.corpus_entity_id]


def test_type_compatibility_uses_qid_and_description_evidence():
    assert type_compatibility("Person", {"instance_of_qids": ["Q5"], "description": ""}) == 1
    assert type_compatibility("Language", {"instance_of_qids": [],
                                           "description": "West Germanic language"}) == 1
    assert type_compatibility("Person", {"instance_of_qids": [], "description": "studio album"}) == 0
