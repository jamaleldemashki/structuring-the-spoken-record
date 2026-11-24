import pytest

from spoken_record.entity_linking import (allowed_candidates, apply_decisions, build_contexts,
                                            fullest_surface, make_decisions, merge_results,
                                            popularity_scores, score_candidates, search_queries, type_compatibility)
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


def entity(entity_id="ce-aquinas", label="Thomas Aquinas", surfaces=None):
    return CorpusEntity(created_by_run_id="resolve", corpus_entity_id=entity_id,
        corpus_id="hop", type="Person", canonical_label=label,
        normalized_keys=[label.casefold()], observed_surfaces=surfaces or [label, "Aquinas"],
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


PEOPLE = ["Q5", "Q20643955"]


def test_queries_add_the_fullest_name_and_people_only_full_text_search():
    local = entity(label="Aquinas", surfaces=["Aquinas", "Thomas Aquinas", "Kilward B"])
    assert fullest_surface(local) == "Thomas Aquinas"
    assert search_queries(local, config()) == [("label", "Aquinas")]
    assert search_queries(local, config(search_fullest_surface=True,
                                        fulltext_instance_types={"Person": PEOPLE})) == [
        ("label", "Aquinas"), ("label", "Thomas Aquinas"),
        ("fulltext", "Aquinas"), ("fulltext", "Thomas Aquinas")]
    # A misspelling with the same number of words is not searched.
    assert fullest_surface(entity(label="W.E.B. Du Bois", surfaces=["W.E.B. Du Bois", "W.E.P. Du Bois"])) is None


def test_merged_results_keep_each_qid_once_with_its_best_rank():
    merged = merge_results([
        ("Augustine", [{"qid": "Q1", "retrieval_rank": 1}, {"qid": "Q8018", "retrieval_rank": 2}]),
        ("Augustine full text", [{"qid": "Q8018", "retrieval_rank": 1}, {"qid": "Q3", "retrieval_rank": 2}]),
    ])
    assert [(row["qid"], row["query"], row["retrieval_rank"]) for row in merged] == [
        ("Q1", "Augustine", 1), ("Q8018", "Augustine full text", 2), ("Q3", "Augustine full text", 3)]


def test_person_candidates_need_a_human_instance_type():
    rows = [{"qid": "Q140036109", "instance_of_qids": ["Q7725634"], "retrieval_rank": 1},
            {"qid": "Q868", "instance_of_qids": ["Q5"], "retrieval_rank": 2},
            {"qid": "Q183403", "instance_of_qids": ["Q20643955"], "retrieval_rank": 3}]
    kept = allowed_candidates(entity(), rows, config(required_instance_types={"Person": PEOPLE}))
    assert [(row["qid"], row["retrieval_rank"]) for row in kept] == [("Q868", 1), ("Q183403", 2)]
    assert allowed_candidates(entity(), rows, config()) == rows
    decision = make_decisions("link", [entity()], [], config(required_instance_types={"Person": PEOPLE}))[0]
    assert decision.decision == "nil" and "allowed type" in decision.reason


def test_popularity_only_compares_namesakes():
    local = entity(label="Locke", surfaces=["Locke"])
    scores = popularity_scores(local, [("John Locke", [], 154), ("Gary Locke", [], 25),
                                       ("Bill Clinton", ["William Jefferson Clinton"], 175),
                                       ("Locke", [], 0)])
    assert scores == [1.0, 25 / 154, 0.0, 0.0]
    # Namesakes without any sitelink: the score says nothing, so nobody loses points.
    assert popularity_scores(local, [("Locke", [], 0), ("Locke", [], None)]) == [1.0, 1.0]


def test_popularity_breaks_a_tie_between_two_namesakes():
    local = entity(label="Augustine", surfaces=["Augustine"])
    raw = {local.corpus_entity_id: [
        {"qid": "Q110735140", "label": "Augustine", "description": "Swedish singer", "aliases": [],
         "instance_of_qids": ["Q5"], "retrieval_rank": 1, "sitelinks": 4},
        {"qid": "Q8018", "label": "Augustine of Hippo", "description": "theologian",
         "aliases": ["Augustine"], "instance_of_qids": ["Q5"], "retrieval_rank": 2, "sitelinks": 227},
    ]}

    class SameContext:
        """Both candidates get the same context score."""

        def encode(self, texts):
            return [[1.0, 0.0] for _ in texts]

    settings = config(name_weight=.30, type_weight=.15, context_weight=.35, popularity_weight=.20,
                      minimum_margin=.10)
    candidates = score_candidates("link", [local], raw, {local.corpus_entity_id: "context"},
                                  settings, SameContext())
    assert [row.qid for row in candidates] == ["Q8018", "Q110735140"]
    assert candidates[0].popularity == 1 and candidates[1].popularity == 4 / 227
    decision = make_decisions("link", [local], list(candidates), settings)[0]
    assert decision.decision == "linked" and decision.selected_qid == "Q8018"


@pytest.mark.parametrize("updates", [dict(popularity_weight=.2),
                                     dict(required_instance_types={"Person": []}),
                                     dict(fulltext_instance_types={"Person": ["human"]})])
def test_invalid_linking_settings_are_refused(updates):
    with pytest.raises(ValueError):
        config(**updates)


RAW_ENTITY = {"labels": {"en": {"language": "en", "value": "Aristotle"}},
              "descriptions": {"en": {"language": "en", "value": "Greek philosopher"}},
              "aliases": {"en": [{"language": "en", "value": "Aristoteles"}]},
              "claims": {"P31": [{"rank": "normal", "mainsnak": {"snaktype": "value",
                                  "datavalue": {"value": {"entity-type": "item", "id": "Q5"}}}},
                                 {"rank": "normal", "mainsnak": {"snaktype": "somevalue"}}],
                         "P18": [{"rank": "normal", "mainsnak": {"datavalue": {"value": "Aristotle.jpg"}}}],
                         "P569": [{"rank": "normal", "mainsnak": {"datavalue": {"value": {"time": "-0384"}}}}]},
              "sitelinks": {"enwiki": {"title": "Aristotle"}, "dewiki": {"title": "Aristoteles"},
                            "commonswiki": {"title": "Category:Aristotle"}}}


def test_the_cache_keeps_every_field_the_linker_reads():
    from spoken_record.entity_linking import WikidataRetriever

    class Raw(WikidataRetriever):
        def _fetch(self, params):
            return {"entities": {"Q868": RAW_ENTITY}}

    class Unreduced(Raw):
        def entities(self, qids, params, keep):
            return super().entities(qids, params, lambda payload: payload)

    settings = config(request_delay_seconds=0)
    assert Raw(settings).details(["Q868"]) == Unreduced(settings).details(["Q868"]) == {"Q868": {
        "aliases": ["Aristoteles"], "instance_of_qids": ["Q5"], "label": "Aristotle",
        "description": "Greek philosopher", "sitelinks": 2}}


def test_the_cache_keeps_every_claim_enrichment_reads():
    from spoken_record.enrichment import fact_fields, trim_entity

    reduced = fact_fields({"entities": {"Q868": {**RAW_ENTITY, "lastrevid": 7}, "Q1": {"missing": ""}}})
    assert trim_entity("Q868", reduced["entities"]["Q868"]) == trim_entity("Q868", {**RAW_ENTITY, "lastrevid": 7})
    assert trim_entity("Q1", reduced["entities"]["Q1"])["missing"] is True
    assert "sitelinks" not in reduced["entities"]["Q868"]


def test_item_details_are_cached_one_by_one_so_new_batches_reuse_them(tmp_path):
    from spoken_record.cache import Cache
    from spoken_record.entity_linking import WikidataRetriever

    class Counting(WikidataRetriever):
        asked = []

        def _fetch(self, params):
            ids = params["ids"].split("|")
            self.asked.append(ids)
            return {"entities": {qid: {**RAW_ENTITY, "labels": {"en": {"value": qid}}} for qid in ids if qid != "Q404"}}

    retriever = Counting(config(request_delay_seconds=0), cache=Cache(tmp_path / "wikidata.sqlite"))
    first = retriever.details(["Q1", "Q2", "Q404"])
    second = retriever.details(["Q2", "Q3", "Q1"])
    assert Counting.asked == [["Q1", "Q2", "Q404"], ["Q3"]]
    assert set(first) == {"Q1", "Q2"} and first["Q2"] == second["Q2"] and second["Q3"]["label"] == "Q3"
    # An item the API did not return is remembered too, and not asked for again.
    retriever.details(["Q404"])
    assert Counting.asked[-1] == ["Q3"]

