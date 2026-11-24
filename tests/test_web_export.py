"""Tests for the knowledge graph and the web app data made from it: edge meaning, passage
offsets and the site config."""

import math
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from spoken_record.graph import co_occurrence, primary_passage
from spoken_record.web_export import co_discussed_edges, episode_code, episode_title, relation_edges, site_row
from spoken_record.web_models import SiteConfig, WebExportConfig


def occ(*pids):
    return [{"passage_id": p, "start": 0, "end": 1} for p in pids]


def test_co_occurrence_counts_distinct_passages_and_the_app_keeps_configured_types():
    found = {"a": occ("p1", "p1", "p2", "p3"), "b": occ("p1", "p2"), "c": occ("p1"), "place": occ("p1", "p2")}
    pairs = co_occurrence(found, 10)
    ab = next(row for row in pairs if (row["source"], row["target"]) == ("a", "b"))
    assert ab["count"] == 2 and ab["evidence"] == ["p1", "p2"]
    assert ab["pmi"] == round(math.log(2 * 10 / (3 * 2)), 4)
    # The graph keeps every pair, also places and single shared passages.
    assert {(row["source"], row["target"]) for row in pairs} >= {("a", "place"), ("a", "c")}
    types = {"a": "Person", "b": "Concept", "c": "Person", "place": "Place"}
    edges = co_discussed_edges(pairs, types, WebExportConfig())
    assert [(e["source"], e["target"], e["count"], e["kind"]) for e in edges] == [("a", "b", 2, "co_discussed")]


def test_only_supported_relations_become_edges_with_their_supported_evidence():
    def evidence(passage, nli, score):
        return {"passage_id": passage, "nli": nli, "score": score}
    relations = [{"source": "s", "target": "o", "predicate": "authored", "decision": "supported",
                  "supported_confidence": 0.81234, "evidence": [evidence("p2", "supported", .81234),
                  evidence("p1", "supported", .7), evidence("p2", "supported", .6), evidence("p3", "abstain", .9)]},
                 {"source": "s", "target": "x", "predicate": "student_of", "decision": "abstain",
                  "supported_confidence": None, "evidence": [evidence("p4", "abstain", .9)]}]
    [edge] = relation_edges(relations, WebExportConfig())
    assert (edge["source"], edge["target"], edge["label"], edge["status"]) == ("s", "o", "wrote", "nli_supported")
    assert edge["evidence"] == ["p1", "p2"] and edge["confidence"] == 0.8123
    assert edge["inverse_label"] == "wrote (by)", "without a configured inverse the direction stays readable"
    [edge] = relation_edges(relations, WebExportConfig(inverse_labels={"authored": "written by"}))
    assert edge["inverse_label"] == "written by"
    assert "group" not in edge
    [edge] = relation_edges(relations, WebExportConfig(predicate_groups={"authored": "Texts"}))
    assert edge["group"] == "Texts"
    assert relation_edges(relations, WebExportConfig(hidden_predicates=["authored"])) == []


def test_primary_passage_is_the_one_containing_the_first_character():
    passages = {"p1": SimpleNamespace(char_start=0, char_end=10), "p2": SimpleNamespace(char_start=10, char_end=20)}
    assert primary_passage(SimpleNamespace(mention_id="m", char_start=12, passage_ids=["p1", "p2"]), passages) == "p2"
    with pytest.raises(ValueError, match="outside"):
        primary_passage(SimpleNamespace(mention_id="m", char_start=25, passage_ids=["p1"]), passages)


def test_site_config_drives_episode_codes_and_lists_every_present_type():
    episode = {"title": "243. Aquinas on Being", "source_key": "ep-243", "facets": {"sub_show": "HoP"}}
    hop = SiteConfig(series_facet="sub_show", episode_number_pattern=r"\s*(\d+)\.")
    assert (episode_code(episode, hop), episode_title(episode, hop)) == ("HoP 243", "Aquinas on Being")
    generic = SiteConfig()
    assert (episode_code(episode, generic), episode_title(episode, generic)) == ("ep-243", "243. Aquinas on Being")
    row = site_row(generic, {"Person", "Language"})
    assert row["entity_types"][-1] == {"type": "Language", "label": "Language", "plural": "Language", "visible_by_default": False}
    assert row["series_label"] is None and row["lifespan_type"] == "Person"
    with pytest.raises(ValidationError):
        SiteConfig(episode_number_pattern=r"\d+")
    with pytest.raises(ValidationError):
        SiteConfig(lived_presets=[{"label": "Backwards", "start": 1500, "end": 500}])
