"""Tests for the web app export: edge meaning, offsets, determinism and tamper checks."""

import json
import math
from types import SimpleNamespace

import pytest

from test_ingestion import workspace
from test_pipeline import full_recipe
from spoken_record.pipeline import run_pipeline, validate_snapshot
from pydantic import ValidationError

from spoken_record.web_export import co_discussed_edges, episode_code, episode_title, primary_passage, relation_edges, site_row
from spoken_record.web_models import SiteConfig, WebExportConfig

WEB_STAGE = {"id": "web", "kind": "export-web", "config": "web.json"}


def occ(*pids):
    return [{"passage_id": p, "start": 0, "end": 1} for p in pids]


def test_co_discussed_counts_distinct_passages_and_respects_types_and_threshold():
    occurrences = {"a": occ("p1", "p1", "p2", "p3"), "b": occ("p1", "p2"), "c": occ("p1"), "place": occ("p1", "p2")}
    types = {"a": "Person", "b": "Concept", "c": "Person", "place": "Place"}
    edges = co_discussed_edges(occurrences, types, 10, WebExportConfig())
    assert [(e["source"], e["target"], e["count"]) for e in edges] == [("a", "b", 2)]
    assert edges[0]["evidence"] == ["p1", "p2"]
    assert edges[0]["pmi"] == round(math.log(2 * 10 / (3 * 2)), 4)
    assert edges[0]["kind"] == "co_discussed"


def test_relation_edges_are_directed_named_and_deduplicate_evidence():
    evidence = [SimpleNamespace(passage_id="p2"), SimpleNamespace(passage_id="p1"), SimpleNamespace(passage_id="p2")]
    assertion = SimpleNamespace(subject_corpus_entity_id="s", object_corpus_entity_id="o", predicate="authored",
                                confidence=0.81234, evidence=evidence)
    [edge] = relation_edges([assertion], "nli_supported", WebExportConfig())
    assert (edge["source"], edge["target"], edge["label"], edge["status"]) == ("s", "o", "wrote", "nli_supported")
    assert edge["evidence"] == ["p1", "p2"] and edge["confidence"] == 0.8123
    other = SimpleNamespace(**{**vars(assertion), "predicate": "student_of"})
    assert relation_edges([other], "unverified", WebExportConfig())[0]["label"] == "student of"


def test_primary_passage_is_the_one_containing_the_first_character():
    passages = {"p1": SimpleNamespace(char_start=0, char_end=10), "p2": SimpleNamespace(char_start=10, char_end=20)}
    assert primary_passage(SimpleNamespace(mention_id="m", char_start=12, passage_ids=["p1", "p2"]), passages) == "p2"
    with pytest.raises(ValueError, match="outside"):
        primary_passage(SimpleNamespace(mention_id="m", char_start=25, passage_ids=["p1"]), passages)


@pytest.fixture
def web_run(workspace, monkeypatch):
    root = full_recipe(workspace, monkeypatch, True, WEB_STAGE)
    (root / "web.json").write_text(WebExportConfig().model_dump_json())
    result = run_pipeline(root, "pipeline.json", "web")
    return root, result


def test_pipeline_exports_valid_web_data(web_run):
    root, result = web_run
    output = root / result["final_output"]
    manifest = validate_snapshot(output)
    assert manifest["stage"] == "export_web" and result["completed_stages"] == 10
    data = {name: json.loads((output / "data" / f"{name}.json").read_bytes()) for name in
            ("episodes", "passages", "entities", "edges", "related", "summary")}
    passages = {p["id"]: p for p in data["passages"]}
    mentions = [json.loads(line) for line in (root / "data/prototype1/runs/web-ner/mentions.jsonl").read_text().splitlines()]
    surfaces = {m["text"] for m in mentions}
    occurrences = [o for e in data["entities"] for o in e["occurrences"]]
    assert len(occurrences) == len(mentions)
    for o in occurrences:
        assert passages[o["passage_id"]]["text"][o["start"]:o["end"]] in surfaces
    assert data["summary"]["relation_status"] == "unverified"
    assert data["summary"]["lifespans"] == "not_enriched"
    assert run_pipeline(root, "pipeline.json", "web", resume=True) == result


def test_tampered_web_data_is_rejected(web_run):
    root, result = web_run
    output = root / result["final_output"]
    path = output / "data/entities.json"
    path.write_bytes(path.read_bytes().replace(b'"label"', b'"labe1"', 1))
    with pytest.raises(ValueError, match="not reproducible"):
        validate_snapshot(output)


def test_site_config_drives_episode_codes_and_lists_every_present_type():
    episode = SimpleNamespace(title="243. Aquinas on Being", source_key="ep-243", facets={"sub_show": "HoP"})
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
