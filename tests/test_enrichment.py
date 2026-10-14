"""Tests for the Wikidata lifespan enrichment: date rules, offline validation, web export."""

import json

import pytest

from test_ingestion import workspace
from test_pipeline import full_recipe
from spoken_record.enrichment import date_range, derive_life, enrich_run, trim_entity, year_interval
from spoken_record.enrichment_models import EnrichmentConfig
from spoken_record.ingestion import dump_json, read_jsonl
from spoken_record.pipeline import run_pipeline, validate_snapshot
from spoken_record.web_export import export_web
from spoken_record.web_models import WebExportConfig


def claim(time, precision=9, rank="normal", **qualifiers):
    snak = lambda t, p: {"snaktype": "value", "datavalue": {"value": {"time": t, "precision": p}}}
    quals = {k: [snak(*v)] for k, v in qualifiers.items() if k in ("P1319", "P1326")}
    if qualifiers.get("circa"):
        quals["P1480"] = [{"snaktype": "value", "datavalue": {"value": {"id": "Q5727902"}}}]
    return {"rank": rank, "mainsnak": snak(time, precision), "qualifiers": quals}


def item(qid="Q1", **claims):
    return trim_entity(qid, {"lastrevid": 7, "claims": claims})


def test_precision_widens_years_and_bce_is_negative():
    assert year_interval("+1225-00-00T00:00:00Z", 9) == (1225, 1225)
    assert year_interval("+1960-00-00T00:00:00Z", 8) == (1960, 1969)
    assert year_interval("+1300-00-00T00:00:00Z", 7) == (1201, 1300)
    assert year_interval("-0500-00-00T00:00:00Z", 7) == (-500, -401)
    assert year_interval("-0384-00-00T00:00:00Z", 9) == (-384, -384)
    assert year_interval("+2000-00-00T00:00:00Z", 6) == (1001, 2000)
    assert year_interval("+1000-00-00T00:00:00Z", 5) is None


def test_best_rank_qualifiers_and_circa():
    rows = item(P570=[claim("+1142-04-21T00:00:00Z", 11, "preferred"), claim("+1150-00-00T00:00:00Z")])["claims"]["P570"]
    assert date_range(rows) == (1142, 1142, False)
    deprecated = item(P569=[claim("+1000-00-00T00:00:00Z", rank="deprecated"), claim("+1033-00-00T00:00:00Z", circa=True)])
    assert date_range(deprecated["claims"]["P569"]) == (1033, 1033, True)
    bounded = item(P569=[claim("-0005-00-00T00:00:00Z", 9, P1319=("-0007-00-00T00:00:00Z", 9), P1326=("-0002-00-00T00:00:00Z", 9))])
    assert date_range(bounded["claims"]["P569"]) == (-7, -2, True)
    conflicting = item(P570=[claim("+1300-00-00T00:00:00Z"), claim("+1303-00-00T00:00:00Z")])
    assert date_range(conflicting["claims"]["P570"]) == (1300, 1303, True)


def test_derive_life_covers_living_unknown_end_and_floruit():
    config = EnrichmentConfig()
    full = derive_life(item(P569=[claim("+1225-00-00T00:00:00Z")], P570=[claim("+1274-00-00T00:00:00Z")]), 2026, config)
    assert (full.start, full.end, full.living, full.approximate, full.basis) == (1225, 1274, False, False, "birth_death")
    living = derive_life(item(P569=[claim("+1946-08-19T00:00:00Z", 11)]), 2026, config)
    assert (living.start, living.end, living.living) == (1946, None, True)
    old = derive_life(item(P569=[claim("+1033-00-00T00:00:00Z")]), 2026, config)
    assert (old.end, old.living) == (None, False)
    floruit = derive_life(item(P1317=[claim("+0500-00-00T00:00:00Z", 7)]), 2026, config)
    assert (floruit.start, floruit.end, floruit.basis, floruit.approximate) == (401, 500, "floruit", True)
    mixed = derive_life(item(P569=[claim("+1000-00-00T00:00:00Z")], P2032=[claim("+1050-00-00T00:00:00Z")]), 2026, config)
    assert (mixed.end, mixed.basis) == (1050, "mixed")
    assert derive_life(item(P106=[]), 2026, config) is None


class FakeFacts:
    """Offline stand-in that returns a fifth-century-BCE philosopher for any QID."""

    def __init__(self, config):
        self.calls = []

    def fetch(self, qids):
        self.calls.append(list(qids))
        philosopher = {"snaktype": "value", "datavalue": {"value": {"id": "Q4964182"}}}
        human = {"snaktype": "value", "datavalue": {"value": {"id": "Q5"}}}
        return {q: trim_entity(q, {"lastrevid": 1, "claims": {
            "P31": [{"rank": "normal", "mainsnak": human}], "P106": [{"rank": "normal", "mainsnak": philosopher}],
            "P569": [claim("-0500-00-00T00:00:00Z", 7)]}}) for q in qids}


@pytest.fixture
def enriched(workspace, monkeypatch):
    root = full_recipe(workspace, monkeypatch, False,
                       {"id": "vault", "kind": "export-obsidian", "config": "vault.concept-navigation-v1.json"})
    run_pipeline(root, "pipeline.json", "full")
    parent = "data/prototype1/runs/full-relations"
    entities = read_jsonl(root / "data/prototype1/runs/full-link/corpus_entities.jsonl")
    panini = next(e for e in entities if e["type"] == "Person")
    (root / "reviewed.jsonl").write_text(json.dumps({"corpus_entity_id": panini["corpus_entity_id"],
                                                     "qid": "Q178031", "reviewer": "test"}) + "\n")
    dump_json(root / "enrich.json", EnrichmentConfig(reviewed_qids="reviewed.jsonl").model_dump())
    monkeypatch.setattr("spoken_record.enrichment.make_retriever", FakeFacts)
    output = enrich_run(root, parent, "enrich.json", "enriched")
    return root, output, panini


def test_enrichment_is_reproducible_offline_and_rejects_tampering(enriched):
    root, output, panini = enriched
    manifest = validate_snapshot(output)
    assert manifest["counts"]["with_lifespan"] == 1 and manifest["counts"]["reviewed"] == 1
    [facts] = read_jsonl(output / "entity_facts.jsonl")
    assert facts["life"] == {"start": -500, "end": None, "living": False, "approximate": True, "basis": "birth_death"}
    assert facts["occupation_qids"] == ["Q4964182"]
    path = output / "wikidata_entities.jsonl"
    path.write_text(path.read_text().replace("-0500", "-0400"))
    with pytest.raises(ValueError):
        validate_snapshot(output)


def test_web_export_carries_lifespans_from_enrichment(enriched):
    root, output, panini = enriched
    dump_json(root / "web.json", WebExportConfig().model_dump())
    web = export_web(root, str(output.relative_to(root)), "web.json", "web-enriched")
    assert validate_snapshot(web)["counts"]["lifespans"] == "enriched"
    entities = {e["id"]: e for e in json.loads((web / "data/entities.json").read_bytes())}
    assert entities[panini["corpus_entity_id"]]["life"]["start"] == -500
    assert entities[panini["corpus_entity_id"]]["qid"] == "Q178031"
