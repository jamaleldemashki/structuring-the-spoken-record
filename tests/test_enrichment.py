"""Tests for the Wikidata enrichment: date rules, images, offline validation, web export."""



from spoken_record.enrichment import (date_range, derive_life, first_image, trim_entity, year_interval)
from spoken_record.enrichment_models import EnrichmentConfig
from spoken_record.web_export import commons_image


def claim(time, precision=9, rank="normal", **qualifiers):
    snak = lambda t, p: {"snaktype": "value", "datavalue": {"value": {"time": t, "precision": p}}}
    quals = {k: [snak(*v)] for k, v in qualifiers.items() if k in ("P1319", "P1326")}
    if qualifiers.get("circa"):
        quals["P1480"] = [{"snaktype": "value", "datavalue": {"value": {"id": "Q5727902"}}}]
    return {"rank": rank, "mainsnak": snak(time, precision), "qualifiers": quals}


def image(name, rank="normal"):
    return {"rank": rank, "mainsnak": {"snaktype": "value", "datavalue": {"value": name, "type": "string"}}}


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


def test_first_best_rank_image_is_used():
    assert first_image(item(P18=[image("Old.jpg"), image("Statue.jpg", "preferred")])) == "Statue.jpg"
    assert first_image(item(P18=[image("Old.jpg", "deprecated")])) is None
    assert first_image(item(P18=[{"rank": "normal", "mainsnak": {"snaktype": "novalue"}}])) is None


def test_commons_links_follow_the_upload_layout():
    links = commons_image("Aristotle Altemps Inv8575.jpg")
    assert links["thumb"] == ("https://upload.wikimedia.org/wikipedia/commons/thumb/a/ae/"
                              "Aristotle_Altemps_Inv8575.jpg/250px-Aristotle_Altemps_Inv8575.jpg")
    assert links["original"] == "https://upload.wikimedia.org/wikipedia/commons/a/ae/Aristotle_Altemps_Inv8575.jpg"
    assert links["page"] == "https://commons.wikimedia.org/wiki/File:Aristotle_Altemps_Inv8575.jpg"
    assert commons_image("Seal.svg")["thumb"].endswith("/250px-Seal.svg.png")
    # A browser can't show a TIFF or a PDF, so those get no image.
    assert commons_image("Scan.tif") is None


class FakeFacts:
    """Offline stand-in that returns a fifth-century-BCE philosopher with a portrait for any QID."""

    def __init__(self, config):
        self.calls = []

    def fetch(self, qids):
        self.calls.append(list(qids))
        philosopher = {"snaktype": "value", "datavalue": {"value": {"id": "Q4964182"}}}
        human = {"snaktype": "value", "datavalue": {"value": {"id": "Q5"}}}
        return {q: trim_entity(q, {"lastrevid": 1, "claims": {
            "P31": [{"rank": "normal", "mainsnak": human}], "P106": [{"rank": "normal", "mainsnak": philosopher}],
            "P569": [claim("-0500-00-00T00:00:00Z", 7)], "P18": [image("Panini bust.jpg")]}}) for q in qids}


def test_a_reviewed_qid_wins_and_facts_carry_lifespan_occupation_and_image():
    from test_consolidation import linked
    from spoken_record.enrichment import build_facts, targets
    from spoken_record.enrichment_models import ReviewedQid
    entities = [linked("ce-1", "Pāṇini", "ep-1", "Q1"), linked("ce-2", "Albert", "ep-1", None, status="ambiguous"),
                linked("ce-3", "Rome", "ep-1", "Q220", kind="Place")]
    reviewed = [ReviewedQid(corpus_entity_id="ce-2", qid="Q60045", reviewer="test")]
    config = EnrichmentConfig()
    wanted = targets(entities, reviewed, config)
    assert wanted == {"ce-1": ("Q1", "linked"), "ce-2": ("Q60045", "reviewed")}
    raw = FakeFacts(config).fetch(sorted({qid for qid, _ in wanted.values()}))
    facts = {row.corpus_entity_id: row for row in build_facts("test", entities, reviewed, raw, 2026, config)}
    assert facts["ce-2"].qid_source == "reviewed" and facts["ce-1"].qid_source == "linked"
    assert facts["ce-1"].life.model_dump() == {"start": -500, "end": None, "living": False,
                                               "approximate": True, "basis": "birth_death"}
    assert facts["ce-1"].occupation_qids == ["Q4964182"] and facts["ce-1"].image == "Panini bust.jpg"
    assert facts["ce-1"].is_human
