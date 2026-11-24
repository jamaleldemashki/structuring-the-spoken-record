"""Tests for the matching keys of terms (lemmatized with spaCy's English tables)."""

import pytest

from spoken_record.resolution_models import TermKeys
from spoken_record.terms import term_key

pytest.importorskip("en_core_web_sm")

SETTINGS = TermKeys(lemmatize_types=["Concept", "School", "Period"], article_types=["Concept", "School", "Period", "Work"],
                    lemmatizer_version="3.8.0")


@pytest.mark.parametrize("surface, key", [
    ("virtues", "virtue"), ("theories", "theory"), ("potentialities", "potentiality"), ("senses", "sense"),
    ("political virtues", "political virtue"), ("theses", "thesis"), ("criteria", "criterion"),
    ("phenomena", "phenomenon"),
    # Known base nouns and Greek or Latin words stay as they are.
    ("analysis", "analysis"), ("ethics", "ethics"), ("metaphysics", "metaphysics"), ("logos", "logos"),
    ("nous", "nous"), ("physis", "physis"),
    # Articles and hyphens.
    ("the soul", "soul"), ("self-awareness", "self awareness"),
])
def test_concept_keys(surface, key):
    assert term_key(surface, "Concept", SETTINGS) == key


def test_unknown_plurals_digits_works_and_names():
    assert term_key("platonists", "School", SETTINGS) == "platonist"
    assert term_key("1270s", "Period", SETTINGS) == "1270s"
    # A title only loses its article, it is not lemmatized.
    assert term_key("the romance of the rose", "Work", SETTINGS) == "romance of the rose"
    assert term_key("the flowing lights", "Work", SETTINGS) == "flowing lights"
    # Names are never touched.
    assert term_key("aquinas", "Person", SETTINGS) == "aquinas"
    assert term_key("athens", "Place", SETTINGS) == "athens"
    assert term_key("virtues", "Concept", None) == "virtues"


def test_virtue_and_virtues_become_one_entity_in_an_episode():
    from test_discovery import navigation_records
    from spoken_record.resolution import resolve_records
    from spoken_record.resolution_models import ResolutionConfig
    records = navigation_records()
    mentions = [m.model_copy(update={"text": text, "type": "Concept", "mention_id": f"m-{i}"})
                for i, (m, text) in enumerate(zip([records[3][0]] * 2, ["virtue", "virtues"]))]
    config = ResolutionConfig(normalization="unicode-nfc-casefold-whitespace-v1", aliases="aliases.jsonl",
                              cross_episode_exact_types=[], term_keys=SETTINGS)
    episode_entities, entities, *_ = resolve_records("t", "hop", mentions, records[2], [], config, "fp")
    assert len(episode_entities) == 1 and entities[0].observed_surfaces == ["virtue", "virtues"]
