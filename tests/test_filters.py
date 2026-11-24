"""Tests for the boilerplate and generic-term filters, and how the graph and app use them."""

import json
from types import SimpleNamespace

import pytest

from test_ingestion import PROFILE
from test_update import loaders, quiet, web_data, write_episode, write_steps
from spoken_record.build import update
from spoken_record.filters import BoilerplateConfig, GenericConfig, boilerplate_spans, generic_entities
from spoken_record.ingestion import dump_json
from spoken_record.validate import validate_build
from spoken_record.web_export import co_discussed_edges
from spoken_record.web_models import WebExportConfig

INTRO = "Hi, I'm Anna Host and you are listening to the Synthetic Podcast, brought to you by the {sponsor}. "


def bundle(episode_id, text):
    return SimpleNamespace(episode=SimpleNamespace(episode_id=episode_id, language="en"),
                           transcript=SimpleNamespace(transcript_version_id=f"t-{episode_id}", text=text))


def test_a_repeated_intro_is_boilerplate_even_with_small_changes():
    episodes = [bundle("e1", INTRO.format(sponsor="Philosophy Department") + "Plato wrote about the soul and the city."),
                bundle("e2", INTRO.format(sponsor="Philosophy Department and the LMU") + "Kant wrote about reason."),
                bundle("e3", "Something else entirely happens in this episode of a different show.")]
    spans = boilerplate_spans(episodes, BoilerplateConfig())
    for row in episodes[:2]:
        [(start, end)] = spans[row.transcript.transcript_version_id]
        assert row.transcript.text[start:end].startswith("Hi, I'm Anna Host")
    assert spans["t-e3"] == []
    # A sentence that only one episode has is never boilerplate.
    assert not any("Plato" in episodes[0].transcript.text[a:b] for a, b in spans["t-e1"])


def test_lowercase_names_are_generic_but_frequent_topics_are_not():
    def entity(entity_id, kind, surfaces, episodes=1, status="ambiguous"):
        return SimpleNamespace(corpus_entity_id=entity_id, type=kind, observed_surfaces=surfaces,
                               episode_count=episodes, link_status=status)
    entities = [entity("philosopher", "Person", ["philosopher"]), entity("plato", "Person", ["Plato"]),
                entity("scholastics", "School", ["scholastics"]), entity("god", "Concept", ["God"], episodes=6),
                entity("aristotle", "Person", ["Aristotle"], episodes=10, status="linked")]
    # "God" in six of ten episodes is a topic, not noise; groups and schools are often lowercase.
    assert generic_entities(entities, GenericConfig()) == {"philosopher"}


def test_every_pair_of_the_configured_types_becomes_an_edge():
    pairs = [{"id": f"wc-{a}{b}", "source": a, "target": b, "count": count, "pmi": 1.0, "evidence": ["p1"]}
             for a, b, count in (("h", "a", 5), ("h", "b", 1), ("h", "w", 2))]
    types = {**dict.fromkeys("hab", "Person"), "w": "Place"}
    kept = {(e["source"], e["target"]) for e in co_discussed_edges(pairs, types, WebExportConfig(min_shared_passages=1))}
    # The app decides which of them to draw, so a pair named only once is still exported.
    assert kept == {("h", "a"), ("h", "b")}
    assert {(e["source"], e["target"]) for e in co_discussed_edges(pairs, types, WebExportConfig(min_shared_passages=2))} == {("h", "a")}


@pytest.fixture
def podcast_with_intros(tmp_path):
    root = tmp_path
    (root / "HoP/01_whisper").mkdir(parents=True)
    # The fake GLiNER tags "Pāṇini", so the intro holds a mention that must not count.
    texts = {"HoP001.json": [INTRO.format(sponsor="Pāṇini Society"),
                             "Pāṇini studies free will and words. ", "He defends free will against his critics. "] * 3,
             "HoP002.json": [INTRO.format(sponsor="Pāṇini Society and the LMU"),
                             "Later Pāṇini returns to free will. ", "He doubts that words are enough. "] * 3}
    audit = [write_episode(root, name, number, parts) for number, (name, parts) in enumerate(texts.items(), 1)]
    (root / "audit.jsonl").write_text("".join(json.dumps(row) + "\n" for row in audit))
    (root / "meta.jsonl").write_text("".join(json.dumps({"filename": name, "title": f"{n}. Synthetic",
                                                          "series": "Medieval"}) + "\n" for n, name in enumerate(texts, 1)))
    (root / "quarantine.jsonl").write_text("")
    dump_json(root / "profile.json", PROFILE)
    dump_json(root / "filters.json", {"schema_version": 1})
    steps = {**write_steps(root), "filters": "filters.json"}
    dump_json(root / "corpus.json", {"schema_version": 1, "corpus_id": "hop", "profile": "profile.json",
              "metadata": "meta.jsonl", "audit": "audit.jsonl", "quarantine": "quarantine.jsonl",
              "episode_codes": {"series_field": "sub_show", "number_fields": ["title"]}, "steps": steps})
    return root


def test_mentions_in_the_intro_are_not_evidence_and_the_build_still_validates(podcast_with_intros):
    root = podcast_with_intros
    build = update(root, "corpus.json", "HoP 1-2", "filtered", loaders=loaders(), progress=quiet)
    graph = json.loads((build / "graph.json").read_bytes())
    assert graph["counts"]["boilerplate_sentences"] == 6
    panini = next(n for n in graph["nodes"] if n["label"] == "Pāṇini")
    assert panini["boilerplate_mentions"] == 6 and len(panini["mentions"]) == 6
    passage = {p["id"]: p for p in graph["passages"]}
    spans = [(b["episode_id"], b["char_start"], b["char_end"]) for b in graph["boilerplate"]]
    for mention in panini["mentions"]:
        row = passage[mention["passage_id"]]
        start = row["char_start"] + mention["start"]
        assert not any(episode == row["episode_id"] and a <= start < b for episode, a, b in spans)
    assert validate_build(root, build)["skipped"] == []
    entities = json.loads(web_data(root, "filtered")["entities.json"])
    shown = next(e for e in entities if e["label"] == "Pāṇini")
    # The app counts his six real mentions, not the six in the intros.
    assert shown["mention_count"] == 6 and len(shown["occurrences"]) == 6
