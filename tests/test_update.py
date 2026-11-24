"""Tests for ``update``: episode folders, the corpus build and the caches.

Two synthetic HoP episodes go through every step with offline fakes for GLiNER, FCoref,
GLiREL, the NLI model, MPNet and Wikidata.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from test_discovery import FakeEncoder
from test_enrichment import FakeFacts
from test_extraction import FakeModel
from test_ingestion import PROFILE, transcript
from spoken_record.adapters.base import sha256
from spoken_record.build import update
from spoken_record.corpus import load_corpus, select_episodes
from spoken_record.ingestion import dump_json, read_jsonl
from spoken_record.verification_models import NLIScores
from spoken_record.web_export import DATA_FILES

REPOSITORY = Path(__file__).resolve().parents[1]
TEXTS = {
    "HoP001.json": ["Pāṇini studies free will and words. ", "He defends free will against his critics. "] * 6,
    "HoP002.json": ["Later Pāṇini returns to free will. ", "He doubts that words are enough. "] * 6,
}


class FakeCoreference:
    """One cluster per transcript: every "Pāṇini" and every "He"."""

    def predict(self, *, texts, max_tokens_in_batch):
        return [FakeClusters(text) for text in texts]


class FakeClusters:
    def __init__(self, text):
        self.text = text

    def get_clusters(self, as_strings=False):
        spans = []
        for needle in ("Pāṇini", "He "):
            start = self.text.find(needle)
            while start >= 0:
                spans.append((start, start + len(needle.strip())))
                start = self.text.find(needle, start + 1)
        return [sorted(spans)]


class FakeGLiREL:
    """A Person argues for a Concept whenever both are in the sentence."""

    def __init__(self):
        self.calls = 0

    def predict_relations(self, tokens, labels, *, threshold, ner, top_k):
        self.calls += 1
        person = next((row for row in ner if row[2] == "Person"), None)
        concept = next((row for row in ner if row[2] == "Concept"), None)
        if not person or not concept:
            return []
        return [{"head_pos": [person[0], person[1] + 1], "tail_pos": [concept[0], concept[1] + 1],
                 "label": "argued for", "score": 0.9}]


class FakeBudget:
    def __init__(self, model, config):
        pass

    def __call__(self, tokens, ner):
        return len(tokens) + 20


class FakeNLI:
    """Supports statements about Pāṇini, not the reversed ones."""

    def __init__(self, root, config):
        pass

    def token_count(self, premise, hypothesis):
        return len(premise.split()) + len(hypothesis.split())

    def predict(self, premise, hypothesis):
        entailment = .95 if hypothesis.startswith("Pāṇini") else .05
        return NLIScores(contradiction=.01, entailment=entailment, neutral=round(.99 - entailment, 2))


class FakeWikidata:
    """Pāṇini is a well known human. Nothing else is found."""

    def __init__(self, config):
        self.cache = None

    def retrieve(self, entities):
        return {e.corpus_entity_id: ([{"qid": "Q178031", "label": "Pāṇini", "description": "Sanskrit grammarian",
                 "aliases": [], "instance_of_qids": ["Q5"], "retrieval_rank": 1, "sitelinks": 40}]
                 if e.canonical_label == "Pāṇini" else []) for e in entities}


class LinkEncoder:
    def encode(self, texts):
        return [[1.0, 0.0] for _ in texts]


def loaders(calls=None):
    """The fakes for ``update``. With ``calls``, every model load is counted."""
    def count(name, value):
        def make(*args):
            if calls is not None:
                calls[name] = calls.get(name, 0) + 1
            return value(*args)
        return make
    return {"gliner": count("gliner", lambda config: FakeModel()),
            "coreference": count("coreference", lambda config: FakeCoreference()),
            "passage_encoder": count("passage_encoder", FakeEncoder),
            "glirel": count("glirel", lambda config: FakeGLiREL()),
            "glirel_budget": count("glirel_budget", FakeBudget),
            "nli": count("nli", FakeNLI),
            "wikidata": count("wikidata", FakeWikidata),
            "link_encoder": count("link_encoder", lambda config: LinkEncoder()),
            "facts": count("facts", FakeFacts)}


def write_episode(root, name, number, parts):
    data = transcript(parts)
    (root / "HoP/01_whisper" / name).write_bytes(data)
    return {"source_path": f"HoP/01_whisper/{name}", "source_sha256": sha256(data), "episode_id": f"ep_{number}",
            "join_tier": "tier1_filename", "content_era": "Medieval", "sub_show": "HOP", "title": f"{number}. Synthetic"}


def write_steps(root) -> dict:
    """The settings file of every step, with small synthetic models."""
    (root / "aliases.jsonl").write_text("")
    dump_json(root / "segmentation.json", {"schema_version": 1, "passage_min_words": 12, "passage_target_words": 18,
              "passage_max_words": 24, "window_max_tokens": 20, "window_overlap_tokens": 5})
    dump_json(root / "labels.json", {"schema_version": 2, "status": "test", "decoding": "flat",
              "labels": {"Person": "person", "Work": "written work", "School": "school",
                         "Concept": "philosophical concept", "Place": "location", "Period": "period",
                         "Language": "language", "Group": "group"}, "provisional_threshold": .5, "changelog": []})
    dump_json(root / "model.json", {"schema_version": 1, "task": "mention_detection", "framework": "gliner",
              "framework_version": "0.2.21", "repo_id": "synthetic/model", "revision": "a" * 40,
              "revision_resolved_at": "2026-09-13", "device": "cpu", "labels_config": "labels.json",
              "words_splitter": "whitespace", "model_max_words": 384, "window_max_tokens": 20,
              "encoder_max_tokens": 512, "window_overlap_tokens": 5, "candidate_floor": .25, "batch_size": 3,
              "smoke_test": {"status": "synthetic"}, "notes": []})
    dump_json(root / "resolve.json", {"schema_version": 1, "normalization": "unicode-nfc-casefold-whitespace-v1",
              "aliases": "aliases.jsonl", "cross_episode_exact_types": ["Place", "Language"]})
    dump_json(root / "web.json", {"schema_version": 2})
    for name in ("identity_refinement.surname-variant.json", "identity_refinement.within-episode.json",
                 "linking.wikidata-mpnet-v5.json", "coreference.fcoref.json",
                 "relations.glirel-proposals-v2.json", "verification.nli-large-v2.json",
                 "discovery.mpnet-v1.json", "enrichment.wikidata-v1.json"):
        shutil.copy(REPOSITORY / "config" / name, root / name)
    return {"segment": "segmentation.json", "extract": "model.json", "resolve": "resolve.json",
            "refine": "identity_refinement.surname-variant.json",
            "refine_episode": "identity_refinement.within-episode.json", "link": "linking.wikidata-mpnet-v5.json",
            "coreference": "coreference.fcoref.json", "relations": "relations.glirel-proposals-v2.json",
            "verify": "verification.nli-large-v2.json", "discover": "discovery.mpnet-v1.json",
            "enrich": "enrichment.wikidata-v1.json", "web": "web.json"}


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path
    (root / "HoP/01_whisper").mkdir(parents=True)
    audit = [write_episode(root, name, number, parts) for number, (name, parts) in enumerate(TEXTS.items(), 1)]
    (root / "audit.jsonl").write_text("".join(json.dumps(row) + "\n" for row in audit))
    (root / "meta.jsonl").write_text("".join(json.dumps({"filename": name, "title": f"{n}. Synthetic",
                                                          "series": "Medieval"}) + "\n" for n, name in enumerate(TEXTS, 1)))
    (root / "quarantine.jsonl").write_text("")
    dump_json(root / "profile.json", PROFILE)
    steps = write_steps(root)
    dump_json(root / "corpus.json", {"schema_version": 1, "corpus_id": "hop", "profile": "profile.json",
              "metadata": "meta.jsonl", "audit": "audit.jsonl", "quarantine": "quarantine.jsonl",
              "episode_codes": {"series_field": "sub_show", "number_fields": ["title"]}, "steps": steps,
              "merge_by_wikidata": ["Person", "Place"]})
    return root, steps


def web_data(root, build_id):
    return {name: (root / "export" / f"{build_id}-web/data" / name).read_bytes() for name in DATA_FILES}


def quiet(message):
    pass


def test_a_build_keeps_exact_offsets_and_links_people_relations_and_wikidata(corpus):
    root, _ = corpus
    update(root, "corpus.json", "HoP 1-2", "built", loaders=loaders(), progress=quiet)
    folder = root / "data/episodes/hop/HoP001"
    text = json.loads((folder / "transcript.json").read_bytes())["text"]
    mentions = read_jsonl(folder / "mentions.jsonl")
    assert mentions and all(text[m["char_start"]:m["char_end"]] == m["text"] for m in mentions)
    decisions = {row["decision"] for row in read_jsonl(folder / "candidates.jsonl")}
    assert {"accepted", "below_threshold", "duplicate"} <= decisions
    links = read_jsonl(folder / "coreference_links.jsonl")
    assert any(row["text"] == "He" and row["decision"] == "linked" for row in links)

    data = {name: json.loads(value) for name, value in web_data(root, "built").items()}
    passages = {p["id"]: p for p in data["passages.json"]}
    surfaces = {m["text"] for name in TEXTS for m in read_jsonl(root / "data/episodes/hop" / name[:-5] / "mentions.jsonl")}
    occurrences = [o for e in data["entities.json"] for o in e["occurrences"]]
    assert occurrences and all(passages[o["passage_id"]]["text"][o["start"]:o["end"]] in surfaces for o in occurrences)
    summary = data["summary.json"]
    assert summary["relation_edges"] >= 1 and summary["relation_status"] == "nli_supported"
    assert summary["lifespans"] == "enriched"
    panini = next(e for e in data["entities.json"] if e["label"] == "Pāṇini")
    assert panini["qid"] == "Q178031" and len(panini["episode_ids"]) == 2 and panini["image"]
    assert panini["life"]["start"] == -500


def test_a_model_failure_stops_the_update(corpus):
    root, _ = corpus
    failing = {**loaders(), "gliner": lambda config: FakeModel(fail=True)}
    with pytest.raises(RuntimeError, match="synthetic failure"):
        update(root, "corpus.json", "HoP 1", "broken", loaders=failing, progress=quiet)
    assert not (root / "data/builds/broken").exists()
    assert not (root / "data/episodes/hop/HoP001/mentions.jsonl").exists()


def test_the_command_line_loads_no_model_library():
    """Block the model libraries in a fresh process, to catch imports that should be lazy."""
    code = """
import importlib.abc
import sys
class BlockModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'fastcoref', 'glirel', 'gliner', 'transformers', 'huggingface_hub', 'torch'}:
            raise ImportError('model library imported: ' + fullname)
sys.meta_path.insert(0, BlockModels())
import spoken_record.cli
sys.argv = ['spoken-record', 'update', '--help']
spoken_record.cli.main()
"""
    env = dict(os.environ, PYTHONPATH=str(REPOSITORY / "src"))
    result = subprocess.run([sys.executable, "-c", code], env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert "--episodes" in result.stdout and "--refresh-wikidata" in result.stdout


def test_a_second_update_reuses_every_episode_and_model_call(corpus):
    root, _ = corpus
    update(root, "corpus.json", "HoP 1-2", "first", loaders=loaders(), progress=quiet)
    calls = {}
    second = update(root, "corpus.json", "HoP 1-2", "second", loaders=loaders(calls), progress=quiet)
    manifest = json.loads((second / "manifest.json").read_bytes())
    assert [row["redone"] for row in manifest["episodes"]] == [[], []]
    # Wikidata clients are created, but every request is answered from the cache.
    assert set(calls) <= {"wikidata", "facts"}
    assert manifest["cache"]["model_misses"] == 0 and manifest["cache"]["wikidata_misses"] == 0
    assert web_data(root, "first") == web_data(root, "second")


def test_adding_an_episode_only_runs_the_models_on_that_episode(corpus):
    root, _ = corpus
    update(root, "corpus.json", "HoP 1", "one", loaders=loaders(), progress=quiet)
    calls = {}
    build = update(root, "corpus.json", "HoP 1-2", "two", loaders=loaders(calls), progress=quiet)
    redone = {Path(row["source_path"]).name: row["redone"]
              for row in json.loads((build / "manifest.json").read_bytes())["episodes"]}
    assert redone == {"HoP001.json": [], "HoP002.json": ["source", "passages", "mentions", "coreference", "graph", "embeddings"]}
    assert calls["gliner"] == calls["coreference"] == calls["passage_encoder"] == 1
    entities = json.loads(web_data(root, "two")["entities.json"])
    assert len(next(e for e in entities if e["label"] == "Pāṇini")["episode_ids"]) == 2


def test_a_metadata_change_reruns_no_model(corpus):
    root, _ = corpus
    update(root, "corpus.json", "HoP 1", "before", loaders=loaders(), progress=quiet)
    meta = (root / "meta.jsonl").read_text().replace("1. Synthetic", "1. Renamed")
    (root / "meta.jsonl").write_text(meta)
    calls = {}
    build = update(root, "corpus.json", "HoP 1", "after", loaders=loaders(calls), progress=quiet)
    [row] = json.loads((build / "manifest.json").read_bytes())["episodes"]
    assert row["redone"] == ["source"] and "gliner" not in calls
    [episode] = json.loads(web_data(root, "after")["episodes.json"])
    assert episode["title"] == "1. Renamed"


def test_episode_selection_by_code(corpus):
    root, _ = corpus
    config, setup = load_corpus(root, "corpus.json")
    codes = config.episode_codes
    assert [row.code for row in select_episodes(setup, codes, "hop 1-2")] == ["HOP 1", "HOP 2"]
    assert [row.code for row in select_episodes(setup, codes, "all")] == ["HOP 1", "HOP 2"]
    assert [row.code for row in select_episodes(setup, codes, "HoP/01_whisper/HoP002.json")] == ["HOP 2"]
    with pytest.raises(ValueError, match="no transcript for episode HoP 3"):
        select_episodes(setup, codes, "HoP 1-3")
    with pytest.raises(ValueError, match="runs backwards"):
        select_episodes(setup, codes, "HoP 2-1")


def test_a_changed_transcript_is_refused_until_the_audit_is_updated(corpus):
    root, _ = corpus
    (root / "HoP/01_whisper/HoP001.json").write_bytes(transcript(["Something else entirely. "] * 6))
    with pytest.raises(ValueError, match="stale_source"):
        update(root, "corpus.json", "HoP 1", "stale", loaders=loaders(), progress=quiet)


def test_an_existing_build_is_never_overwritten(corpus):
    root, _ = corpus
    update(root, "corpus.json", "HoP 1", "same", loaders=loaders(), progress=quiet)
    with pytest.raises(FileExistsError):
        update(root, "corpus.json", "HoP 1", "same", loaders=loaders(), progress=quiet)


def test_validate_passes_on_a_fresh_build_and_catches_an_edited_file(corpus):
    from spoken_record.validate import validate_build
    root, _ = corpus
    build = update(root, "corpus.json", "HoP 1-2", "checked", loaders=loaders(), progress=quiet)
    report = validate_build(root, build)
    assert report["skipped"] == [] and len(report["checks"]) == 7
    graph = build / "graph.json"
    graph.write_bytes(graph.read_bytes().replace(b"Synthetic", b"Edited"))
    with pytest.raises(ValueError, match="changed: graph.json"):
        validate_build(root, build)


def test_validate_redoes_the_link_decisions_from_their_scores(corpus):
    from spoken_record.adapters.base import sha256
    from spoken_record.validate import validate_build
    root, _ = corpus
    build = update(root, "corpus.json", "HoP 1-2", "scores", loaders=loaders(), progress=quiet)
    path = build / "wikidata_candidates.jsonl"
    rows = read_jsonl(path)
    rows[0]["rank_score"] = 0.1
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))
    # Even with the hashes written again, the decision no longer follows from the score.
    manifest = json.loads((build / "manifest.json").read_bytes())
    manifest["artifact_sha256"][path.name] = sha256(path.read_bytes())
    dump_json(build / "manifest.json", manifest)
    export = root / manifest["export"] / "manifest.json"
    export_manifest = json.loads(export.read_bytes())
    export_manifest["build_manifest_sha256"] = sha256((build / "manifest.json").read_bytes())
    dump_json(export, export_manifest)
    with pytest.raises(ValueError, match="link decisions"):
        validate_build(root, build)


def test_an_older_build_stays_checkable_after_the_settings_change(corpus):
    from spoken_record.validate import validate_build
    root, steps = corpus
    old = update(root, "corpus.json", "HoP 1-2", "old", loaders=loaders(), progress=quiet)
    settings = json.loads((root / steps["relations"]).read_bytes())
    settings["acceptance_threshold"] = 0.95
    dump_json(root / steps["relations"], settings)
    new = update(root, "corpus.json", "HoP 1-2", "new", loaders=loaders(), progress=quiet)
    assert [row["redone"] for row in json.loads((new / "manifest.json").read_bytes())["episodes"]] == [["graph"]] * 2
    report = validate_build(root, old)
    assert len(report["checks"]) == 6 and "a later build updated the episode folders" in report["skipped"][0]
    assert validate_build(root, new)["skipped"] == []


class PronounTagger(FakeModel):
    """Like the fake GLiNER, but it also tags "He" as a person, as the real model does."""

    def batch_predict_entities(self, texts, labels, **kwargs):
        rows = super().batch_predict_entities(texts, labels, **kwargs)
        for text, found in zip(texts, rows):
            start = text.find("He ")
            while start >= 0:
                found.append({"start": start, "end": start + 2, "text": "He", "label": "person", "score": .9})
                start = text.find("He ", start + 1)
        return rows


def test_a_pronoun_tagged_as_a_person_is_left_to_coreference(corpus):
    root, steps = corpus
    settings = json.loads((root / steps["resolve"]).read_bytes())
    dump_json(root / steps["resolve"], {**settings, "not_names": ["he", "she", "I", "you"]})
    update(root, "corpus.json", "HoP 1", "pronouns", loaders={**loaders(), "gliner": lambda config: PronounTagger()},
           progress=quiet)
    folder = root / "data/episodes/hop/HoP001"
    assert any(m["text"] == "He" for m in read_jsonl(folder / "mentions.jsonl"))
    assert not any(e["canonical_label"] == "He" for e in read_jsonl(folder / "entities.jsonl"))
    # The relation goes through the FCoref link from "He" to Pāṇini, not through an entity "He".
    candidates = read_jsonl(folder / "relation_candidates.jsonl")
    assert candidates and all(c["head_text"] != "He" or c["head_endpoint_kind"] == "coreference_reference"
                              for c in candidates)
    entities = json.loads(web_data(root, "pronouns")["entities.json"])
    assert not any(e["label"] == "He" for e in entities)
