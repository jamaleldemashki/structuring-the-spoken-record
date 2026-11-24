"""A second, unrelated podcast runs through every step with configuration only.

Nothing here looks like History of Philosophy: a different transcript layout, the generic
Whisper JSON adapter, metadata matched by file name, no corpus audit, episodes chosen by
season and number, and its own web app settings. The same offline fakes as in the HoP test
replace the models and Wikidata.
"""

import json

import pytest

from test_ingestion import transcript
from test_update import loaders, quiet, web_data, write_steps
from spoken_record.build import update
from spoken_record.corpus import load_corpus, select_episodes
from spoken_record.ingestion import dump_json

SOURCE = "Podcasts/Longitude/whisper/season-2/ep07.json"


@pytest.fixture
def other_podcast(tmp_path):
    root = tmp_path
    (root / SOURCE).parent.mkdir(parents=True)
    (root / SOURCE).write_bytes(transcript(["Pāṇini studies free will and words. ",
                                            "He defends free will against his critics. "] * 6))
    dump_json(root / "profile.json", {"schema_version": 1, "adapter": "whisper_json",
                                      "transcripts_dir": "Podcasts/Longitude/whisper"})
    (root / "episodes.jsonl").write_text(json.dumps({
        "file": "season-2/ep07.json", "title": "Ep. 7 — Clocks at Sea", "date": "2022-05-17",
        "audio_url": "https://example.org/longitude/ep07.mp3", "page_url": "https://example.org/longitude/7",
        "facets": {"season": "Season 2"}}) + "\n")
    steps = write_steps(root)
    dump_json(root / "web.json", {"schema_version": 2, "site": {
        "archive_id": "longitude", "title": "Longitude", "subtitle": "a podcast about navigation",
        "series_facet": "season", "series_label": "Season", "episode_number_pattern": r"Ep\. (\d+) —",
        "entity_types": [{"type": "Person", "label": "Person", "plural": "people"},
                         {"type": "Concept", "label": "Idea", "plural": "ideas"}],
        "lifespan_type": None, "search_placeholder": "Search people and ideas…"}})
    dump_json(root / "corpus.json", {"schema_version": 1, "corpus_id": "longitude", "profile": "profile.json",
              "metadata": "episodes.jsonl", "episode_codes": {"series_field": "facets.season",
              "number_fields": ["title"], "number_pattern": r"Ep\. (\d+)", "path_field": "file",
              "path_in_transcripts_dir": True}, "steps": steps})
    return root


def test_episodes_of_another_podcast_are_chosen_by_season_and_number(other_podcast):
    config, setup = load_corpus(other_podcast, "corpus.json")
    [episode] = select_episodes(setup, config.episode_codes, "season 2 7")
    assert (episode.code, episode.source_path, episode.expected_sha256) == ("Season 2 7", SOURCE, None)


def test_second_podcast_runs_every_step_to_the_web_app(other_podcast):
    root = other_podcast
    update(root, "corpus.json", "Season 2 7", "longitude", loaders=loaders(), progress=quiet)
    data = {name: json.loads(value) for name, value in web_data(root, "longitude").items()}
    assert data["site.json"]["title"] == "Longitude" and data["site.json"]["series_label"] == "Season"
    assert data["site.json"]["lifespan_type"] is None
    [episode] = data["episodes.json"]
    assert (episode["code"], episode["title"], episode["series"]) == ("Season 2 7", "Clocks at Sea", "Season 2")
    assert episode["audio_url"] == "https://example.org/longitude/ep07.mp3"
    assert data["summary.json"]["episodes"] == 1 and data["summary.json"]["relation_edges"] >= 1
