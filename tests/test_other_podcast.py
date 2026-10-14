"""A second, unrelated podcast runs through every stage with configuration only.

Nothing here looks like History of Philosophy: a different transcript layout, the generic
Whisper JSON adapter, metadata matched by file name, no corpus audit, a season facet and
its own web app config. The same offline fakes as in the HoP pipeline test replace GLiNER,
FCoref, GLiREL, Wikidata and MPNet.
"""

import json

import pytest

from test_extraction import transcript
from test_pipeline import full_recipe
from spoken_record.adapters.base import sha256
from spoken_record.ingestion import dump_json, ingest
from spoken_record.pipeline import run_pipeline, validate_snapshot

SOURCE = "Podcasts/Longitude/whisper/season-2/ep07.json"


@pytest.fixture
def other_podcast(tmp_path):
    (tmp_path / SOURCE).parent.mkdir(parents=True)
    (tmp_path / SOURCE).write_bytes(transcript())
    dump_json(tmp_path / "profile.json", {"schema_version": 1, "adapter": "whisper_json",
                                          "transcripts_dir": "Podcasts/Longitude/whisper"})
    dump_json(tmp_path / "config.json", {"schema_version": 1, "corpus_id": "longitude",
                                         "profile": "profile.json", "metadata": "episodes.jsonl"})
    dump_json(tmp_path / "pilot.json", {"schema_version": 1, "episodes": [
        {"source_path": SOURCE, "source_sha256": sha256(transcript())}]})
    (tmp_path / "episodes.jsonl").write_text(json.dumps({
        "file": "season-2/ep07.json", "title": "Ep. 7 — Clocks at Sea", "date": "2022-05-17",
        "audio_url": "https://example.org/longitude/ep07.mp3", "page_url": "https://example.org/longitude/7",
        "facets": {"season": "Season 2"}}) + "\n")
    return tmp_path, SOURCE


def test_generic_adapter_reads_metadata_by_file_and_needs_no_audit(other_podcast):
    root, _ = other_podcast
    episode = json.loads((ingest(root, "config.json", "pilot.json", "plain") / "episodes.jsonl").read_text())
    assert (episode["title"], episode["publication_date"], episode["metadata_method"]) == \
        ("Ep. 7 — Clocks at Sea", "2022-05-17", "exact_file")
    assert episode["facets"] == {"season": "Season 2"} and episode["audio_url"].endswith("ep07.mp3")


def test_unknown_adapter_and_missing_metadata_fail_clearly(other_podcast):
    root, _ = other_podcast
    dump_json(root / "profile.json", {"schema_version": 1, "adapter": "srt", "transcripts_dir": "Podcasts"})
    with pytest.raises(ValueError, match="unknown transcript adapter 'srt'"):
        ingest(root, "config.json", "pilot.json", "bad-adapter")
    dump_json(root / "profile.json", {"schema_version": 1, "adapter": "whisper_json",
                                      "transcripts_dir": "Podcasts/Longitude/whisper"})
    (root / "episodes.jsonl").write_text("")
    run = ingest(root, "config.json", "pilot.json", "no-metadata")
    outcome = json.loads((run / "outcomes.jsonl").read_text())
    assert outcome["status"] == "failed" and outcome["reason"].startswith("metadata_join_unresolved")


def test_second_podcast_runs_every_stage_to_the_web_app(other_podcast, monkeypatch):
    root = full_recipe(other_podcast, monkeypatch, True, {"id": "web", "kind": "export-web", "config": "web.json"})
    dump_json(root / "web.json", {"schema_version": 2, "site": {
        "archive_id": "longitude", "title": "Longitude", "subtitle": "a podcast about navigation",
        "series_facet": "season", "series_label": "Season", "episode_number_pattern": r"Ep\. (\d+) —",
        "entity_types": [{"type": "Person", "label": "Person", "plural": "people"},
                         {"type": "Place", "label": "Place", "plural": "places"}],
        "lifespan_type": None, "search_placeholder": "Search people and places…"}})
    result = run_pipeline(root, "pipeline.json", "longitude")
    assert result["completed_stages"] == 10
    output = root / result["final_output"]
    assert validate_snapshot(output)["stage"] == "export_web"
    data = {n: json.loads((output / "data" / f"{n}.json").read_bytes()) for n in ("site", "episodes", "summary")}
    assert data["site"]["title"] == "Longitude" and data["site"]["series_label"] == "Season"
    assert data["site"]["lifespan_type"] is None
    [episode] = data["episodes"]
    assert (episode["code"], episode["title"], episode["series"]) == ("Season 2 7", "Clocks at Sea", "Season 2")
    assert episode["audio_url"] == "https://example.org/longitude/ep07.mp3"
    assert data["summary"]["episodes"] == 1
