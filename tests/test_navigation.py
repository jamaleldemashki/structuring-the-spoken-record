"""Tests for navigation: co-mentions are evidence to look at, never identity merges."""

import json

import pytest

from spoken_record.extraction_models import ExtractedMention
from spoken_record.navigation import build_concept_navigation
from spoken_record.resolution import resolve_records
from spoken_record.resolution_models import ResolutionConfig
from spoken_record.vault import build_vault_files, navigation_counts, note_paths, validate_links
from spoken_record.vault_models import VaultConfig
from test_vault import records


def navigation_records():
    """Two local concepts with the same name sharing a person, plus a school and a work."""
    source = records()
    texts = {"e1": "Thomas Aquinas discusses form, form and Academy in Book.",
             "e2": "Thomas Aquinas discusses form."}
    for i, episode in enumerate(source[0]):
        text = texts[episode.episode_id]
        source[1][i] = source[1][i].model_copy(update={"text": text})
        source[2][i] = source[2][i].model_copy(update={"text": text, "char_end": len(text),
            "word_count": len(text.split()), "start_s": None, "end_s": None,
            "timing_precision": "unavailable", "source_segment_ids": []})
        # Repeated names are separate spans in one passage, not extra passages.
        terms = [("form", "Concept"), ("Academy", "School"), ("Book", "Work")]
        for term, kind in terms:
            start = 0
            while (position := text.find(term, start)) >= 0:
                source[3].append(ExtractedMention(created_by_run_id="extract",
                    mention_id=f"{episode.episode_id}-{term}-{position}", episode_id=episode.episode_id,
                    transcript_version_id=f"t-{episode.episode_id}", char_start=position,
                    char_end=position + len(term), text=term, type=kind, score=.9,
                    passage_ids=[f"p-{episode.episode_id}"], source_window_ids=["w-one", "w-overlap"],
                    extraction_fingerprint="x"))
                start = position + len(term)
    config = ResolutionConfig(normalization="unicode-nfc-casefold-whitespace-v1",
        aliases="aliases.jsonl", cross_episode_exact_types=["Place"])
    episode_entities, entities, assignments, _, queue = resolve_records(
        "resolve", "hop", source[3], source[2], [], config, "fingerprint")
    return (*source[:4], list(episode_entities), list(entities), list(assignments), list(queue))


def test_navigation_counts_distinct_passages_and_preserves_local_identities():
    """Same labels stay separate entities, and repeated mentions don't inflate pair counts."""
    source = navigation_records()
    report = build_concept_navigation(source[2], source[3], source[5], source[6])
    assert report["counts"]["concept_entities"] == 2
    assert report["counts"]["school_entities"] == 1
    assert report["counts"]["cross_episode_topics"] == 0
    assert report["counts"]["shared_passage_pairs"] == 5
    assert all(row["passage_count"] == 1 for row in report["connections"])
    assert any(len(e["topic_mention_ids"]) == 2 for row in report["connections"] for e in row["evidence"])
    reversed_report = build_concept_navigation(list(reversed(source[2])), list(reversed(source[3])),
                                               list(reversed(source[5])), list(reversed(source[6])))
    assert reversed_report == report


def test_navigation_rejects_cross_transcript_or_missing_assignments():
    """A broken parent join must not create co-occurrence evidence."""
    source = navigation_records()
    wrong = source[3][0].model_copy(update={"passage_ids": ["p-e2"]})
    with pytest.raises(ValueError, match="invalid passage provenance"):
        build_concept_navigation(source[2], [wrong, *source[3][1:]], source[5], source[6])
    with pytest.raises(ValueError, match="exactly one assignment"):
        build_concept_navigation(source[2], source[3], source[5], source[6][1:])


def test_navigation_export_is_optional_bounded_and_grounded():
    """Both directions link to the source blocks, and every pair survives the display limit."""
    source = navigation_records()
    config = VaultConfig(vault_name="Navigation", concept_navigation=True, navigation_max_neighbors=1)
    files = build_vault_files(config, *source)
    validate_links(files, config, source[2])
    report = json.loads(files["_meta/concept_navigation.json"])
    assert len(report["connections"]) == 5
    _, paths = note_paths(config, source[0], source[5])
    topic = next(e for e in source[5] if e.type == "Concept" and e.episode_ids == ["e1"])
    note = files[f"{paths[topic.corpus_entity_id]}.md"].decode()
    assert "Showing 1 of 2 neighbors" in note
    assert "timestamp unavailable" in note
    assert "membership, influence, agreement" in note
    assert "#^p-e1" in note
    person = next(e for e in source[5] if e.type == "Person")
    assert "Concepts and schools in shared passages" in files[f"{paths[person.corpus_entity_id]}.md"].decode()
    index = files["00_Index/Explore concepts and schools.md"].decode()
    assert "No Concept or School identity" in index
    assert "form — Episode e1" in index and "form — Episode e2" in index
    assert navigation_counts(files)["navigation_shared_passage_pairs"] == 5
    inventory = files["90_Review/Shared passage inventory.md"].decode()
    assert f"^nav-{topic.corpus_entity_id}" in inventory
    assert "All neighbors and shared-passage evidence" in note
    # Neighbours shown or cut by the cap both still have readable evidence.
    assert "Book" in inventory and "Thomas Aquinas" in inventory
    legacy = build_vault_files(VaultConfig(vault_name="Navigation"), *source)
    explicit_off = build_vault_files(config.model_copy(update={"concept_navigation": False}), *source)
    assert legacy == explicit_off
    assert navigation_counts(legacy) == {}


def test_empty_topic_coverage_is_reported_without_inventing_nodes():
    """Input with only people still exports a useful coverage note."""
    source = records()
    config = VaultConfig(vault_name="No topics", concept_navigation=True)
    files = build_vault_files(config, *source)
    validate_links(files, config, source[2])
    index = files["00_Index/Explore concepts and schools.md"].decode()
    assert "No Concept entities detected" in index and "No School entities detected" in index
    assert navigation_counts(files)["navigation_shared_passage_pairs"] == 0
