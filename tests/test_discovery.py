"""Tests for passage discovery: source coverage, ranking and the evidence it keeps."""

import pytest

from spoken_record.discovery import (check_embeddings, compute_discovery, normalized, prepare_inputs,
                                      rank_connections, token_ranges)
from spoken_record.discovery_models import DiscoveryConfig
from spoken_record.extraction_models import ExtractedMention
from spoken_record.models import Episode, TranscriptVersion
from spoken_record.passage_models import Passage
from spoken_record.resolution import resolve_records
from spoken_record.resolution_models import ResolutionConfig


def navigation_records():
    """Two episodes that share a person and have local concepts, a school and a work."""
    texts = {"e1": "Thomas Aquinas discusses form, form and Academy in Book.",
             "e2": "Thomas Aquinas discusses form."}
    episodes = [Episode(created_by_run_id="source", episode_id=eid, corpus_id="hop", source_key=f"{eid}.json",
        source_path=f"HoP/{eid}.json", title=f"Episode {eid}", language="en", metadata_status="matched",
        metadata_method="exact") for eid in texts]
    transcripts = [TranscriptVersion(created_by_run_id="source", transcript_version_id=f"t-{eid}", episode_id=eid,
        kind="raw_asr", text=text, source_sha256="a" * 64, adapter="test", adapter_version="1")
        for eid, text in texts.items()]
    passages = [Passage(created_by_run_id="passages", passage_id=f"p-{eid}", episode_id=eid,
        transcript_version_id=f"t-{eid}", index=0, char_start=0, char_end=len(text), text=text,
        word_count=len(text.split()), timing_precision="unavailable", source_segment_ids=[],
        method="sentence_packing", segmentation_fingerprint="s") for eid, text in texts.items()]
    mentions = []
    for eid, text in texts.items():
        for term, kind in (("Thomas Aquinas", "Person"), ("form", "Concept"), ("Academy", "School"), ("Book", "Work")):
            start = 0
            while (position := text.find(term, start)) >= 0:
                mention_id = f"m-{eid}" if kind == "Person" else f"{eid}-{term}-{position}"
                mentions.append(ExtractedMention(created_by_run_id="extract", mention_id=mention_id, episode_id=eid,
                    transcript_version_id=f"t-{eid}", char_start=position, char_end=position + len(term),
                    text=term, type=kind, score=.9, passage_ids=[f"p-{eid}"], source_window_ids=["w"],
                    extraction_fingerprint="x"))
                start = position + len(term)
    config = ResolutionConfig(normalization="unicode-nfc-casefold-whitespace-v1", aliases="aliases.jsonl",
                              cross_episode_exact_types=["Place"])
    episode_entities, entities, assignments, _, queue = resolve_records(
        "resolve", "hop", mentions, passages, [], config, "fingerprint")
    return (episodes, transcripts, passages, mentions, list(episode_entities), list(entities),
            list(assignments), list(queue))


def config(**changes):
    """Same vector size as the real model, but fake vectors and no model loading."""
    return DiscoveryConfig(revision="a"*40, **changes)


class FakeEncoder:
    """Unit vectors and full token spans, so the results are always the same."""

    def __init__(self, root, settings):
        self.settings = settings

    def encode(self, texts):
        """Identical vectors on purpose, to get a tie in the scores."""
        return [{"vector": [1.] + [0.]*767, "token_count": 5, "special_tokens": 2,
                 "chunks": [{"token_start": 0, "token_end": 5, "encoder_tokens": 7}]} for _ in texts]


def example():
    """Two different passages that share a person and have local topic names."""
    records = navigation_records()
    inputs = prepare_inputs(records, [])
    vectors = [{**row, "passage_id": item["passage_id"], "input_sha256": item["input_sha256"]}
               for row, item in zip(FakeEncoder(None, config()).encode([r["input_text"] for r in inputs]), inputs)]
    return records, inputs, vectors


def test_token_partition_and_invalid_vectors():
    """A last, shorter chunk still covers the rest of a long passage."""
    assert token_ranges(770, 382) == [(0, 382), (382, 764), (764, 770)]
    assert token_ranges(382, 382) == [(0, 382)]
    for vector in ([0., 0.], [float("nan"), 1.]):
        with pytest.raises(ValueError):
            normalized(vector)
    _, inputs, vectors = example()
    check_embeddings(inputs, vectors, config())
    vectors[0]["chunks"][0]["token_end"] = 4
    with pytest.raises(ValueError, match="omit tokens"):
        check_embeddings(inputs, vectors, config())


def test_reviewed_exclusions_preserve_offsets_and_remove_affected_mentions():
    """Only reviewed spans are removed, the rest of the text keeps its tail."""
    records = navigation_records()
    p = records[2][0]
    end = len("Thomas Aquinas")
    exclusion = {"transcript_version_id": p.transcript_version_id,
                 "char_start": 0, "char_end": end, "review_status": "reviewed"}
    rows = prepare_inputs(records, [exclusion])
    row = next(r for r in rows if r["passage_id"] == p.passage_id)
    assert row["input_text"] == p.text[end:]
    assert row["source_spans"] == [[end, len(p.text)]]
    assert not any("m-e1" in ids for ids in row["entity_mentions"].values())
    assert records[2][0].text == p.text
    with pytest.raises(ValueError, match="only reviewed"):
        prepare_inputs(records, [{**exclusion, "review_status": "proposed"}])
    empty = prepare_inputs(records, [{**exclusion, "char_end": len(p.text)}])
    assert empty[0]["status"] == "empty_after_exclusions"


def test_rankings_are_cross_episode_and_keep_methods_separate():
    """Shared entities only feed the overlap baseline, they don't make topics the same."""
    _, inputs, vectors = example()
    scores, links, counts = rank_connections(inputs, vectors, config())
    assert len(scores) == 4 and len(links) == 4
    assert all(r["source_episode_id"] != r["target_episode_id"] for r in links)
    assert all(r["status"] == "suggested" for r in links)
    assert {r["method"] for r in links} == {"mpnet_cosine", "shared_entity_jaccard"}
    assert counts["common_selected_pairs"] == 2
    overlap = next(r for r in links if r["method"] == "shared_entity_jaccard")
    assert len(overlap["shared_entity_ids"]) == 1  # Person shared; two local Concepts stay distinct.
    assert overlap["score"] < 1.
    inputs[1]["duplicate_key"] = inputs[0]["duplicate_key"]
    assert rank_connections(inputs, vectors, config())[2]["excluded_duplicate_pairs"] == 2
    assert rank_connections(inputs, vectors, config())[1] == []
    inputs[1]["episode_id"] = inputs[0]["episode_id"]
    assert rank_connections(inputs, vectors, config())[2]["excluded_same_episode_pairs"] == 2


def test_no_entity_overlap_can_still_have_embedding_suggestions_and_abstention():
    """No shared names is fine, and a weak cosine gives no graph link."""
    _, inputs, vectors = example()
    for row in inputs:
        row["entity_mentions"] = {}
    links = rank_connections(inputs, vectors, config())[1]
    assert len(links) == 2 and all(r["method"] == "mpnet_cosine" for r in links)
    vectors[1]["vector"] = [0., 1.] + [0.]*766
    assert rank_connections(inputs, vectors, config())[1] == []


def test_top_k_ties_use_stable_target_ids():
    """Exactly k per source in any input order, and the rejected ranks are kept."""
    _, inputs, vectors = example()
    inputs.append({**inputs[1], "passage_id": "p-third", "episode_id": "e3", "duplicate_key": "third"})
    vectors.append({**vectors[1], "passage_id": "p-third"})
    scores, links, _ = rank_connections(inputs, vectors, config(top_k=1))
    chosen = [r for r in links if r["source_passage_id"] == "p-e1" and r["method"] == "mpnet_cosine"]
    assert chosen[0]["target_passage_id"] == "p-e2"
    assert any(r["decision"] == "outside_top_k" for r in scores)


def test_a_model_failure_stops_discovery_instead_of_finding_nothing():
    """A model failure must not look like a result with no related passages."""
    def fail(texts):
        raise RuntimeError("model failed")
    with pytest.raises(RuntimeError, match="model failed"):
        compute_discovery(navigation_records(), [], config(), fail)
    _, _, _, connections, counts = compute_discovery(
        navigation_records(), [], config(), FakeEncoder(None, config()).encode)
    assert counts["mpnet_cosine_connections"] == 2 and len(connections) == 4
