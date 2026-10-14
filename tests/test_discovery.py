"""Tests for passage discovery: source coverage, ranking, saved runs and the evidence shown."""

import json

import pytest

from spoken_record.adapters.base import sha256
from spoken_record.discovery import (check_embeddings, discover_run, normalized, prepare_inputs,
                                      rank_connections, token_ranges, validate_run)
from spoken_record.discovery_models import DiscoveryConfig
from spoken_record.ingestion import dump_json
from spoken_record.vault import build_vault_files, validate_links
from spoken_record.vault_models import VaultConfig
from test_navigation import navigation_records


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


def test_discovery_vault_shows_two_passages_and_separate_comparison():
    """Suggested passage notes keep the source text, anchors and the uncertainty labels."""
    records, inputs, vectors = example()
    _, links, counts = rank_connections(inputs, vectors, config())
    view = {"run_id": "discovery-test", "inputs": inputs, "connections": links, "counts": counts}
    settings = VaultConfig(vault_name="Discovery", concept_navigation=True)
    files = build_vault_files(settings, *records, discovery=view)
    validate_links(files, settings, records[2])
    nodes = [data.decode() for name, data in files.items() if name.startswith("40_Discussions/")]
    assert len(nodes) == 2
    assert "40_Discussions/Episode e1 — Passage 1.md" in files
    assert all("retrieval suggestions, not factual relations" in text for text in nodes)
    assert all("timestamp unavailable" in text for text in nodes)
    assert all("Cosine similarity: 1.000" in text for text in nodes)
    review = files["90_Review/Passage discovery comparison.md"].decode()
    assert "Shared-entity Jaccard" in review and "MPNet cosine" in review
    assert not any(name.startswith("30_Relations/") for name in files)
    no_view = build_vault_files(settings, *records)
    assert not any(name.startswith("40_Discussions/") for name in no_view)


def test_snapshot_roundtrip_and_rehashed_ranking_tampering(tmp_path, monkeypatch):
    """A saved run validates offline and fails when scores are edited, even with new hashes."""
    import spoken_record.discovery as module
    records = navigation_records()
    sources = {"identity_run_id": "parent", "passage_run_id": "passages",
               "identity_manifest_sha256": "a"*64, "passage_manifest_sha256": "b"*64}
    monkeypatch.setattr(module, "load_inputs", lambda root, parent: (records, [], sources))
    parent = tmp_path / "data/prototype1/runs/parent"
    parent.mkdir(parents=True)
    dump_json(parent/"manifest.json", {"stage": "resolve", "status": "complete"})
    dump_json(tmp_path/"discovery.json", config().model_dump(mode="json"))
    output = discover_run(tmp_path, str(parent.relative_to(tmp_path)), "discovery.json",
                          "discovery-test", backend_factory=FakeEncoder)
    assert validate_run(output)["counts"]["mpnet_cosine_connections"] == 2
    with pytest.raises(FileExistsError):
        discover_run(tmp_path, str(parent.relative_to(tmp_path)), "discovery.json", "discovery-test")
    path = output/"passage_connections.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["score"] = .123
    path.write_text("".join(json.dumps(r)+"\n" for r in rows))
    manifest = json.loads((output/"manifest.json").read_text())
    manifest["artifact_sha256"][path.name] = sha256(path.read_bytes())
    dump_json(output/"manifest.json", manifest)
    with pytest.raises(ValueError, match="rankings not reproducible"):
        validate_run(output)


def test_backend_failure_does_not_publish(tmp_path, monkeypatch):
    """A model failure must not produce a valid run with no suggestions."""
    import spoken_record.discovery as module
    monkeypatch.setattr(module, "load_inputs", lambda root, parent: (navigation_records(), [], {}))
    parent = tmp_path/"data/prototype1/runs/parent"
    parent.mkdir(parents=True)
    dump_json(parent/"manifest.json", {})
    dump_json(tmp_path/"discovery.json", config().model_dump(mode="json"))
    def fail(root, settings):
        """Pretend the checkpoint is missing, before anything is written."""
        raise RuntimeError("model failed")
    with pytest.raises(RuntimeError, match="model failed"):
        discover_run(tmp_path, "data/prototype1/runs/parent", "discovery.json", "failed", backend_factory=fail)
    assert not (parent.parent/"failed").exists()
