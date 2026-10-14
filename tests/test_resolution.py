from spoken_record.extraction_models import ExtractedMention
from spoken_record.passage_models import Passage
from spoken_record.resolution import normalize_surface, pairwise, resolve_records, validate_aliases
from spoken_record.resolution_models import AliasRule, ResolutionConfig
from spoken_record.models import Episode


def passage(episode_id, passage_id):
    return Passage(created_by_run_id="passages", passage_id=passage_id, episode_id=episode_id,
        transcript_version_id=f"t-{episode_id}", index=0, char_start=0, char_end=100,
        text="x" * 100, word_count=1, timing_precision="unavailable", source_segment_ids=[],
        method="token_fallback", segmentation_fingerprint="segment-v1")


def mention(episode_id, passage_id, mention_id, text, entity_type, start):
    return ExtractedMention(created_by_run_id="extract", mention_id=mention_id, episode_id=episode_id,
        transcript_version_id=f"t-{episode_id}", char_start=start, char_end=start + len(text), text=text,
        type=entity_type, score=.8, source_window_ids=[f"w-{mention_id}"], passage_ids=[passage_id],
        extraction_fingerprint="extract-v1")


def episode(episode_id):
    return Episode(created_by_run_id="source", episode_id=episode_id, corpus_id="hop",
        source_key=f"{episode_id}.json", source_path=f"HoP/{episode_id}.json", title=episode_id,
        language="en", metadata_status="matched", metadata_method="exact")


def config():
    return ResolutionConfig(normalization="unicode-nfc-casefold-whitespace-v1",
        aliases="aliases.jsonl", cross_episode_exact_types=["Place"])


def fixture_records():
    passages = [passage("e1", "p1"), passage("e2", "p2")]
    mentions = [
        mention("e1", "p1", "m1", "Thomas Aquinas", "Person", 0),
        mention("e2", "p2", "m2", "Thomas Aquinas", "Person", 0),
        mention("e1", "p1", "m3", "Aquinas", "Person", 20),
        mention("e1", "p1", "m4", "AQUINAS", "Person", 30),
        mention("e2", "p2", "m5", "Aquinas", "Person", 20),
        mention("e1", "p1", "m6", "Paris", "Place", 40),
        mention("e2", "p2", "m7", "Paris", "Place", 40),
        mention("e1", "p1", "m8", "form", "Concept", 50),
        mention("e2", "p2", "m9", "form", "Concept", 50),
    ]
    return mentions, passages, [episode("e1"), episode("e2")]


def test_normalization_is_narrow_and_diacritic_preserving():
    assert normalize_surface("  PĀṆINI\t") == "pāṇini"
    assert normalize_surface("Pāṇini") == "pāṇini"
    assert normalize_surface("Panini") != normalize_surface("Pāṇini")
    assert normalize_surface("Aquinas,") != normalize_surface("Aquinas")


def test_conservative_resolution_merges_only_allowed_exact_keys():
    mentions, passages, episodes = fixture_records()
    rows = resolve_records("resolve", "hop", mentions, passages, [], config(), "fingerprint")
    episode_entities, corpus_entities, assignments, decisions, queue = rows
    assert len(assignments) == len(mentions) == len({a.mention_id for a in assignments})
    assert len(episode_entities) == 8  # e1 Aquinas case variants form one episode entity.

    grouped = {(entity.type, tuple(entity.observed_surfaces)): entity for entity in corpus_entities}
    thomas = next(e for e in corpus_entities if e.canonical_label == "Thomas Aquinas")
    paris = next(e for e in corpus_entities if e.canonical_label == "Paris" and e.episode_count == 2)
    assert (thomas.episode_count, thomas.resolution_method, thomas.resolution_status) == (2, "provisional_exact", "provisional")
    assert (paris.episode_count, paris.resolution_method) == (2, "provisional_exact")
    assert len([e for e in corpus_entities if e.type == "Person" and "Aquinas" in e.observed_surfaces]) == 2
    assert len([e for e in corpus_entities if e.type == "Concept" and e.canonical_label == "form"]) == 2
    assert {(candidate.type, candidate.normalized_key, candidate.reason) for candidate in queue} == {
        ("Person", "aquinas", "single_token_person"),
        ("Concept", "form", "type_requires_review"),
    }
    assert len(decisions) == len(episode_entities)
    assert all(a.corpus_entity_id in {e.corpus_entity_id for e in corpus_entities} for a in assignments)


def test_reviewed_alias_can_merge_variants_and_block_exact_place():
    mentions, passages, episodes = fixture_records()
    rules = [
        AliasRule(alias_id="alias-aquinas", corpus_id="hop", type="Person", identity_id="person-aquinas",
            canonical_label="Thomas Aquinas", aliases=["Thomas Aquinas", "Aquinas"], decision="merge",
            scope="corpus", evidence_passage_ids=["p1", "p2"], reviewer="author",
            reviewed_at="2026-09-13", note="synthetic review"),
        AliasRule(alias_id="block-paris", corpus_id="hop", type="Place", identity_id="place-paris",
            canonical_label="Paris", aliases=["Paris"], decision="block", scope="corpus",
            evidence_passage_ids=["p1", "p2"], reviewer="author", reviewed_at="2026-09-13",
            note="synthetic ambiguity"),
    ]
    validate_aliases(rules, mentions, passages, episodes)
    _, corpus_entities, assignments, decisions, queue = resolve_records(
        "resolve", "hop", mentions, passages, rules, config(), "fingerprint")
    aquinas = next(e for e in corpus_entities if e.registry_key == "reviewed:person-aquinas")
    assert aquinas.mention_count == 5 and aquinas.episode_count == 2
    assert aquinas.resolution_method == "reviewed_alias" and aquinas.resolution_status == "reviewed"
    assert len([e for e in corpus_entities if e.type == "Place"]) == 2
    assert all(d.method == "blocked" for d in decisions if d.corpus_entity_id in {
        e.corpus_entity_id for e in corpus_entities if e.type == "Place"})
    assert next(q for q in queue if q.type == "Place").reason == "blocked_by_rule"
    assert len(assignments) == len(mentions)


def test_pairwise_metrics_count_false_merges_and_splits():
    rows = [
        {"episode_id": "e1", "gold_identity": "a", "cluster": "x"},
        {"episode_id": "e2", "gold_identity": "a", "cluster": "y"},
        {"episode_id": "e2", "gold_identity": "b", "cluster": "y"},
    ]
    assert pairwise(rows, "cluster") == {
        "pairs_considered": 3, "gold_same_pairs": 1, "predicted_same_pairs": 1,
        "tp": 0, "fp": 1, "fn": 1, "false_merges": 1, "false_splits": 1,
        "precision": 0.0, "recall": 0.0, "f1": 0.0,
    }
    cross = pairwise(rows, "cluster", cross_episode=True)
    assert cross["pairs_considered"] == 2 and cross["fn"] == 1 and cross["fp"] == 0
