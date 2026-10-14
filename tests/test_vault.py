from pathlib import Path

import pytest

from spoken_record.extraction_models import ExtractedMention
from spoken_record.coreference_models import CoreferenceLink
from spoken_record.models import Episode, TranscriptVersion
from spoken_record.passage_models import Passage
from spoken_record.relation_models import RelationAssertion, RelationEvidence
from spoken_record.resolution import resolve_records
from spoken_record.resolution_models import CorpusEntity, ResolutionConfig
from spoken_record.vault import build_vault_files, episode_reference, markdown_text, note_paths, validate_links
from spoken_record.vault_models import VaultConfig


def records():
    texts = {"e1": "Thomas Aquinas discusses [[form]].", "e2": "# Thomas Aquinas revisited."}
    episodes = [Episode(created_by_run_id="source", episode_id=eid, corpus_id="hop",
        source_key=f"{eid}.json", source_path=f"HoP/{eid}.json", title=f"Episode {eid}", language="en",
        metadata_status="matched", metadata_method="exact") for eid in texts]
    transcripts = [TranscriptVersion(created_by_run_id="source", transcript_version_id=f"t-{eid}",
        episode_id=eid, kind="raw_asr", text=text, source_sha256="a" * 64, adapter="test",
        adapter_version="1") for eid, text in texts.items()]
    passages = [Passage(created_by_run_id="passages", passage_id=f"p-{eid}", episode_id=eid,
        transcript_version_id=f"t-{eid}", index=0, char_start=0, char_end=len(text), text=text,
        word_count=len(text.split()), start_s=10, end_s=20, timing_precision="source_segment",
        source_segment_ids=[f"s-{eid}"], method="sentence_packing", segmentation_fingerprint="s")
        for eid, text in texts.items()]
    mentions = []
    for eid, text in texts.items():
        surface = "Thomas Aquinas"
        start = text.index(surface)
        mentions.append(ExtractedMention(created_by_run_id="extract", mention_id=f"m-{eid}", episode_id=eid,
            transcript_version_id=f"t-{eid}", char_start=start, char_end=start + len(surface), text=surface,
            type="Person", score=.9, source_window_ids=[f"w-{eid}"], passage_ids=[f"p-{eid}"],
            extraction_fingerprint="x"))
    resolution_config = ResolutionConfig(normalization="unicode-nfc-casefold-whitespace-v1",
        aliases="aliases.jsonl", cross_episode_exact_types=["Place"])
    episode_entities, corpus_entities, assignments, _, queue = resolve_records(
        "resolve", "hop", mentions, passages, [], resolution_config, "fingerprint")
    return episodes, transcripts, passages, mentions, list(episode_entities), list(corpus_entities), list(assignments), list(queue)


def test_markdown_source_escaping_prevents_accidental_graph_links():
    rendered = markdown_text("[[Entity]] *emphasis*\n# heading\n1. list\n<tag>")
    assert "[[Entity]]" not in rendered
    assert "\\[\\[Entity\\]\\]" in rendered
    assert "\\# heading" in rendered and "1\\. list" in rendered


def test_vault_projection_has_complete_notes_and_working_passage_links():
    config = VaultConfig(vault_name="Synthetic vault")
    source = records()
    files = build_vault_files(config, *source)
    validate_links(files, config, source[2])
    assert len([name for name in files if name.startswith("10_Episodes/")]) == 2
    assert len([name for name in files if name.startswith("20_Entities/")]) == 1
    entity = source[5][0]
    episode_paths, entity_paths = note_paths(config, source[0], source[5])
    assert episode_paths["e1"] == "10_Episodes/Episode e1"
    assert entity_paths[entity.corpus_entity_id] == "20_Entities/Person/Thomas Aquinas"
    entity_note = files[f"{entity_paths[entity.corpus_entity_id]}.md"].decode()
    assert f"[[{episode_paths['e1']}#^p-e1|Episode e1 · 00:10–00:20]]" in entity_note
    assert f"[[{episode_paths['e2']}#^p-e2|Episode e2 · 00:10–00:20]]" in entity_note
    assert "coreference_status: \"not_attempted\"" in entity_note
    assert "\\[\\[form\\]\\]" in files[f"{episode_paths['e1']}.md"].decode()
    assert "^p-e1" in files[f"{episode_paths['e1']}.md"].decode()
    assert "05_Corpus/Corpus.md" in files and "START HERE.md" in files


def test_vault_validation_rejects_a_broken_target():
    config = VaultConfig(vault_name="Synthetic vault")
    source = records()
    files = build_vault_files(config, *source)
    files["00_Index/Home.md"] += b"\n[[20_Entities/missing|Missing]]\n"
    with pytest.raises(ValueError, match="broken vault link"):
        validate_links(files, config, source[2])


def test_duplicate_entity_labels_use_short_episode_context():
    episodes = [Episode(created_by_run_id="source", episode_id=f"e{i}", corpus_id="hop",
        source_key=f"HoP24{i}.json", source_path=f"HoP/HoP24{i}.json",
        title=f"24{i}. A deliberately long episode title about a philosopher", language="en",
        metadata_status="matched", metadata_method="exact", facets={"sub_show": "HOP"})
        for i in (3, 4)]
    entities = [CorpusEntity(created_by_run_id="resolve", corpus_entity_id=f"ce-{i}", corpus_id="hop",
        type="Person", canonical_label="Aristotle", normalized_keys=["aristotle"],
        observed_surfaces=["Aristotle"], reviewed_aliases=[], episode_entity_ids=[f"ee-{i}"],
        episode_ids=[f"e{i}"], mention_count=1, episode_count=1, resolution_method="episode_local",
        resolution_status="unresolved", registry_key=f"episode:ee-{i}") for i in (3, 4)]
    hop = VaultConfig.model_validate_json(Path(__file__).resolve().parents[1].joinpath("config/vault.concept-navigation-v1.json").read_bytes())
    episode_paths, entity_paths = note_paths(hop, episodes, entities)
    assert episode_reference(episodes[0], hop) == "HoP 243"
    assert episode_paths == {"e3": "10_Episodes/HoP 243", "e4": "10_Episodes/HoP 244"}
    assert set(entity_paths.values()) == {
        "20_Entities/Person/Aristotle — HoP 243", "20_Entities/Person/Aristotle — HoP 244"}


def test_coreference_projection_adds_passage_evidence_without_new_nodes():
    config = VaultConfig(vault_name="Synthetic vault")
    source = records()
    entity = source[5][0]
    text = source[1][0].text
    start = text.index("discusses")
    # This fake link uses any token to test the rendering. The real coreference stage only
    # links the configured pronouns.
    link = CoreferenceLink(created_by_run_id="coref", reference_id="cr-1", episode_id="e1",
        transcript_version_id="t-e1", cluster_id="fc-1", char_start=start,
        char_end=start + len("discusses"), text="discusses", reference_kind="third_person_pronoun",
        passage_ids=["p-e1"], anchor_mention_ids=["m-e1"],
        episode_entity_ids=[source[4][0].episode_entity_id], corpus_entity_id=entity.corpus_entity_id,
        decision="linked", reason="single_corpus_entity")
    files = build_vault_files(config, *source, [link])
    _, entity_paths = note_paths(config, source[0], source[5])
    entity_note = files[f"{entity_paths[entity.corpus_entity_id]}.md"].decode()
    episode_note = files["10_Episodes/Episode e1.md"].decode()
    assert 'coreference_status: "provisional_fcoref"' in entity_note
    assert "## Provisional coreferential references" in entity_note
    assert "**Provisional coreference:**" in episode_note
    assert len([name for name in files if name.startswith("20_Entities/")]) == 1


def test_linked_entity_projection_exposes_qid_and_score():
    """A linking run is readable in the vault, and an abstention is not shown as a link."""
    config = VaultConfig(vault_name="Synthetic vault")
    source = records()
    linked = source[5][0].model_copy(update={"created_by_run_id": "link",
        "wikidata_qid": "Q9438", "link_status": "linked", "link_confidence": .83})
    source[5][0] = linked
    files = build_vault_files(config, *source)
    _, entity_paths = note_paths(config, source[0], source[5])
    note = files[f"{entity_paths[linked.corpus_entity_id]}.md"].decode()
    assert "https://www.wikidata.org/wiki/Q9438" in note
    assert "provisional score 0.830" in note
    assert 'link_status: "linked"' in note


def test_relation_projection_adds_visible_provisional_graph_node():
    """Accepted GLiREL output is visible, but not presented as a reviewed fact."""
    config = VaultConfig(vault_name="Synthetic vault")
    source = records()
    aquinas = source[5][0]
    albert = CorpusEntity(created_by_run_id="resolve", corpus_entity_id="ce-albert",
        corpus_id="hop", type="Person", canonical_label="Albert the Great",
        normalized_keys=["albert the great"], observed_surfaces=["Albert"], reviewed_aliases=[],
        episode_entity_ids=["ee-albert"], episode_ids=["e1"], mention_count=1, episode_count=1,
        resolution_method="episode_local", resolution_status="unresolved",
        registry_key="episode:ee-albert")
    source[5].append(albert)
    text = source[1][0].text
    reference_start = text.index("discusses")
    reference = CoreferenceLink(created_by_run_id="coref", reference_id="cr-relation",
        episode_id="e1", transcript_version_id="t-e1", cluster_id="fc-relation",
        char_start=reference_start, char_end=reference_start + len("discusses"), text="discusses",
        reference_kind="third_person_pronoun", passage_ids=["p-e1"],
        anchor_mention_ids=["m-e1"], episode_entity_ids=[source[4][0].episode_entity_id],
        corpus_entity_id=aquinas.corpus_entity_id, decision="linked",
        reason="single_corpus_entity")
    assertion = RelationAssertion(created_by_run_id="relations", assertion_id="ra-1",
        subject_corpus_entity_id=aquinas.corpus_entity_id, predicate="student_of",
        object_corpus_entity_id=albert.corpus_entity_id,
        evidence=[RelationEvidence(passage_id="p-e1", head_mention_id="cr-relation",
                                   tail_mention_id="m-albert",
                                   head_endpoint_kind="coreference_reference", score=.91)],
        confidence=.91)
    files = build_vault_files(config, *source, [reference], [assertion])
    validate_links(files, config, source[2])
    relation_notes = [name for name in files if name.startswith("30_Relations/")]
    assert len(relation_notes) == 1
    note = files[relation_notes[0]].decode()
    assert 'kind: "provisional_relation"' in note
    assert "coreference_assisted: true" in note
    assert "Thomas Aquinas — student of — Albert the Great" in note
    assert "[[20_Entities/Person/Thomas Aquinas|Thomas Aquinas]]" in note
    assert "[[20_Entities/Person/Albert the Great|Albert the Great]]" in note
    assert "[[10_Episodes/Episode e1#^p-e1|Episode e1 · 00:10–00:20]]" in note
    assert "subject “discusses” → Thomas Aquinas" in note
    assert 'path:\\"30_Relations\\"' in files[".obsidian/graph.json"].decode()

    # Withheld relations stay readable in the review folder but get no graph node.
    withheld = assertion.model_copy(update={"assertion_id": "ra-withheld", "predicate": "criticized"})
    verification = {"assertions": [assertion, withheld], "decisions": [
        {"assertion_id": "ra-1", "decision": "supported", "reason": "supported_evidence", "candidate_ids": ["v1"]},
        {"assertion_id": "ra-withheld", "decision": "abstain", "reason": "no_supported_evidence", "candidate_ids": ["v2"]}],
        "evidence": [
            {"candidate_id": "v1", "passage_id": "p-e1", "decision": "supported", "reason": "nli_entailment",
             "scores": {"entailment": .9, "contradiction": .02, "neutral": .08}},
            {"candidate_id": "v2", "passage_id": "p-e1", "decision": "abstain", "reason": "insufficient_entailment",
             "scores": {"entailment": .2, "contradiction": .1, "neutral": .7}}]}
    verified = build_vault_files(config, *source, [reference], [assertion], verification)
    validate_links(verified, config, source[2])
    assert len([name for name in verified if name.startswith("30_Relations/")]) == 1
    assert 'verification_status: "model_supported"' in verified[relation_notes[0]].decode()
    review = verified["90_Review/Relation verification.md"].decode()
    assert "criticized" in review and "**abstain**" in review and "**supported**" in review


def test_episode_naming_comes_only_from_configuration():
    """Another podcast's numbering works through the config. Without rules, titles are used."""
    episode = Episode(created_by_run_id="source", episode_id="e1", corpus_id="other",
        source_key="s2/e07.json", source_path="Other/s2/e07.json", title="Ep. 7 — The Long Now",
        language="en", metadata_status="matched", metadata_method="exact_file", facets={"season": "S2"})
    other = VaultConfig(vault_name="Other", series_facet="season", episode_number_pattern=r"Ep\. (\d+)")
    assert episode_reference(episode, other) == "S2 7"
    assert episode_reference(episode, VaultConfig(vault_name="Plain")) == "Ep. 7 — The Long Now"
    with pytest.raises(ValueError):
        VaultConfig(vault_name="Bad", source_key_pattern=r"(\d+)")
