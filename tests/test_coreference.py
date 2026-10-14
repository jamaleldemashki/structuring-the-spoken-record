from spoken_record.coreference import project_clusters
from spoken_record.coreference_models import CoreferenceConfig
from spoken_record.extraction_models import ExtractedMention
from spoken_record.models import TranscriptVersion
from spoken_record.passage_models import Passage
from spoken_record.resolution_models import MentionAssignment


def config():
    return CoreferenceConfig(framework="fastcoref", framework_version="2.1.6",
        architecture="FCoref", repo_id="biu-nlp/f-coref",
        revision="e91dfff12879495d882ba9460d0c5d5dd44ade59", device="cpu",
        offline_only=True, max_tokens_in_batch=2048,
        reference_forms=["he", "him", "his", "himself", "she", "her", "hers", "herself"])


def mention(text, surface, mention_id, episode_entity_id, corpus_entity_id):
    start = text.index(surface)
    row = ExtractedMention(created_by_run_id="extract", mention_id=mention_id,
        episode_id="e1", transcript_version_id="t1", char_start=start,
        char_end=start + len(surface), text=surface, type="Person", score=.9,
        source_window_ids=[f"w-{mention_id}"], passage_ids=["p1"],
        extraction_fingerprint="extract-v1")
    assignment = MentionAssignment(created_by_run_id="resolve", mention_id=mention_id,
        episode_entity_id=episode_entity_id, corpus_entity_id=corpus_entity_id)
    return row, assignment


def fixture():
    text = "Thomas Aquinas discussed being. He wrote. Plato answered him."
    transcript = TranscriptVersion(created_by_run_id="source", transcript_version_id="t1",
        episode_id="e1", kind="raw_asr", text=text, source_sha256="0" * 64,
        adapter="test", adapter_version="1")
    passage = Passage(created_by_run_id="passages", passage_id="p1", episode_id="e1",
        transcript_version_id="t1", index=0, char_start=0, char_end=len(text), text=text,
        word_count=len(text.split()), timing_precision="unavailable", source_segment_ids=[],
        method="token_fallback", segmentation_fingerprint="segment-v1")
    thomas, thomas_assignment = mention(text, "Thomas Aquinas", "m1", "ee1", "ce-aquinas")
    plato, plato_assignment = mention(text, "Plato", "m2", "ee2", "ce-plato")
    return text, transcript, [passage], [thomas, plato], [thomas_assignment, plato_assignment]


def span(text, value, after=0):
    start = text.index(value, after)
    return start, start + len(value)


def test_pronoun_links_only_when_cluster_has_one_named_person_identity():
    text, transcript, passages, mentions, assignments = fixture()
    thomas = span(text, "Thomas Aquinas")
    he = span(text, "He")
    plato = span(text, "Plato")
    him = span(text, "him")
    clusters, links = project_clusters("coref", "e1", transcript,
        [[thomas, he], [plato, him]], passages, mentions, assignments, config(), "fingerprint")
    assert len(clusters) == 2 and len(links) == 2
    assert [(row.text, row.corpus_entity_id, row.decision) for row in links] == [
        ("He", "ce-aquinas", "linked"), ("him", "ce-plato", "linked")]
    assert all(row.passage_ids == ["p1"] for row in links)


def test_ambiguous_and_unanchored_clusters_abstain():
    text, transcript, passages, mentions, assignments = fixture()
    thomas, he, plato, him = (span(text, "Thomas Aquinas"), span(text, "He"),
                              span(text, "Plato"), span(text, "him"))
    _, links = project_clusters("coref", "e1", transcript,
        [[thomas, plato, he], [span(text, "answered"), him]],
        passages, mentions, assignments, config(), "fingerprint")
    assert [row.decision for row in links] == ["abstain", "abstain"]
    assert [row.reason for row in links] == ["ambiguous_person_anchors", "no_named_person_anchor"]


def test_first_and_second_person_forms_are_outside_the_probe():
    text, transcript, passages, mentions, assignments = fixture()
    thomas = span(text, "Thomas Aquinas")
    _, links = project_clusters("coref", "e1", transcript, [[thomas, thomas]],
                                passages, mentions, assignments, config(), "fingerprint")
    assert not links


def test_lowercase_role_noun_cannot_anchor_a_person_reference():
    text = "the philosopher spoke because he disagreed"
    transcript = TranscriptVersion(created_by_run_id="source", transcript_version_id="t1",
        episode_id="e1", kind="raw_asr", text=text, source_sha256="0" * 64,
        adapter="test", adapter_version="1")
    passage = Passage(created_by_run_id="passages", passage_id="p1", episode_id="e1",
        transcript_version_id="t1", index=0, char_start=0, char_end=len(text), text=text,
        word_count=len(text.split()), timing_precision="unavailable", source_segment_ids=[],
        method="token_fallback", segmentation_fingerprint="segment-v1")
    role, assignment = mention(text, "philosopher", "m-role", "ee-role", "ce-role")
    _, links = project_clusters("coref", "e1", transcript,
        [[span(text, "philosopher"), span(text, "he")]], [passage], [role], [assignment],
        config(), "fingerprint")
    assert links[0].decision == "abstain"
    assert links[0].reason == "no_named_person_anchor"
