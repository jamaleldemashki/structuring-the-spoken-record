"""Tests for the identity and pronoun coreference review packets."""

import pytest
from pydantic import ValidationError

from spoken_record.adapters.base import sha256
from spoken_record.coreference_models import CoreferenceLink
from spoken_record.reference_evaluation import (
    review_markdown,
    score_coreference_subset,
    score_identity,
    select_coreference_links,
)
from spoken_record.reference_evaluation_models import (
    CoreferenceEvaluationAnnotation,
    CoreferenceEvaluationSample,
    CoreferenceTarget,
    EvidencePassage,
    IdentityEvaluationAnnotation,
    IdentityEvaluationSample,
    ReferenceEvaluationConfig,
)
from spoken_record.reference_review import reviewed_annotations, worksheet_rows


def config(maximum=8):
    """Small evaluation config for the sampling tests."""
    return ReferenceEvaluationConfig(identity_types=["Person"], include_all_identity_candidates=True,
        coreference_max_examples=maximum, identity_max_evidence_passages=2,
        sampling_method="decision-stratified-stable-hash-v1", sampling_salt="tests",
        included_reference_kinds=["third_person_pronoun"], excluded_reference_forms=["i"])


def link(index, decision, text="he"):
    """One valid synthetic model decision."""
    target = "ce-a" if decision == "linked" else None
    return CoreferenceLink(created_by_run_id="coref", reference_id=f"r-{decision}-{index}",
        episode_id=f"e{index % 3}", transcript_version_id=f"t{index % 3}", cluster_id=f"c{index}",
        char_start=index * 10, char_end=index * 10 + len(text), text=text,
        reference_kind="third_person_pronoun", passage_ids=[f"p{index}"],
        anchor_mention_ids=[f"m{index}"] if decision == "linked" else [],
        episode_entity_ids=[f"ee{index}"] if decision == "linked" else [],
        corpus_entity_id=target, decision=decision,
        reason="single_corpus_entity" if decision == "linked" else "no_named_person_anchor")


def evidence(text="Alpha"):
    """Minimal passage for the scoring fixtures."""
    return EvidencePassage(passage_id="p1", episode_id="e1", episode_title="Episode",
        passage_index=0, char_start=0, char_end=len(text), text=text,
        text_sha256=sha256(text.encode()))


def identity_sample(identifier, decision):
    """Minimal identity candidate with a chosen model decision."""
    return IdentityEvaluationSample(candidate_id=identifier, type="Person",
        left_corpus_entity_id=f"left-{identifier}", right_corpus_entity_id=f"right-{identifier}",
        left_label="Alpha", right_label="A. Alpha", episode_ids=["e1", "e2"],
        model_decision=decision, model_reason="fixture", retrieval_methods=["exact_key"],
        surface_similarity=1, lexical_context_similarity=.5, neighbor_similarity=0,
        combined_score=.7, evidence=[evidence()])


def coreference_sample(identifier, decision, target="ce-a"):
    """Minimal coreference item with one Person target to pick."""
    candidate = CoreferenceTarget(corpus_entity_id=target, canonical_label="Alpha",
        observed_surfaces=["Alpha"], sources=["local_context"], model_anchor_mentions=0)
    predicted = target if decision == "linked" else None
    return CoreferenceEvaluationSample(reference_id=identifier, episode_id="e1",
        episode_title="Episode", transcript_version_id="t1", char_start=0, char_end=2,
        text="he", reference_kind="third_person_pronoun", model_decision=decision,
        model_reason="fixture", predicted_corpus_entity_id=predicted,
        predicted_label="Alpha" if predicted else None, candidate_targets=[candidate], evidence=[evidence("he")])


def test_coreference_sampling_balances_decisions_and_excludes_speaker_forms():
    """The packet must not include first-person speaker references."""
    rows = [link(index, "linked") for index in range(12)]
    rows += [link(index + 20, "abstain") for index in range(12)]
    rows += [link(99, "linked", "I")]
    selected = select_coreference_links(rows, config())
    assert len(selected) == 8
    assert sum(row.decision == "linked" for row in selected) == 4
    assert sum(row.decision == "abstain" for row in selected) == 4
    assert all(row.text.casefold() != "i" for row in selected)
    assert selected == select_coreference_links(rows, config())


def test_reviewed_annotations_require_complete_and_consistent_judgments():
    """Incomplete reviews and links without a target must fail validation."""
    with pytest.raises(ValidationError):
        IdentityEvaluationAnnotation(candidate_id="c1", status="reviewed", gold_decision="same")
    with pytest.raises(ValidationError):
        CoreferenceEvaluationAnnotation(reference_id="r1", status="reviewed", gold_decision="link",
                                         reviewer="reviewer", reviewed_at="2026-09-22")


def test_identity_scoring_reports_false_merges_and_false_splits():
    """An abstained pair that should merge counts as a missed merge."""
    samples = [identity_sample("a", "merge"), identity_sample("b", "abstain"),
               identity_sample("c", "merge")]
    annotations = [IdentityEvaluationAnnotation(candidate_id="a", status="reviewed",
        gold_decision="same", reviewer="r", reviewed_at="2026-09-22"),
        IdentityEvaluationAnnotation(candidate_id="b", status="reviewed",
        gold_decision="same", reviewer="r", reviewed_at="2026-09-22"),
        IdentityEvaluationAnnotation(candidate_id="c", status="reviewed",
        gold_decision="different", reviewer="r", reviewed_at="2026-09-22")]
    result = score_identity(samples, annotations)
    assert result["merge"] == {"tp": 1, "fp": 1, "fn": 1,
                               "precision": .5, "recall": .5, "f1": .5}
    assert result["false_merges"] == 1 and result["false_splits"] == 1
    assert result["candidate_retrieval_recall"] is None


def test_coreference_scoring_uses_exact_target_and_abstention_quality():
    """A wrong target is both a false positive link and a missed correct link."""
    samples = [coreference_sample("correct", "linked"), coreference_sample("wrong", "linked"),
               coreference_sample("good-abstain", "abstain"),
               coreference_sample("missed", "abstain")]
    annotations = {
        "correct": CoreferenceEvaluationAnnotation(reference_id="correct", status="reviewed",
            gold_decision="link", gold_corpus_entity_id="ce-a", reviewer="r", reviewed_at="2026-09-22"),
        "wrong": CoreferenceEvaluationAnnotation(reference_id="wrong", status="reviewed",
            gold_decision="link", gold_corpus_entity_id="ce-b", reviewer="r", reviewed_at="2026-09-22"),
        "good-abstain": CoreferenceEvaluationAnnotation(reference_id="good-abstain", status="reviewed",
            gold_decision="no_link", reviewer="r", reviewed_at="2026-09-22"),
        "missed": CoreferenceEvaluationAnnotation(reference_id="missed", status="reviewed",
            gold_decision="link", gold_corpus_entity_id="ce-a", reviewer="r", reviewed_at="2026-09-22"),
    }
    result = score_coreference_subset(samples, annotations)
    assert result["link"] == {"tp": 1, "fp": 1, "fn": 2,
                              "precision": .5, "recall": 1 / 3, "f1": .4}
    assert result["coverage"] == .5
    assert result["target_accuracy_on_attempted_linkable"] == .5
    assert result["abstention_precision"] == .5


def test_readable_review_hides_model_answers_and_marks_reference_span():
    """Reviewers see the evidence but not the model's decision."""
    rendered = review_markdown([identity_sample("a", "merge")],
                               [coreference_sample("r", "linked")])
    assert "Model: **" not in rendered
    assert "→" not in rendered
    assert "score `0.700`" not in rendered
    assert "**he**" in rendered


def test_blind_worksheet_applies_valid_identity_and_coreference_judgments():
    """The editable worksheet columns turn into complete review records."""
    identity = [identity_sample("a", "merge")]
    coreference = [coreference_sample("r", "linked")]
    identity_gold = [IdentityEvaluationAnnotation(candidate_id="a")]
    coreference_gold = [CoreferenceEvaluationAnnotation(reference_id="r")]
    rows = worksheet_rows(identity, coreference, identity_gold, coreference_gold)
    assert all("model" not in field for field in rows[0])
    rows[0]["decision"] = "same"
    rows[1]["decision"] = "link"
    rows[1]["target_corpus_entity_id"] = "ce-a"
    reviewed_identity, reviewed_coreference = reviewed_annotations(
        identity, coreference, identity_gold, coreference_gold, rows, "reviewer", "2026-09-22")
    assert reviewed_identity[0].gold_decision == "same"
    assert reviewed_coreference[0].gold_corpus_entity_id == "ce-a"
    assert all(row.status == "reviewed" for row in reviewed_identity + reviewed_coreference)


def test_blind_worksheet_rejects_changed_context_and_unknown_targets():
    """Only the three review columns may be edited in the spreadsheet."""
    identity = [identity_sample("a", "merge")]
    coreference = [coreference_sample("r", "linked")]
    identity_gold = [IdentityEvaluationAnnotation(candidate_id="a")]
    coreference_gold = [CoreferenceEvaluationAnnotation(reference_id="r")]
    rows = worksheet_rows(identity, coreference, identity_gold, coreference_gold)
    rows[0]["prompt"] = "changed"
    with pytest.raises(ValueError, match="immutable worksheet columns"):
        reviewed_annotations(identity, coreference, identity_gold, coreference_gold,
                             rows, "reviewer", "2026-09-22")
    rows = worksheet_rows(identity, coreference, identity_gold, coreference_gold)
    rows[1]["decision"], rows[1]["target_corpus_entity_id"] = "link", "ce-unknown"
    with pytest.raises(ValueError, match="invalid coreference target"):
        reviewed_annotations(identity, coreference, identity_gold, coreference_gold,
                             rows, "reviewer", "2026-09-22")
