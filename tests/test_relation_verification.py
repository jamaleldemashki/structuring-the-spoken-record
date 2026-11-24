"""Tests for NLI verification: policy, provenance, withheld evidence and saved runs."""

from types import SimpleNamespace

import pytest

from test_relation_contexts import contexts
from spoken_record.relations import project_predictions, build_assertions
from spoken_record.relation_verification import (aggregate, evidence_policy, prepare_evidence,
                                                verify_evidence)
from spoken_record.verification_models import NLIScores, VerificationConfig


def config(**updates):
    """Real label order, with a made-up revision and template."""
    values = dict(repo_id="test/nli", revision="a"*40,
        label_order=["contradiction", "entailment", "neutral"],
        templates={"authored": "{subject} wrote {object}."})
    values.update(updates)
    return VerificationConfig(**values)


def score(entailment=.9, contradiction=.02):
    """Normalized fake scores, no model needed."""
    return NLIScores(entailment=entailment, contradiction=contradiction,
                     neutral=1-entailment-contradiction)


def evidence_fixture(**limits):
    """Distant antecedent example with an unrelated person in between."""
    (inputs, _, _, windows), args = contexts(**limits)
    text, passages, mentions, assignments, reference, mapping, relation_config = args
    tokens, ner, spans, start, end = inputs[0]
    by_id = {row.source_id: list(span) for span, row in spans.items()}
    candidates = list(project_predictions("relations", passages[1], [dict(head_pos=by_id["he"],
        tail_pos=by_id["b"], label="author of", score=.9)], spans, mapping, relation_config, "fp", start, end))
    assertions = list(build_assertions("relations", candidates))
    transcripts = [SimpleNamespace(text=text, transcript_version_id="t", episode_id="e")]
    entities = [SimpleNamespace(corpus_entity_id=f"ce-{mid}", canonical_label=name)
                for mid, name in (("a", "Augustine"), ("z", "Zeno"), ("b", "Book of Reason"))]
    return (candidates, assertions, windows, transcripts, mentions, assignments,
            entities, [reference], relation_config.model_dump(mode="json"))


def prepared(inputs, settings=None):
    """Prepare the one candidate in the fixture against its context."""
    candidates, _, windows, transcripts, mentions, assignments, _, references, _ = inputs
    return prepare_evidence("verify", candidates[0], windows, transcripts, mentions,
                            assignments, references, settings or config(), True)


class FakeBackend:
    """Fake NLI adapter that shows the test both hypothesis directions."""

    def token_count(self, premise, hypothesis):
        """Count both strings plus a fixed number of special tokens."""
        return len(premise.split()) + len(hypothesis.split()) + 3

    def predict(self, premise, hypothesis):
        """Only support the intended direction in this example."""
        return score() if hypothesis.startswith("Augustine wrote") else score(.03, .9)


def test_original_premise_and_recorded_anchor_form_the_hypothesis():
    """No nearby name is substituted and no made-up statement goes into the premise."""
    inputs = evidence_fixture()
    row = prepared(inputs)
    assert row.hypothesis == "Augustine wrote Book of Reason."
    assert row.reverse_hypothesis == "Book of Reason wrote Augustine."
    assert row.premise == inputs[3][0].text
    assert "He wrote" in row.premise and "Zeno disagreed" in row.premise
    result = verify_evidence(row, FakeBackend(), config())
    assert result.decision == "supported"
    assert result.scores.entailment == .9 and result.reverse_scores.contradiction == .9


@pytest.mark.parametrize("forward,reverse,decision,reason", [
    (score(), score(.03,.9), "supported", "nli_entailment"),
    (score(.02,.9), score(.03,.9), "contradicted", "nli_contradiction"),
    (score(.3,.1), score(.03,.9), "abstain", "insufficient_entailment"),
    (score(), score(), "abstain", "both_directions_supported"),
])
def test_policy_does_not_treat_neutral_or_both_directions_as_support(forward, reverse, decision, reason):
    """Score and direction checks don't depend on the original GLiREL score."""
    assert evidence_policy(forward, reverse, config()) == (decision, reason)


def test_missing_antecedent_and_unknown_predicate_never_call_model():
    """Uncertain upstream resolution and unsupported templates stay visible as such."""
    missing = prepared(evidence_fixture(max_context_tokens=12))
    assert missing.reason == "missing_antecedent"
    assert verify_evidence(missing, None, config()).decision == "abstain"
    unsupported = prepared(evidence_fixture(), config(templates={}))
    assert unsupported.reason == "unsupported_predicate"
    assert verify_evidence(unsupported, None, config(templates={})).scores is None


def test_pair_budget_never_silently_truncates():
    """With a small budget we abstain before the adapter's forward method is called."""
    class NeverPredict(FakeBackend):
        """Fails if inference is attempted in this over-budget test."""

        def predict(self, premise, hypothesis):
            """No model call after a failed token budget check."""
            raise AssertionError("over-budget inference")

    result = verify_evidence(prepared(evidence_fixture()), NeverPredict(), config(max_tokens=16))
    assert result.reason == "token_budget_exceeded" and result.scores is None


def test_conflicting_evidence_is_withheld_and_neutral_evidence_is_not_promoted():
    """One supported row doesn't carry unverified evidence into the exported assertion."""
    inputs = evidence_fixture()
    candidate = inputs[0][0]
    other = candidate.model_copy(update={"candidate_id": "other", "score": .99})
    candidates = [candidate, other]
    assertions = list(build_assertions("r", candidates))
    row = verify_evidence(prepared(inputs), FakeBackend(), config())
    neutral = row.model_copy(update={"candidate_id": "other", "decision": "abstain", "reason": "insufficient_entailment"})
    decisions, supported = aggregate("verify", candidates, assertions, [row, neutral])
    assert decisions[0].decision == "supported"
    assert len(supported[0].evidence) == 1 and supported[0].confidence == .9
    contradiction = neutral.model_copy(update={"decision": "contradicted", "reason": "nli_contradiction"})
    decisions, supported = aggregate("verify", candidates, assertions, [row, contradiction])
    assert decisions[0].reason == "conflicting_evidence" and supported == []


def test_invalid_templates_and_unnormalized_scores_are_rejected():
    """Unknown template fields and raw logits raise an error."""
    with pytest.raises(ValueError):
        config(templates={"authored": "{subject.__class__} wrote {object}."})
    with pytest.raises(ValueError):
        NLIScores(entailment=.9, contradiction=.9, neutral=.1)
