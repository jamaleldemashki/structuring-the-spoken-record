"""Checks GLiREL relation candidates against the transcript with an NLI model.

This stage can't add triples, merge identities, guess speakers or change the source text.
It tests both directions and abstains when the support is too weak.
"""

from pathlib import Path

from spoken_record.adapters.base import sha256
from spoken_record.backends import EvidenceVerifier
from spoken_record.ingestion import workspace_path
from spoken_record.relations import build_assertions
from spoken_record.verification_models import (NLIScores, VerificationConfig,
                                               VerificationDecision, VerificationEvidence)


class NLIBackend:
    """Sequence-pair classifier, loaded lazily and offline. Inputs are never truncated."""

    def __init__(self, root: Path, config: VerificationConfig):
        """Load the pinned weights (safetensors) and check the label order and input budget."""
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.config = config
        # Loading from a local folder also stops some transformers versions from asking the
        # Hub for tokenizer metadata even when local_files_only is set.
        cache_dir = str(workspace_path(root, config.cache_dir)) if config.cache_dir else None
        local_path = snapshot_download(config.repo_id, revision=config.revision,
            cache_dir=cache_dir, local_files_only=True)
        options = dict(local_files_only=True, trust_remote_code=False)
        self.tokenizer = AutoTokenizer.from_pretrained(local_path, **options)
        self.model = AutoModelForSequenceClassification.from_pretrained(local_path,
            use_safetensors=True, **options).to(config.device).eval()
        labels = [self.model.config.id2label[i].lower() for i in range(self.model.config.num_labels)]
        if labels != config.label_order:
            raise ValueError(f"checkpoint class order differs from configuration: {labels}")
        if config.max_tokens > self.model.config.max_position_embeddings:
            raise ValueError("verification token budget exceeds encoder positions")

    def token_count(self, premise: str, hypothesis: str) -> int:
        """Token count for both texts plus special tokens. Nothing is dropped."""
        return len(self.tokenizer(premise, hypothesis, truncation=False)["input_ids"])

    def predict(self, premise: str, hypothesis: str) -> NLIScores:
        """Classify one pair, after checking the budget."""
        if self.token_count(premise, hypothesis) > self.config.max_tokens:
            raise ValueError("NLI input exceeds the configured token budget")
        inputs = self.tokenizer(premise, hypothesis, truncation=False, return_tensors="pt")
        inputs = {name: value.to(self.config.device) for name, value in inputs.items()}
        with self.torch.inference_mode():
            scores = self.model(**inputs).logits.softmax(dim=-1)[0].cpu().tolist()
        return NLIScores.model_validate(dict(zip(self.config.label_order, scores, strict=True)))


def evidence_policy(scores, reverse_scores, config):
    """Apply the fixed thresholds. They are not calibrated and ignore the GLiREL score."""
    if scores.contradiction >= config.contradiction_threshold:
        return "contradicted", "nli_contradiction"
    if scores.entailment < config.entailment_threshold:
        return "abstain", "insufficient_entailment"
    if scores.entailment - max(scores.neutral, scores.contradiction) < config.minimum_margin:
        return "abstain", "insufficient_class_margin"
    if reverse_scores is not None and reverse_scores.entailment >= config.entailment_threshold:
        return "abstain", "both_directions_supported"
    return "supported", "nli_entailment"


def prepare_evidence(run_id, candidate, contexts, transcripts, mentions, assignments, references, config, asymmetric):
    """Build the hypothesis from entity names seen in the text. The premise is the source text as is.

    A pronoun only uses earlier anchors of its own cluster that are inside the context and
    have a compatible type. No alias is inserted and no speaker name is guessed.
    """
    context = next(row for row in contexts if row.passage_id == candidate.passage_id
                   and row.char_start == candidate.context_char_start
                   and row.char_end == candidate.context_char_end)
    transcript = next(row for row in transcripts if row.transcript_version_id == context.transcript_version_id)
    premise = transcript.text[context.char_start:context.char_end]
    mention_by_id = {row.mention_id: row for row in mentions}
    assignment_by_id = {row.mention_id: row for row in assignments}
    reference_by_id = {row.reference_id: row for row in references}

    def name(kind, endpoint_id):
        """Use the named surface, or the pronoun's own antecedent in the context. Never a random nearby name.
        """
        if kind == "mention":
            return mention_by_id[endpoint_id].text
        reference = reference_by_id[endpoint_id]
        anchors = [mention_by_id[mid] for mid in reference.anchor_mention_ids
                   if mid in context.included_anchor_ids and mid in mention_by_id
                   and mention_by_id[mid].char_end <= reference.char_start
                   and assignment_by_id[mid].corpus_entity_id == reference.corpus_entity_id]
        return max(anchors, key=lambda row: (row.char_end, row.mention_id)).text if anchors else None

    subject = name(candidate.head_endpoint_kind, candidate.head_mention_id)
    object_text = name(candidate.tail_endpoint_kind, candidate.tail_mention_id)
    template = config.templates.get(candidate.canonical_predicate)
    reason = "missing_antecedent" if subject is None or object_text is None else "unsupported_predicate" if template is None else "pending"
    hypothesis = template.format(subject=subject, object=object_text) if reason == "pending" else None
    reverse = template.format(subject=object_text, object=subject) if hypothesis and asymmetric else None
    return VerificationEvidence(created_by_run_id=run_id, candidate_id=candidate.candidate_id,
        passage_id=candidate.passage_id, context_id=context.context_id, subject_text=subject,
        object_text=object_text, premise=premise, premise_sha256=sha256(premise.encode()),
        hypothesis=hypothesis, reverse_hypothesis=reverse, decision="abstain", reason=reason)


def verify_evidence(row, backend: EvidenceVerifier | None, config):
    """Missing anchors and pairs over budget become abstentions, without running the model."""
    if row.reason != "pending":
        return row
    count = backend.token_count(row.premise, row.hypothesis)
    reverse_count = backend.token_count(row.premise, row.reverse_hypothesis) if row.reverse_hypothesis else None
    result = row.model_copy(update={"input_tokens": count, "reverse_input_tokens": reverse_count})
    if count > config.max_tokens or (reverse_count is not None and reverse_count > config.max_tokens):
        return result.model_copy(update={"reason": "token_budget_exceeded"})
    scores = NLIScores.model_validate(backend.predict(row.premise, row.hypothesis))
    reverse = NLIScores.model_validate(backend.predict(row.premise, row.reverse_hypothesis)) if row.reverse_hypothesis else None
    decision, reason = evidence_policy(scores, reverse, config)
    return result.model_copy(update={"scores": scores, "reverse_scores": reverse,
                                     "decision": decision, "reason": reason})


def aggregate(run_id, candidates, assertions, evidence):
    """Hold back conflicting assertions. Promoted triples keep only their supported evidence."""
    by_candidate = {row.candidate_id: row for row in evidence}
    decisions, retained = [], []
    for assertion in assertions:
        matching = [row for row in candidates if (row.head_corpus_entity_id, row.canonical_predicate,
            row.tail_corpus_entity_id) == (assertion.subject_corpus_entity_id, assertion.predicate,
                                          assertion.object_corpus_entity_id)]
        supported = [row for row in matching if by_candidate[row.candidate_id].decision == "supported"]
        contradicted = [row.candidate_id for row in matching if by_candidate[row.candidate_id].decision == "contradicted"]
        if supported and contradicted:
            decision, reason = "abstain", "conflicting_evidence"
        elif supported:
            decision, reason = "supported", "supported_evidence"
            retained.extend(supported)
        elif contradicted:
            decision, reason = "contradicted", "contradicted_evidence"
        else:
            decision, reason = "abstain", "no_supported_evidence"
        decisions.append(VerificationDecision(created_by_run_id=run_id, assertion_id=assertion.assertion_id,
            decision=decision, reason=reason, candidate_ids=sorted(row.candidate_id for row in matching),
            supported_candidate_ids=sorted(row.candidate_id for row in supported),
            contradicted_candidate_ids=sorted(contradicted)))
    return decisions, list(build_assertions(run_id, retained))


def compute_verification(run_id, candidates, assertions, contexts, transcripts, mentions, assignments,
                         references, relation_labels, config, make_backend):
    """NLI check of every accepted relation candidate. ``make_backend`` is only called when at
    least one pair needs the model."""
    asymmetric = {row["canonical_predicate"]: row["asymmetric"] for row in relation_labels}
    prepared = [prepare_evidence(run_id, row, contexts, transcripts, mentions, assignments,
                references, config, asymmetric[row.canonical_predicate]) for row in candidates]
    backend = make_backend() if any(row.reason == "pending" for row in prepared) else None
    evidence = [verify_evidence(row, backend, config) for row in prepared]
    decisions, supported = aggregate(run_id, candidates, assertions, evidence)
    return evidence, decisions, supported
