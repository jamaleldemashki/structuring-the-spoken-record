"""Checks GLiREL relation candidates against the transcript with an NLI model.

This stage can't add triples, merge identities, guess speakers or change the source text.
It tests both directions and abstains when the support is too weak.
"""

import importlib.metadata
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Callable
from uuid import uuid4

from spoken_record import __version__
from spoken_record.adapters.base import sha256
from spoken_record.backends import EvidenceVerifier
from spoken_record.ingestion import dump_json, dump_jsonl, read_jsonl, workspace_path
from spoken_record.relation_models import RelationAssertion, RelationCandidate, RelationContext
from spoken_record.relations import (build_assertions, context_transcripts, load_chain,
                                      validate_run as validate_relations)
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
        local_path = snapshot_download(config.repo_id, revision=config.revision,
            cache_dir=str(workspace_path(root, config.cache_dir)), local_files_only=True)
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


def load_inputs(parent: Path):
    """Load relations that have context records, and the evidence chain behind them."""
    manifest = validate_relations(parent)
    if manifest["counts"].get("failed", 0):
        raise ValueError("verification requires a relation run without failed passages")
    if "relation_contexts.jsonl" not in manifest["artifact_sha256"]:
        raise ValueError("verification requires antecedent-window relation contexts")
    upstream = parent.parent / manifest["parent_run_id"]
    _, mentions, assignments, entities, references = load_chain(upstream)
    candidates = [RelationCandidate.model_validate(row) for row in read_jsonl(parent / "relation_candidates.jsonl")
                  if row["decision"] == "accepted"]
    assertions = [RelationAssertion.model_validate(row) for row in read_jsonl(parent / "relation_assertions.jsonl")]
    contexts = [RelationContext.model_validate(row) for row in read_jsonl(parent / "relation_contexts.jsonl")]
    return (candidates, assertions, contexts, context_transcripts(upstream), mentions,
            assignments, entities, references, json.loads((parent / "config.json").read_bytes()))


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


def summary(evidence, decisions):
    """Counts for candidate evidence and for assertion decisions, kept apart."""
    return {"evidence_records": len(evidence), "assertions": len(decisions), "failed": 0,
        **{f"{status}_evidence": sum(row.decision == status for row in evidence)
           for status in ("supported", "contradicted", "abstain")},
        **{f"{status}_assertions": sum(row.decision == status for row in decisions)
           for status in ("supported", "contradicted", "abstain")}}


def review_text(decisions, assertions, entities):
    """Readable list of decisions. NLI output is not presented as a reviewed fact."""
    names = {row.corpus_entity_id: row.canonical_label for row in entities}
    by_id = {row.assertion_id: row for row in assertions}
    lines = ["# Relation verification", "", "Automated NLI decisions; uncalibrated and not human-reviewed.", ""]
    for decision in decisions:
        row = by_id[decision.assertion_id]
        lines.extend([f"- {names[row.subject_corpus_entity_id]} → {row.predicate} → {names[row.object_corpus_entity_id]}",
                      f"  - {decision.decision}: {decision.reason}"])
    return "\n".join(lines) + "\n"


def verify_run(root: Path, input_name: str, config_name: str, run_id: str | None = None,
               *, backend_factory: Callable[[Path, VerificationConfig], EvidenceVerifier] | None = None) -> Path:
    """Write the verification run. If the backend fails, nothing is written."""
    root, started = root.resolve(), perf_counter()
    run_id = run_id or f"verify-{uuid4().hex[:12]}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]{0,95}", run_id):
        raise ValueError("invalid verification run ID")
    parent = workspace_path(root, input_name)
    if parent.parent != root / "data/prototype1/runs":
        raise ValueError("verification input must be a sibling relation snapshot")
    output = parent.parent / run_id
    if output.exists():
        raise FileExistsError(output)
    config_path = workspace_path(root, config_name)
    config_bytes = config_path.read_bytes()
    config = VerificationConfig.model_validate_json(config_bytes)
    parent_hash = sha256((parent / "manifest.json").read_bytes())
    candidates, assertions, contexts, transcripts, mentions, assignments, entities, references, relation_config = load_inputs(parent)
    asymmetric = {row["canonical_predicate"]: row["asymmetric"] for row in relation_config["labels"]}
    prepared = [prepare_evidence(run_id, row, contexts, transcripts, mentions, assignments,
                references, config, asymmetric[row.canonical_predicate]) for row in candidates]
    backend = (backend_factory or NLIBackend)(root, config) if any(row.reason == "pending" for row in prepared) else None
    evidence = [verify_evidence(row, backend, config) for row in prepared]
    decisions, supported = aggregate(run_id, candidates, assertions, evidence)
    if config_path.read_bytes() != config_bytes or sha256((parent / "manifest.json").read_bytes()) != parent_hash:
        raise ValueError("verification inputs changed during inference")
    output.mkdir(parents=True, exist_ok=False)
    dump_jsonl(output / "verification_evidence.jsonl", evidence)
    dump_jsonl(output / "verification_decisions.jsonl", decisions)
    dump_jsonl(output / "relation_assertions.jsonl", supported)
    dump_json(output / "config.json", config.model_dump(mode="json"))
    counts = summary(evidence, decisions)
    dump_json(output / "summary.json", counts)
    (output / "review.md").write_text(review_text(decisions, assertions, entities), encoding="utf-8")
    files = ("verification_evidence.jsonl", "verification_decisions.jsonl", "relation_assertions.jsonl",
             "config.json", "summary.json", "review.md")
    code_root = Path(__file__).parent
    dump_json(output / "manifest.json", {"schema_version": 1, "stage": "relation_verification",
        "run_id": run_id, "status": "complete", "package_version": __version__,
        "created_utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": perf_counter()-started,
        "parent_run_id": parent.name, "parent_manifest_sha256": parent_hash,
        "input_config": config_name, "input_config_sha256": sha256(config_bytes),
        "model": {"repo_id": config.repo_id, "revision": config.revision, "label_order": config.label_order},
        "dependencies": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "pydantic")},
        "counts": counts, "artifact_sha256": {name: sha256((output/name).read_bytes()) for name in files},
        "code_sha256": {str(path.relative_to(code_root)): sha256(path.read_bytes()) for path in sorted(code_root.rglob("*.py"))}})
    validate_run(output)
    return output


def validate_run(output: Path) -> dict:
    """Rebuild hypotheses and decisions from the saved scores, without the model."""
    output = output.resolve()
    manifest = json.loads((output / "manifest.json").read_bytes())
    if (manifest.get("schema_version"), manifest.get("stage"), manifest.get("status")) != (1, "relation_verification", "complete"):
        raise ValueError("unsupported verification snapshot")
    root = output.parents[3]
    parent = output.parent / manifest["parent_run_id"]
    if sha256((parent / "manifest.json").read_bytes()) != manifest["parent_manifest_sha256"]:
        raise ValueError("verification parent changed")
    config_bytes = workspace_path(root, manifest["input_config"]).read_bytes()
    if sha256(config_bytes) != manifest["input_config_sha256"]:
        raise ValueError("verification configuration changed")
    config = VerificationConfig.model_validate_json(config_bytes)
    if config.model_dump(mode="json") != json.loads((output/"config.json").read_bytes()):
        raise ValueError("saved verification configuration differs")
    if manifest["model"] != {"repo_id": config.repo_id, "revision": config.revision, "label_order": config.label_order}:
        raise ValueError("verification model provenance differs from configuration")
    names = {"verification_evidence.jsonl", "verification_decisions.jsonl", "relation_assertions.jsonl",
             "config.json", "summary.json", "review.md"}
    if set(manifest["artifact_sha256"]) != names:
        raise ValueError("unexpected verification artifact set")
    for name, digest in manifest["artifact_sha256"].items():
        if sha256((output/name).read_bytes()) != digest:
            raise ValueError(f"verification artifact changed: {name}")
    candidates, assertions, contexts, transcripts, mentions, assignments, entities, references, relation_config = load_inputs(parent)
    asymmetric = {row["canonical_predicate"]: row["asymmetric"] for row in relation_config["labels"]}
    evidence = [VerificationEvidence.model_validate(row) for row in read_jsonl(output/"verification_evidence.jsonl")]
    if [row.candidate_id for row in evidence] != [row.candidate_id for row in candidates]:
        raise ValueError("verification must cover every accepted candidate exactly once")
    for candidate, saved in zip(candidates, evidence):
        prepared = prepare_evidence(manifest["run_id"], candidate, contexts, transcripts, mentions,
            assignments, references, config, asymmetric[candidate.canonical_predicate])
        for field in ("created_by_run_id", "candidate_id", "passage_id", "context_id", "subject_text",
                      "object_text", "premise", "premise_sha256", "hypothesis", "reverse_hypothesis"):
            if getattr(prepared, field) != getattr(saved, field):
                raise ValueError(f"verification evidence differs: {field}")
        if prepared.reason != "pending":
            expected = prepared
        else:
            if saved.input_tokens is None or (bool(saved.reverse_hypothesis) != (saved.reverse_input_tokens is not None)):
                raise ValueError("verification lacks complete budget accounting")
            expected = prepared.model_copy(update={"input_tokens": saved.input_tokens,
                                                   "reverse_input_tokens": saved.reverse_input_tokens})
            if max(saved.input_tokens, saved.reverse_input_tokens or 0) > config.max_tokens:
                expected = expected.model_copy(update={"reason": "token_budget_exceeded"})
            else:
                if saved.scores is None or (bool(saved.reverse_hypothesis) != (saved.reverse_scores is not None)):
                    raise ValueError("verification lacks directional NLI scores")
                decision, reason = evidence_policy(saved.scores, saved.reverse_scores, config)
                expected = expected.model_copy(update={"scores": saved.scores, "reverse_scores": saved.reverse_scores,
                                                       "decision": decision, "reason": reason})
        if saved != expected:
            raise ValueError("verification decision is not reproducible")
    decisions, supported = aggregate(manifest["run_id"], candidates, assertions, evidence)
    if decisions != [VerificationDecision.model_validate(row) for row in read_jsonl(output/"verification_decisions.jsonl")]:
        raise ValueError("verification assertion decisions differ")
    if supported != [RelationAssertion.model_validate(row) for row in read_jsonl(output/"relation_assertions.jsonl")]:
        raise ValueError("supported assertion evidence differs")
    counts = summary(evidence, decisions)
    if counts != manifest["counts"] or counts != json.loads((output/"summary.json").read_bytes()):
        raise ValueError("verification summary differs")
    if review_text(decisions, assertions, entities) != (output/"review.md").read_text(encoding="utf-8"):
        raise ValueError("verification review differs")
    return manifest
