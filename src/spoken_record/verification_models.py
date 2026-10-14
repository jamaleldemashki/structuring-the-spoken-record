"""Data models for relation verification. Its results are provisional."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from spoken_record.models import Record


class VerificationConfig(BaseModel):
    """NLI model (pinned), hypothesis templates and thresholds. The thresholds are not calibrated."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    schema_version: Literal[1] = 1
    repo_id: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    cache_dir: str = ".cache/huggingface/hub"
    offline_only: Literal[True] = True
    device: Literal["cpu"] = "cpu"
    max_tokens: int = Field(default=512, ge=16)
    entailment_threshold: float = Field(default=.8, ge=0, le=1)
    minimum_margin: float = Field(default=.2, ge=0, le=1)
    contradiction_threshold: float = Field(default=.8, ge=0, le=1)
    label_order: list[Literal["contradiction", "entailment", "neutral"]]
    templates: dict[str, str]

    @model_validator(mode="after")
    def valid_inventory(self) -> Self:
        """The label map must be complete and templates may only use known fields."""
        from string import Formatter
        if len(self.label_order) != 3 or set(self.label_order) != {"contradiction", "entailment", "neutral"}:
            raise ValueError("NLI requires exactly one of each class")
        for template in self.templates.values():
            parsed = list(Formatter().parse(template))
            if {name for _, name, _, _ in parsed if name is not None} != {"subject", "object"}:
                raise ValueError("hypotheses must contain only subject and object fields")
            if any(spec or conversion for _, _, spec, conversion in parsed):
                raise ValueError("hypothesis formatting directives are not supported")
        return self


class NLIScores(BaseModel):
    """Softmax scores from the classifier, not calibrated probabilities that something is true."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    contradiction: float = Field(ge=0, le=1)
    entailment: float = Field(ge=0, le=1)
    neutral: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def normalized(self) -> Self:
        """Catch broken adapters instead of taking any logits as scores."""
        if abs(self.contradiction + self.entailment + self.neutral - 1) > 1e-5:
            raise ValueError("NLI scores must sum to one")
        return self


class VerificationEvidence(Record):
    """One accepted GLiREL evidence item, its premise text and the NLI decision."""

    candidate_id: str
    passage_id: str
    context_id: str
    subject_text: str | None
    object_text: str | None
    premise: str
    premise_sha256: str
    hypothesis: str | None
    reverse_hypothesis: str | None
    input_tokens: int | None = Field(default=None, ge=1)
    reverse_input_tokens: int | None = Field(default=None, ge=1)
    scores: NLIScores | None = None
    reverse_scores: NLIScores | None = None
    decision: Literal["supported", "contradicted", "abstain"]
    reason: str


class VerificationDecision(Record):
    """How well the evidence supports a relation. Not a human review and not a claim of truth."""

    assertion_id: str
    decision: Literal["supported", "contradicted", "abstain"]
    reason: str
    candidate_ids: list[str]
    supported_candidate_ids: list[str]
    contradicted_candidate_ids: list[str]
