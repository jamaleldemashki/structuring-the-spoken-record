"""Data models for the identity-pair and pronoun coreference review."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from spoken_record.evaluation import EntityType


class StrictEvaluationModel(BaseModel):
    """Unknown fields are an error, so typos in annotations get caught."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ReferenceEvaluationConfig(StrictEvaluationModel):
    """Packet size limits and exclusions."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    schema_version: Literal[1] = 1
    identity_types: list[EntityType] = Field(min_length=1)
    include_all_identity_candidates: Literal[True] = True
    coreference_max_examples: int = Field(gt=0)
    identity_max_evidence_passages: int = Field(gt=0)
    sampling_method: Literal["decision-stratified-stable-hash-v1"]
    sampling_salt: str = Field(min_length=1)
    included_reference_kinds: list[Literal["third_person_pronoun"]] = Field(min_length=1)
    excluded_reference_forms: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_values(self) -> Self:
        """Lists must be normalized and must not overlap."""
        if len(set(self.identity_types)) != len(self.identity_types):
            raise ValueError("duplicate identity type")
        if len(set(self.included_reference_kinds)) != len(self.included_reference_kinds):
            raise ValueError("duplicate reference kind")
        if any(value != value.casefold().strip() or not value for value in self.excluded_reference_forms):
            raise ValueError("excluded reference forms must be nonempty normalized strings")
        if len(set(self.excluded_reference_forms)) != len(self.excluded_reference_forms):
            raise ValueError("duplicate excluded reference form")
        return self


class EvidencePassage(StrictEvaluationModel):
    """Copy of a source passage needed to judge one sample."""

    passage_id: str
    episode_id: str
    episode_title: str
    passage_index: int = Field(ge=0)
    char_start: int = Field(ge=0)
    char_end: int = Field(gt=0)
    start_s: float | None = Field(default=None, ge=0)
    end_s: float | None = Field(default=None, ge=0)
    text: str = Field(min_length=1)
    text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def valid_range(self) -> Self:
        """The copied text must match its character range in the transcript."""
        if self.char_end - self.char_start != len(self.text):
            raise ValueError("evidence passage range differs from text")
        if (self.start_s is None) != (self.end_s is None):
            raise ValueError("evidence passage times must both be present or absent")
        if self.start_s is not None and self.end_s < self.start_s:
            raise ValueError("evidence passage times are reversed")
        return self


class IdentityEvaluationSample(StrictEvaluationModel):
    """One retrieved entity pair with the full model decision."""

    candidate_id: str
    type: EntityType
    left_corpus_entity_id: str
    right_corpus_entity_id: str
    left_label: str
    right_label: str
    episode_ids: list[str] = Field(min_length=1)
    model_decision: Literal["merge", "separate", "abstain"]
    model_reason: str
    retrieval_methods: list[str] = Field(min_length=1)
    surface_similarity: float = Field(ge=0, le=1)
    lexical_context_similarity: float = Field(ge=0, le=1)
    neighbor_similarity: float = Field(ge=0, le=1)
    combined_score: float = Field(ge=0, le=1)
    evidence: list[EvidencePassage] = Field(min_length=1)


class IdentityEvaluationAnnotation(StrictEvaluationModel):
    """The reviewer's own judgment for one candidate pair."""

    candidate_id: str
    status: Literal["unreviewed", "reviewed"] = "unreviewed"
    gold_decision: Literal["same", "different", "uncertain"] | None = None
    reviewer: str | None = None
    reviewed_at: str | None = None
    annotation_seconds: float | None = Field(default=None, ge=0)
    notes: str = ""

    @model_validator(mode="after")
    def complete_review(self) -> Self:
        """A reviewed row needs a decision, a reviewer and a timestamp."""
        fields = (self.gold_decision, (self.reviewer or "").strip(), (self.reviewed_at or "").strip())
        if self.status == "reviewed" and not all(fields):
            raise ValueError("reviewed identity annotation is incomplete")
        if self.status == "unreviewed" and any((self.gold_decision, self.reviewer, self.reviewed_at,
                                                 self.annotation_seconds is not None)):
            raise ValueError("unreviewed identity annotation carries review data")
        return self


class CoreferenceTarget(StrictEvaluationModel):
    """A Person entity the reviewer can choose as a target."""

    corpus_entity_id: str
    canonical_label: str
    observed_surfaces: list[str] = Field(min_length=1)
    sources: list[Literal["model_anchor", "local_context"]] = Field(min_length=1)
    model_anchor_mentions: int = Field(ge=0)

    @model_validator(mode="after")
    def valid_sources(self) -> Self:
        """Source flags must be unique and the anchor counts must match them."""
        if len(set(self.sources)) != len(self.sources):
            raise ValueError("duplicate coreference target source")
        if ("model_anchor" in self.sources) != (self.model_anchor_mentions > 0):
            raise ValueError("coreference target anchor count disagrees with sources")
        return self


class CoreferenceEvaluationSample(StrictEvaluationModel):
    """One pronoun with the model's anchors and the passage context."""

    reference_id: str
    episode_id: str
    episode_title: str
    transcript_version_id: str
    char_start: int = Field(ge=0)
    char_end: int = Field(gt=0)
    text: str = Field(min_length=1)
    reference_kind: Literal["third_person_pronoun"]
    model_decision: Literal["linked", "abstain"]
    model_reason: str
    predicted_corpus_entity_id: str | None = None
    predicted_label: str | None = None
    candidate_targets: list[CoreferenceTarget]
    evidence: list[EvidencePassage] = Field(min_length=1)

    @model_validator(mode="after")
    def valid_prediction(self) -> Self:
        """A linked sample needs a target, an abstention must not have one."""
        linked = self.model_decision == "linked"
        if linked != (self.predicted_corpus_entity_id is not None):
            raise ValueError("coreference model decision and target disagree")
        if linked != (self.predicted_label is not None):
            raise ValueError("coreference model decision and target label disagree")
        if self.char_end - self.char_start != len(self.text):
            raise ValueError("coreference sample range differs from text")
        target_ids = [row.corpus_entity_id for row in self.candidate_targets]
        if len(set(target_ids)) != len(target_ids):
            raise ValueError("duplicate coreference candidate target")
        if linked and self.predicted_corpus_entity_id not in target_ids:
            raise ValueError("predicted coreference target is unavailable to reviewer")
        aligned = [row for row in self.evidence if row.char_start <= self.char_start and
                   self.char_end <= row.char_end and
                   row.text[self.char_start - row.char_start:self.char_end - row.char_start] == self.text]
        if not aligned:
            raise ValueError("coreference reference is not aligned to an evidence passage")
        return self


class CoreferenceEvaluationAnnotation(StrictEvaluationModel):
    """The reviewed target, or the decision that the pronoun should not be linked."""

    reference_id: str
    status: Literal["unreviewed", "reviewed"] = "unreviewed"
    gold_decision: Literal["link", "no_link", "ambiguous", "out_of_scope"] | None = None
    gold_corpus_entity_id: str | None = None
    reviewer: str | None = None
    reviewed_at: str | None = None
    annotation_seconds: float | None = Field(default=None, ge=0)
    notes: str = ""

    @model_validator(mode="after")
    def complete_review(self) -> Self:
        """Exactly one target for a reviewed link, none otherwise."""
        if self.status == "reviewed":
            if self.gold_decision is None or not (self.reviewer or "").strip() or not (self.reviewed_at or "").strip():
                raise ValueError("reviewed coreference annotation is incomplete")
            if (self.gold_decision == "link") != (self.gold_corpus_entity_id is not None):
                raise ValueError("coreference gold decision and target disagree")
        elif any((self.gold_decision, self.gold_corpus_entity_id, self.reviewer, self.reviewed_at,
                  self.annotation_seconds is not None)):
            raise ValueError("unreviewed coreference annotation carries review data")
        return self
