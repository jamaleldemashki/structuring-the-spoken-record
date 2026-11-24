"""Data models for within-episode coreference, tied to passages."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from spoken_record.models import Record


class CoreferenceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[1] = 1
    framework: Literal["fastcoref"]
    framework_version: str
    architecture: Literal["FCoref"]
    repo_id: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    device: Literal["cpu"]
    offline_only: bool
    max_tokens_in_batch: int = Field(gt=0)
    reference_forms: list[str] = Field(min_length=1)
    anchor_type: Literal["Person"] = "Person"
    require_capitalized_anchor: bool = True
    include_nominals: Literal[False] = False

    @model_validator(mode="after")
    def valid_forms(self) -> Self:
        if any(form != form.casefold() or not form.strip() for form in self.reference_forms):
            raise ValueError("coreference forms must be nonempty case-folded strings")
        if len(set(self.reference_forms)) != len(self.reference_forms):
            raise ValueError("duplicate coreference form")
        return self


class CoreferenceSpan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    char_start: int = Field(ge=0)
    char_end: int = Field(gt=0)
    text: str = Field(min_length=1)

    @model_validator(mode="after")
    def valid_span(self) -> Self:
        if self.char_end - self.char_start != len(self.text):
            raise ValueError("coreference span length differs from text")
        return self


class ModelCoreferenceCluster(Record):
    cluster_id: str
    episode_id: str
    transcript_version_id: str
    spans: list[CoreferenceSpan] = Field(min_length=2)
    model_fingerprint: str


class CoreferenceLink(Record):
    reference_id: str
    episode_id: str
    transcript_version_id: str
    cluster_id: str
    char_start: int = Field(ge=0)
    char_end: int = Field(gt=0)
    text: str = Field(min_length=1)
    reference_kind: Literal["third_person_pronoun"]
    passage_ids: list[str] = Field(min_length=1)
    anchor_mention_ids: list[str]
    episode_entity_ids: list[str]
    corpus_entity_id: str | None = None
    decision: Literal["linked", "abstain"]
    reason: Literal["single_corpus_entity", "no_named_person_anchor", "ambiguous_person_anchors"]
    method: Literal["fcoref-cluster-person-anchor-v1"] = "fcoref-cluster-person-anchor-v1"

    @model_validator(mode="after")
    def valid_decision(self) -> Self:
        if self.char_end - self.char_start != len(self.text):
            raise ValueError("coreference reference length differs from text")
        if (self.decision == "linked") != (self.corpus_entity_id is not None):
            raise ValueError("coreference decision contradicts target")
        if self.decision == "linked" and not self.anchor_mention_ids:
            raise ValueError("linked reference has no named anchor")
        return self
