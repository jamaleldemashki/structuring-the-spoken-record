"""Data models for the contextual identity-candidate analysis."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from spoken_record.evaluation import EntityType
from spoken_record.models import Record


class ContextResolutionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[1] = 1
    candidate_types: list[EntityType]
    suggestion_methods: list[Literal["exact_key", "person_name_containment", "fuzzy_surface"]]
    block_conflicting_numbers: bool = True
    fuzzy_similarity_floor: float = Field(ge=0, le=1)
    suggestion_threshold: float = Field(ge=0, le=1)
    minimum_context_similarity: float = Field(ge=0, le=1)
    surface_weight: float = Field(ge=0, le=1)
    lexical_context_weight: float = Field(ge=0, le=1)
    neighbor_weight: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def valid_weights(self) -> Self:
        if len(set(self.candidate_types)) != len(self.candidate_types):
            raise ValueError("duplicate candidate type")
        if len(set(self.suggestion_methods)) != len(self.suggestion_methods):
            raise ValueError("duplicate suggestion method")
        if abs(self.surface_weight + self.lexical_context_weight + self.neighbor_weight - 1) > 1e-9:
            raise ValueError("contextual score weights must sum to one")
        return self


class ContextIdentityCandidate(Record):
    candidate_id: str
    type: EntityType
    left_corpus_entity_id: str
    right_corpus_entity_id: str
    left_label: str
    right_label: str
    episode_ids: list[str] = Field(min_length=1)
    retrieval_methods: list[Literal["exact_key", "person_name_containment", "fuzzy_surface"]] = Field(min_length=1)
    surface_similarity: float = Field(ge=0, le=1)
    lexical_context_similarity: float = Field(ge=0, le=1)
    neighbor_similarity: float = Field(ge=0, le=1)
    combined_score: float = Field(ge=0, le=1)
    decision: Literal["suggest_merge", "abstain"]
    reason: str
    evidence_passage_ids: list[str] = Field(min_length=1)
