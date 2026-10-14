"""Data models for the automatic identity refinement stage."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from spoken_record.evaluation import EntityType
from spoken_record.models import Record


RetrievalMethod = Literal[
    "exact_key",
    "person_name_containment",
    "person_initial_alignment",
    "person_name_block",
    "person_surname_variant",
    "fuzzy_surface",
]

Contradiction = Literal[
    "conflicting_numbers",
    "conflicting_person_qualifier",
    "conflicting_person_forename",
    "conflicting_reviewed_identity",
]


class IdentityRefinementConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[1] = 1
    candidate_types: list[EntityType]
    merge_methods: list[RetrievalMethod]
    fuzzy_similarity_floor: float = Field(ge=0, le=1)
    # The default keeps runs made before this rule existed valid.
    person_surname_similarity_floor: float = Field(default=.9, ge=0, le=1)
    merge_threshold: float = Field(ge=0, le=1)
    minimum_context_similarity: float = Field(ge=0, le=1)
    surface_weight: float = Field(ge=0, le=1)
    lexical_context_weight: float = Field(ge=0, le=1)
    neighbor_weight: float = Field(ge=0, le=1)
    block_conflicting_numbers: bool = True
    separate_incompatible_person_names: bool = True
    maximum_component_entities: int = Field(ge=2)

    @model_validator(mode="after")
    def valid_values(self) -> Self:
        if len(set(self.candidate_types)) != len(self.candidate_types):
            raise ValueError("duplicate candidate type")
        if len(set(self.merge_methods)) != len(self.merge_methods):
            raise ValueError("duplicate merge method")
        if abs(self.surface_weight + self.lexical_context_weight + self.neighbor_weight - 1) > 1e-9:
            raise ValueError("refinement score weights must sum to one")
        return self


class IdentityCandidateDecision(Record):
    candidate_id: str
    type: EntityType
    left_corpus_entity_id: str
    right_corpus_entity_id: str
    left_label: str
    right_label: str
    episode_ids: list[str] = Field(min_length=1)
    retrieval_methods: list[RetrievalMethod] = Field(min_length=1)
    surface_similarity: float = Field(ge=0, le=1)
    lexical_context_similarity: float = Field(ge=0, le=1)
    neighbor_similarity: float = Field(ge=0, le=1)
    combined_score: float = Field(ge=0, le=1)
    contradictions: list[Contradiction] = Field(default_factory=list)
    decision: Literal["merge", "separate", "abstain"]
    reason: str
    evidence_passage_ids: list[str] = Field(min_length=1)
    method: Literal["indexed-name-tfidf-context-v1"] = "indexed-name-tfidf-context-v1"


class EntityRedirect(Record):
    from_corpus_entity_id: str
    to_corpus_entity_id: str
    candidate_ids: list[str] = Field(min_length=1)
    reason: Literal["provisional_automatic_merge"] = "provisional_automatic_merge"
