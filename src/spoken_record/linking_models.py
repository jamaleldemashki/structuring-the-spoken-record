"""Data models for Wikidata entity linking."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from spoken_record.evaluation import EntityType
from spoken_record.models import Record


class EntityLinkingConfig(BaseModel):
    """Settings for search, ranking and abstention."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[1] = 1
    entity_types: list[EntityType]
    api_url: str = "https://www.wikidata.org/w/api.php"
    language: str = "en"
    top_k: int = Field(default=5, ge=1, le=20)
    request_timeout_seconds: float = Field(default=20, gt=0, le=120)
    request_delay_seconds: float = Field(default=.05, ge=0, le=5)
    retries: int = Field(default=3, ge=1, le=10)
    max_context_characters: int = Field(default=2400, ge=200, le=12000)
    embedding_repo_id: str
    embedding_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    embedding_device: Literal["cpu", "mps", "cuda"] = "cpu"
    embedding_batch_size: int = Field(default=16, ge=1, le=128)
    embedding_max_tokens: int = Field(default=384, ge=32, le=512)
    name_weight: float = Field(ge=0, le=1)
    type_weight: float = Field(ge=0, le=1)
    context_weight: float = Field(ge=0, le=1)
    keyword_type_score: float = Field(default=1, ge=0, le=1)
    nil_threshold: float = Field(default=0, ge=0, le=1)
    linked_threshold: float = Field(ge=0, le=1)
    minimum_margin: float = Field(ge=0, le=1)
    single_candidate_context_floor: float = Field(default=0, ge=0, le=1)
    single_token_person_context_floor: float = Field(default=0, ge=0, le=1)
    # Extra description words that count as type evidence, on top of the English defaults
    # (for example {"Person": ["composer", "drummer"]} for a music podcast).
    extra_type_keywords: dict[EntityType, list[str]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid_weights_and_types(self) -> Self:
        if len(set(self.entity_types)) != len(self.entity_types):
            raise ValueError("duplicate entity-linking type")
        if abs(self.name_weight + self.type_weight + self.context_weight - 1) > 1e-9:
            raise ValueError("entity-linking weights must sum to one")
        if self.nil_threshold > self.linked_threshold:
            raise ValueError("NIL threshold cannot exceed the linked threshold")
        return self


class WikidataCandidate(Record):
    """One Wikidata search result with its ranking features."""

    candidate_id: str
    corpus_entity_id: str
    query: str
    qid: str = Field(pattern=r"^Q[1-9][0-9]*$")
    label: str
    description: str
    aliases: list[str]
    instance_of_qids: list[str]
    retrieval_rank: int = Field(ge=1)
    name_similarity: float = Field(ge=0, le=1)
    type_compatibility: float = Field(ge=0, le=1)
    context_similarity: float = Field(ge=0, le=1)
    rank_score: float = Field(ge=0, le=1)


class LinkDecision(Record):
    """Linked, ambiguous or NIL decision for one corpus entity."""

    decision_id: str
    corpus_entity_id: str
    entity_type: EntityType
    canonical_label: str
    candidate_ids: list[str]
    selected_qid: str | None = Field(default=None, pattern=r"^Q[1-9][0-9]*$")
    decision: Literal["linked", "ambiguous", "nil"]
    confidence: float | None = Field(default=None, ge=0, le=1)
    margin: float | None = Field(default=None, ge=0, le=1)
    reason: str

    @model_validator(mode="after")
    def valid_decision(self) -> Self:
        if self.decision == "linked" and (self.selected_qid is None or self.confidence is None):
            raise ValueError("linked decision requires a selected QID and confidence")
        if self.decision != "linked" and self.selected_qid is not None:
            raise ValueError("abstaining decisions cannot select a QID")
        if self.decision == "ambiguous" and not self.candidate_ids:
            raise ValueError("ambiguous requires retrieved candidates")
        return self
