"""Data models for Wikidata entity linking."""

import re
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from spoken_record.models import EntityType
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
    # Wikidata refuses requests while its servers lag more than this many seconds.
    # None never sends maxlag, so a slow server doesn't stop the run.
    maxlag: int | None = Field(default=5, ge=1, le=60)
    # Also search the fullest name seen in the transcripts, when it has more words than the
    # canonical label (for example "Thomas Aquinas" next to "Aquinas").
    search_fullest_surface: bool = False
    # Full-text search limited to items with one of these P31 values, per type. It finds
    # well known people the label search misses (Augustine of Hippo for "Augustine").
    fulltext_instance_types: dict[EntityType, list[str]] = Field(default_factory=dict)
    # Candidates of these types must have one of these P31 values, everything else is dropped.
    required_instance_types: dict[EntityType, list[str]] = Field(default_factory=dict)
    # How well known an item is, compared with its namesakes among the candidates for the
    # same entity: its Wikipedia sitelinks divided by the most sitelinks among them. A lesser
    # known philosopher only loses points next to a much better known namesake.
    popularity_weight: float = Field(default=0, ge=0, le=1)

    @model_validator(mode="after")
    def valid_weights_and_types(self) -> Self:
        if len(set(self.entity_types)) != len(self.entity_types):
            raise ValueError("duplicate entity-linking type")
        if abs(self.name_weight + self.type_weight + self.context_weight +
               self.popularity_weight - 1) > 1e-9:
            raise ValueError("entity-linking weights must sum to one")
        for name in ("fulltext_instance_types", "required_instance_types"):
            for qids in getattr(self, name).values():
                if not qids or any(not re.fullmatch(r"Q[1-9][0-9]*", qid) for qid in qids):
                    raise ValueError(f"{name} needs a list of QIDs")
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
    # Number of Wikipedia sitelinks and the score made from it. Both stay empty when the
    # popularity score is not used.
    sitelinks: int | None = Field(default=None, ge=0)
    popularity: float = Field(default=0, ge=0, le=1)
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
