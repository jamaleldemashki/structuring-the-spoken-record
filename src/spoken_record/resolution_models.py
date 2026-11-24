"""Data models for resolution: Mention to EpisodeEntity to CorpusEntity."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from spoken_record.models import EntityType
from spoken_record.models import Record


class TermKeys(BaseModel):
    """How the matching keys of terms are built (see terms.py)."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    lemmatize_types: list[EntityType] = Field(default_factory=list)
    article_types: list[EntityType] = Field(default_factory=list)
    articles: list[str] = Field(default_factory=lambda: ["the", "a", "an"])
    lemmatizer: Literal["en_core_web_sm"] = "en_core_web_sm"
    lemmatizer_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")


class ResolutionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[1] = 1
    normalization: Literal["unicode-nfc-casefold-whitespace-v1"]
    aliases: str
    cross_episode_exact_types: list[EntityType]
    cross_episode_multitoken_person: bool = True
    single_token_person_requires_review: bool = True
    all_other_types_require_review: bool = True
    # Words that are never a name, even when the NER model tags them ("I", "you", "she").
    # Mentions that consist of one of them are dropped before entities are formed. Pronouns
    # are left to coreference, which links them to a named person or abstains.
    not_names: list[str] = Field(default_factory=list)
    # Lemmatized matching keys for terms ("virtue" and "virtues"). Without it, only the
    # Unicode form, case and spaces are normalized.
    term_keys: TermKeys | None = None

    @model_validator(mode="after")
    def unique_types(self) -> Self:
        if len(set(self.cross_episode_exact_types)) != len(self.cross_episode_exact_types):
            raise ValueError("duplicate cross-episode exact type")
        if "Person" in self.cross_episode_exact_types:
            raise ValueError("Person uses the explicit multi-token rule")
        return self


class AliasRule(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1] = 1
    alias_id: str
    corpus_id: str
    type: EntityType
    identity_id: str
    canonical_label: str = Field(min_length=1)
    aliases: list[str] = Field(min_length=1)
    decision: Literal["merge", "block"]
    scope: Literal["corpus", "episode", "mentions"]
    episode_ids: list[str] = Field(default_factory=list)
    mention_ids: list[str] = Field(default_factory=list)
    evidence_passage_ids: list[str] = Field(min_length=1)
    reviewer: str = Field(min_length=1)
    reviewed_at: str
    note: str

    @model_validator(mode="after")
    def valid_scope(self) -> Self:
        if self.scope == "corpus" and (self.episode_ids or self.mention_ids):
            raise ValueError("corpus alias must not carry narrower targets")
        if self.scope == "episode" and (not self.episode_ids or self.mention_ids):
            raise ValueError("episode alias requires only episode targets")
        if self.scope == "mentions" and (not self.mention_ids or self.episode_ids):
            raise ValueError("mention alias requires only mention targets")
        if len(set(self.aliases)) != len(self.aliases):
            raise ValueError("duplicate literal alias")
        return self


class EpisodeEntity(Record):
    episode_entity_id: str
    episode_id: str
    transcript_version_id: str
    type: EntityType
    canonical_surface: str
    normalized_keys: list[str] = Field(min_length=1)
    mention_ids: list[str] = Field(min_length=1)
    passage_ids: list[str] = Field(min_length=1)
    mention_count: int = Field(gt=0)
    is_singleton: bool
    mean_score: float = Field(ge=0, le=1)
    max_score: float = Field(ge=0, le=1)
    resolution_method: Literal["exact_normalized", "reviewed_alias", "mention_block"]
    resolution_status: Literal["provisional", "reviewed", "unresolved"]
    alias_rule_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def valid_counts(self) -> Self:
        if self.mention_count != len(self.mention_ids) or self.is_singleton != (self.mention_count == 1):
            raise ValueError("episode entity counts disagree")
        if self.mean_score > self.max_score + 1e-12:
            raise ValueError("episode entity score aggregation disagrees")
        return self


class CorpusEntity(Record):
    corpus_entity_id: str
    corpus_id: str
    type: EntityType
    canonical_label: str
    normalized_keys: list[str] = Field(min_length=1)
    observed_surfaces: list[str] = Field(min_length=1)
    reviewed_aliases: list[str]
    episode_entity_ids: list[str] = Field(min_length=1)
    episode_ids: list[str] = Field(min_length=1)
    mention_count: int = Field(gt=0)
    episode_count: int = Field(gt=0)
    resolution_method: Literal["episode_local", "provisional_exact", "reviewed_alias", "blocked",
                               "automatic_context", "same_wikidata_item"]
    resolution_status: Literal["provisional", "reviewed", "unresolved"]
    # Filled in by linking. "not_attempted" tells an entity of a type that is not linked
    # apart from a NIL or ambiguous decision.
    wikidata_qid: str | None = Field(default=None, pattern=r"^Q[1-9][0-9]*$")
    link_status: Literal["not_attempted", "linked", "nil", "ambiguous"] = "not_attempted"
    link_confidence: float | None = Field(default=None, ge=0, le=1)
    registry_key: str

    @model_validator(mode="after")
    def valid_counts(self) -> Self:
        if self.episode_count != len(self.episode_ids) or len(set(self.episode_ids)) != self.episode_count:
            raise ValueError("corpus entity episode counts disagree")
        return self


class MentionAssignment(Record):
    mention_id: str
    episode_entity_id: str
    corpus_entity_id: str


class ResolutionDecision(Record):
    decision_id: str
    episode_entity_id: str
    corpus_entity_id: str
    method: Literal["episode_local", "provisional_exact", "reviewed_alias", "blocked", "automatic_context"]
    status: Literal["provisional", "reviewed", "unresolved"]
    alias_rule_ids: list[str] = Field(default_factory=list)
    reason: str


class ReviewCandidate(Record):
    candidate_id: str
    type: EntityType
    normalized_key: str
    observed_surfaces: list[str]
    episode_entity_ids: list[str] = Field(min_length=2)
    episode_ids: list[str] = Field(min_length=2)
    mention_count: int = Field(gt=1)
    status: Literal["needs_review"] = "needs_review"
    reason: Literal["single_token_person", "type_requires_review", "blocked_by_rule"]
