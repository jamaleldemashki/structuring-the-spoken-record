"""Data models for the Wikidata enrichment stage."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from spoken_record.models import Record


class EnrichmentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[1] = 1
    api_url: str = "https://www.wikidata.org/w/api.php"
    request_timeout_seconds: float = Field(default=20, gt=0)
    request_delay_seconds: float = Field(default=0.25, ge=0)
    retries: int = Field(default=3, ge=1, le=10)
    # maxlag lets Wikimedia tell us to wait when its servers lag. null leaves it out,
    # which is fine for small read-only runs.
    maxlag: int | None = Field(default=5, ge=1)
    entity_types: list[Literal["Person"]] = Field(default_factory=lambda: ["Person"], min_length=1)
    # Optional JSONL file with reviewed QIDs for entities the linker left ambiguous or NIL.
    reviewed_qids: str | None = None
    # Someone without a death date who was born less than this many years ago counts as living.
    living_max_age_years: int = Field(default=110, ge=1)


class ReviewedQid(BaseModel):
    """A reviewed decision that assigns a Wikidata item to a local entity."""

    model_config = ConfigDict(extra="forbid")
    corpus_entity_id: str
    qid: str = Field(pattern=r"^Q[1-9][0-9]*$")
    reviewer: str = Field(min_length=1)
    note: str = ""


class Lifespan(BaseModel):
    """Widest plausible lifespan in years (negative = BCE). Used by the overlap filter."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    start: int | None
    end: int | None
    living: bool
    approximate: bool
    basis: Literal["birth_death", "floruit", "mixed"]


class EntityFacts(Record):
    corpus_entity_id: str
    qid: str = Field(pattern=r"^Q[1-9][0-9]*$")
    qid_source: Literal["linked", "reviewed"]
    life: Lifespan | None
    is_human: bool
    # Best-rank occupations (P106), so each archive can filter on the roles that matter to it.
    occupation_qids: list[str] = Field(default_factory=list)
