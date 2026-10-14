"""Data models for extracted mentions and the extraction config."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from spoken_record.evaluation import EntityType
from spoken_record.models import Record


class ExtractionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    schema_version: Literal[1] = 1
    task: Literal["mention_detection"]
    framework: Literal["gliner"]
    framework_version: str
    repo_id: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    revision_resolved_at: str
    device: Literal["cpu", "mps"]
    labels_config: str
    words_splitter: Literal["whitespace"]
    model_max_words: int = Field(gt=1)
    encoder_max_tokens: int = Field(default=512, gt=1)
    window_max_tokens: int = Field(gt=1)
    window_overlap_tokens: int = Field(ge=0)
    candidate_floor: float = Field(ge=0, le=1)
    batch_size: int = Field(gt=0)
    smoke_test: dict
    notes: list[str]

    @model_validator(mode="after")
    def valid_budget(self) -> Self:
        if self.window_max_tokens > self.model_max_words:
            raise ValueError("window maximum exceeds checkpoint word maximum")
        if self.window_overlap_tokens >= self.window_max_tokens:
            raise ValueError("window overlap prevents progress")
        return self


class ExtractionCandidate(Record):
    candidate_id: str
    window_id: str
    episode_id: str
    transcript_version_id: str
    char_start: int = Field(ge=0)
    char_end: int = Field(gt=0)
    text: str = Field(min_length=1)
    type: EntityType
    model_label: str
    score: float = Field(ge=0, le=1)
    decision: Literal["accepted", "below_threshold", "duplicate", "overlap"]
    selected_mention_id: str | None = None

    @model_validator(mode="after")
    def valid_candidate(self) -> Self:
        if self.char_end - self.char_start != len(self.text):
            raise ValueError("candidate text length differs from range")
        if (self.decision in {"accepted", "duplicate"}) != (self.selected_mention_id is not None):
            raise ValueError("candidate decision contradicts selected mention")
        return self


class ExtractedMention(Record):
    mention_id: str
    episode_id: str
    transcript_version_id: str
    char_start: int = Field(ge=0)
    char_end: int = Field(gt=0)
    text: str = Field(min_length=1)
    type: EntityType
    score: float = Field(ge=0, le=1)
    source_window_ids: list[str] = Field(min_length=1)
    passage_ids: list[str] = Field(min_length=1)
    extraction_fingerprint: str

    @model_validator(mode="after")
    def valid_mention(self) -> Self:
        if self.char_end - self.char_start != len(self.text):
            raise ValueError("mention text length differs from range")
        if len(set(self.source_window_ids)) != len(self.source_window_ids):
            raise ValueError("duplicate source window reference")
        return self
