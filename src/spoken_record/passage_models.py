"""Passages are the evidence units we cite. Windows only chunk text for a model and can be rebuilt."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from spoken_record.models import Record


class SegmentationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[1] = 1
    passage_min_words: int = Field(default=150, gt=0)
    passage_target_words: int = Field(default=200, gt=0)
    passage_max_words: int = Field(default=300, gt=0)
    window_max_tokens: int = Field(default=200, gt=1)
    window_overlap_tokens: int = Field(default=40, ge=0)

    @model_validator(mode="after")
    def ordered_limits(self) -> Self:
        if not self.passage_min_words <= self.passage_target_words <= self.passage_max_words:
            raise ValueError("passage limits must satisfy min <= target <= max")
        if self.window_overlap_tokens >= self.window_max_tokens:
            raise ValueError("window overlap must be smaller than its budget")
        return self


class Passage(Record):
    passage_id: str
    episode_id: str
    transcript_version_id: str
    index: int = Field(ge=0)
    char_start: int = Field(ge=0)
    char_end: int = Field(gt=0)
    text: str = Field(min_length=1)
    word_count: int = Field(ge=0)
    start_s: float | None = Field(default=None, ge=0)
    end_s: float | None = Field(default=None, ge=0)
    timing_precision: Literal["source_segment", "unavailable"]
    source_segment_ids: list[str]
    method: Literal["sentence_packing", "paragraph_fallback", "token_fallback", "whitespace_only"]
    flags: list[str] = Field(default_factory=list)
    segmentation_fingerprint: str

    @model_validator(mode="after")
    def valid_range(self) -> Self:
        if self.char_end - self.char_start != len(self.text):
            raise ValueError("passage text length differs from its range")
        if self.word_count != len(self.text.split()):
            raise ValueError("passage word count differs from its text")
        if (self.start_s is None) != (self.end_s is None):
            raise ValueError("both passage times must be present or absent")
        if self.start_s is not None and self.end_s < self.start_s:
            raise ValueError("passage times reversed")
        if (self.start_s is None) != (self.timing_precision == "unavailable"):
            raise ValueError("timing precision contradicts times")
        return self


class ModelWindow(Record):
    window_id: str
    episode_id: str
    transcript_version_id: str
    index: int = Field(ge=0)
    char_start: int = Field(ge=0)
    char_end: int = Field(gt=0)
    passage_ids: list[str] = Field(min_length=1)
    token_start: int = Field(ge=0)
    token_end: int = Field(gt=0)
    token_count: int = Field(gt=0)
    token_budget: int = Field(gt=0)
    overlap_previous_tokens: int = Field(ge=0)
    tokenizer: str
    budget_verified_for_model: bool = False
    window_fingerprint: str

    @model_validator(mode="after")
    def valid_range(self) -> Self:
        if self.char_end <= self.char_start:
            raise ValueError("empty model window")
        if self.token_end - self.token_start != self.token_count or self.token_count > self.token_budget:
            raise ValueError("invalid window token count or exceeded budget")
        if self.overlap_previous_tokens >= self.token_count:
            raise ValueError("window must include new tokens")
        return self
