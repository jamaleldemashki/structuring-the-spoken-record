"""Core source records. Character spans are zero-based and end-exclusive, counted in
Unicode code points.
"""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    schema_version: Literal[1] = 1
    created_by_run_id: str = Field(min_length=1)


class Episode(Record):
    episode_id: str
    corpus_id: str
    source_key: str
    source_path: str
    title: str
    language: str
    source_url: str | None = None
    audio_url: str | None = None
    publication_date: str | None = None
    metadata_status: Literal["matched", "missing", "ambiguous"]
    metadata_method: str
    facets: dict[str, str | None] = Field(default_factory=dict)
    legacy_audit_id: str | None = None


class TranscriptVersion(Record):
    transcript_version_id: str
    episode_id: str
    kind: Literal["raw_asr", "plain_text"]
    text: str = Field(min_length=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    adapter: str
    adapter_version: str


class SourceSegment(Record):
    segment_id: str
    transcript_version_id: str
    source_segment_id: str
    index: int = Field(ge=0)
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    text: str
    start_s: float | None = Field(default=None, ge=0)
    end_s: float | None = Field(default=None, ge=0)
    speaker_label: str | None = None
    quality: dict[str, float | None] = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid_span(self) -> Self:
        if self.char_end - self.char_start != len(self.text):
            raise ValueError("segment character range does not match text length")
        if (self.start_s is None) != (self.end_s is None):
            raise ValueError("segment timestamps must both be present or both absent")
        if self.start_s is not None and self.end_s < self.start_s:
            raise ValueError("segment ends before it starts")
        return self


class SourceBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")
    episode: Episode
    transcript: TranscriptVersion
    segments: list[SourceSegment]

    @model_validator(mode="after")
    def integrity(self) -> Self:
        if self.episode.episode_id != self.transcript.episode_id:
            raise ValueError("transcript refers to a different episode")
        cursor = 0
        previous_start = -1.0
        ids: set[str] = set()
        for index, segment in enumerate(self.segments):
            if segment.transcript_version_id != self.transcript.transcript_version_id:
                raise ValueError("segment refers to a different transcript")
            if segment.segment_id in ids or segment.index != index:
                raise ValueError("duplicate segment ID or non-contiguous index")
            ids.add(segment.segment_id)
            if segment.char_start != cursor:
                raise ValueError("segment coverage has a gap or overlap")
            if self.transcript.text[segment.char_start:segment.char_end] != segment.text:
                raise ValueError("segment does not reproduce the source text")
            cursor = segment.char_end
            if segment.start_s is not None:
                if segment.start_s < previous_start:
                    raise ValueError("segment start times are not monotonic")
                previous_start = segment.start_s
            if segment.created_by_run_id != self.episode.created_by_run_id:
                raise ValueError("segment run provenance does not match episode")
        if self.transcript.created_by_run_id != self.episode.created_by_run_id:
            raise ValueError("transcript run provenance does not match episode")
        if self.segments and cursor != len(self.transcript.text):
            raise ValueError("segment coverage misses the transcript tail")
        if self.transcript.kind == "raw_asr" and not self.segments:
            raise ValueError("raw ASR must provide aligned segments")
        return self
