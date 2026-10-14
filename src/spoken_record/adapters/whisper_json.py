"""Whisper JSON transcripts for any podcast, with episode metadata from a JSONL file.

Whisper (and tools built on it) write `{"text", "language", "segments": [{"id", "start",
"end", "text", ...}]}`. This adapter reads that format and matches each transcript to one
metadata row by its file name. Nothing about a specific podcast is hard-coded: titles,
dates, audio links and grouping facets (series, season, era, ...) all come from the metadata.

Metadata JSONL, one object per episode:

    {"file": "season1/ep001.json", "title": "...", "date": "2021-03-04",
     "audio_url": "https://...mp3", "page_url": "https://...", "language": "en",
     "facets": {"series": "Main feed", "season": "1"}}

`file` is the path relative to the profile's `transcripts_dir` and is the only required
field. A transcript without a row gets metadata status `missing`.
"""

import json
from pathlib import PurePosixPath

from spoken_record.adapters.base import sha256, stable_id
from spoken_record.models import Episode, SourceBundle, SourceSegment, TranscriptVersion

ADAPTER = "whisper_json"
ADAPTER_VERSION = "1"
QUALITY_FIELDS = ("avg_logprob", "no_speech_prob", "compression_ratio", "temperature")
METADATA_FIELDS = {"file", "title", "date", "audio_url", "page_url", "language", "facets"}


def parse_whisper(data: bytes) -> dict:
    """Parse a Whisper JSON file. The segment texts joined together must give the full text."""
    raw = json.loads(data)
    if not isinstance(raw, dict) or not isinstance(raw.get("text"), str):
        raise ValueError("missing transcript text")
    raw_segments = raw.get("segments")
    if not isinstance(raw_segments, list) or not raw_segments:
        raise ValueError("missing source segments")
    if any(not isinstance(s, dict) or not isinstance(s.get("text"), str) for s in raw_segments):
        raise ValueError("invalid segment text")
    if "".join(s["text"] for s in raw_segments) != raw["text"]:
        raise ValueError("source_alignment_error: top-level text differs from segment concatenation")
    return raw


def whisper_segments(raw: dict, tv_id: str, run_id: str) -> list[SourceSegment]:
    """Turn Whisper segments into SourceSegments with character offsets into the full text."""
    segments = []
    offset = 0
    for index, source in enumerate(raw["segments"]):
        text = source["text"]
        segments.append(SourceSegment(
            created_by_run_id=run_id, segment_id=stable_id("seg", tv_id, str(index)),
            transcript_version_id=tv_id, source_segment_id=str(source.get("id", index)),
            index=index, char_start=offset, char_end=offset + len(text), text=text,
            start_s=source.get("start"), end_s=source.get("end"),
            quality={k: source[k] for k in QUALITY_FIELDS if k in source},
        ))
        offset += len(text)
    return segments


class FileMetadata:
    """Metadata rows looked up by transcript path. Duplicates and unknown fields raise an error."""

    def __init__(self, rows: list[dict]):
        self.by_file: dict[str, dict] = {}
        for row in rows:
            if not isinstance(row.get("file"), str) or not row["file"].strip():
                raise ValueError("every metadata row needs a 'file'")
            if unknown := set(row) - METADATA_FIELDS:
                raise ValueError(f"unknown metadata fields: {sorted(unknown)}")
            facets = row.get("facets", {})
            if not isinstance(facets, dict) or any(not isinstance(v, (str, type(None))) for v in facets.values()):
                raise ValueError("metadata 'facets' must map names to strings")
            key = PurePosixPath(row["file"].replace("\\", "/")).as_posix()
            if key in self.by_file:
                raise ValueError(f"duplicate metadata row for {key}")
            self.by_file[key] = row

    def match(self, source_key: str) -> tuple[dict | None, str, str]:
        row = self.by_file.get(source_key)
        return (row, "matched", "exact_file") if row else (None, "missing", "unresolved")


class WhisperJsonAdapter:
    def __init__(self, metadata: FileMetadata):
        self.metadata = metadata

    def ingest(self, data: bytes, *, corpus_id: str, source_key: str,
               source_path: str, run_id: str) -> SourceBundle:
        raw = parse_whisper(data)
        meta, status, method = self.metadata.match(source_key)
        meta = meta or {}
        ep_id = stable_id("ep", corpus_id, source_key)
        digest = sha256(data)
        tv_id = stable_id("tv", ep_id, digest, ADAPTER, ADAPTER_VERSION)
        episode = Episode(
            created_by_run_id=run_id, episode_id=ep_id, corpus_id=corpus_id,
            source_key=source_key, source_path=source_path,
            title=meta.get("title") or PurePosixPath(source_key).stem,
            language=meta.get("language") or raw.get("language") or "und",
            source_url=meta.get("page_url"), audio_url=meta.get("audio_url"),
            publication_date=meta.get("date"), metadata_status=status, metadata_method=method,
            facets=dict(meta.get("facets", {})),
        )
        transcript = TranscriptVersion(
            created_by_run_id=run_id, transcript_version_id=tv_id, episode_id=ep_id,
            kind="raw_asr", text=raw["text"], source_sha256=digest,
            adapter=ADAPTER, adapter_version=ADAPTER_VERSION,
        )
        return SourceBundle(episode=episode, transcript=transcript, segments=whisper_segments(raw, tv_id, run_id))

    def audit_differs(self, bundle: SourceBundle, row: dict) -> bool:
        """True if the episode disagrees with its audit row (join method or facets)."""
        return (row.get("join_tier", bundle.episode.metadata_method) != bundle.episode.metadata_method
                or any(bundle.episode.facets.get(k) != v for k, v in row.get("facets", {}).items()))
