"""Adapter for the History of Philosophy transcripts.

Metadata is joined the same two ways as in the corpus audit: exact file name first, then
show and episode number. An ambiguous match is an error, titles are never guessed.
"""

import re
from collections import defaultdict
from pathlib import PurePosixPath

from spoken_record.adapters.base import sha256, stable_id
from spoken_record.adapters.whisper_json import parse_whisper, whisper_segments
from spoken_record.models import Episode, SourceBundle, TranscriptVersion

ADAPTER_VERSION = "1"


class MetadataIndex:
    def __init__(self, records: list[dict], profile: dict):
        self.by_filename: dict[str, list[dict]] = defaultdict(list)
        self.by_number: dict[tuple, list[dict]] = defaultdict(list)
        self.profile = profile
        shows = "|".join(re.escape(s) for s in profile["known_shows"])
        self.transcript_pattern = re.compile(
            rf"({shows})[_\-\s]*0*(\d+)([a-z]?(?=$|[_\-\s]))?", re.I
        )
        for record in records:
            if filename := record.get("filename"):
                # Accept both / and \ in paths. Titles are never fuzzy-matched.
                stem = PurePosixPath(filename.replace("\\", "/")).stem
                self.by_filename[stem].append(record)
            match = re.match(r"^\s*(\d+)\s*([a-z]?)(?=[.\s\-:]|$)", record.get("nav_title") or "", re.I)
            series = record.get("series")
            group = profile["series_to_group"].get(series)
            if series in profile["main_series"]:
                group = "HOP"
            if match and group:
                self.by_number[(group.upper(), int(match[1]), match[2].lower())].append(record)

    def match(self, source_key: str) -> tuple[dict | None, str, str]:
        stem = PurePosixPath(source_key).stem
        candidates = self.by_filename.get(stem, [])
        if candidates:
            return (candidates[0], "matched", "tier1_filename") if len(candidates) == 1 else (None, "ambiguous", "tier1_filename")
        match = self.transcript_pattern.search(stem)
        if match:
            key = (match[1].upper(), int(match[2]), (match[3] or "").lower())
            candidates = self.by_number.get(key, [])
            if len(candidates) == 1:
                return candidates[0], "matched", "tier2_group_number"
            if candidates:
                return None, "ambiguous", "tier2_group_number"
        return None, "missing", "unresolved"

    def show(self, source_key: str) -> str | None:
        match = self.transcript_pattern.search(source_key)
        return match[1].upper() if match else None


class HoPWhisperAdapter:
    def __init__(self, metadata: MetadataIndex):
        self.metadata = metadata

    def ingest(self, data: bytes, *, corpus_id: str, source_key: str,
               source_path: str, run_id: str) -> SourceBundle:
        raw = parse_whisper(data)
        metadata, status, method = self.metadata.match(source_key)
        meta = metadata or {}
        ep_id = stable_id("ep", corpus_id, source_key)
        digest = sha256(data)
        tv_id = stable_id("tv", ep_id, digest, "hop_whisper", ADAPTER_VERSION)
        episode = Episode(
            created_by_run_id=run_id, episode_id=ep_id, corpus_id=corpus_id,
            source_key=source_key, source_path=source_path,
            title=meta.get("title") or meta.get("nav_title") or PurePosixPath(source_key).stem,
            language=raw.get("language") or "und", source_url=meta.get("link"),
            audio_url=meta.get("audio_url"), publication_date=meta.get("pub_date"),
            metadata_status=status, metadata_method=method,
            facets={"content_era": meta.get("series"), "sub_show": self.metadata.show(source_key)},
        )
        transcript = TranscriptVersion(
            created_by_run_id=run_id, transcript_version_id=tv_id, episode_id=ep_id,
            kind="raw_asr", text=raw["text"], source_sha256=digest,
            adapter="hop_whisper", adapter_version=ADAPTER_VERSION,
        )
        return SourceBundle(episode=episode, transcript=transcript, segments=whisper_segments(raw, tv_id, run_id))

    def audit_differs(self, bundle: SourceBundle, row: dict) -> bool:
        """True if the episode disagrees with the audit on join tier or content era."""
        return (bundle.episode.metadata_method != row["join_tier"]
                or bundle.episode.facets["content_era"] != row.get("content_era"))
