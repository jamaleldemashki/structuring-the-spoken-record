import hashlib
import json
from typing import Protocol

from spoken_record.models import SourceBundle


def stable_id(prefix: str, *parts: str) -> str:
    """Short hash ID from the parts. The parts are JSON-encoded, so ("ab", "c") and
    ("a", "bc") give different IDs. Display names are never used as identity.
    """
    raw = json.dumps(parts, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return f"{prefix}-{hashlib.sha256(raw).hexdigest()[:24]}"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class TranscriptAdapter(Protocol):
    def ingest(
        self, data: bytes, *, corpus_id: str, source_key: str,
        source_path: str, run_id: str,
    ) -> SourceBundle: ...
