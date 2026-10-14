"""Plain-text adapter without timestamps. The bytes are decoded as they are, so CRLF
line endings and Unicode stay unchanged.
"""

from spoken_record.adapters.base import sha256, stable_id
from spoken_record.models import Episode, SourceBundle, TranscriptVersion


class PlainTextAdapter:
    def ingest(self, data: bytes, *, corpus_id: str, source_key: str,
               source_path: str, run_id: str) -> SourceBundle:
        ep_id = stable_id("ep", corpus_id, source_key)
        digest = sha256(data)
        return SourceBundle(
            episode=Episode(
                created_by_run_id=run_id, episode_id=ep_id, corpus_id=corpus_id,
                source_key=source_key, source_path=source_path, title=source_key,
                language="und", metadata_status="missing", metadata_method="not_supplied",
            ),
            transcript=TranscriptVersion(
                created_by_run_id=run_id,
                transcript_version_id=stable_id("tv", ep_id, digest, "plain_text", "1"),
                episode_id=ep_id, kind="plain_text", text=data.decode("utf-8"),
                source_sha256=digest, adapter="plain_text", adapter_version="1",
            ),
            segments=[],
        )
