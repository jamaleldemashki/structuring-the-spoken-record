"""Reading transcripts and their metadata, with the checks against the corpus audit."""

import json
from pathlib import Path


from spoken_record.adapters.base import sha256
from spoken_record.adapters.hop_whisper import HoPWhisperAdapter, MetadataIndex
from spoken_record.adapters.whisper_json import FileMetadata, WhisperJsonAdapter
from spoken_record.models import SourceBundle


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def workspace_path(root: Path, name: str) -> Path:
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("paths must be relative to the workspace without '..'")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("input path escapes workspace")
    return path


def index_rows(rows: list[dict]) -> dict[str, dict]:
    result = {}
    for row in rows:
        key = row["source_path"]
        if key in result:
            raise ValueError(f"duplicate audit source_path: {key}")
        result[key] = row
    return result


def dump_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def dump_jsonl(path: Path, rows: list) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            data = row.model_dump(mode="json") if hasattr(row, "model_dump") else row
            handle.write(json.dumps(data, ensure_ascii=False, sort_keys=True) + "\n")


ADAPTERS = ("hop_whisper", "whisper_json")


def make_adapter(profile: dict, metadata_rows: list[dict]):
    """Each podcast has a profile that names its transcript format and folder."""
    if profile.get("adapter") == "hop_whisper":
        return HoPWhisperAdapter(MetadataIndex(metadata_rows, profile))
    if profile.get("adapter") == "whisper_json":
        return WhisperJsonAdapter(FileMetadata(metadata_rows))
    raise ValueError(f"unknown transcript adapter {profile.get('adapter')!r}; supported: {', '.join(ADAPTERS)}")


class IngestSetup:
    """Everything needed to read transcripts of one corpus: config, profile, adapter, audit."""

    def __init__(self, root: Path, config: dict, config_path: Path):
        self.root, self.config, self.config_path = root, config, config_path
        self.profile_path = workspace_path(root, self.config["profile"])
        self.profile = json.loads(self.profile_path.read_bytes())
        if self.profile.get("schema_version") != 1:
            raise ValueError("unsupported configuration schema version")
        if self.profile.get("adapter") not in ADAPTERS:
            raise ValueError(f"unknown transcript adapter {self.profile.get('adapter')!r}; "
                             f"supported: {', '.join(ADAPTERS)}")
        if not isinstance(self.config.get("corpus_id"), str) or not self.config["corpus_id"].strip():
            raise ValueError("corpus_id must be a nonempty namespace")
        self.corpus_id = self.config["corpus_id"]
        optional = {key: workspace_path(root, self.config[key]) if self.config.get(key) else None
                    for key in ("metadata", "audit", "quarantine")}
        self.metadata_path, self.audit_path, self.quarantine_path = (
            optional["metadata"], optional["audit"], optional["quarantine"])
        self.require_metadata = self.config.get("require_metadata", True)
        if self.require_metadata and self.metadata_path is None:
            raise ValueError("metadata is required unless require_metadata is false")
        self.metadata_rows = read_jsonl(self.metadata_path) if self.metadata_path else []
        self.audited = index_rows(read_jsonl(self.audit_path)) if self.audit_path else None
        self.quarantined = index_rows(read_jsonl(self.quarantine_path)) if self.quarantine_path else {}
        if self.audited is not None and set(self.audited) & set(self.quarantined):
            raise ValueError("audit and quarantine overlap")
        self.source_root = workspace_path(root, self.profile["transcripts_dir"])
        self.adapter = make_adapter(self.profile, self.metadata_rows)

    def support_files(self) -> list[Path]:
        return [p for p in (self.config_path, self.profile_path, self.metadata_path,
                            self.audit_path, self.quarantine_path) if p]

    def source_path(self, name: str) -> Path:
        path = workspace_path(self.root, name)
        if not path.is_relative_to(self.source_root) or path.suffix != ".json":
            raise ValueError("selected transcript must be a JSON file in the configured source directory")
        return path

    def skip_reason(self, name: str) -> tuple[str, str] | None:
        """Why a transcript is not read at all: quarantined, or missing from the audit."""
        if name in self.quarantined:
            return "excluded", self.quarantined[name]["reason_code"]
        if self.audited is not None and name not in self.audited:
            return "failed", "not_in_accepted_audit"
        return None

    def read(self, name: str, data: bytes, expected_sha256: str | None, run_id: str) -> SourceBundle:
        """Turn the bytes of one transcript into a SourceBundle, after the checksum and
        metadata checks. Raises ValueError with a short reason."""
        path = self.source_path(name)
        digest = sha256(data)
        audited = self.audited
        if (expected_sha256 is not None and digest != expected_sha256) or \
                (audited is not None and digest != audited[name]["source_sha256"]):
            raise ValueError("stale_source: the transcript differs from the checksum in the audit")
        bundle = self.adapter.ingest(
            data, corpus_id=self.corpus_id, source_key=path.relative_to(self.source_root).as_posix(),
            source_path=name, run_id=run_id,
        )
        if self.require_metadata and bundle.episode.metadata_status != "matched":
            raise ValueError("metadata_join_unresolved: no unique metadata row for this transcript")
        if audited is not None:
            audit = audited[name]
            if self.adapter.audit_differs(bundle, audit):
                raise ValueError("stale_metadata: join differs from accepted audit")
            bundle.episode.legacy_audit_id = audit.get("episode_id")
        return bundle
