"""The corpus config and choosing episodes by their code ("HoP 230-254").

A corpus config describes one podcast: its transcript profile (format and folder), its
metadata, an optional audit and quarantine list, how episode codes are read, and the
settings file of every step. Episodes are chosen from the audit when there is one, otherwise
from the metadata, so nobody has to write file paths or checksums by hand.
"""

import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from spoken_record.models import EntityType
from spoken_record.ingestion import IngestSetup, workspace_path


class EpisodeCodes(BaseModel):
    """How to read an episode code from a catalog row. Dotted names reach into nested
    fields, for example ``facets.season``."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    series_field: str | None = None
    # Fields that can hold the episode number, tried in this order.
    number_fields: list[str] = Field(default_factory=lambda: ["title"])
    number_pattern: str = r"\s*(\d+)\s*[.-]"
    # Field with the transcript path. Audit rows have the full path, metadata rows usually
    # only the file name inside the transcript folder.
    path_field: str = "source_path"
    path_in_transcripts_dir: bool = False


class StepConfigs(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    segment: str
    extract: str
    resolve: str
    refine: str
    # Name-variant rules inside one episode. Without it, the refine settings are used.
    refine_episode: str | None = None
    link: str
    coreference: str
    relations: str
    verify: str
    discover: str
    enrich: str
    web: str
    # Boilerplate and generic-term filters (filters.py). Without it nothing is filtered.
    filters: str | None = None
    # Vectors for asking questions in the app (search_index.py). Without it the app only
    # has the name search.
    search: str | None = None


class CorpusConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[1] = 1
    corpus_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    profile: str
    metadata: str | None = None
    audit: str | None = None
    quarantine: str | None = None
    # Without it, transcripts with no title, date or audio link are refused.
    require_metadata: bool = True
    episode_codes: EpisodeCodes = Field(default_factory=EpisodeCodes)
    steps: StepConfigs
    # Entities of these types that are linked to the same Wikidata item become one entity.
    merge_by_wikidata: list[EntityType] = Field(default_factory=list)


class SelectedEpisode(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    code: str | None
    source_path: str
    # The checksum the audit recorded, if there is an audit.
    expected_sha256: str | None


def field(row: dict, name: str | None):
    value = row
    for part in (name or "").split("."):
        if not part or not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def catalog(setup: IngestSetup, codes: EpisodeCodes) -> list[SelectedEpisode]:
    """Every episode that can be selected, with its code. Quarantined files are left out."""
    rows = list(setup.audited.values()) if setup.audited is not None else setup.metadata_rows
    transcripts_dir = Path(setup.profile["transcripts_dir"])
    result = []
    for row in rows:
        path = field(row, codes.path_field)
        if not isinstance(path, str) or not path:
            continue
        source_path = (transcripts_dir / path).as_posix() if codes.path_in_transcripts_dir else path
        if source_path in setup.quarantined:
            continue
        series = field(row, codes.series_field) if codes.series_field else None
        match = next((found for name in codes.number_fields
                      if (found := re.match(codes.number_pattern, str(field(row, name) or "")))), None)
        code = f"{series} {int(match[1])}" if match and series else None
        expected = row.get("source_sha256") if setup.audited is not None else None
        result.append(SelectedEpisode(code=code, source_path=source_path, expected_sha256=expected))
    return sorted(result, key=lambda row: row.source_path)


def select_episodes(setup: IngestSetup, codes: EpisodeCodes, text: str) -> list[SelectedEpisode]:
    """Parse a selection such as "HoP 230-254, HAP 54", "all", or transcript paths.

    Codes are compared without case. An unknown code or a code that matches more than one
    transcript stops the selection, so a typo never quietly drops an episode.
    """
    available = catalog(setup, codes)
    by_code: dict[tuple[str, int], list[SelectedEpisode]] = {}
    for row in available:
        if row.code:
            series, _, number = row.code.rpartition(" ")
            by_code.setdefault((series.casefold(), int(number)), []).append(row)
    by_path = {row.source_path: row for row in available}
    chosen: dict[str, SelectedEpisode] = {}
    for token in (part.strip() for part in text.split(",")):
        if not token:
            continue
        if token.casefold() == "all":
            chosen.update(by_path)
            continue
        if token.endswith(".json"):
            if token not in by_path:
                raise ValueError(f"transcript is not in the catalog (or is quarantined): {token}")
            chosen[token] = by_path[token]
            continue
        match = re.fullmatch(r"(.*?)\s*(\d+)(?:\s*-\s*(\d+))?", token)
        if not match:
            raise ValueError(f"can't read episode selection: {token!r}")
        series, first = match[1].strip().casefold(), int(match[2])
        last = int(match[3]) if match[3] else first
        if last < first:
            raise ValueError(f"episode range runs backwards: {token!r}")
        for number in range(first, last + 1):
            rows = by_code.get((series, number), [])
            if not rows:
                raise ValueError(f"no transcript for episode {match[1].strip()} {number}")
            if len(rows) > 1:
                raise ValueError(f"episode {match[1].strip()} {number} matches several transcripts: "
                                 + ", ".join(row.source_path for row in rows))
            chosen[rows[0].source_path] = rows[0]
    if not chosen:
        raise ValueError("the selection is empty")
    return [chosen[path] for path in sorted(chosen)]


def load_corpus(root: Path, name: str) -> tuple[CorpusConfig, IngestSetup]:
    path = workspace_path(root, name)
    config = CorpusConfig.model_validate_json(path.read_bytes())
    return config, IngestSetup(root, config.model_dump(), path)
