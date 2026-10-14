"""Config for the Obsidian vault export.

Schema 2 moved the podcast-specific naming (series codes, episode numbers, corpus title)
out of the code and into this config.
"""

import re
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class VaultConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[2] = 2
    vault_name: str = Field(min_length=1)
    corpus_title: str = Field(default="Podcast corpus", min_length=1)
    # Short episode references like "HoP 243" are a series code plus the episode number.
    # series_facet is the facet that holds the series, series_codes maps its values (ignoring
    # case) to display codes. Without codes the facet value is used as it is.
    series_facet: str | None = None
    series_codes: dict[str, str] = Field(default_factory=dict)
    # Regex for the start of the title, group 1 is the episode number.
    episode_number_pattern: str | None = None
    # Fallback regex searched in the file name: group 1 is the series, group 2 the number.
    source_key_pattern: str | None = None
    index_folder: Literal["00_Index"] = "00_Index"
    corpus_folder: Literal["05_Corpus"] = "05_Corpus"
    episode_folder: Literal["10_Episodes"] = "10_Episodes"
    entity_folder: Literal["20_Entities"] = "20_Entities"
    relation_folder: Literal["30_Relations"] = "30_Relations"
    review_folder: Literal["90_Review"] = "90_Review"
    include_all_entities: Literal[True] = True
    include_singletons: Literal[True] = True
    passage_navigation: Literal["obsidian_block_reference"] = "obsidian_block_reference"
    graph_exclude_paths: list[str] = Field(default_factory=lambda: ["00_Index", "90_Review"])
    # Off by default so older configs render exactly as before.
    concept_navigation: bool = False
    navigation_max_neighbors: int = Field(default=8, ge=1, le=50)
    navigation_max_evidence: int = Field(default=3, ge=1, le=20)

    @model_validator(mode="after")
    def distinct_folders(self) -> Self:
        folders = [self.index_folder, self.corpus_folder, self.episode_folder,
                   self.entity_folder, self.relation_folder, self.review_folder]
        if len(set(folders)) != len(folders):
            raise ValueError("vault folders must be distinct")
        for pattern, groups in ((self.episode_number_pattern, 1), (self.source_key_pattern, 2)):
            if pattern is not None and re.compile(pattern).groups < groups:
                raise ValueError(f"naming pattern {pattern!r} needs {groups} capture group(s)")
        return self
