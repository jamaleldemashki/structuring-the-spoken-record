"""Config for the web app data export.

Everything podcast-specific the app shows (name, facet labels, type labels, time presets)
comes from `site`, so one build of the app works for any archive the pipeline exports.
"""

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

EntityType = Literal["Person", "Work", "School", "Concept", "Place", "Period", "Language", "Group"]


class TypeDisplay(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    type: EntityType
    label: str = Field(min_length=1)
    plural: str = Field(min_length=1)
    visible_by_default: bool = True


class TimePreset(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    label: str = Field(min_length=1)
    # Years; negative years are BCE.
    start: int
    end: int

    @model_validator(mode="after")
    def ordered(self):
        if self.start > self.end:
            raise ValueError("time preset must start before it ends")
        return self


class SiteConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    archive_id: str = Field(default="archive", pattern=r"^[a-z0-9][a-z0-9-]{0,63}$")
    title: str = Field(default="Podcast archive", min_length=1)
    subtitle: str = ""
    # Episode facets used as the "series" and "era" groups. None hides them.
    series_facet: str | None = None
    series_label: str = "Series"
    era_facet: str | None = None
    era_label: str = "Era"
    # Regex whose first group is the episode number, matched at the start of the title.
    # The matched part is cut from the title that is shown.
    episode_number_pattern: str | None = None
    # Order matters: the first three types get the three main colors.
    entity_types: list[TypeDisplay] = Field(default_factory=lambda: [
        TypeDisplay(type="Person", label="Person", plural="people"),
        TypeDisplay(type="Concept", label="Concept", plural="concepts"),
        TypeDisplay(type="Work", label="Work", plural="works"),
        TypeDisplay(type="School", label="School", plural="schools"),
        TypeDisplay(type="Place", label="Place", plural="places", visible_by_default=False),
    ])
    # Type whose lifespans drive the "lived" filter and the timeline. None turns both off.
    lifespan_type: EntityType | None = "Person"
    lived_presets: list[TimePreset] = Field(default_factory=list)
    search_placeholder: str = "Search people, concepts, works…"

    @model_validator(mode="after")
    def consistent(self):
        types = [t.type for t in self.entity_types]
        if len(set(types)) != len(types):
            raise ValueError("entity_types must not repeat a type")
        if self.episode_number_pattern is not None and re.compile(self.episode_number_pattern).groups < 1:
            raise ValueError("episode_number_pattern needs a group capturing the number")
        return self


class WebExportConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[2] = 2
    site: SiteConfig = Field(default_factory=SiteConfig)
    # Types whose shared passages become "discussed together" edges.
    co_discussed_types: list[EntityType] = Field(
        default_factory=lambda: ["Person", "Work", "School", "Concept"], min_length=1)
    min_shared_passages: int = Field(default=2, ge=1)
    include_passage_text: bool = True
    predicate_labels: dict[str, str] = Field(default_factory=lambda: {"authored": "wrote"})
    discovery_method: Literal["mpnet_cosine", "shared_entity_jaccard"] = "mpnet_cosine"
