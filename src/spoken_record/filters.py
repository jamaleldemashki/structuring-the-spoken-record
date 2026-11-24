"""Two filters that need several episodes: boilerplate sentences and generic terms.

Both only decide what counts as evidence in the graph and what the app shows. They never
change a transcript, an episode folder or an entity.

**Boilerplate**: intros, sponsor lines and sign-offs that come back in many episodes ("Hi,
I'm Peter Adamson, and you're listening to the History of Philosophy podcast, brought to you
with the support of ..."). A sentence counts as boilerplate when most of its five-word
sequences also occur in several other chosen episodes. Small changes (a new sponsor, the
episode title) don't stop that. Mentions inside such a sentence are not evidence, and the
related-passage ranking leaves the sentence out.

**Generic terms**: words like "philosopher" or "woman" that the model tagged as a name. For
types that are names (Person, Place, Work, Language) an entity whose every spelling is
lowercase is generic. How often a term comes up is no sign: in a philosophy podcast "soul"
and "intellect" are in many episodes because they are the topic. Generic entities stay in
the graph and in search, but get no edges in the app.
"""

import math
import re
from collections import defaultdict
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from spoken_record.models import EntityType
from spoken_record.passages import sentence_boundaries

WORD = re.compile(r"[^\W_]+", re.UNICODE)


class BoilerplateConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    shingle_words: int = Field(default=5, ge=2)
    # A five-word sequence is repeated when it occurs in this many episodes, or in this share
    # of the chosen episodes if that is more.
    min_episodes: int = Field(default=2, ge=2)
    min_episode_share: float = Field(default=0.1, ge=0, le=1)
    # Share of a sentence's sequences that must be repeated.
    covered_share: float = Field(default=0.8, gt=0, le=1)
    min_sentence_words: int = Field(default=6, ge=1)


class GenericConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    # Groups and schools are left out on purpose: in English they are often written in
    # lowercase ("the scholastics", "black nationalists").
    proper_name_types: list[EntityType] = Field(default_factory=lambda: ["Person", "Place", "Work", "Language"])


class FiltersConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[1] = 1
    boilerplate: BoilerplateConfig = Field(default_factory=BoilerplateConfig)
    generic: GenericConfig = Field(default_factory=GenericConfig)


def sentences(transcript, language: str) -> list[tuple[int, int]]:
    bounds = sentence_boundaries(transcript.text, language)
    return list(zip(bounds, bounds[1:])) if bounds else [(0, len(transcript.text))]


def shingles(words: list[str], size: int) -> set[tuple[str, ...]]:
    return {tuple(words[i:i + size]) for i in range(len(words) - size + 1)}


def boilerplate_spans(bundles, settings: BoilerplateConfig) -> dict[str, list[tuple[int, int]]]:
    """Character ranges of boilerplate sentences, per transcript version."""
    parts = {}
    seen = defaultdict(set)
    for bundle in bundles:
        transcript = bundle.transcript
        rows = []
        for start, end in sentences(transcript, bundle.episode.language):
            words = [w.casefold() for w in WORD.findall(transcript.text[start:end])]
            found = shingles(words, settings.shingle_words) if len(words) >= settings.min_sentence_words else set()
            rows.append((start, end, found))
            for shingle in found:
                seen[shingle].add(bundle.episode.episode_id)
        parts[transcript.transcript_version_id] = rows
    needed = max(settings.min_episodes, math.ceil(settings.min_episode_share * len(bundles)))
    result = {}
    for transcript_id, rows in parts.items():
        result[transcript_id] = [(start, end) for start, end, found in rows if found and
                                 sum(len(seen[s]) >= needed for s in found) >= settings.covered_share * len(found)]
    return result


def inside(spans: list[tuple[int, int]], start: int, end: int) -> bool:
    return any(a <= start and end <= b for a, b in spans)


def generic_entities(entities, settings: GenericConfig) -> set[str]:
    return {entity.corpus_entity_id for entity in entities
            if entity.type in settings.proper_name_types and not any(
                character.isupper() for surface in entity.observed_surfaces for character in surface)}
