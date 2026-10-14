"""Packs sentences into passages. Each passage is an exact slice of the source, with
simple fallbacks when no sentences are found.
"""

import json
import re
from functools import lru_cache

import spacy

from spoken_record.adapters.base import sha256, stable_id
from spoken_record.models import SourceBundle, SourceSegment
from spoken_record.passage_models import Passage, SegmentationConfig

PASSAGE_VERSION = "1"


@lru_cache(maxsize=1)
def english_splitter():
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    return nlp


def passage_fingerprint(config: SegmentationConfig, language: str) -> str:
    # Window settings are left out on purpose, changing them must not change passage IDs.
    fields = {k: v for k, v in config.model_dump().items() if k.startswith("passage_")}
    return sha256(json.dumps({"config": fields, "language": language, "algorithm": PASSAGE_VERSION,
                              "spacy": spacy.__version__}, sort_keys=True).encode())


def sentence_boundaries(text: str, language: str) -> list[int]:
    """Sentence boundaries, or [] if there are no usable English sentence boundaries."""
    if language != "en" or not text.strip():
        return []
    doc = english_splitter()(text)
    sentences = [s for s in doc.sents if s.text.strip()]
    # One long block without punctuation is not a real sentence, use the fallback.
    if len(sentences) == 1 and not re.search(r"[.!?][\s\"'’”)]*$", text):
        return []
    starts = [s.start_char for s in sentences]
    return [0, *starts[1:], len(text)] if starts else []


def fallback_ranges(text: str, config: SegmentationConfig) -> list[tuple[int, int, str]]:
    if not text.strip():
        return [(0, len(text), "whitespace_only")]
    boundaries = [0, *[m.end() for m in re.finditer(r"(?:\r?\n)[ \t]*(?:\r?\n)", text)], len(text)]
    boundaries = sorted(set(boundaries))
    result = []
    for start, end in zip(boundaries, boundaries[1:]):
        tokens = list(re.finditer(r"\S+", text[start:end]))
        if len(tokens) <= config.passage_max_words:
            method = "paragraph_fallback" if len(boundaries) > 2 else "token_fallback"
            result.append((start, end, method))
            continue
        cuts = [start]
        cuts += [start + tokens[i].start() for i in range(config.passage_target_words, len(tokens), config.passage_target_words)]
        cuts.append(end)
        result.extend((a, b, "token_fallback") for a, b in zip(cuts, cuts[1:]))
    return result


def passage_ranges(text: str, config: SegmentationConfig, language: str) -> list[tuple[int, int, str]]:
    boundaries = sentence_boundaries(text, language)
    if not boundaries:
        return fallback_ranges(text, config)
    result = []
    index = 0
    while index < len(boundaries) - 1:
        options = []
        for next_index in range(index + 1, len(boundaries)):
            count = len(text[boundaries[index]:boundaries[next_index]].split())
            if count > config.passage_max_words:
                break
            options.append((next_index, count))
        preferred = [(i, n) for i, n in options if n >= config.passage_min_words]
        if preferred:
            end_index = min(preferred, key=lambda pair: (abs(pair[1] - config.passage_target_words), pair[0]))[0]
        elif options:
            end_index = options[-1][0]
        else:
            # A long sentence is never split just to keep the passage short.
            end_index = index + 1
        result.append((boundaries[index], boundaries[end_index], "sentence_packing"))
        index = end_index
    # Greedy packing can leave a very short last passage. Merge it into the one before
    # if that stays under the maximum. Sentence boundaries are kept.
    if len(result) > 1:
        tail_start, tail_end, tail_method = result[-1]
        previous_start, previous_end, previous_method = result[-2]
        if (len(text[tail_start:tail_end].split()) < config.passage_min_words and
                len(text[previous_start:tail_end].split()) <= config.passage_max_words and
                tail_method == previous_method):
            result[-2:] = [(previous_start, tail_end, previous_method)]
    return result


def evidence_segments(segments: list[SourceSegment], start: int, end: int) -> list[SourceSegment]:
    return [s for s in segments if s.char_start < end and s.char_end > start]


def time_bounds(segments: list[SourceSegment]) -> tuple[float | None, float | None]:
    if not segments or any(s.start_s is None for s in segments):
        return None, None
    return min(s.start_s for s in segments), max(s.end_s for s in segments)


def build_passages(bundle: SourceBundle, config: SegmentationConfig, run_id: str) -> list[Passage]:
    fingerprint = passage_fingerprint(config, bundle.episode.language)
    transcript = bundle.transcript
    result = []
    for index, (start, end, method) in enumerate(passage_ranges(transcript.text, config, bundle.episode.language)):
        text = transcript.text[start:end]
        count = len(text.split())
        segments = evidence_segments(bundle.segments, start, end)
        start_s, end_s = time_bounds(segments)
        flags = []
        if count > config.passage_max_words:
            flags.append("long_sentence")
        if count < config.passage_min_words:
            flags.append("short_passage")
        if method != "sentence_packing":
            flags.append("fallback_segmentation")
        result.append(Passage(
            created_by_run_id=run_id, passage_id=stable_id("p", transcript.transcript_version_id, fingerprint, str(start), str(end)),
            episode_id=bundle.episode.episode_id, transcript_version_id=transcript.transcript_version_id,
            index=index, char_start=start, char_end=end, text=text, word_count=count,
            start_s=start_s, end_s=end_s, timing_precision="source_segment" if start_s is not None else "unavailable",
            source_segment_ids=[s.segment_id for s in segments], method=method, flags=flags,
            segmentation_fingerprint=fingerprint,
        ))
    return result
