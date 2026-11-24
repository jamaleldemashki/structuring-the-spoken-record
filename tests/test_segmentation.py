from dataclasses import dataclass

import pytest
from pydantic import ValidationError

from test_ingestion import parse, transcript
from spoken_record.adapters.plain_text import PlainTextAdapter
from spoken_record.passage_models import SegmentationConfig
from spoken_record.passages import build_passages, check_passages, sentence_boundaries
from spoken_record.windows import WhitespaceBudget, build_windows, require_model_ready

CONFIG = SegmentationConfig(passage_min_words=5, passage_target_words=8, passage_max_words=12,
                            window_max_tokens=6, window_overlap_tokens=2)


def text_bundle(text, language="en"):
    bundle = PlainTextAdapter().ingest(text.encode(), corpus_id="synthetic", source_key="sample.txt",
                                      source_path="sample.txt", run_id="source")
    bundle.episode.language = language
    return bundle


def derive(bundle, config=CONFIG, run="derived"):
    passages = build_passages(bundle, config, run)
    windows = build_windows(bundle, passages, config, run)
    check_passages(bundle, passages)
    return passages, windows


def test_sentences_tile_unicode_source_with_exact_segment_times():
    bundle = parse(transcript(["  Pāṇini studies language. Thomas studies philosophy.\r\n",
                               "Saint Thomas writes about knowledge. 😀 The discussion continues.  "]))
    passages, windows = derive(bundle)
    assert "".join(p.text for p in passages) == bundle.transcript.text
    boundaries = sentence_boundaries(bundle.transcript.text, "en")
    assert all(p.char_start in boundaries and p.char_end in boundaries for p in passages)
    assert all(p.timing_precision == "source_segment" for p in passages)
    assert all(w.budget_verified_for_model is False for w in windows)
    with pytest.raises(ValueError, match="provisional"):
        require_model_ready(windows)


def test_nearest_sentence_boundary_tie_prefers_earlier():
    bundle = text_bundle("One two three. Four five six seven. Eight nine ten eleven. Twelve thirteen fourteen fifteen.")
    config = SegmentationConfig(passage_min_words=3, passage_target_words=5, passage_max_words=8,
                                window_max_tokens=6, window_overlap_tokens=1)
    passages, _ = derive(bundle, config)
    # 3 and 7 words are equally close to 5. Keep the earlier boundary.
    assert passages[0].word_count == 3


def test_short_sentence_tail_is_merged_when_combination_fits():
    bundle = text_bundle("One two three four five six. Seven eight nine ten eleven twelve. Tail ends.")
    config = SegmentationConfig(passage_min_words=5, passage_target_words=6, passage_max_words=9,
                                window_max_tokens=6, window_overlap_tokens=1)
    passages, _ = derive(bundle, config)
    assert [p.word_count for p in passages] == [6, 8]
    assert not passages[-1].flags


def test_long_sentence_remains_one_passage_but_has_complete_windows():
    bundle = text_bundle(" ".join(f"word{i}" for i in range(39)) + ".  ")
    passages, windows = derive(bundle)
    assert len(passages) == 1 and "long_sentence" in passages[0].flags
    assert windows[0].char_start == 0
    assert windows[-1].char_end == len(bundle.transcript.text)
    assert windows[-1].token_end == 39
    assert all(w.token_count <= CONFIG.window_max_tokens for w in windows)
    assert all(w.overlap_previous_tokens == 2 for w in windows[1:])
    assert set(i for w in windows for i in range(w.token_start, w.token_end)) == set(range(39))


@pytest.mark.parametrize("text,language,methods", [
    ("alpha beta gamma\n\nsecond paragraph text\r\n\r\nend", "und", {"paragraph_fallback"}),
    (" ".join(["unpunctuated"] * 40), "en", {"token_fallback"}),
    ("这是一个例子。没有英文句子。", "zh", {"token_fallback"}),
    (" \r\n\t ", "en", {"whitespace_only"}),
])
def test_fallbacks_preserve_every_character_without_inventing_time(text, language, methods):
    bundle = text_bundle(text, language)
    passages, windows = derive(bundle)
    assert "".join(p.text for p in passages) == text
    assert {p.method for p in passages} == methods
    assert all(p.start_s is None and p.end_s is None and not p.source_segment_ids for p in passages)
    if not text.strip():
        assert windows == []


def test_passage_identity_independent_of_windows_but_changes_with_segmentation():
    bundle = text_bundle("One two three. Four five six. Seven eight nine. Ten eleven twelve.")
    first, _ = derive(bundle)
    changed_windows = CONFIG.model_copy(update={"window_max_tokens": 9})
    second, _ = derive(bundle, changed_windows, run="other")
    assert [p.passage_id for p in first] == [p.passage_id for p in second]
    changed_passages = CONFIG.model_copy(update={"passage_target_words": 10})
    third, _ = derive(bundle, changed_passages)
    assert [p.passage_id for p in first] != [p.passage_id for p in third]


def test_windows_bridge_passages_and_cover_boundary_names():
    bundle = text_bundle("First second third. Thomas Aquinas wrote. Fourth fifth sixth. Other words follow.")
    config = SegmentationConfig(passage_min_words=3, passage_target_words=3, passage_max_words=3,
                                window_max_tokens=5, window_overlap_tokens=2)
    passages, windows = derive(bundle, config)
    assert any(len(w.passage_ids) > 1 for w in windows)
    start = bundle.transcript.text.index("Thomas Aquinas")
    end = start + len("Thomas Aquinas")
    assert any(w.char_start <= start and w.char_end >= end for w in windows)


def test_budget_can_shrink_windows_but_never_truncate_or_stall():
    @dataclass(frozen=True)
    class TightBudget(WhitespaceBudget):
        name: str = "test-double-token-cost"

        def fits(self, text, max_tokens):
            return 2 * len(self.spans(text)) <= max_tokens

    bundle = text_bundle(" ".join(["token"] * 20))
    passages = build_passages(bundle, CONFIG, "r")
    windows = build_windows(bundle, passages, CONFIG, "r", TightBudget())
    assert all(w.token_count <= 3 for w in windows)
    assert windows[-1].char_end == len(bundle.transcript.text)
    too_much_overlap = CONFIG.model_copy(update={"window_overlap_tokens": 3})
    with pytest.raises(ValueError, match="overlap"):
        build_windows(bundle, passages, too_much_overlap, "r", TightBudget())


@pytest.mark.parametrize("values", [
    {"passage_min_words": 10, "passage_target_words": 5},
    {"window_max_tokens": 40, "window_overlap_tokens": 40},
    {"window_overlap_tokens": -1},
])
def test_invalid_configuration(values):
    with pytest.raises(ValidationError):
        SegmentationConfig(**values)


def test_missing_or_changed_passages_fail_the_check():
    bundle = text_bundle(" ".join(["word"] * 40))
    passages, _ = derive(bundle)
    with pytest.raises(ValueError, match="cover"):
        check_passages(bundle, passages[:-1])
    with pytest.raises(ValueError, match="reproduce"):
        check_passages(bundle, [passages[0].model_copy(update={"text": "changed"}), *passages[1:]])


def test_passage_timing_gives_each_segment_offset_and_times():
    from spoken_record.graph import timing
    bundle = parse(transcript(["  Pāṇini studies language. Thomas studies philosophy.\r\n",
                               "Saint Thomas writes about knowledge. 😀 The discussion continues.  "]))
    passages, _ = derive(bundle)
    by_id = {s.segment_id: s for s in bundle.segments}
    for passage in passages:
        rows = timing(passage, by_id)
        assert [row[0] for row in rows] == sorted(row[0] for row in rows)
        assert rows[0][0] <= 0 and rows[-1][1] >= len(passage.text)
        assert all(row[0] < row[1] and row[2] <= row[3] for row in rows)
        assert rows[0][2] == passage.start_s and rows[-1][3] == passage.end_s
