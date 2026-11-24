import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from spoken_record.models import TYPES
from spoken_record.extraction import (GLiNERWhitespaceBudget, TOKEN_PATTERN, resolve_candidates,
                                      verify_encoder_batch)
from spoken_record.passage_models import Passage


class WhitespaceTokenSplitter:
    def __call__(self, text):
        for match in TOKEN_PATTERN.finditer(text):
            yield match.group(), match.start(), match.end()


class FakeWordsSplitter:
    def __init__(self):
        self.splitter = WhitespaceTokenSplitter()

    def __call__(self, text):
        yield from self.splitter(text)


class FakeModel:
    def __init__(self, fail=False):
        self.config = SimpleNamespace(max_len=384)
        self.data_processor = SimpleNamespace(words_splitter=FakeWordsSplitter())
        self.model = SimpleNamespace(token_rep_layer=SimpleNamespace(bert_layer=SimpleNamespace(
            model=SimpleNamespace(config=SimpleNamespace(max_position_embeddings=512)))))
        self.fail = fail

    def prepare_model_inputs(self, texts, labels):
        tokens = [list(TOKEN_PATTERN.finditer(text)) for text in texts]
        lengths = [len(row) + 2 * len(labels) + 2 for row in tokens]
        attention = FakeMatrix([[1] * length for length in lengths])
        words = FakeMatrix([list(range(1, len(row) + 1)) for row in tokens])
        return {"attention_mask": attention, "words_mask": words}, {"seq_length": [len(row) for row in tokens]}

    def batch_predict_entities(self, texts, labels, **kwargs):
        if self.fail:
            raise RuntimeError("synthetic failure")
        rows = []
        for text in texts:
            found = []
            for needle, label, confidence in (("Pāṇini", "person", .9), ("free will", "philosophical concept", .8),
                                               ("words", "philosophical concept", .3)):
                start = text.find(needle)
                while start >= 0:
                    found.append({"start": start, "end": start + len(needle), "text": needle,
                                  "label": label, "score": confidence})
                    start = text.find(needle, start + len(needle))
            rows.append(found)
        return rows


class FakeNumber:
    def __init__(self, value): self.value = value
    def item(self): return self.value


class FakeVector(list):
    def max(self): return FakeNumber(max(self, default=0))
    def tolist(self): return list(self)


class FakeMatrix(list):
    def __init__(self, rows): super().__init__(FakeVector(row) for row in rows)
    def sum(self, dim):
        assert dim == 1
        return FakeVector([sum(row) for row in self])


def test_flat_resolution_tiebreaks_and_exact_duplicate_windows():
    passage = Passage(created_by_run_id="p", passage_id="p1", episode_id="e", transcript_version_id="t",
        index=0, char_start=0, char_end=20, text="abcdefghijklmnopqrst", word_count=1,
        timing_precision="unavailable", source_segment_ids=[], method="token_fallback",
        segmentation_fingerprint="x")
    base = {"created_by_run_id": "r", "episode_id": "e", "transcript_version_id": "t",
            "model_label": "person"}
    raw = [
        {**base, "candidate_id": "c1", "window_id": "w1", "char_start": 2, "char_end": 6,
         "text": "cdef", "type": "Person", "score": .8},
        {**base, "candidate_id": "c2", "window_id": "w2", "char_start": 2, "char_end": 6,
         "text": "cdef", "type": "Person", "score": .7},
        {**base, "candidate_id": "c3", "window_id": "w1", "char_start": 3, "char_end": 7,
         "text": "defg", "type": "Person", "score": .75},
        {**base, "candidate_id": "c4", "window_id": "w1", "char_start": 10, "char_end": 12,
         "text": "kl", "type": "Person", "score": .4},
    ]
    candidates, mentions = resolve_candidates(raw, .5, "fingerprint", [passage])
    decisions = {c.candidate_id: c.decision for c in candidates}
    assert decisions == {"c1": "accepted", "c2": "duplicate", "c3": "overlap", "c4": "below_threshold"}
    assert len(mentions) == 1 and mentions[0].source_window_ids == ["w1", "w2"]


def test_configuration_inventory_is_eight_types():
    root = Path(__file__).resolve().parents[1]
    labels = json.loads((root / "config/entity_labels.development-calibrated.json").read_bytes())
    model = json.loads((root / "config/model.development-calibrated.json").read_bytes())
    assert set(labels["labels"]) == set(TYPES)
    assert model["window_max_tokens"] <= model["model_max_words"]
    assert model["candidate_floor"] <= labels["provisional_threshold"]


def test_encoder_prompt_budget_is_checked_before_inference():
    model = FakeModel()
    budget = GLiNERWhitespaceBudget("a" * 40)
    config = SimpleNamespace(encoder_max_tokens=25)
    with pytest.raises(ValueError, match="encoder token limit"):
        verify_encoder_batch(model, [" ".join(["word"] * 10)], list(TYPES), config, budget)
