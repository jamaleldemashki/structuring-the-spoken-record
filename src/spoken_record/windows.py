"""Overlapping text windows. The whitespace budget here is not the GLiNER token budget."""

import json
import re
from dataclasses import dataclass
from typing import Protocol

from spoken_record.adapters.base import sha256, stable_id
from spoken_record.models import SourceBundle
from spoken_record.passage_models import ModelWindow, Passage, SegmentationConfig


class WindowBudget(Protocol):
    name: str
    verified_for_model: bool

    def spans(self, text: str) -> list[tuple[int, int]]: ...

    def fits(self, text: str, max_tokens: int) -> bool: ...


@dataclass(frozen=True)
class WhitespaceBudget:
    name: str = "whitespace-v1"
    verified_for_model: bool = False

    def spans(self, text: str) -> list[tuple[int, int]]:
        return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]

    def fits(self, text: str, max_tokens: int) -> bool:
        return len(self.spans(text)) <= max_tokens


def build_windows(bundle: SourceBundle, passages: list[Passage], config: SegmentationConfig,
                  run_id: str, budget: WindowBudget | None = None) -> list[ModelWindow]:
    budget = budget or WhitespaceBudget()
    transcript = bundle.transcript
    text = transcript.text
    spans = budget.spans(text)
    if any(a < 0 or b <= a or b > len(text) or (i and a < spans[i-1][1]) for i, (a, b) in enumerate(spans)):
        raise ValueError("budget tokenizer returned invalid source spans")
    if not spans:
        if text.strip():
            raise ValueError("budget tokenizer omitted non-whitespace input")
        return []
    fingerprint = sha256(json.dumps({"algorithm": "windows-1", "tokenizer": budget.name,
        "verified": budget.verified_for_model, "max_tokens": config.window_max_tokens,
        "overlap": config.window_overlap_tokens}, sort_keys=True).encode())
    result = []
    start_token, previous_end = 0, 0
    while start_token < len(spans):
        end_token = min(start_token + config.window_max_tokens, len(spans))
        char_start = 0 if start_token == 0 else spans[start_token][0]
        # Trailing whitespace belongs to the token before it, also at the end of the text.
        char_end = spans[end_token][0] if end_token < len(spans) else len(text)
        while not budget.fits(text[char_start:char_end], config.window_max_tokens):
            end_token -= 1
            if end_token <= max(start_token, previous_end):
                raise ValueError("model budget cannot fit required overlap plus a new token; reduce overlap or handle oversized token")
            char_end = spans[end_token][0]
        overlap = max(0, previous_end - start_token)
        result.append(ModelWindow(
            created_by_run_id=run_id,
            window_id=stable_id("w", transcript.transcript_version_id, fingerprint, str(char_start), str(char_end)),
            episode_id=bundle.episode.episode_id, transcript_version_id=transcript.transcript_version_id,
            index=len(result), char_start=char_start, char_end=char_end,
            passage_ids=[p.passage_id for p in passages if p.char_start < char_end and p.char_end > char_start],
            token_start=start_token, token_end=end_token, token_count=end_token-start_token,
            token_budget=config.window_max_tokens, overlap_previous_tokens=overlap,
            tokenizer=budget.name, budget_verified_for_model=budget.verified_for_model,
            window_fingerprint=fingerprint,
        ))
        if end_token == len(spans):
            break
        next_start = end_token - config.window_overlap_tokens
        if next_start <= start_token:
            raise ValueError("overlap prevents model-window progress")
        start_token, previous_end = next_start, end_token
    return result


def require_model_ready(windows: list[ModelWindow]) -> None:
    """Model code must not use these provisional windows, it has to build its own."""
    if any(not window.budget_verified_for_model for window in windows):
        raise ValueError("provisional windows: validate actual model/tokenizer/label budgets before inference")
