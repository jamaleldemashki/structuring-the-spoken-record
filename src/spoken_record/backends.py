"""Interfaces between the stages and the models.

A backend only returns model output (spans, scores, vectors). What that output means is
decided in the stage code. Using a different model needs a new adapter, a model config that
describes it correctly, and tests for offsets and input limits. Changing the checkpoint name
alone is not enough. The GLiNER backend also exposes its tokenizer so the input budget can
be checked, and keeps its existing loader hook.
"""

from typing import Protocol
from spoken_record.verification_models import NLIScores

from spoken_record.resolution_models import CorpusEntity


class RelationBackend(Protocol):
    """Labels, scores and token spans (end exclusive) for the given endpoints."""

    def predict_relations(self, tokens: list[str], labels: list[str], *, threshold: float,
                          ner: list[list], top_k: int) -> list[dict]:
        """Raw relation predictions. Grounding and acceptance happen in the stage."""
        ...


class CoreferencePrediction(Protocol):
    """Clusters as character offsets (end exclusive) into the input text."""

    def get_clusters(self, *, as_strings: bool = False) -> list[list[tuple[int, int]]]:
        """Cluster spans. The transcript text is left as it is."""
        ...


class CoreferenceBackend(Protocol):
    """Predicts mention clusters. Which corpus entity they belong to is decided later."""

    def predict(self, *, texts: list[str], max_tokens_in_batch: int) -> list[CoreferencePrediction]:
        """One prediction per input transcript, in input order."""
        ...


class CandidateRetriever(Protocol):
    """Fetches Wikidata candidates for each local CorpusEntity ID."""

    def retrieve(self, entities: list[CorpusEntity]) -> dict[str, list[dict]]:
        """QID, label, description, aliases, type evidence and search rank."""
        ...


class TextEncoder(Protocol):
    """Encodes context and candidate text into vectors that can be compared."""

    def encode(self, texts: list[str]) -> list[list[float]]:
        """One finite vector per text, all of the same length, in input order."""
        ...


class EvidenceVerifier(Protocol):
    """Scores premise/hypothesis pairs, with an input budget and a fixed label order."""

    def token_count(self, premise: str, hypothesis: str) -> int:
        """Token count of the whole pair, special tokens included."""
        ...

    def predict(self, premise: str, hypothesis: str) -> NLIScores:
        """Entailment, contradiction and neutral scores. The evidence text is not changed."""
        ...
