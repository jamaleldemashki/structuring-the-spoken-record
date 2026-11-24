"""Vectors for asking the archive a question in the web app (``search.json``).

The app answers a question like "Is there any relation between Thomas Aquinas and
Aristotle when it comes to the soul?" in two parts. Names are found by matching the words of
the question against the entity names. The rest of the question ("the soul") is compared
by meaning: with the passages, to rank the evidence, and with the names of concepts,
schools and works, so "mind" can still find "intellect". Both need vectors made with the
same model that runs in the browser for the question.

The model is all-MiniLM-L6-v2: small enough to load in a browser (the int8 ONNX version is
23 MB) and a sentence encoder, not a text generator. Here it runs in PyTorch, in the browser
as the int8 ONNX conversion of the same weights. The int8 model gives slightly different
numbers, but the same passages come out on top.

The vectors are stored as int8 (each value times 127, rounded), which keeps the file at a
quarter of float32 and changes cosine similarities by less than 0.01.
"""

import base64
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from spoken_record.models import EntityType


class SearchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    schema_version: Literal[1] = 1
    repo_id: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    # The same weights as ONNX for transformers.js in the browser.
    browser_repo_id: str
    browser_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    browser_dtype: Literal["q8", "fp32"] = "q8"
    cache_dir: str | None = None
    device: Literal["cpu"] = "cpu"
    max_tokens: int = Field(default=256, ge=8, le=512)
    batch_size: int = Field(default=16, ge=1, le=64)
    dimension: int = Field(default=384, gt=0)
    # Types whose names are compared by meaning with the question. People and places are
    # found by their names only: "Aristotle" means nothing more than the person.
    topic_types: list[EntityType] = Field(default_factory=lambda: ["Concept", "School", "Work", "Period", "Group"])


def quantize(vectors: list[list[float]]) -> str:
    """Normalized vectors as int8, row after row, base64 encoded."""
    data = bytearray()
    for vector in vectors:
        data.extend(max(-127, min(127, round(v * 127))) & 0xFF for v in vector)
    return base64.b64encode(bytes(data)).decode("ascii")


def search_payload(config: SearchConfig, passages: list[dict], nodes: list[dict], encode) -> dict:
    """The vectors for the app: one per passage and one per topic name.

    ``passages`` and ``nodes`` are the rows of graph.json. Generic terms are left out of the
    topics because they get no edges in the app. ``encode`` takes texts and returns one
    encoder result (with its ``vector``) per text."""
    texts = [p["text"] for p in passages]
    topics = [n for n in nodes if n["type"] in config.topic_types and not n.get("generic")]
    vectors = [row["vector"] for row in encode(texts + [n["label"] for n in topics])] if texts or topics else []
    if any(len(v) != config.dimension for v in vectors):
        raise ValueError("search vectors have the wrong dimension")
    return {"schema_version": 1,
            "model": {"repo_id": config.browser_repo_id, "revision": config.browser_revision,
                      "dtype": config.browser_dtype, "pooling": "mean", "normalize": True,
                      "max_tokens": config.max_tokens},
            "dimension": config.dimension, "scale": 127,
            "passages": {"ids": [p["id"] for p in passages], "data": quantize(vectors[:len(texts)])},
            "topics": {"ids": [n["id"] for n in topics], "data": quantize(vectors[len(texts):])}}
