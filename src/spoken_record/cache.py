"""A small key-value cache in SQLite for Wikidata answers and model outputs.

The key of an entry is a hash of everything that can change the result: what was asked
(the model and its revision, or the Wikidata request) and the exact input. So an entry can
never be outdated. A new model revision or a changed input simply gives a new key. Values
are stored as JSON, and floats come back exactly as they were stored.

The real data of a build stays in readable JSON files. This cache only saves us from
asking a model or Wikidata the same question twice.
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from spoken_record.adapters.base import sha256


def plain(value):
    """The same value with NumPy or torch numbers turned into plain Python numbers, the way
    it comes back from the cache."""
    return json.loads(json.dumps(value, ensure_ascii=False,
                                 default=lambda item: item.tolist() if hasattr(item, "tolist") else str(item)))


class Cache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.connection = sqlite3.connect(path)
        self.connection.execute("CREATE TABLE IF NOT EXISTS entries "
                                "(key TEXT PRIMARY KEY, value TEXT NOT NULL, created_utc TEXT NOT NULL)")
        self.hits = self.misses = 0

    @staticmethod
    def key(*parts) -> str:
        """Hash of the parts. They must be JSON-serializable, dict keys are sorted."""
        return sha256(json.dumps(parts, ensure_ascii=False, sort_keys=True,
                                 separators=(",", ":")).encode("utf-8"))

    def get(self, key: str):
        row = self.connection.execute("SELECT value FROM entries WHERE key = ?", (key,)).fetchone()
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        return json.loads(row[0])

    def put(self, key: str, value) -> None:
        self.connection.execute("INSERT OR REPLACE INTO entries VALUES (?, ?, ?)",
                                (key, json.dumps(plain(value), ensure_ascii=False),
                                 datetime.now(timezone.utc).isoformat()))
        self.connection.commit()

    def clear(self) -> None:
        self.connection.execute("DELETE FROM entries")
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()


class CachedEncoder:
    """Text encoder that only loads the model for texts it has not seen.

    Texts that are missing are encoded together, in their original order, so a first build
    sends the model the same batches as before.
    """

    def __init__(self, cache: Cache, name: str, settings: dict, load):
        self.cache, self.name, self.settings, self.load = cache, name, settings, load
        self.model = None

    def encode(self, texts: list[str]):
        keys = [self.cache.key(self.name, self.settings, text) for text in texts]
        found = [self.cache.get(key) for key in keys]
        missing = [i for i, value in enumerate(found) if value is None]
        if missing:
            self.model = self.model or self.load()
            encoded = self.model.encode([texts[i] for i in missing])
            if len(encoded) != len(missing):
                raise ValueError("encoder must return one result per text")
            for i, value in zip(missing, encoded):
                self.cache.put(keys[i], value)
                found[i] = plain(value)
        return found


class CachedRelationModel:
    """GLiREL predictions and token counts by their exact input. GLiREL is loaded on the
    first input that is not cached."""

    def __init__(self, cache: Cache, config, load, make_budget):
        self.cache, self.config, self.load, self.make_budget = cache, config, load, make_budget
        self.settings = {"repo_id": config.repo_id, "revision": config.revision,
                         "encoder_repo_id": config.encoder_repo_id,
                         "encoder_revision": config.encoder_revision,
                         "labels": config.prompts()}
        self.model = self.real_budget = None

    def _model(self):
        if self.model is None:
            self.model = self.load(self.config)
        return self.model

    def predict_relations(self, tokens, labels, *, threshold, ner, top_k):
        key = self.cache.key("glirel", self.settings, tokens, labels, threshold, ner, top_k)
        cached = self.cache.get(key)
        if cached is None:
            cached = self._model().predict_relations(tokens, labels, threshold=threshold, ner=ner, top_k=top_k)
            self.cache.put(key, cached)
            cached = plain(cached)
        return cached

    def budget(self, tokens, ner) -> int:
        key = self.cache.key("glirel-tokens", self.settings, tokens, ner)
        cached = self.cache.get(key)
        if cached is None:
            if self.real_budget is None:
                self.real_budget = self.make_budget(self._model(), self.config)
            cached = int(self.real_budget(tokens, ner))
            self.cache.put(key, cached)
        return cached


class CachedVerifier:
    """NLI scores and token counts by premise and hypothesis. The model is loaded on the
    first pair that is not cached."""

    def __init__(self, cache: Cache, config, load):
        self.cache, self.config, self.load = cache, config, load
        self.settings = {"repo_id": config.repo_id, "revision": config.revision,
                         "label_order": config.label_order}
        self.model = None

    def _model(self):
        if self.model is None:
            self.model = self.load()
        return self.model

    def token_count(self, premise: str, hypothesis: str) -> int:
        key = self.cache.key("nli-tokens", self.settings, premise, hypothesis)
        cached = self.cache.get(key)
        if cached is None:
            cached = int(self._model().token_count(premise, hypothesis))
            self.cache.put(key, cached)
        return cached

    def predict(self, premise: str, hypothesis: str):
        from spoken_record.verification_models import NLIScores

        key = self.cache.key("nli", self.settings, premise, hypothesis)
        cached = self.cache.get(key)
        if cached is None:
            cached = NLIScores.model_validate(self._model().predict(premise, hypothesis)).model_dump(mode="json")
            self.cache.put(key, cached)
        return NLIScores.model_validate(cached)
