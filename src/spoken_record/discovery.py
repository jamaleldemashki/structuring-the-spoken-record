"""Finds related passages across episodes.

It never merges entities, and a high similarity is not treated as a statement about the
content. The passage vectors are stored per episode, so the ranking can be redone without
running the model again.
"""

import math
from collections import defaultdict
from pathlib import Path

from spoken_record.adapters.base import sha256, stable_id
from spoken_record.discovery_models import DiscoveryConfig
from spoken_record.ingestion import workspace_path


def token_ranges(length: int, capacity: int) -> list[tuple[int, int]]:
    """Split the content tokens into chunks so every token is used once. Special tokens
    are counted separately.
    """
    if length < 1 or capacity < 1:
        raise ValueError("discovery requires nonempty tokens and positive chunk capacity")
    return [(start, min(length, start + capacity)) for start in range(0, length, capacity)]


def normalized(vector: list[float]) -> list[float]:
    """Reject empty, zero or malformed vectors before computing cosine similarity."""
    if not vector or any(not math.isfinite(v) for v in vector):
        raise ValueError("discovery vector must be finite and nonempty")
    norm = math.sqrt(sum(v * v for v in vector))
    if norm < 1e-12:
        raise ValueError("discovery vector has zero norm")
    return [v / norm for v in vector]


class PassageEncoder:
    """MPNet encoder at a pinned revision, with attention-masked mean pooling over all tokens."""

    def __init__(self, root: Path, config: DiscoveryConfig):
        """Load from the local snapshot folder so the tokenizer doesn't try to go online."""
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModel, AutoTokenizer
        self.config, self.torch = config, torch
        cache = str(workspace_path(root, config.cache_dir)) if config.cache_dir else None
        local = snapshot_download(config.repo_id, revision=config.revision,
                                  cache_dir=cache, local_files_only=True)
        self.tokenizer = AutoTokenizer.from_pretrained(local, local_files_only=True, trust_remote_code=False)
        self.model = AutoModel.from_pretrained(local, local_files_only=True, trust_remote_code=False,
                                               use_safetensors=True).to(config.device).eval()
        self.special_tokens = self.tokenizer.num_special_tokens_to_add(pair=False)
        if config.max_tokens > self.tokenizer.model_max_length:
            raise ValueError("discovery token budget exceeds tokenizer limit")

    def encode(self, texts: list[str]) -> list[dict]:
        """One normalized vector per passage, plus the token ranges that went into it.

        Each chunk vector is the normalized mean of its token states. The passage vector is the
        token-weighted mean of the chunks, normalized again. This is a simple retrieval baseline
        and not the same as encoding the whole passage at once.
        """
        output = []
        for text in texts:
            ids = self.tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
            spans = token_ranges(len(ids), self.config.max_tokens - self.special_tokens)
            chunks, vectors = [], []
            for start in range(0, len(spans), self.config.batch_size):
                batch_spans = spans[start:start + self.config.batch_size]
                prepared = [self.tokenizer.prepare_for_model(ids[a:b], add_special_tokens=True,
                    truncation=False, return_attention_mask=True) for a, b in batch_spans]
                batch = self.tokenizer.pad(prepared, padding=True, return_tensors="pt")
                batch = {k: v.to(self.config.device) for k, v in batch.items()}
                with self.torch.inference_mode():
                    states = self.model(**batch).last_hidden_state
                    mask = batch["attention_mask"].unsqueeze(-1).expand(states.size()).float()
                    pooled = (states * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                    pooled = self.torch.nn.functional.normalize(pooled, p=2, dim=1)
                vectors.extend(pooled.cpu().tolist())
                chunks.extend({"token_start": a, "token_end": b,
                               "encoder_tokens": b - a + self.special_tokens} for a, b in batch_spans)
            vector = normalized([sum(v[i] * (b-a) for v, (a, b) in zip(vectors, spans, strict=True))
                                 / len(ids) for i in range(self.config.dimension)])
            output.append({"vector": vector, "token_count": len(ids),
                           "special_tokens": self.special_tokens, "chunks": chunks})
        return output


def prepare_inputs(records, exclusions: list[dict]) -> list[dict]:
    """Leave excluded spans (boilerplate, reviewed credits) out of the retrieval text.

    The transcript and passages are not changed. The remaining ranges are joined with
    newlines only for the model input, and their offsets and both hashes are recorded.
    """
    passages, mentions, assignments = records[2], records[3], records[6]
    if any(e.get("review_status") not in ("reviewed", "automatic") for e in exclusions):
        raise ValueError("discovery accepts only reviewed or automatic source exclusions")
    by_mention = {a.mention_id: a.corpus_entity_id for a in assignments}
    by_passage = defaultdict(list)
    for mention in mentions:
        for pid in set(mention.passage_ids):
            by_passage[pid].append(mention)
    rows = []
    for p in sorted(passages, key=lambda row: row.passage_id):
        removed = sorted((max(p.char_start, e["char_start"]), min(p.char_end, e["char_end"]))
            for e in exclusions if e["transcript_version_id"] == p.transcript_version_id
            and e["char_start"] < p.char_end and e["char_end"] > p.char_start)
        spans, cursor = [], p.char_start
        for a, b in removed:
            if cursor < a:
                spans.append([cursor, a])
            cursor = max(cursor, b)
        if cursor < p.char_end:
            spans.append([cursor, p.char_end])
        text = "\n".join(p.text[a-p.char_start:b-p.char_start] for a, b in spans)
        members = defaultdict(set)
        for mention in by_passage[p.passage_id]:
            if not any(a < mention.char_end and b > mention.char_start for a, b in removed):
                members[by_mention[mention.mention_id]].add(mention.mention_id)
        rows.append({"passage_id": p.passage_id, "episode_id": p.episode_id,
            "transcript_version_id": p.transcript_version_id, "source_sha256": sha256(p.text.encode()),
            "source_spans": spans, "excluded_spans": [list(s) for s in removed],
            "input_text": text, "input_sha256": sha256(text.encode()),
            "duplicate_key": sha256(" ".join(text.casefold().split()).encode()),
            "entity_mentions": {cid: sorted(mids) for cid, mids in sorted(members.items())},
            "status": "ready" if text.strip() else "empty_after_exclusions"})
    return rows


def check_embeddings(inputs, embeddings, config):
    """Check vector shape, normalization and the saved token chunks without loading the model."""
    ready = [row for row in inputs if row["status"] == "ready"]
    if [e["passage_id"] for e in embeddings] != [r["passage_id"] for r in ready]:
        raise ValueError("discovery embedding coverage/order differs from inputs")
    for item, row in zip(embeddings, ready, strict=True):
        if item["input_sha256"] != row["input_sha256"]:
            raise ValueError("discovery embedding source hash differs")
        if (len(item["vector"]) != config.dimension or
                any(not math.isfinite(v) for v in item["vector"]) or
                abs(sum(v*v for v in item["vector"]) - 1) > 1e-4):
            raise ValueError("invalid discovery vector shape or normalization")
        tokens, special = item["token_count"], item["special_tokens"]
        if type(tokens) is not int or type(special) is not int or tokens < 1 or not 1 <= special < config.max_tokens:
            raise ValueError("invalid discovery token counts")
        expected = [{"token_start": a, "token_end": b, "encoder_tokens": b-a+special}
                    for a, b in token_ranges(tokens, config.max_tokens-special)]
        if item["chunks"] != expected:
            raise ValueError("discovery chunks omit tokens or exceed budget")


def rank_connections(inputs, embeddings, config):
    """Compare every eligible pair of passages from different episodes, with two rankings.

    Top-k is taken per source passage, so the links are directed. Similar passages can still
    take opposite views. The Jaccard baseline only uses named entities found in the passages.
    Neither ranking changes any entity identities.
    """
    ready = [r for r in inputs if r["status"] == "ready"]
    vectors = {r["passage_id"]: r["vector"] for r in embeddings}
    scores, selected = [], []
    same_episode = duplicates = 0
    for source in ready:
        candidates = {"mpnet_cosine": [], "shared_entity_jaccard": []}
        for target in ready:
            if source["passage_id"] == target["passage_id"]:
                continue
            if source["episode_id"] == target["episode_id"]:
                same_episode += 1
                continue
            if source["duplicate_key"] == target["duplicate_key"]:
                duplicates += 1
                continue
            a, b = set(source["entity_mentions"]), set(target["entity_mentions"])
            common = sorted(a & b)
            cosine = max(-1., min(1., sum(x*y for x, y in zip(
                vectors[source["passage_id"]], vectors[target["passage_id"]], strict=True))))
            jaccard = len(common) / len(a | b) if a | b else 0.
            for method, score in (("mpnet_cosine", cosine), ("shared_entity_jaccard", jaccard)):
                candidates[method].append({"source_passage_id": source["passage_id"],
                    "target_passage_id": target["passage_id"], "source_episode_id": source["episode_id"],
                    "target_episode_id": target["episode_id"], "method": method, "score": score,
                    "shared_entity_ids": common, "source_input_sha256": source["input_sha256"],
                    "target_input_sha256": target["input_sha256"]})
        for method, rows in candidates.items():
            floor = config.minimum_cosine if method == "mpnet_cosine" else config.minimum_jaccard
            for rank, row in enumerate(sorted(rows, key=lambda r: (-r["score"], r["target_passage_id"])), 1):
                row.update({"rank": rank, "decision": "below_threshold" if row["score"] < floor else
                            "outside_top_k" if rank > config.top_k else "selected"})
                scores.append(row)
                if row["decision"] == "selected":
                    selected.append({**row, "connection_id": stable_id("pc", method,
                        row["source_passage_id"], row["target_passage_id"]), "status": "suggested"})
    counts = {"passages": len(inputs), "ready_passages": len(ready), "failed": 0,
        "skipped_passages": len(inputs)-len(ready), "scored_directed_pairs": len(scores)//2,
        "excluded_same_episode_pairs": same_episode, "excluded_duplicate_pairs": duplicates,
        "passages_with_exclusions": sum(bool(r["excluded_spans"]) for r in inputs),
        "multi_chunk_passages": sum(len(r["chunks"]) > 1 for r in embeddings)}
    for method in ("mpnet_cosine", "shared_entity_jaccard"):
        rows = [r for r in selected if r["method"] == method]
        counts[f"{method}_connections"] = len(rows)
        counts[f"{method}_sources"] = len({r["source_passage_id"] for r in rows})
    mp = {(r["source_passage_id"], r["target_passage_id"]) for r in selected if r["method"] == "mpnet_cosine"}
    shared = {(r["source_passage_id"], r["target_passage_id"]) for r in selected if r["method"] == "shared_entity_jaccard"}
    counts.update({"common_selected_pairs": len(mp & shared), "mpnet_only_pairs": len(mp-shared),
                   "jaccard_only_pairs": len(shared-mp)})
    return scores, selected, counts


def compute_discovery(records, exclusions, config, encode):
    """Encode the passages and rank related passages in other episodes. ``encode`` gets the
    passage texts and is only called when there is at least one."""
    inputs = prepare_inputs(records, exclusions)
    ready = [r for r in inputs if r["status"] == "ready"]
    encoded = encode([r["input_text"] for r in ready]) if ready else []
    embeddings = [{**e, "passage_id": r["passage_id"], "input_sha256": r["input_sha256"]}
                  for r, e in zip(ready, encoded, strict=True)]
    check_embeddings(inputs, embeddings, config)
    scores, connections, counts = rank_connections(inputs, embeddings, config)
    return inputs, embeddings, scores, connections, counts
