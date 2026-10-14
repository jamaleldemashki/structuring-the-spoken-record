"""Finds related passages across episodes.

The output wraps the parent run. It never merges entities, and a high similarity is not
treated as a statement about the content. The embeddings are saved, so the rankings can be
rebuilt and checked later without running the model again.
"""

import importlib.metadata
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from spoken_record.adapters.base import sha256, stable_id
from spoken_record.discovery_models import DiscoveryConfig
from spoken_record.ingestion import dump_json, dump_jsonl, read_jsonl, workspace_path


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


def load_inputs(root: Path, parent: Path):
    """Read both source runs through one validated chain so they can't get out of sync."""
    from spoken_record.vault import vault_input, load_resolution_chain
    manifest = json.loads((parent / "manifest.json").read_bytes())
    if manifest["stage"] not in {"resolve", "refine_resolution", "entity_linking", "coreference",
                                  "relations", "relation_verification"}:
        raise ValueError("discovery requires an identity-compatible upstream snapshot")
    if manifest.get("status") != "complete" or manifest.get("counts", {}).get("failed", 0):
        raise ValueError("discovery parent is incomplete or contains failures")
    identity, _, _ = vault_input(root, parent)
    records = load_resolution_chain(root, identity)
    identity_manifest = json.loads((identity / "manifest.json").read_bytes())
    extraction = parent.parent / identity_manifest.get("extraction_run_id", identity_manifest["parent_run_id"])
    extraction_manifest = json.loads((extraction / "manifest.json").read_bytes())
    passage_run = parent.parent / extraction_manifest["parent_run_id"]
    sources = {"identity_run_id": identity.name, "passage_run_id": passage_run.name,
        "identity_manifest_sha256": sha256((identity / "manifest.json").read_bytes()),
        "passage_manifest_sha256": sha256((passage_run / "manifest.json").read_bytes())}
    return records, read_jsonl(passage_run / "exclusions.jsonl"), sources


def prepare_inputs(records, exclusions: list[dict]) -> list[dict]:
    """Leave reviewed exclusions (like credits) out of the retrieval text.

    The transcript and passages are not changed. The remaining ranges are joined with
    newlines only for the model input, and their offsets and both hashes are recorded.
    """
    passages, mentions, assignments = records[2], records[3], records[6]
    if any(e.get("review_status") != "reviewed" for e in exclusions):
        raise ValueError("discovery accepts only reviewed source exclusions")
    from spoken_record.navigation import build_concept_navigation
    # Still run the provenance check when the corpus has no Concepts or Schools.
    build_concept_navigation(passages, mentions, records[5], assignments)
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


ARTIFACTS = ("config.json", "passage_inputs.jsonl", "passage_embeddings.jsonl",
             "passage_scores.jsonl", "passage_connections.jsonl", "summary.json")


def discover_run(root: Path, input_name: str, config_name: str, run_id: str | None = None,
                 *, backend_factory=None) -> Path:
    """Write the discovery run. If the model fails, nothing is written (no empty result)."""
    root, started = root.resolve(), perf_counter()
    run_id = run_id or f"discovery-{uuid4().hex[:12]}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]{0,95}", run_id):
        raise ValueError("invalid discovery run ID")
    parent = workspace_path(root, input_name)
    output = workspace_path(root, f"data/prototype1/runs/{run_id}")
    if parent.parent != output.parent or parent == output:
        raise ValueError("discovery requires a sibling snapshot")
    if output.exists():
        raise FileExistsError(output)
    raw_config = workspace_path(root, config_name).read_bytes()
    config = DiscoveryConfig.model_validate_json(raw_config)
    parent_hash = sha256((parent/"manifest.json").read_bytes())
    records, exclusions, sources = load_inputs(root, parent)
    inputs = prepare_inputs(records, exclusions)
    ready = [r for r in inputs if r["status"] == "ready"]
    backend = (backend_factory or PassageEncoder)(root, config) if ready else None
    encoded = backend.encode([r["input_text"] for r in ready]) if ready else []
    embeddings = [{**e, "passage_id": r["passage_id"], "input_sha256": r["input_sha256"]}
                  for r, e in zip(ready, encoded, strict=True)]
    check_embeddings(inputs, embeddings, config)
    scores, connections, counts = rank_connections(inputs, embeddings, config)
    if (workspace_path(root, config_name).read_bytes() != raw_config or
            sha256((parent/"manifest.json").read_bytes()) != parent_hash):
        raise ValueError("discovery inputs changed during inference")
    output.mkdir(parents=True, exist_ok=False)
    dump_json(output/"config.json", config.model_dump(mode="json"))
    for name, data in (("passage_inputs", inputs), ("passage_embeddings", embeddings),
                       ("passage_scores", scores), ("passage_connections", connections)):
        dump_jsonl(output/f"{name}.jsonl", data)
    dump_json(output/"summary.json", counts)
    code_root = Path(__file__).parent
    dump_json(output/"manifest.json", {"schema_version": 1, "stage": "passage_discovery",
        "status": "complete", "run_id": run_id, "parent_run_id": parent.name,
        "parent_manifest_sha256": parent_hash, "source_snapshots": sources,
        "input_config": config_name, "input_config_sha256": sha256(raw_config),
        "model": {"repo_id": config.repo_id, "revision": config.revision, "pooling": config.pooling},
        "created_utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": perf_counter()-started,
        "dependencies": {n: importlib.metadata.version(n) for n in ("torch", "transformers", "pydantic")},
        "code_sha256": {str(p.relative_to(code_root)): sha256(p.read_bytes()) for p in sorted(code_root.rglob("*.py"))},
        "counts": counts, "artifact_sha256": {n: sha256((output/n).read_bytes()) for n in ARTIFACTS}})
    validate_run(output)
    return output


def validate_run(output: Path) -> dict:
    """Rebuild inputs and rankings from the saved vectors, without running the model.

    Hashes and chunk coverage are checked. The model output itself can only be checked by
    running the pinned model again.
    """
    output = output.resolve()
    root = output.parents[3]
    manifest = json.loads((output/"manifest.json").read_bytes())
    if (manifest.get("schema_version"), manifest.get("stage"), manifest.get("status")) != (1, "passage_discovery", "complete"):
        raise ValueError("unsupported discovery snapshot")
    if manifest["run_id"] != output.name or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]{0,95}", manifest["parent_run_id"]):
        raise ValueError("invalid discovery snapshot ID")
    parent = output.parent / manifest["parent_run_id"]
    if parent == output or sha256((parent/"manifest.json").read_bytes()) != manifest["parent_manifest_sha256"]:
        raise ValueError("discovery parent changed")
    if set(manifest["artifact_sha256"]) != set(ARTIFACTS):
        raise ValueError("unexpected discovery artifacts")
    for name, digest in manifest["artifact_sha256"].items():
        if sha256((output/name).read_bytes()) != digest:
            raise ValueError(f"discovery artifact changed: {name}")
    raw_config = workspace_path(root, manifest["input_config"]).read_bytes()
    config = DiscoveryConfig.model_validate_json(raw_config)
    if sha256(raw_config) != manifest["input_config_sha256"] or config.model_dump(mode="json") != json.loads((output/"config.json").read_bytes()):
        raise ValueError("discovery configuration changed")
    if manifest["model"] != {"repo_id": config.repo_id, "revision": config.revision, "pooling": config.pooling}:
        raise ValueError("discovery model provenance differs")
    records, exclusions, sources = load_inputs(root, parent)
    if sources != manifest["source_snapshots"]:
        raise ValueError("discovery source snapshot mismatch")
    inputs = prepare_inputs(records, exclusions)
    if inputs != read_jsonl(output/"passage_inputs.jsonl"):
        raise ValueError("discovery source inputs differ")
    embeddings = read_jsonl(output/"passage_embeddings.jsonl")
    check_embeddings(inputs, embeddings, config)
    scores, connections, counts = rank_connections(inputs, embeddings, config)
    if scores != read_jsonl(output/"passage_scores.jsonl") or connections != read_jsonl(output/"passage_connections.jsonl"):
        raise ValueError("discovery rankings not reproducible")
    if counts != manifest["counts"] or counts != json.loads((output/"summary.json").read_bytes()):
        raise ValueError("discovery summary differs")
    return manifest
