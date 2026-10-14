"""Finds pairs of corpus entities that might be the same and compares their passage
contexts. The output is only a list of suggestions for review, nothing gets merged here.
"""

import importlib.metadata
import itertools
import json
import math
import platform
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from spoken_record import __version__
from spoken_record.adapters.base import sha256, stable_id
from spoken_record.context_models import ContextIdentityCandidate, ContextResolutionConfig
from spoken_record.evaluation import Annotation, load_sample
from spoken_record.extraction_models import ExtractedMention
from spoken_record.ingestion import dump_json, dump_jsonl, read_jsonl, workspace_path
from spoken_record.passage_models import Passage
from spoken_record.resolution import normalize_surface, validate_run as validate_resolution
from spoken_record.resolution_models import CorpusEntity, MentionAssignment


WORD = re.compile(r"[^\W\d_]+(?:['’-][^\W\d_]+)*", re.UNICODE)
STOPWORDS = frozenset("a an and are as at be been being but by did do does for from had has have he her hers him his i if in into is it its me my no not of on or our she so than that the their them then there these they this those to was we were what when which who will with would you your".split())


def tokens(text: str) -> list[str]:
    return [token.casefold() for token in WORD.findall(text) if token.casefold() not in STOPWORDS and len(token) > 1]


def surface_relation(left: CorpusEntity, right: CorpusEntity, fuzzy_floor: float,
                     block_conflicting_numbers: bool = True):
    methods, score = set(), 0.0
    for a, b in itertools.product(left.normalized_keys, right.normalized_keys):
        if a == b:
            methods.add("exact_key")
            score = 1.0
            continue
        a_numbers, b_numbers = set(re.findall(r"\d+", a)), set(re.findall(r"\d+", b))
        if block_conflicting_numbers and a_numbers and b_numbers and a_numbers != b_numbers:
            continue
        a_tokens, b_tokens = a.split(), b.split()
        short, long = (a_tokens, b_tokens) if len(a_tokens) <= len(b_tokens) else (b_tokens, a_tokens)
        if left.type == "Person" and len(short) < len(long) and short == long[-len(short):]:
            methods.add("person_name_containment")
            score = max(score, .9)
        similarity = SequenceMatcher(None, a, b).ratio()
        if similarity >= fuzzy_floor:
            methods.add("fuzzy_surface")
            score = max(score, similarity)
    return tuple(sorted(methods)), score


def cosine(left: Counter, right: Counter, idf: dict[str, float]) -> float:
    shared = sorted(set(left) & set(right))
    dot = math.fsum(left[t] * right[t] * idf[t] ** 2 for t in shared)
    left_norm = math.sqrt(math.fsum(left[t] ** 2 * idf[t] ** 2 for t in sorted(left)))
    right_norm = math.sqrt(math.fsum(right[t] ** 2 * idf[t] ** 2 for t in sorted(right)))
    value = dot / (left_norm * right_norm) if left_norm and right_norm else 0.0
    return min(1.0, max(0.0, value))


def jaccard(left: set[str], right: set[str]) -> float:
    return len(left & right) / len(left | right) if left or right else 0.0


def review_view(candidates: tuple[ContextIdentityCandidate, ...], passages: list[Passage]) -> str:
    passage_by_id = {passage.passage_id: passage for passage in passages}
    lines = ["# Contextual identity candidate review", "",
             "> [!warning] Advisory output",
             "> `suggest_merge` means inspect the evidence. It does not change corpus identities or the Obsidian vault.", ""]
    suggestions = [candidate for candidate in candidates if candidate.decision == "suggest_merge"]
    lines.extend(["## Suggested merges", ""])
    if not suggestions:
        lines.extend(["No pairs met the conservative suggestion rules.", ""])
    for candidate in suggestions:
        lines.extend([
            f"### {candidate.left_label} ↔ {candidate.right_label}", "",
            f"- Type: `{candidate.type}`",
            f"- Retrieval: `{', '.join(candidate.retrieval_methods)}`",
            f"- Score: `{candidate.combined_score:.3f}` (surface `{candidate.surface_similarity:.3f}`, "
            f"lexical context `{candidate.lexical_context_similarity:.3f}`, neighbors `{candidate.neighbor_similarity:.3f}`)",
            f"- Entity IDs: `{candidate.left_corpus_entity_id}` / `{candidate.right_corpus_entity_id}`",
            f"- Reason: {candidate.reason}",
            "- Evidence:",
        ])
        for passage_id in candidate.evidence_passage_ids:
            excerpt = " ".join(passage_by_id[passage_id].text.split())
            if len(excerpt) > 320:
                excerpt = excerpt[:317].rstrip() + "..."
            lines.append(f"  - `{passage_id}` — {excerpt}")
        lines.append("")
    lines.extend(["## Abstentions", "",
                  "| Type | Left | Right | Retrieval | Score | Reason |",
                  "|---|---|---|---|---:|---|"])
    for candidate in candidates:
        if candidate.decision != "abstain":
            continue
        values = (candidate.type, candidate.left_label, candidate.right_label,
                  ", ".join(candidate.retrieval_methods), f"{candidate.combined_score:.3f}", candidate.reason)
        lines.append("| " + " | ".join(value.replace("|", "\\|") for value in values) + " |")
    return "\n".join(lines) + "\n"


def build_candidates(run_id: str, entities: list[CorpusEntity], mentions: list[ExtractedMention],
                     assignments: list[MentionAssignment], passages: list[Passage],
                     config: ContextResolutionConfig) -> tuple[ContextIdentityCandidate, ...]:
    passage_by_id = {p.passage_id: p for p in passages}
    assignment_by_mention = {a.mention_id: a for a in assignments}
    entity_passages = defaultdict(set)
    passage_entities = defaultdict(set)
    for mention in mentions:
        corpus_entity_id = assignment_by_mention[mention.mention_id].corpus_entity_id
        for passage_id in mention.passage_ids:
            entity_passages[corpus_entity_id].add(passage_id)
            passage_entities[passage_id].add(corpus_entity_id)
    contexts, neighbors = {}, defaultdict(set)
    for entity in entities:
        context_text = "\n".join(passage_by_id[pid].text for pid in sorted(entity_passages[entity.corpus_entity_id]))
        remove = {token for key in entity.normalized_keys for token in tokens(key)}
        contexts[entity.corpus_entity_id] = Counter(t for t in tokens(context_text) if t not in remove)
        for passage_id in entity_passages[entity.corpus_entity_id]:
            neighbors[entity.corpus_entity_id].update(passage_entities[passage_id] - {entity.corpus_entity_id})
    document_frequency = Counter(token for context in contexts.values() for token in context)
    document_count = max(1, len(contexts))
    idf = {token: math.log((1 + document_count) / (1 + count)) + 1 for token, count in document_frequency.items()}

    rows = []
    eligible = [entity for entity in entities if entity.type in config.candidate_types]
    for left, right in itertools.combinations(sorted(eligible, key=lambda e: e.corpus_entity_id), 2):
        if left.type != right.type:
            continue
        methods, surface_score = surface_relation(left, right, config.fuzzy_similarity_floor,
                                                  config.block_conflicting_numbers)
        if not methods:
            continue
        lexical = cosine(contexts[left.corpus_entity_id], contexts[right.corpus_entity_id], idf)
        neighbor = jaccard(neighbors[left.corpus_entity_id], neighbors[right.corpus_entity_id])
        combined = min(1.0, max(0.0, config.surface_weight * surface_score +
                               config.lexical_context_weight * lexical + config.neighbor_weight * neighbor))
        enough_context = lexical >= config.minimum_context_similarity or neighbor >= config.minimum_context_similarity
        allowed_method = any(method in config.suggestion_methods for method in methods)
        decision = "suggest_merge" if allowed_method and combined >= config.suggestion_threshold and enough_context else "abstain"
        if decision == "suggest_merge":
            reason = "conservative surface relation has corroborating passage context; human review required"
        elif not allowed_method:
            reason = "retrieved for comparison, but this surface method is not allowed to suggest an identity merge"
        else:
            reason = "conservative surface relation lacks enough contextual support for a merge suggestion"
        evidence = sorted(entity_passages[left.corpus_entity_id] | entity_passages[right.corpus_entity_id],
                          key=lambda pid: (passage_by_id[pid].episode_id, passage_by_id[pid].index, pid))
        rows.append(ContextIdentityCandidate(created_by_run_id=run_id,
            candidate_id=stable_id("cc", left.corpus_entity_id, right.corpus_entity_id,
                                   config.model_dump_json()), type=left.type,
            left_corpus_entity_id=left.corpus_entity_id, right_corpus_entity_id=right.corpus_entity_id,
            left_label=left.canonical_label, right_label=right.canonical_label,
            episode_ids=sorted(set(left.episode_ids) | set(right.episode_ids)),
            retrieval_methods=list(methods), surface_similarity=surface_score,
            lexical_context_similarity=lexical, neighbor_similarity=neighbor, combined_score=combined,
            decision=decision, reason=reason, evidence_passage_ids=evidence))
    return tuple(sorted(rows, key=lambda row: (-row.combined_score, row.candidate_id)))


def evaluate_candidates(packet: Path, mentions: list[ExtractedMention], assignments: list[MentionAssignment],
                        candidates: tuple[ContextIdentityCandidate, ...]) -> dict:
    _, samples = load_sample(packet)
    gold = [Annotation.model_validate(r) for r in read_jsonl(packet / "annotations.jsonl")]
    mention_map = {(m.episode_id, m.char_start, m.char_end, m.type): m for m in mentions}
    assignment_map = {a.mention_id: a.corpus_entity_id for a in assignments}
    entity_gold = defaultdict(set)
    aligned_known = 0
    for annotation in gold:
        sample = samples[annotation.passage_id]
        for gold_mention in annotation.mentions:
            if gold_mention.identity_id is None:
                continue
            predicted = mention_map.get((sample["episode_id"], sample["char_start"] + gold_mention.start,
                                         sample["char_start"] + gold_mention.end, gold_mention.type))
            if predicted:
                aligned_known += 1
                entity_gold[assignment_map[predicted.mention_id]].add(gold_mention.identity_id)
    positive_pairs = set()
    for identity in {identity for values in entity_gold.values() for identity in values}:
        members = sorted(entity for entity, values in entity_gold.items() if identity in values)
        positive_pairs.update(itertools.combinations(members, 2))
    candidate_map = {(min(c.left_corpus_entity_id, c.right_corpus_entity_id),
                      max(c.left_corpus_entity_id, c.right_corpus_entity_id)): c for c in candidates}
    retrieved = positive_pairs & set(candidate_map)
    suggested = {pair for pair, candidate in candidate_map.items() if candidate.decision == "suggest_merge"}
    evaluated_negative = {pair for pair in candidate_map if entity_gold[pair[0]] and entity_gold[pair[1]] and
                          entity_gold[pair[0]].isdisjoint(entity_gold[pair[1]])}
    true_suggestions = suggested & positive_pairs
    false_suggestions = suggested & evaluated_negative
    evaluable_pairs = positive_pairs | evaluated_negative
    evaluated_suggestions = suggested & evaluable_pairs
    return {"schema_version": 1, "metric": "gold-split-corpus-entity-pairs-v1",
            "aligned_known_identity_mentions": aligned_known,
            "gold_split_entity_pairs": len(positive_pairs), "retrieved_gold_split_pairs": len(retrieved),
            "suggested_gold_split_pairs": len(true_suggestions),
            "evaluated_negative_candidate_pairs": len(evaluated_negative),
            "evaluated_suggestion_pairs": len(evaluated_suggestions),
            "unevaluated_suggestion_pairs": len(suggested - evaluable_pairs),
            "false_merge_suggestions_on_evaluated_pairs": len(false_suggestions),
            "candidate_recall": len(retrieved) / len(positive_pairs) if positive_pairs else None,
            "suggestion_recall": len(true_suggestions) / len(positive_pairs) if positive_pairs else None,
            "suggestion_precision_on_evaluated_pairs": len(true_suggestions) / len(evaluated_suggestions)
                if true_suggestions or false_suggestions else None,
            "missed_gold_split_pairs": [list(pair) for pair in sorted(positive_pairs - set(candidate_map))],
            "notes": ["Only exact span/type aligned mentions with known reviewed identities contribute.",
                      "Unlabelled candidate pairs are unknown, not assumed negative.",
                      "suggest_merge remains a review proposal and does not alter corpus identities."]}


def load_inputs(parent: Path):
    manifest = validate_resolution(parent)
    extraction = parent.parent / manifest["parent_run_id"]
    extraction_manifest = json.loads((extraction / "manifest.json").read_bytes())
    passage_run = extraction.parent / extraction_manifest["parent_run_id"]
    return ([CorpusEntity.model_validate(r) for r in read_jsonl(parent / "corpus_entities.jsonl")],
            [MentionAssignment.model_validate(r) for r in read_jsonl(parent / "assignments.jsonl")],
            [ExtractedMention.model_validate(r) for r in read_jsonl(extraction / "mentions.jsonl")],
            [Passage.model_validate(r) for r in read_jsonl(passage_run / "passages.jsonl")])


def analyze_run(root: Path, input_name: str, config_name: str, packet_name: str,
                run_id: str | None = None) -> Path:
    """Write the candidate pairs as a new run. The parent's assignments stay as they are."""
    root, started, now = root.resolve(), perf_counter(), datetime.now(timezone.utc)
    run_id = run_id or f"context-{now:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]{0,95}", run_id):
        raise ValueError("invalid run ID")
    parent = workspace_path(root, input_name)
    if parent.parent != root / "data/prototype1/runs":
        raise ValueError("input must be a resolution snapshot")
    output = parent.parent / run_id
    if output.exists():
        raise FileExistsError(output)
    parent_manifest = validate_resolution(parent)
    parent_digest = sha256((parent / "manifest.json").read_bytes())
    config_path = workspace_path(root, config_name)
    config_bytes = config_path.read_bytes()
    config = ContextResolutionConfig.model_validate_json(config_bytes)
    entities, assignments, mentions, passages = load_inputs(parent)
    candidates = build_candidates(run_id, entities, mentions, assignments, passages, config)
    packet = workspace_path(root, packet_name)
    evaluation = evaluate_candidates(packet, mentions, assignments, candidates)
    support = {
        str(config_path.relative_to(root)): sha256(config_bytes),
        str((packet / "manifest.json").relative_to(root)): sha256((packet / "manifest.json").read_bytes()),
        str((packet / "sample.jsonl").relative_to(root)): sha256((packet / "sample.jsonl").read_bytes()),
        str((packet / "annotations.jsonl").relative_to(root)): sha256((packet / "annotations.jsonl").read_bytes()),
    }
    summary = {"candidates": len(candidates),
               "suggest_merge": sum(c.decision == "suggest_merge" for c in candidates),
               "abstain": sum(c.decision == "abstain" for c in candidates),
               "by_type": dict(sorted(Counter(c.type for c in candidates).items())),
               "by_retrieval_method": dict(sorted(Counter(method for c in candidates for method in c.retrieval_methods).items()))}
    validate_resolution(parent)
    if sha256((parent / "manifest.json").read_bytes()) != parent_digest or config_path.read_bytes() != config_bytes:
        raise ValueError("parent or context configuration changed during analysis")
    for name, digest in support.items():
        if sha256(workspace_path(root, name).read_bytes()) != digest:
            raise ValueError("evaluation input changed during context analysis")
    output.mkdir(parents=True, exist_ok=False)
    dump_jsonl(output / "context_candidates.jsonl", list(candidates))
    dump_json(output / "config.json", config.model_dump(mode="json"))
    dump_json(output / "summary.json", summary)
    dump_json(output / "evaluation.json", evaluation)
    (output / "review.md").write_text(review_view(candidates, passages), encoding="utf-8")
    artifacts = ("context_candidates.jsonl", "config.json", "summary.json", "evaluation.json", "review.md")
    code_root = Path(__file__).parent
    manifest = {"schema_version": 1, "stage": "analyze_resolution", "run_id": run_id, "status": "complete",
        "package_version": __version__, "started_utc": now.isoformat(),
        "elapsed_seconds": round(perf_counter() - started, 4), "python": platform.python_version(),
        "dependencies": {"pydantic": importlib.metadata.version("pydantic")},
        "parent_run_id": parent.name, "parent_manifest_sha256": parent_digest,
        "input_config": config_name, "input_config_sha256": sha256(config_bytes),
        "evaluation_packet": packet_name, "support_sha256": support, "counts": summary,
        "code_sha256": {str(p.relative_to(code_root)): sha256(p.read_bytes()) for p in sorted(code_root.rglob("*.py"))},
        "artifact_sha256": {name: sha256((output / name).read_bytes()) for name in artifacts}}
    dump_json(output / "manifest.json", manifest)
    validate_run(output)
    return output


def validate_run(output: Path) -> dict:
    output = output.resolve()
    manifest = json.loads((output / "manifest.json").read_bytes())
    if manifest.get("schema_version") != 1 or manifest.get("stage") != "analyze_resolution" or manifest.get("status") != "complete":
        raise ValueError("unsupported contextual-resolution analysis")
    root = output.parents[3]
    parent = output.parent / manifest["parent_run_id"]
    if sha256((parent / "manifest.json").read_bytes()) != manifest["parent_manifest_sha256"]:
        raise ValueError("parent resolution manifest changed")
    config_path = workspace_path(root, manifest["input_config"])
    if sha256(config_path.read_bytes()) != manifest["input_config_sha256"]:
        raise ValueError("context configuration changed")
    expected_artifacts = {"context_candidates.jsonl", "config.json", "summary.json", "evaluation.json", "review.md"}
    if set(manifest["artifact_sha256"]) != expected_artifacts:
        raise ValueError("unexpected context-analysis artifact set")
    for name, digest in manifest["artifact_sha256"].items():
        if sha256((output / name).read_bytes()) != digest:
            raise ValueError(f"context artifact changed: {name}")
    for name, digest in manifest["support_sha256"].items():
        if sha256(workspace_path(root, name).read_bytes()) != digest:
            raise ValueError(f"context-analysis support changed: {name}")
    config = ContextResolutionConfig.model_validate_json((output / "config.json").read_bytes())
    entities, assignments, mentions, passages = load_inputs(parent)
    expected = build_candidates(manifest["run_id"], entities, mentions, assignments, passages, config)
    actual = tuple(ContextIdentityCandidate.model_validate(r) for r in read_jsonl(output / "context_candidates.jsonl"))
    if expected != actual:
        raise ValueError("context candidates are not reproducible")
    summary = {"candidates": len(actual), "suggest_merge": sum(c.decision == "suggest_merge" for c in actual),
               "abstain": sum(c.decision == "abstain" for c in actual),
               "by_type": dict(sorted(Counter(c.type for c in actual).items())),
               "by_retrieval_method": dict(sorted(Counter(m for c in actual for m in c.retrieval_methods).items()))}
    if summary != manifest["counts"] or summary != json.loads((output / "summary.json").read_bytes()):
        raise ValueError("context analysis counts differ")
    if review_view(actual, passages) != (output / "review.md").read_text(encoding="utf-8"):
        raise ValueError("context review sheet is not reproducible")
    evaluation = evaluate_candidates(workspace_path(root, manifest["evaluation_packet"]), mentions, assignments,
                                     actual)
    if evaluation != json.loads((output / "evaluation.json").read_bytes()):
        raise ValueError("context evaluation is not reproducible")
    return manifest
