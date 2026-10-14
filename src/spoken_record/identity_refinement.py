"""Automatic merging of Person identities by name, abstaining when unsure. The merges
are provisional.
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
from spoken_record.context_resolution import cosine, jaccard, tokens
from spoken_record.extraction_models import ExtractedMention
from spoken_record.identity_models import (EntityRedirect, IdentityCandidateDecision,
                                           IdentityRefinementConfig)
from spoken_record.ingestion import dump_json, dump_jsonl, read_jsonl, workspace_path
from spoken_record.passage_models import Passage
from spoken_record.resolution import (evaluate_resolution, normalize_surface,
                                      validate_run as validate_resolution)
from spoken_record.resolution_models import (AliasRule, CorpusEntity, EpisodeEntity,
                                             MentionAssignment, ResolutionConfig,
                                             ResolutionDecision, ReviewCandidate)


NAME_WORD = re.compile(r"[^\W_]+", re.UNICODE)
QUALIFIERS = frozenset({"of", "from"})
PERSON_DESCRIPTORS = frozenset({"aunt", "brother", "father", "great", "her", "his", "king",
                                "mother", "my", "our", "pope", "queen", "saint", "sister", "st",
                                "their", "uncle", "your"})


def name_tokens(value: str) -> tuple[str, ...]:
    return tuple(token.casefold() for token in NAME_WORD.findall(value))


def compact(value: str) -> str:
    return "".join(name_tokens(value))


def character_grams(value: str) -> set[str]:
    value = compact(value)
    if len(value) < 3:
        return {value} if value else set()
    return {value[index:index + 3] for index in range(len(value) - 2)}


def qualified_name(parts: tuple[str, ...]) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
    for index, token in enumerate(parts):
        if token in QUALIFIERS and index and index < len(parts) - 1:
            return parts[:index], parts[index + 1:]
    return None


def initial_alignment(left: tuple[str, ...], right: tuple[str, ...]) -> bool:
    if len(left) != len(right) or len(left) < 2 or left[-1] != right[-1]:
        return False
    pairs = zip(left[:-1], right[:-1])
    return all(a == b or len(a) == 1 and b.startswith(a) or len(b) == 1 and a.startswith(b)
               for a, b in pairs)


def indexed_pairs(entities: list[CorpusEntity], config: IdentityRefinementConfig) -> set[tuple[str, str]]:
    """Find candidate pairs through inverted indexes instead of comparing every pair."""
    eligible = {entity.corpus_entity_id: entity for entity in entities if entity.type in config.candidate_types}
    blocks: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for entity in eligible.values():
        for key in entity.normalized_keys:
            blocks[(entity.type, "exact", key)].add(entity.corpus_entity_id)
            for gram in character_grams(key):
                blocks[(entity.type, "gram", gram)].add(entity.corpus_entity_id)
            if entity.type == "Person":
                parts = name_tokens(key)
                if parts:
                    blocks[(entity.type, "surname", parts[-1])].add(entity.corpus_entity_id)
                qualified = qualified_name(parts)
                if qualified:
                    blocks[(entity.type, "qualified", " ".join(qualified[0]))].add(entity.corpus_entity_id)
    pairs = set()
    for members in blocks.values():
        pairs.update(itertools.combinations(sorted(members), 2))
    return pairs


def surface_features(left: CorpusEntity, right: CorpusEntity, config: IdentityRefinementConfig):
    methods, contradictions, score = set(), set(), 0.0
    for left_key, right_key in itertools.product(left.normalized_keys, right.normalized_keys):
        left_numbers = set(re.findall(r"\d+", left_key))
        right_numbers = set(re.findall(r"\d+", right_key))
        if config.block_conflicting_numbers and left_numbers and right_numbers and left_numbers != right_numbers:
            contradictions.add("conflicting_numbers")
            continue
        if left_key == right_key:
            methods.add("exact_key")
            score = 1.0
        left_parts, right_parts = name_tokens(left_key), name_tokens(right_key)
        if left.type == "Person" and left_parts and right_parts:
            short, long = (left_parts, right_parts) if len(left_parts) <= len(right_parts) else (right_parts, left_parts)
            if len(short) < len(long) and short == long[-len(short):]:
                methods.add("person_name_containment")
                score = max(score, .9)
            if initial_alignment(left_parts, right_parts):
                methods.add("person_initial_alignment")
                score = max(score, .92)
            if left_parts[-1] == right_parts[-1] or (qualified_name(left_parts) and qualified_name(right_parts)
                    and qualified_name(left_parts)[0] == qualified_name(right_parts)[0]):
                methods.add("person_name_block")
            # This finds ASR spelling variants like Douglas/Douglass. It is not in merge_methods,
            # so context similarity alone can't turn such a pair into an automatic merge.
            surname_variants_enabled = "person_surname_similarity_floor" in config.model_fields_set
            if surname_variants_enabled and len(short) == 1 and len(long) > 1 and short[0] != long[-1]:
                surname_similarity = SequenceMatcher(None, short[0], long[-1]).ratio()
                if surname_similarity >= config.person_surname_similarity_floor:
                    methods.add("person_surname_variant")
                    score = max(score, surname_similarity)
            if config.separate_incompatible_person_names:
                left_qualified, right_qualified = qualified_name(left_parts), qualified_name(right_parts)
                if left_qualified and right_qualified and left_qualified[0] == right_qualified[0] \
                        and left_qualified[1] != right_qualified[1]:
                    contradictions.add("conflicting_person_qualifier")
                elif left_parts[-1] == right_parts[-1] and len(left_parts) > 1 and len(right_parts) > 1 \
                        and not initial_alignment(left_parts, right_parts) \
                        and not (len(short) < len(long) and short == long[-len(short):]):
                    left_given, right_given = left_parts[:-1], right_parts[:-1]
                    if all(len(token) > 1 for token in (*left_given, *right_given)) and not \
                            set((*left_given, *right_given)) & PERSON_DESCRIPTORS:
                        contradictions.add("conflicting_person_forename")
        similarity = SequenceMatcher(None, left_key, right_key).ratio()
        if similarity >= config.fuzzy_similarity_floor:
            methods.add("fuzzy_surface")
            score = max(score, similarity)
    if left.resolution_status == right.resolution_status == "reviewed" \
            and left.registry_key != right.registry_key:
        contradictions.add("conflicting_reviewed_identity")
    return tuple(sorted(methods)), tuple(sorted(contradictions)), score


def build_contexts(entities: list[CorpusEntity], mentions: list[ExtractedMention],
                   assignments: list[MentionAssignment], passages: list[Passage]):
    passage_by_id = {passage.passage_id: passage for passage in passages}
    assignment_by_mention = {assignment.mention_id: assignment for assignment in assignments}
    entity_passages: dict[str, set[str]] = defaultdict(set)
    passage_entities: dict[str, set[str]] = defaultdict(set)
    for mention in mentions:
        corpus_entity_id = assignment_by_mention[mention.mention_id].corpus_entity_id
        for passage_id in mention.passage_ids:
            entity_passages[corpus_entity_id].add(passage_id)
            passage_entities[passage_id].add(corpus_entity_id)
    contexts, neighbors = {}, defaultdict(set)
    for entity in entities:
        context_text = "\n".join(passage_by_id[pid].text for pid in sorted(entity_passages[entity.corpus_entity_id]))
        remove = {token for key in entity.normalized_keys for token in tokens(key)}
        contexts[entity.corpus_entity_id] = Counter(token for token in tokens(context_text) if token not in remove)
        for passage_id in entity_passages[entity.corpus_entity_id]:
            neighbors[entity.corpus_entity_id].update(passage_entities[passage_id] - {entity.corpus_entity_id})
    frequencies = Counter(token for context in contexts.values() for token in context)
    count = max(1, len(contexts))
    idf = {token: math.log((1 + count) / (1 + frequency)) + 1 for token, frequency in frequencies.items()}
    return passage_by_id, entity_passages, contexts, neighbors, idf


def build_decisions(run_id: str, entities: list[CorpusEntity], mentions: list[ExtractedMention],
                    assignments: list[MentionAssignment], passages: list[Passage],
                    config: IdentityRefinementConfig) -> tuple[IdentityCandidateDecision, ...]:
    entity_by_id = {entity.corpus_entity_id: entity for entity in entities}
    passage_by_id, entity_passages, contexts, neighbors, idf = build_contexts(
        entities, mentions, assignments, passages)
    rows = []
    for left_id, right_id in sorted(indexed_pairs(entities, config)):
        left, right = entity_by_id[left_id], entity_by_id[right_id]
        methods, contradictions, surface = surface_features(left, right, config)
        if not methods:
            continue
        lexical = cosine(contexts[left_id], contexts[right_id], idf)
        neighbor = jaccard(neighbors[left_id], neighbors[right_id])
        combined = min(1.0, max(0.0, math.fsum((config.surface_weight * surface,
            config.lexical_context_weight * lexical, config.neighbor_weight * neighbor))))
        enough_context = lexical >= config.minimum_context_similarity or neighbor >= config.minimum_context_similarity
        merge_method = any(method in config.merge_methods for method in methods)
        if contradictions:
            decision = "separate"
            reason = "name or reviewed-identity evidence conflicts: " + ", ".join(contradictions)
        elif merge_method and combined >= config.merge_threshold and enough_context:
            decision = "merge"
            reason = "compatible name forms have corroborating passage context; automatic link remains provisional"
        elif not merge_method:
            decision = "abstain"
            reason = "retrieved by an ambiguous or fuzzy name block without an allowed merge relation"
        else:
            decision = "abstain"
            reason = "compatible name forms lack the configured contextual support for an automatic merge"
        evidence = sorted(entity_passages[left_id] | entity_passages[right_id],
                          key=lambda pid: (passage_by_id[pid].episode_id, passage_by_id[pid].index, pid))
        rows.append(IdentityCandidateDecision(created_by_run_id=run_id,
            # exclude_unset keeps the IDs of runs made before the optional rules existed.
            candidate_id=stable_id("ic", left_id, right_id,
                                   config.model_dump_json(exclude_unset=True)), type=left.type,
            left_corpus_entity_id=left_id, right_corpus_entity_id=right_id,
            left_label=left.canonical_label, right_label=right.canonical_label,
            episode_ids=sorted(set(left.episode_ids) | set(right.episode_ids)),
            retrieval_methods=list(methods), surface_similarity=surface,
            lexical_context_similarity=lexical, neighbor_similarity=neighbor,
            combined_score=combined, contradictions=list(contradictions), decision=decision,
            reason=reason, evidence_passage_ids=evidence))
    return tuple(sorted(rows, key=lambda row: (-row.combined_score, row.candidate_id)))


class Components:
    def __init__(self, ids):
        ids = list(ids)
        self.parent = {value: value for value in ids}
        self.members = {value: {value} for value in ids}

    def find(self, value):
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left, right):
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        survivor, absorbed = sorted((left_root, right_root))
        self.parent[absorbed] = survivor
        self.members[survivor].update(self.members.pop(absorbed))


def consistent_components(entities: list[CorpusEntity], decisions: tuple[IdentityCandidateDecision, ...],
                          maximum: int):
    components = Components(entity.corpus_entity_id for entity in entities)
    separate = {frozenset((row.left_corpus_entity_id, row.right_corpus_entity_id))
                for row in decisions if row.decision == "separate"}
    final = []
    for row in decisions:
        if row.decision != "merge":
            final.append(row)
            continue
        left = components.members[components.find(row.left_corpus_entity_id)]
        right = components.members[components.find(row.right_corpus_entity_id)]
        combined = left | right
        conflicts = any(frozenset(pair) in separate for pair in itertools.product(left, right))
        if len(combined) > maximum or conflicts:
            reason = ("merge abstained because transitive closure would exceed the configured component limit"
                      if len(combined) > maximum else
                      "merge abstained because transitive closure would violate an explicit separate decision")
            final.append(row.model_copy(update={"decision": "abstain", "reason": reason}))
            continue
        components.union(row.left_corpus_entity_id, row.right_corpus_entity_id)
        final.append(row)
    return components, tuple(final)


def canonical_rank(entity: CorpusEntity):
    parts = name_tokens(entity.canonical_label)
    initials = sum(len(token) == 1 for token in parts)
    full = sum(len(token) > 1 and token not in QUALIFIERS for token in parts)
    return (entity.resolution_status == "reviewed", full, -initials, entity.mention_count,
            len(entity.canonical_label), entity.canonical_label.casefold())


def refine_records(run_id: str, corpus_entities: list[CorpusEntity], episode_entities: list[EpisodeEntity],
                   assignments: list[MentionAssignment], base_decisions: list[ResolutionDecision],
                   queue: list[ReviewCandidate], candidates: tuple[IdentityCandidateDecision, ...],
                   config: IdentityRefinementConfig):
    entity_by_id = {entity.corpus_entity_id: entity for entity in corpus_entities}
    components, candidates = consistent_components(corpus_entities, candidates,
                                                    config.maximum_component_entities)
    groups = [sorted(members) for _, members in sorted(components.members.items())]
    old_to_new, refined, redirects = {}, [], []
    merge_candidates = [row for row in candidates if row.decision == "merge"]
    for member_ids in groups:
        members = [entity_by_id[value] for value in member_ids]
        if len(members) == 1:
            refined.append(members[0])
            old_to_new[member_ids[0]] = member_ids[0]
            continue
        canonical = max(members, key=canonical_rank)
        survivor_id = canonical.corpus_entity_id
        connecting = sorted(row.candidate_id for row in merge_candidates
                            if row.left_corpus_entity_id in member_ids and row.right_corpus_entity_id in member_ids)
        refined.append(CorpusEntity(created_by_run_id=run_id, corpus_entity_id=survivor_id,
            corpus_id=canonical.corpus_id, type=canonical.type, canonical_label=canonical.canonical_label,
            normalized_keys=sorted({key for entity in members for key in entity.normalized_keys}),
            observed_surfaces=sorted({surface for entity in members for surface in entity.observed_surfaces},
                                     key=lambda value: (normalize_surface(value), value)),
            reviewed_aliases=sorted({alias for entity in members for alias in entity.reviewed_aliases}),
            episode_entity_ids=sorted({value for entity in members for value in entity.episode_entity_ids}),
            episode_ids=sorted({value for entity in members for value in entity.episode_ids}),
            mention_count=sum(entity.mention_count for entity in members),
            episode_count=len({value for entity in members for value in entity.episode_ids}),
            resolution_method="automatic_context", resolution_status="provisional",
            registry_key=f"automatic:{survivor_id}"))
        for member_id in member_ids:
            old_to_new[member_id] = survivor_id
            if member_id != survivor_id:
                redirects.append(EntityRedirect(created_by_run_id=run_id,
                    from_corpus_entity_id=member_id, to_corpus_entity_id=survivor_id,
                    candidate_ids=connecting))
    refined_assignments = [assignment.model_copy(update={"created_by_run_id": run_id,
        "corpus_entity_id": old_to_new[assignment.corpus_entity_id]}) for assignment in assignments]
    base_by_episode = {decision.episode_entity_id: decision for decision in base_decisions}
    refined_decisions = []
    for episode_entity in episode_entities:
        base = base_by_episode[episode_entity.episode_entity_id]
        corpus_entity_id = old_to_new[base.corpus_entity_id]
        changed = corpus_entity_id != base.corpus_entity_id or any(
            entity.corpus_entity_id == corpus_entity_id and entity.resolution_method == "automatic_context"
            for entity in refined)
        refined_decisions.append(ResolutionDecision(created_by_run_id=run_id,
            decision_id=stable_id("rd", episode_entity.episode_entity_id, corpus_entity_id,
                                  "indexed-name-tfidf-context-v1"),
            episode_entity_id=episode_entity.episode_entity_id, corpus_entity_id=corpus_entity_id,
            method="automatic_context" if changed else base.method,
            status="provisional" if changed else base.status, alias_rule_ids=base.alias_rule_ids,
            reason=("provisional automatic name/context component" if changed else base.reason)))
    remaining_queue = [candidate for candidate in queue if len({old_to_new[
        next(assignment.corpus_entity_id for assignment in assignments
             if assignment.episode_entity_id == episode_entity_id)]
        for episode_entity_id in candidate.episode_entity_ids}) > 1]
    return (tuple(sorted(refined, key=lambda row: row.corpus_entity_id)),
            tuple(sorted(refined_assignments, key=lambda row: row.mention_id)),
            tuple(sorted(refined_decisions, key=lambda row: row.decision_id)),
            tuple(sorted(remaining_queue, key=lambda row: row.candidate_id)),
            tuple(sorted(redirects, key=lambda row: row.from_corpus_entity_id)), candidates)


def review_view(candidates: tuple[IdentityCandidateDecision, ...]) -> str:
    lines = ["# Automatic identity refinement", "",
             "> [!warning] Provisional output",
             "> Automatic merges improve navigation but are not human-reviewed identity facts.", ""]
    for decision in ("merge", "separate", "abstain"):
        rows = [row for row in candidates if row.decision == decision]
        lines.extend([f"## {decision.title()} ({len(rows)})", "",
                      "| Type | Left | Right | Retrieval | Score | Reason |",
                      "|---|---|---|---|---:|---|"])
        for row in rows:
            values = (row.type, row.left_label, row.right_label, ", ".join(row.retrieval_methods),
                      f"{row.combined_score:.3f}", row.reason)
            lines.append("| " + " | ".join(value.replace("|", "\\|") for value in values) + " |")
        lines.append("")
    return "\n".join(lines)


def load_base(parent: Path):
    manifest = validate_resolution(parent)
    extraction = parent.parent / manifest["parent_run_id"]
    extraction_manifest = json.loads((extraction / "manifest.json").read_bytes())
    passage_run = extraction.parent / extraction_manifest["parent_run_id"]
    return (manifest, extraction,
        [EpisodeEntity.model_validate(row) for row in read_jsonl(parent / "episode_entities.jsonl")],
        [CorpusEntity.model_validate(row) for row in read_jsonl(parent / "corpus_entities.jsonl")],
        [MentionAssignment.model_validate(row) for row in read_jsonl(parent / "assignments.jsonl")],
        [ResolutionDecision.model_validate(row) for row in read_jsonl(parent / "decisions.jsonl")],
        [ReviewCandidate.model_validate(row) for row in read_jsonl(parent / "review_queue.jsonl")],
        [AliasRule.model_validate(row) for row in read_jsonl(parent / "aliases.jsonl")],
        ResolutionConfig.model_validate_json((parent / "config.json").read_bytes()),
        [ExtractedMention.model_validate(row) for row in read_jsonl(extraction / "mentions.jsonl")],
        [Passage.model_validate(row) for row in read_jsonl(passage_run / "passages.jsonl")])


def summary_of(episode_entities, entities, assignments, decisions, queue, redirects, candidates):
    return {"mentions": len(assignments), "episode_entities": len(episode_entities),
        "corpus_entities": len(entities), "assignments": len(assignments), "decisions": len(decisions),
        "review_candidates": len(queue), "redirects": len(redirects),
        "identity_candidates": len(candidates),
        "automatic_merges": sum(row.decision == "merge" for row in candidates),
        "automatic_separates": sum(row.decision == "separate" for row in candidates),
        "automatic_abstentions": sum(row.decision == "abstain" for row in candidates),
        "automatic_components": sum(entity.resolution_method == "automatic_context" for entity in entities),
        "cross_episode_entities": sum(entity.episode_count > 1 for entity in entities),
        "unresolved_entities": sum(entity.resolution_status == "unresolved" for entity in entities)}


def refine_run(root: Path, input_name: str, config_name: str, run_id: str | None = None) -> Path:
    """Apply the provisional Person merges that pass the consistency checks."""
    root, started, now = root.resolve(), perf_counter(), datetime.now(timezone.utc)
    run_id = run_id or f"refine-{now:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]{0,95}", run_id):
        raise ValueError("invalid run ID")
    parent = workspace_path(root, input_name)
    if parent.parent != root / "data/prototype1/runs":
        raise ValueError("input must be a resolution snapshot")
    output = parent.parent / run_id
    if output.exists():
        raise FileExistsError(output)
    parent_digest = sha256((parent / "manifest.json").read_bytes())
    config_path = workspace_path(root, config_name)
    config_bytes = config_path.read_bytes()
    config = IdentityRefinementConfig.model_validate_json(config_bytes)
    (base_manifest, extraction, episode_entities, corpus_entities, assignments, base_decisions,
     queue, rules, base_config, mentions, passages) = load_base(parent)
    candidates = build_decisions(run_id, corpus_entities, mentions, assignments, passages, config)
    entities, assignments, decisions, queue, redirects, candidates = refine_records(
        run_id, corpus_entities, episode_entities, assignments, base_decisions, queue, candidates, config)
    # Gold data isn't needed. If the parent run had an evaluation packet, it is scored too.
    evaluation = {"status": "not_evaluated", "reason": "no evaluation packet supplied"}
    if base_manifest["evaluation_packet"]:
        packet = workspace_path(root, base_manifest["evaluation_packet"])
        evaluation = evaluate_resolution(packet, mentions, list(assignments), rules, base_config)
    if sha256((parent / "manifest.json").read_bytes()) != parent_digest or config_path.read_bytes() != config_bytes:
        raise ValueError("parent or identity-refinement configuration changed")
    output.mkdir(parents=True, exist_ok=False)
    artifacts = {"episode_entities.jsonl": episode_entities, "corpus_entities.jsonl": entities,
        "assignments.jsonl": assignments, "decisions.jsonl": decisions, "review_queue.jsonl": queue,
        "aliases.jsonl": rules, "identity_candidates.jsonl": candidates, "redirects.jsonl": redirects}
    for name, rows in artifacts.items():
        dump_jsonl(output / name, list(rows))
    dump_json(output / "config.json", config.model_dump(mode="json"))
    dump_json(output / "base_config.json", base_config.model_dump(mode="json"))
    dump_json(output / "evaluation.json", evaluation)
    summary = summary_of(episode_entities, entities, assignments, decisions, queue, redirects, candidates)
    dump_json(output / "summary.json", summary)
    (output / "review.md").write_text(review_view(candidates), encoding="utf-8")
    names = [*artifacts, "config.json", "base_config.json", "evaluation.json", "summary.json", "review.md"]
    code_root = Path(__file__).parent
    manifest = {"schema_version": 1, "stage": "refine_resolution", "run_id": run_id,
        "status": "complete", "package_version": __version__, "started_utc": now.isoformat(),
        "elapsed_seconds": round(perf_counter() - started, 4), "python": platform.python_version(),
        "dependencies": {"pydantic": importlib.metadata.version("pydantic")},
        "parent_run_id": parent.name, "parent_manifest_sha256": parent_digest,
        "extraction_run_id": extraction.name, "input_config": config_name,
        "input_config_sha256": sha256(config_bytes), "evaluation_packet": base_manifest["evaluation_packet"],
        "counts": summary,
        "code_sha256": {str(path.relative_to(code_root)): sha256(path.read_bytes())
                        for path in sorted(code_root.rglob("*.py"))},
        "artifact_sha256": {name: sha256((output / name).read_bytes()) for name in names}}
    dump_json(output / "manifest.json", manifest)
    validate_run(output)
    return output


def validate_run(output: Path) -> dict:
    output = output.resolve()
    manifest = json.loads((output / "manifest.json").read_bytes())
    if manifest.get("schema_version") != 1 or manifest.get("stage") != "refine_resolution" \
            or manifest.get("status") != "complete":
        raise ValueError("unsupported identity-refinement snapshot")
    root = output.parents[3]
    parent = output.parent / manifest["parent_run_id"]
    if sha256((parent / "manifest.json").read_bytes()) != manifest["parent_manifest_sha256"]:
        raise ValueError("parent resolution manifest changed")
    config_path = workspace_path(root, manifest["input_config"])
    if sha256(config_path.read_bytes()) != manifest["input_config_sha256"]:
        raise ValueError("identity-refinement configuration changed")
    expected_names = {"episode_entities.jsonl", "corpus_entities.jsonl", "assignments.jsonl",
        "decisions.jsonl", "review_queue.jsonl", "aliases.jsonl", "identity_candidates.jsonl",
        "redirects.jsonl", "config.json", "base_config.json", "evaluation.json", "summary.json",
        "review.md"}
    if set(manifest["artifact_sha256"]) != expected_names:
        raise ValueError("unexpected identity-refinement artifact set")
    for name, digest in manifest["artifact_sha256"].items():
        if sha256((output / name).read_bytes()) != digest:
            raise ValueError(f"identity-refinement artifact changed: {name}")
    config = IdentityRefinementConfig.model_validate_json((output / "config.json").read_bytes())
    (base_manifest, _, episode_entities, corpus_entities, base_assignments, base_decisions,
     base_queue, rules, base_config, mentions, passages) = load_base(parent)
    expected_candidates = build_decisions(manifest["run_id"], corpus_entities, mentions,
                                          base_assignments, passages, config)
    expected = refine_records(manifest["run_id"], corpus_entities, episode_entities, base_assignments,
                              base_decisions, base_queue, expected_candidates, config)
    entities, assignments, decisions, queue, redirects, candidates = expected
    model_files = ((CorpusEntity, "corpus_entities.jsonl", entities),
        (MentionAssignment, "assignments.jsonl", assignments),
        (ResolutionDecision, "decisions.jsonl", decisions),
        (ReviewCandidate, "review_queue.jsonl", queue), (EntityRedirect, "redirects.jsonl", redirects),
        (IdentityCandidateDecision, "identity_candidates.jsonl", candidates))
    for model, name, expected_rows in model_files:
        actual = tuple(model.model_validate(row) for row in read_jsonl(output / name))
        if actual != expected_rows:
            raise ValueError(f"identity-refinement records are not reproducible: {name}")
    actual_episode_entities = tuple(EpisodeEntity.model_validate(row)
                                    for row in read_jsonl(output / "episode_entities.jsonl"))
    if actual_episode_entities != tuple(episode_entities):
        raise ValueError("identity refinement changed EpisodeEntities")
    if tuple(AliasRule.model_validate(row) for row in read_jsonl(output / "aliases.jsonl")) != tuple(rules):
        raise ValueError("identity refinement changed reviewed aliases")
    if ResolutionConfig.model_validate_json((output / "base_config.json").read_bytes()) != base_config:
        raise ValueError("identity refinement changed its base resolution configuration")
    entity_ids = {entity.corpus_entity_id for entity in entities}
    if len(assignments) != len(mentions) or {row.mention_id for row in assignments} != {row.mention_id for row in mentions}:
        raise ValueError("every mention must have one refined assignment")
    if any(row.corpus_entity_id not in entity_ids for row in assignments):
        raise ValueError("refined assignment refers to an unknown CorpusEntity")
    summary = summary_of(episode_entities, entities, assignments, decisions, queue, redirects, candidates)
    if summary != manifest["counts"] or summary != json.loads((output / "summary.json").read_bytes()):
        raise ValueError("identity-refinement counts differ")
    if review_view(candidates) != (output / "review.md").read_text(encoding="utf-8"):
        raise ValueError("identity-refinement review is not reproducible")
    evaluation = {"status": "not_evaluated", "reason": "no evaluation packet supplied"}
    if base_manifest["evaluation_packet"]:
        packet = workspace_path(root, base_manifest["evaluation_packet"])
        evaluation = evaluate_resolution(packet, mentions, list(assignments), rules, base_config)
    if evaluation != json.loads((output / "evaluation.json").read_bytes()):
        raise ValueError("identity-refinement evaluation is not reproducible")
    return manifest
