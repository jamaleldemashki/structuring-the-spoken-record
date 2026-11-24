"""Automatic merging of Person identities by name, abstaining when unsure. The merges
are provisional.
"""

import itertools
import math
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher

from spoken_record.adapters.base import stable_id
from spoken_record.extraction_models import ExtractedMention
from spoken_record.identity_models import (EntityRedirect, IdentityCandidateDecision,
                                           IdentityRefinementConfig)
from spoken_record.passage_models import Passage
from spoken_record.resolution import normalize_surface
from spoken_record.resolution_models import (CorpusEntity, EpisodeEntity, MentionAssignment,
                                             ResolutionDecision, ReviewCandidate)


NAME_WORD = re.compile(r"[^\W_]+", re.UNICODE)
QUALIFIERS = frozenset({"of", "from"})
PERSON_DESCRIPTORS = frozenset({"aunt", "brother", "father", "great", "her", "his", "king",
                                "mother", "my", "our", "pope", "queen", "saint", "sister", "st",
                                "their", "uncle", "your"})


WORD = re.compile(r"[^\W\d_]+(?:['’-][^\W\d_]+)*", re.UNICODE)
STOPWORDS = frozenset("a an and are as at be been being but by did do does for from had has have he her hers him "
                      "his i if in into is it its me my no not of on or our she so than that the their them then "
                      "there these they this those to was we were what when which who will with would you your".split())


def tokens(text: str) -> list[str]:
    """Lowercase words for the context comparison, without stopwords and one-letter words."""
    return [token.casefold() for token in WORD.findall(text) if token.casefold() not in STOPWORDS and len(token) > 1]


def cosine(left: Counter, right: Counter, idf: dict[str, float]) -> float:
    """TF-IDF cosine of two word counts."""
    shared = sorted(set(left) & set(right))
    dot = math.fsum(left[t] * right[t] * idf[t] ** 2 for t in shared)
    left_norm = math.sqrt(math.fsum(left[t] ** 2 * idf[t] ** 2 for t in sorted(left)))
    right_norm = math.sqrt(math.fsum(right[t] ** 2 * idf[t] ** 2 for t in sorted(right)))
    value = dot / (left_norm * right_norm) if left_norm and right_norm else 0.0
    return min(1.0, max(0.0, value))


def jaccard(left: set[str], right: set[str]) -> float:
    return len(left & right) / len(left | right) if left or right else 0.0


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
            if len(short) == 1 and len(long) > 1 and short[0] != long[-1]:
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
            candidate_id=stable_id("ic", left_id, right_id, config.model_dump_json()), type=left.type,
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
        # Merges made inside an episode are kept as they are. The size limit is only there to
        # stop name-variant merges across episodes from snowballing.
        too_big = len(combined) > maximum and row.retrieval_methods != ["same_episode_entity"]
        if too_big or conflicts:
            reason = ("merge abstained because transitive closure would exceed the configured component limit"
                      if too_big else
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
