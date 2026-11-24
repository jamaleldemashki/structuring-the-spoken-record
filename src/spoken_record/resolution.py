"""Conservative entity resolution (mention to episode entity to corpus entity), by
normalized name and reviewed aliases.
"""

import json
import unicodedata
from collections import Counter, defaultdict

from spoken_record.adapters.base import sha256, stable_id
from spoken_record.extraction_models import ExtractedMention
from spoken_record.models import Episode
from spoken_record.passage_models import Passage
from spoken_record.terms import term_key
from spoken_record.resolution_models import (AliasRule, CorpusEntity, EpisodeEntity, MentionAssignment,
                                             ResolutionConfig, ResolutionDecision, ReviewCandidate)


def normalize_surface(text: str) -> str:
    """Identity key: NFC, casefold, single spaces. Kept narrow on purpose."""
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def match_key(mention: ExtractedMention, config: ResolutionConfig) -> str:
    """The key that decides whether two mentions are the same thing. For terms it is
    lemmatized (see terms.py), for names it is the normalized surface."""
    return term_key(normalize_surface(mention.text), mention.type, config.term_keys)


def rule_keys(rule: AliasRule) -> set[str]:
    return {normalize_surface(value) for value in [rule.canonical_label, *rule.aliases]}


def rule_applies(rule: AliasRule, mention: ExtractedMention) -> bool:
    if normalize_surface(mention.text) not in rule_keys(rule):
        return False
    if rule.scope == "corpus":
        return True
    if rule.scope == "episode":
        return mention.episode_id in rule.episode_ids
    return mention.mention_id in rule.mention_ids


def select_surface(mentions: list[ExtractedMention]) -> str:
    counts = Counter(m.text for m in mentions)
    max_scores = {surface: max(m.score for m in mentions if m.text == surface) for surface in counts}
    return sorted(counts, key=lambda s: (-counts[s], -max_scores[s], -len(s), normalize_surface(s), s))[0]


def name_mentions(mentions: list[ExtractedMention], config: ResolutionConfig) -> list[ExtractedMention]:
    """The mentions that can be names: without those that are only a word from ``not_names``."""
    words = {normalize_surface(word) for word in config.not_names}
    return [mention for mention in mentions if normalize_surface(mention.text) not in words]


def rules_for(rules: list[AliasRule], mentions: list[ExtractedMention], episodes: list[Episode]) -> list[AliasRule]:
    """The reviewed rules that can apply to the chosen episodes. A rule about episodes or
    mentions that are not in the build is left out, it says nothing about this build."""
    episode_ids = {e.episode_id for e in episodes}
    mention_ids = {m.mention_id for m in mentions}
    return [rule for rule in rules if rule.scope == "corpus"
            or (rule.scope == "episode" and set(rule.episode_ids) & episode_ids)
            or (rule.scope == "mentions" and set(rule.mention_ids) & mention_ids)]


def validate_aliases(rules: list[AliasRule], mentions: list[ExtractedMention], episodes: list[Episode]) -> None:
    """Check the reviewed rules of a build. The passages a rule names as evidence are not
    checked, because they can lie in episodes that are not in the build."""
    if len({r.alias_id for r in rules}) != len(rules):
        raise ValueError("duplicate alias rule ID")
    for rule in rules:
        if rule.corpus_id not in {e.corpus_id for e in episodes}:
            raise ValueError("alias rule uses a different corpus namespace")
        normalized_aliases = [normalize_surface(value) for value in rule.aliases]
        if any(not value for value in normalized_aliases):
            raise ValueError("alias normalizes to an empty key")
        if len(set(normalized_aliases)) != len(normalized_aliases):
            raise ValueError("alias rule has duplicate normalized aliases")
    # A mention can't be covered by two reviewed rules that disagree (merge and block).
    for mention in mentions:
        applicable = [r for r in rules if r.type == mention.type and rule_applies(r, mention)]
        merge_ids = {r.identity_id for r in applicable if r.decision == "merge"}
        if len(merge_ids) > 1 or (merge_ids and any(r.decision == "block" for r in applicable)):
            raise ValueError("conflicting reviewed alias decisions")


def resolve_records(run_id: str, corpus_id: str, mentions: list[ExtractedMention], passages: list[Passage],
                    rules: list[AliasRule], config: ResolutionConfig, fingerprint: str):
    passage_by_id = {p.passage_id: p for p in passages}
    by_episode_group = defaultdict(list)
    mention_merge_rule, mention_block_rules = {}, defaultdict(list)
    for mention in mentions:
        applicable = [r for r in rules if r.type == mention.type and rule_applies(r, mention)]
        merges = [r for r in applicable if r.decision == "merge"]
        blocks = [r for r in applicable if r.decision == "block"]
        if merges:
            mention_merge_rule[mention.mention_id] = merges[0]
            group = (mention.episode_id, mention.type, "alias", merges[0].identity_id)
        elif any(r.scope == "mentions" for r in blocks):
            group = (mention.episode_id, mention.type, "blocked", mention.mention_id)
        else:
            group = (mention.episode_id, mention.type, "exact", match_key(mention, config))
        mention_block_rules[mention.mention_id] = blocks
        by_episode_group[group].append(mention)

    episode_entities, internal = [], {}
    for group, members in sorted(by_episode_group.items()):
        episode_id, entity_type, method, key = group
        aliases = sorted({mention_merge_rule[m.mention_id].alias_id for m in members
                          if m.mention_id in mention_merge_rule})
        normalized = sorted({match_key(m, config) for m in members})
        canonical = mention_merge_rule[members[0].mention_id].canonical_label if method == "alias" else select_surface(members)
        entity_id = stable_id("ee", episode_id, entity_type, method, key)
        passage_ids = sorted({pid for m in members for pid in m.passage_ids},
                             key=lambda pid: (passage_by_id[pid].index, pid))
        status = "reviewed" if method == "alias" else "unresolved" if method == "blocked" else "provisional"
        episode_entities.append(EpisodeEntity(
            created_by_run_id=run_id, episode_entity_id=entity_id, episode_id=episode_id,
            transcript_version_id=members[0].transcript_version_id, type=entity_type,
            canonical_surface=canonical, normalized_keys=normalized,
            mention_ids=sorted(m.mention_id for m in members), passage_ids=passage_ids,
            mention_count=len(members), is_singleton=len(members) == 1,
            mean_score=sum(m.score for m in members) / len(members), max_score=max(m.score for m in members),
            resolution_method={"alias": "reviewed_alias", "blocked": "mention_block"}.get(method, "exact_normalized"),
            resolution_status=status, alias_rule_ids=aliases))
        internal[entity_id] = {"members": members, "method": method, "key": key,
                               "merge_rule": mention_merge_rule.get(members[0].mention_id),
                               "blocks": [r for m in members for r in mention_block_rules[m.mention_id]]}

    by_corpus_group = defaultdict(list)
    for entity in episode_entities:
        info = internal[entity.episode_entity_id]
        if info["method"] == "alias":
            corpus_key = (entity.type, "reviewed", info["key"])
        elif info["blocks"]:
            corpus_key = (entity.type, "blocked", entity.episode_entity_id)
        elif entity.type in config.cross_episode_exact_types:
            corpus_key = (entity.type, "exact", entity.normalized_keys[0])
        elif entity.type == "Person" and ((config.cross_episode_multitoken_person and
                len(entity.normalized_keys) == 1 and len(entity.normalized_keys[0].split()) > 1) or
                not config.single_token_person_requires_review):
            corpus_key = (entity.type, "exact", entity.normalized_keys[0])
        elif entity.type != "Person" and not config.all_other_types_require_review:
            corpus_key = (entity.type, "exact", entity.normalized_keys[0])
        else:
            corpus_key = (entity.type, "local", entity.episode_entity_id)
        by_corpus_group[corpus_key].append(entity)

    corpus_entities, decisions, episode_to_corpus = [], [], {}
    # Note keys whose surface shows up in several episodes. That separates a normal
    # singleton from a case where we chose not to merge across episodes.
    shared_keys = Counter((e.type, key) for e in episode_entities for key in e.normalized_keys)
    for group, members in sorted(by_corpus_group.items()):
        entity_type, policy, key = group
        episode_ids = sorted({e.episode_id for e in members})
        mention_ids = [mid for e in members for mid in e.mention_ids]
        mention_id_set = set(mention_ids)
        member_mentions = [m for m in mentions if m.mention_id in mention_id_set]
        alias_rule = internal[members[0].episode_entity_id]["merge_rule"] if policy == "reviewed" else None
        if policy == "reviewed":
            method, status, registry = "reviewed_alias", "reviewed", f"reviewed:{alias_rule.identity_id}"
            canonical, reviewed_aliases = alias_rule.canonical_label, sorted(set(alias_rule.aliases))
        elif policy == "exact" and len(episode_ids) > 1:
            method, status, registry = "provisional_exact", "provisional", f"exact:{entity_type}:{key}"
            canonical, reviewed_aliases = select_surface(member_mentions), []
        elif policy == "blocked":
            method, status, registry = "blocked", "unresolved", f"blocked:{members[0].episode_entity_id}"
            canonical, reviewed_aliases = members[0].canonical_surface, []
        else:
            method = "episode_local"
            unresolved = any(shared_keys[(entity_type, norm)] > 1 for norm in members[0].normalized_keys)
            status, registry = ("unresolved" if unresolved else "provisional"), f"episode:{members[0].episode_entity_id}"
            canonical, reviewed_aliases = members[0].canonical_surface, []
        corpus_id_value = stable_id("ce", corpus_id, entity_type, registry)
        corpus_entities.append(CorpusEntity(
            created_by_run_id=run_id, corpus_entity_id=corpus_id_value, corpus_id=corpus_id,
            type=entity_type, canonical_label=canonical,
            normalized_keys=sorted({k for e in members for k in e.normalized_keys}),
            observed_surfaces=sorted({m.text for m in member_mentions}, key=lambda s: (normalize_surface(s), s)),
            reviewed_aliases=reviewed_aliases,
            episode_entity_ids=sorted(e.episode_entity_id for e in members), episode_ids=episode_ids,
            mention_count=len(mention_ids), episode_count=len(episode_ids),
            resolution_method=method, resolution_status=status, registry_key=registry))
        for entity in members:
            episode_to_corpus[entity.episode_entity_id] = corpus_id_value
            rule_ids = entity.alias_rule_ids or sorted({r.alias_id for r in internal[entity.episode_entity_id]["blocks"]})
            reason = {"reviewed_alias": "reviewed alias identity",
                      "provisional_exact": "type-allowed exact normalized key across episodes",
                      "blocked": "reviewed rule blocks automatic grouping",
                      "episode_local": "cross-episode merge requires review"}[method]
            decisions.append(ResolutionDecision(
                created_by_run_id=run_id,
                decision_id=stable_id("rd", entity.episode_entity_id, corpus_id_value, fingerprint),
                episode_entity_id=entity.episode_entity_id, corpus_entity_id=corpus_id_value,
                method=method, status=status, alias_rule_ids=rule_ids, reason=reason))

    mention_to_episode = {mid: e.episode_entity_id for e in episode_entities for mid in e.mention_ids}
    assignments = [MentionAssignment(created_by_run_id=run_id, mention_id=m.mention_id,
        episode_entity_id=mention_to_episode[m.mention_id],
        corpus_entity_id=episode_to_corpus[mention_to_episode[m.mention_id]]) for m in mentions]

    candidate_groups = defaultdict(list)
    for entity in episode_entities:
        for key in entity.normalized_keys:
            candidate_groups[(entity.type, key)].append(entity)
    queue = []
    for (entity_type, key), members in sorted(candidate_groups.items()):
        episode_ids = sorted({e.episode_id for e in members})
        corpus_ids = {episode_to_corpus[e.episode_entity_id] for e in members}
        if len(episode_ids) < 2 or len(corpus_ids) == 1:
            continue
        blocks = [r for e in members for r in internal[e.episode_entity_id]["blocks"]]
        reason = "blocked_by_rule" if blocks else "single_token_person" if entity_type == "Person" else "type_requires_review"
        queue.append(ReviewCandidate(
            created_by_run_id=run_id, candidate_id=stable_id("rc", corpus_id, entity_type, key),
            type=entity_type, normalized_key=key,
            observed_surfaces=sorted({m.text for e in members for m in internal[e.episode_entity_id]["members"]}),
            episode_entity_ids=sorted(e.episode_entity_id for e in members), episode_ids=episode_ids,
            mention_count=sum(e.mention_count for e in members), reason=reason))
    return (tuple(sorted(episode_entities, key=lambda r: r.episode_entity_id)),
            tuple(sorted(corpus_entities, key=lambda r: r.corpus_entity_id)),
            tuple(sorted(assignments, key=lambda r: r.mention_id)),
            tuple(sorted(decisions, key=lambda r: r.decision_id)),
            tuple(sorted(queue, key=lambda r: r.candidate_id)))


def resolution_fingerprint(config: ResolutionConfig, rules: list[AliasRule]) -> str:
    payload = {"algorithm": "conservative-resolution-1", "config": config.model_dump(mode="json"),
               "aliases": [r.model_dump(mode="json") for r in rules]}
    return sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode())
