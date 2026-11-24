"""Entities from the bottom up: first inside one episode, then across episodes.

Inside an episode, mentions are grouped into episode entities with the resolution rules
(same name, reviewed aliases) and then the name-variant rules of the refinement step, so
"Aquinas" and "Thomas Aquinas" in one episode become one entity. Pronouns and relations
are decided against these episode entities, so an episode never depends on other episodes.

Across episodes, the same rules run over all mentions, and every merge made inside an
episode is passed in as a fixed decision that comes first. What an episode joined stays
joined. Pronoun links and relations are then lifted from episode entities to corpus
entities.
"""

import itertools
from collections import Counter, defaultdict

from spoken_record.adapters.base import stable_id
from spoken_record.identity_models import EntityRedirect, IdentityCandidateDecision
from spoken_record.identity_refinement import build_decisions, canonical_rank, name_tokens, refine_records
from spoken_record.resolution import normalize_surface
from spoken_record.resolution_models import CorpusEntity
from spoken_record.relations import build_assertions
from spoken_record.resolution import resolve_records


def consolidate_episode(run_id, corpus_id, episode_id, mentions, passages, rules, resolution_config,
                        resolution_fingerprint, refinement_config):
    """Episode entities of one episode. Their IDs start with ``eu-`` and are unique per episode,
    so they can never be mistaken for corpus entities."""
    episode_entities, units, assignments, decisions, queue = resolve_records(
        run_id, corpus_id, mentions, passages, rules, resolution_config, resolution_fingerprint)
    candidates = unambiguous_merges(build_decisions(run_id, list(units), mentions, list(assignments),
                                                    passages, refinement_config)
                                    + first_name_merges(run_id, list(units), list(assignments), mentions, passages))
    units, assignments, _, _, _, _ = refine_records(run_id, list(units), list(episode_entities),
        list(assignments), list(decisions), list(queue), candidates, refinement_config)
    renamed = {unit.corpus_entity_id: stable_id("eu", episode_id, unit.corpus_entity_id) for unit in units}
    units = sorted((unit.model_copy(update={"corpus_entity_id": renamed[unit.corpus_entity_id]}) for unit in units),
                   key=lambda row: row.corpus_entity_id)
    assignments = sorted((row.model_copy(update={"corpus_entity_id": renamed[row.corpus_entity_id]})
                          for row in assignments), key=lambda row: row.mention_id)
    return list(episode_entities), units, assignments


# Words that start an epithet or a place of origin after a first name: "Alexander of Hales",
# "Albert the Great", "Ibn Rushd" is not one of them because it has no first name before.
EPITHET_WORDS = frozenset({"of", "from", "the", "de", "da", "di", "du", "von", "van", "ibn"})


def first_name_merges(run_id, entities, assignments, mentions, passages):
    """Merge decisions for a one-word person name and the one full name in the same episode
    that starts with it and goes on with an epithet or place ("Alexander" and "Alexander of
    Hales", "William" and "William of Ockham").

    The full name has to come first. A podcast introduces a person with the full name and then
    shortens it. If the one-word name comes earlier it is likely someone else: in the Aquinas
    episode "Boethius" is the Roman philosopher, and Boethius of Dacia is only named later.
    With two such full names ("William of Ockham", "William of Auvergne") the one-word name is
    left alone."""
    people = [e for e in entities if e.type == "Person"]
    full = defaultdict(list)
    for entity in people:
        parts = name_tokens(entity.canonical_label)
        if len(parts) >= 3 and parts[1] in EPITHET_WORDS:
            full[parts[0]].append(entity)
    passage_by_id = {row.passage_id: row for row in passages}
    passages_of, first = defaultdict(set), {}
    entity_of = {row.mention_id: row.corpus_entity_id for row in assignments}
    for mention in mentions:
        entity_id = entity_of[mention.mention_id]
        passages_of[entity_id].update(mention.passage_ids)
        first[entity_id] = min(first.get(entity_id, mention.char_start), mention.char_start)
    rows = []
    for entity in people:
        parts = name_tokens(entity.canonical_label)
        if len(parts) != 1 or len(full.get(parts[0], [])) != 1:
            continue
        target = full[parts[0]][0]
        if first[target.corpus_entity_id] > first[entity.corpus_entity_id]:
            continue
        left, right = sorted((entity, target), key=lambda row: row.corpus_entity_id)
        rows.append(IdentityCandidateDecision(created_by_run_id=run_id,
            candidate_id=stable_id("ic", left.corpus_entity_id, right.corpus_entity_id, "person_first_name"),
            type="Person", left_corpus_entity_id=left.corpus_entity_id, right_corpus_entity_id=right.corpus_entity_id,
            left_label=left.canonical_label, right_label=right.canonical_label,
            episode_ids=sorted(set(left.episode_ids) | set(right.episode_ids)),
            retrieval_methods=["person_first_name"], surface_similarity=1.0, lexical_context_similarity=1.0,
            neighbor_similarity=1.0, combined_score=1.0, decision="merge",
            reason="the first name of the one full name with an epithet or place in this episode",
            evidence_passage_ids=sorted(passages_of[left.corpus_entity_id] | passages_of[right.corpus_entity_id],
                key=lambda pid: (passage_by_id[pid].index, pid))))
    return tuple(rows)


def unambiguous_merges(candidates):
    """A name that would merge with two entities that are known to be different people
    ("Bacon" with "Roger Bacon" and "Francis Bacon") merges with neither."""
    partners = defaultdict(set)
    for row in candidates:
        if row.decision == "merge":
            partners[row.left_corpus_entity_id].add(row.right_corpus_entity_id)
            partners[row.right_corpus_entity_id].add(row.left_corpus_entity_id)
    separate = {frozenset((row.left_corpus_entity_id, row.right_corpus_entity_id))
                for row in candidates if row.decision == "separate"}
    ambiguous = {entity for entity, others in partners.items()
                 if any(frozenset(pair) in separate for pair in itertools.combinations(sorted(others), 2))}
    return tuple(row.model_copy(update={"decision": "abstain",
                 "reason": "the shorter name fits more than one person in this episode"})
                 if row.decision == "merge" and ambiguous & {row.left_corpus_entity_id, row.right_corpus_entity_id}
                 else row for row in candidates)


def episode_merges(run_id, entities, assignments, unit_assignments, mentions, passages):
    """Fixed merge decisions for corpus entities whose mentions share one episode entity."""
    corpus_of = {row.mention_id: row.corpus_entity_id for row in assignments}
    entity_by_id = {row.corpus_entity_id: row for row in entities}
    passage_by_id = {row.passage_id: row for row in passages}
    passages_of_mention = {row.mention_id: row.passage_ids for row in mentions}
    groups, evidence = defaultdict(set), defaultdict(set)
    for row in unit_assignments:
        groups[row.corpus_entity_id].add(corpus_of[row.mention_id])
        evidence[row.corpus_entity_id].update(passages_of_mention[row.mention_id])
    rows = []
    for unit_id, corpus_ids in sorted(groups.items()):
        ordered = sorted(corpus_ids)
        for right in ordered[1:]:
            left_entity, right_entity = entity_by_id[ordered[0]], entity_by_id[right]
            rows.append(IdentityCandidateDecision(created_by_run_id=run_id,
                candidate_id=stable_id("ic", ordered[0], right, "same_episode_entity", unit_id),
                type=left_entity.type, left_corpus_entity_id=ordered[0], right_corpus_entity_id=right,
                left_label=left_entity.canonical_label, right_label=right_entity.canonical_label,
                episode_ids=sorted(set(left_entity.episode_ids) | set(right_entity.episode_ids)),
                retrieval_methods=["same_episode_entity"], surface_similarity=1.0,
                lexical_context_similarity=1.0, neighbor_similarity=1.0, combined_score=1.0,
                decision="merge", reason="already one entity inside the episode",
                evidence_passage_ids=sorted(evidence[unit_id], key=lambda pid: (
                    passage_by_id[pid].episode_id, passage_by_id[pid].index, pid))))
    return tuple(rows)


def corpus_identities(run_id, corpus_id, mentions, passages, unit_assignments, rules, resolution_config,
                      resolution_fingerprint, refinement_config):
    """Corpus entities over all episodes, keeping every merge made inside an episode."""
    episode_entities, entities, assignments, decisions, queue = resolve_records(
        run_id, corpus_id, mentions, passages, rules, resolution_config, resolution_fingerprint)
    fixed = episode_merges(run_id, entities, assignments, unit_assignments, mentions, passages)
    joined = {frozenset((row.left_corpus_entity_id, row.right_corpus_entity_id)) for row in fixed}
    candidates = build_decisions(run_id, list(entities), mentions, list(assignments), passages, refinement_config)
    # Bottom up: a corpus-level "separate" can't undo a merge made inside an episode.
    candidates = tuple(row for row in candidates if not (row.decision == "separate" and
                       frozenset((row.left_corpus_entity_id, row.right_corpus_entity_id)) in joined))
    # A one-word name that an episode joined to a full name ("Boethius" to "Boethius of Dacia")
    # stands for that person now. Merging it with the same one-word name from another episode
    # would chain other people into the full name (the Roman Boethius of another episode), so
    # such merges are left out.
    one_word = {e.corpus_entity_id for e in entities if all(len(key.split()) == 1 for key in e.normalized_keys)}
    anchored = {side for row in fixed for side, other in ((row.left_corpus_entity_id, row.right_corpus_entity_id),
                (row.right_corpus_entity_id, row.left_corpus_entity_id)) if side in one_word and other not in one_word}
    candidates = tuple(row.model_copy(update={"decision": "abstain",
                       "reason": "the one-word name stands for a full name in one of the episodes"})
                       if row.decision == "merge" and {row.left_corpus_entity_id, row.right_corpus_entity_id} & anchored
                       else row for row in candidates)
    return refine_records(run_id, list(entities), list(episode_entities), list(assignments), list(decisions),
                          list(queue), fixed + candidates, refinement_config)


def supported_by_namesake(entities, link_decisions, wikidata_candidates, linked_threshold, types, type_weight=0.0):
    """Entities that just missed their own link, but whose best candidate is the item a
    namesake is confidently linked to. Three "Bonaventure"s with one to three mentions each can
    miss the margin or the context floor by a hair, while four others are linked to St.
    Bonaventure.

    An entity joins only when its own score reaches the link threshold, its best candidate is
    that item, and its name is the item's Wikidata label or one of its aliases. The last check
    keeps "Alexander" (here Alexander of Hales) away from Alexander the Great, whose names on
    Wikidata do not include plain "Alexander".

    The namesake can also be of another type, when the entity recognizer typed a single mention
    wrongly: one "Averroes" of 23 was typed as a school, and a person fits "school" badly, which
    alone kept its score (0.68) under the threshold. Such an entity joins when its score reaches
    the threshold with the type penalty taken out, it was heard with exactly the same spelling as
    the linked namesake (so a lowercase idea such as "republic" stays away from Plato's Republic),
    and the namesake has more mentions. It then takes the namesake's type. Two entities that were
    both linked on their own under different types stay apart. Returns {entity ID: (type, QID)}."""
    linked = defaultdict(list)
    for e in entities:
        if e.type in types and e.link_status == "linked" and e.wikidata_qid:
            linked[e.wikidata_qid].append(e)
    by_id = {row.candidate_id: row for row in wikidata_candidates}
    entity_by_id = {e.corpus_entity_id: e for e in entities}
    result = {}
    for decision in link_decisions:
        entity = entity_by_id.get(decision.corpus_entity_id)
        if (entity is None or entity.link_status == "linked" or decision.decision != "ambiguous"
                or decision.confidence is None or not decision.candidate_ids):
            continue
        best = by_id[decision.candidate_ids[0]]
        names = {normalize_surface(value) for value in (best.label, *best.aliases)}
        if not names & set(entity.normalized_keys):
            continue
        if any(e.type == entity.type for e in linked[best.qid]):
            if decision.confidence >= linked_threshold:
                result[entity.corpus_entity_id] = (entity.type, best.qid)
            continue
        penalty = type_weight * (1 - getattr(best, "type_compatibility", 1.0))
        others = [e for e in linked[best.qid] if set(e.observed_surfaces) & set(entity.observed_surfaces)
                  and e.mention_count > entity.mention_count]
        if decision.confidence + penalty >= linked_threshold and len({e.type for e in others}) == 1:
            result[entity.corpus_entity_id] = (others[0].type, best.qid)
    return result


def merge_by_wikidata(run_id, entities, assignments, decisions, redirects, identity_candidates,
                      link_decisions, types, supported=None):
    """Entities of the same type with the same confident Wikidata link become one entity
    ("Summa" and "Summa Theologiae", or Plato in two episodes). ``supported`` adds entities
    that join a linked namesake (see supported_by_namesake). Two entities the name rules
    marked as different people are never joined, even with the same link."""
    supported = supported or {}
    separate = {frozenset((row.left_corpus_entity_id, row.right_corpus_entity_id))
                for row in identity_candidates if row.decision == "separate"}
    decision_of = {row.corpus_entity_id: row.decision_id for row in link_decisions}
    groups = defaultdict(list)
    for entity in entities:
        if entity.type in types and entity.link_status == "linked" and entity.wikidata_qid:
            groups[(entity.type, entity.wikidata_qid)].append(entity)
        elif entity.corpus_entity_id in supported:
            groups[supported[entity.corpus_entity_id]].append(entity)
    target, merged = {}, []
    for (_, qid), members in sorted(groups.items()):
        clusters = []
        # Linked members first, so the survivor is always one that was linked on its own.
        for entity in sorted(members, key=lambda e: (e.link_status == "linked", canonical_rank(e)), reverse=True):
            home = next((cluster for cluster in clusters if not any(
                frozenset((entity.corpus_entity_id, other.corpus_entity_id)) in separate for other in cluster)), None)
            if home is None:
                clusters.append([entity])
            else:
                home.append(entity)
        for cluster in clusters:
            # A group without an entity that was linked on its own is not merged or linked.
            if len(cluster) < 2 or cluster[0].link_status != "linked":
                continue
            survivor = cluster[0]
            for entity in cluster:
                target[entity.corpus_entity_id] = survivor.corpus_entity_id
            merged.append(CorpusEntity(created_by_run_id=run_id, corpus_entity_id=survivor.corpus_entity_id,
                corpus_id=survivor.corpus_id, type=survivor.type, canonical_label=survivor.canonical_label,
                normalized_keys=sorted({key for e in cluster for key in e.normalized_keys}),
                observed_surfaces=sorted({s for e in cluster for s in e.observed_surfaces},
                                         key=lambda value: (normalize_surface(value), value)),
                reviewed_aliases=sorted({a for e in cluster for a in e.reviewed_aliases}),
                episode_entity_ids=sorted({v for e in cluster for v in e.episode_entity_ids}),
                episode_ids=sorted({v for e in cluster for v in e.episode_ids}),
                mention_count=sum(e.mention_count for e in cluster),
                episode_count=len({v for e in cluster for v in e.episode_ids}),
                resolution_method="same_wikidata_item", resolution_status="provisional",
                registry_key=f"wikidata:{qid}", wikidata_qid=qid,
                link_status="linked", link_confidence=max(e.link_confidence or 0 for e in cluster
                                                          if e.link_status == "linked")))
    if not target:
        return entities, assignments, decisions, redirects
    absorbed = {old for old, new in target.items() if old != new}
    survivors = {row.corpus_entity_id: row for row in merged}
    entities = sorted([survivors.get(e.corpus_entity_id, e) for e in entities if e.corpus_entity_id not in absorbed],
                      key=lambda row: row.corpus_entity_id)

    def moved(value):
        return target.get(value, value)

    assignments = [row.model_copy(update={"corpus_entity_id": moved(row.corpus_entity_id)}) for row in assignments]
    decisions = [row.model_copy(update={"corpus_entity_id": moved(row.corpus_entity_id)}) for row in decisions]
    redirects = [row.model_copy(update={"to_corpus_entity_id": moved(row.to_corpus_entity_id)}) for row in redirects]
    redirects += [EntityRedirect(created_by_run_id=run_id, from_corpus_entity_id=old, to_corpus_entity_id=new,
                                 candidate_ids=sorted({decision_of[old], decision_of[new]}), reason="same_wikidata_item")
                  for old, new in sorted(target.items()) if old != new]
    return entities, assignments, decisions, sorted(redirects, key=lambda row: row.from_corpus_entity_id)


def unit_map(unit_assignments, assignments) -> dict[str, str]:
    """Corpus entity of every episode entity: the one most of its mentions went to."""
    corpus_of = {row.mention_id: row.corpus_entity_id for row in assignments}
    votes = defaultdict(Counter)
    for row in unit_assignments:
        votes[row.corpus_entity_id][corpus_of[row.mention_id]] += 1
    return {unit: max(sorted(counts), key=lambda value: counts[value]) for unit, counts in votes.items()}


def lift_coreference(links, units):
    return [row.model_copy(update={"corpus_entity_id": units[row.corpus_entity_id]})
            if row.corpus_entity_id else row for row in links]


def lift_relations(run_id, candidates, units):
    """Relation candidates between corpus entities, grouped into assertions.

    Both directions of a proposal stay, the NLI check decides between them.
    """
    lifted = []
    for row in candidates:
        head, tail = units[row.head_corpus_entity_id], units[row.tail_corpus_entity_id]
        update = {"head_corpus_entity_id": head, "tail_corpus_entity_id": tail}
        if row.decision == "accepted" and head == tail:
            update.update(decision="self_relation", reason="both mentions resolve to the same CorpusEntity")
        lifted.append(row.model_copy(update=update))
    lifted = tuple(sorted(lifted, key=lambda row: row.candidate_id))
    return lifted, build_assertions(run_id, list(lifted))
