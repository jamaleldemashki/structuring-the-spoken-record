"""Concept and school navigation, built from named mentions. No new identity claims.

A pair only means that two existing entities are mentioned in the same passage. Repeated
mentions and overlapping windows don't increase the passage count.
"""

from collections import defaultdict

from spoken_record.extraction_models import ExtractedMention
from spoken_record.passage_models import Passage
from spoken_record.resolution_models import CorpusEntity, MentionAssignment


def build_concept_navigation(passages: list[Passage], mentions: list[ExtractedMention],
                             entities: list[CorpusEntity],
                             assignments: list[MentionAssignment]) -> dict:
    """Occurrences and shared-passage pairs, always in the same order.

    Only named mentions count. Coreference and relation predictions have their own rules and
    are not used as evidence here. All pairs are kept, the vault decides how many to show.
    """
    passage_by_id = {p.passage_id: p for p in passages}
    entity_by_id = {e.corpus_entity_id: e for e in entities}
    assignment_by_id = {a.mention_id: a for a in assignments}
    if (len(passage_by_id) != len(passages) or len(entity_by_id) != len(entities)
            or len(assignment_by_id) != len(assignments)
            or len({m.mention_id for m in mentions}) != len(mentions)):
        raise ValueError("navigation requires unique source record IDs")
    if set(assignment_by_id) != {m.mention_id for m in mentions}:
        raise ValueError("navigation requires exactly one assignment per mention")
    passage_members = defaultdict(lambda: defaultdict(set))
    entity_passages = defaultdict(set)
    for mention in mentions:
        entity_id = assignment_by_id[mention.mention_id].corpus_entity_id
        if entity_id not in entity_by_id:
            raise ValueError("navigation assignment references missing entity")
        for passage_id in set(mention.passage_ids):
            passage = passage_by_id.get(passage_id)
            if (passage is None or passage.episode_id != mention.episode_id
                    or passage.transcript_version_id != mention.transcript_version_id
                    or max(passage.char_start, mention.char_start) >= min(passage.char_end, mention.char_end)):
                raise ValueError("navigation mention has invalid passage provenance")
            passage_members[passage_id][entity_id].add(mention.mention_id)
            entity_passages[entity_id].add(passage_id)

    topics = sorted(e.corpus_entity_id for e in entities if e.type in {"Concept", "School"})
    pairs = defaultdict(list)
    entries = []
    for topic_id in topics:
        pids = sorted(entity_passages[topic_id])
        entries.append({"corpus_entity_id": topic_id, "entity_type": entity_by_id[topic_id].type,
            "passage_ids": pids, "episode_ids": sorted({passage_by_id[p].episode_id for p in pids}),
            "mention_ids": sorted({mid for p in pids for mid in passage_members[p][topic_id]})})
        for pid in pids:
            for other_id, mids in sorted(passage_members[pid].items()):
                if entity_by_id[other_id].type not in {"Person", "Work"}:
                    continue
                pairs[(topic_id, other_id)].append({"passage_id": pid,
                    "episode_id": passage_by_id[pid].episode_id,
                    "topic_mention_ids": sorted(passage_members[pid][topic_id]),
                    "other_mention_ids": sorted(mids)})
    connections = [{"topic_entity_id": topic, "other_entity_id": other,
        "passage_count": len(evidence),
        "episode_count": len({row["episode_id"] for row in evidence}), "evidence": evidence}
        for (topic, other), evidence in sorted(pairs.items())]
    counts = {"concept_entities": sum(e["entity_type"] == "Concept" for e in entries),
        "school_entities": sum(e["entity_type"] == "School" for e in entries),
        "topics_with_evidence": sum(bool(e["passage_ids"]) for e in entries),
        "cross_episode_topics": sum(len(e["episode_ids"]) > 1 for e in entries),
        "topics_without_people_or_works": len(topics) - len({row["topic_entity_id"] for row in connections}),
        "shared_passage_pairs": len(connections),
        "pair_passage_evidence": sum(row["passage_count"] for row in connections)}
    return {"schema_version": 1, "method": "named_mention_shared_passage",
        "source_layer": "derived_navigation", "status": "unreviewed_mentions",
        "counts": counts, "topics": entries, "connections": connections}
