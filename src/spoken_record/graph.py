"""The knowledge graph of a build: ``graph.json``.

The graph holds the facts of a build with their evidence, and nothing about how the app
shows them:

* episodes with their metadata and facets
* passages: exact character ranges of the transcript, with times and text, and the timing
  of each transcript segment inside the passage (so the app can follow the audio)
* nodes: one per corpus entity, with its names, Wikidata link, lifespan, occupations, image
  and every mention (passage and character range inside it); mentions in boilerplate are
  only counted, and generic terms are marked (see filters.py)
* relations: every relation GLiREL found between two entities, with the NLI decision and
  every piece of evidence, also the ones the NLI check withheld
* co-occurrence: every pair of entities named in the same passages, with the count and PMI
* related passages: suggestions from both ranking methods
* boilerplate: the sentences that were not used as evidence

The web export (web_export.py) and later an RDF export read only this file.
"""

import itertools
import json
import math
from collections import defaultdict

from spoken_record.adapters.base import stable_id
from spoken_record.filters import inside

GRAPH_VERSION = 1


def primary_passage(mention, passages_by_id: dict) -> str:
    """The passage that contains the first character of the mention."""
    for pid in mention.passage_ids:
        p = passages_by_id[pid]
        if p.char_start <= mention.char_start < p.char_end:
            return pid
    raise ValueError(f"mention {mention.mention_id} starts outside its recorded passages")


def occurrences(passages, mentions, assignments, boilerplate=None, skipped=None) -> dict[str, list[dict]]:
    """All mentions of each entity, as ranges inside their passage (cut at its end). Mentions
    in boilerplate are left out and counted in ``skipped``."""
    by_id = {p.passage_id: p for p in passages}
    entity_of = {a.mention_id: a.corpus_entity_id for a in assignments}
    result = defaultdict(list)
    for m in mentions:
        if boilerplate and inside(boilerplate.get(m.transcript_version_id, []), m.char_start, m.char_end):
            skipped[entity_of[m.mention_id]] += 1
            continue
        pid = primary_passage(m, by_id)
        p = by_id[pid]
        result[entity_of[m.mention_id]].append(
            {"passage_id": pid, "start": m.char_start - p.char_start, "end": min(m.char_end, p.char_end) - p.char_start})
    for rows in result.values():
        rows.sort(key=lambda r: (r["passage_id"], r["start"], r["end"]))
    return result


def co_occurrence(found: dict[str, list[dict]], passage_count: int) -> list[dict]:
    """Every pair of entities that share a passage, with the shared passages and the PMI.

    PMI compares how often two entities share a passage with what chance would give, so
    generic pairs rank below specific ones. It only describes co-occurrence, not a relation.
    """
    passages_of = {cid: {o["passage_id"] for o in rows} for cid, rows in found.items()}
    members = defaultdict(set)
    for cid, pids in passages_of.items():
        for pid in pids:
            members[pid].add(cid)
    shared = defaultdict(set)
    for pid, cids in members.items():
        for a, b in itertools.combinations(sorted(cids), 2):
            shared[(a, b)].add(pid)
    edges = []
    for (a, b), pids in sorted(shared.items()):
        pmi = math.log(len(pids) * passage_count / (len(passages_of[a]) * len(passages_of[b])))
        edges.append({"id": stable_id("wc", a, b), "source": a, "target": b, "count": len(pids),
                      "pmi": round(pmi, 4), "evidence": sorted(pids)})
    return edges


def relations(candidates, assertions, evidence, decisions, passages, boilerplate) -> list[dict]:
    """Every relation with its NLI decision. Each piece of evidence keeps its own NLI result,
    and says whether it lies in boilerplate."""
    transcript_of = {p.passage_id: p.transcript_version_id for p in passages}

    def in_boilerplate(row):
        spans = boilerplate.get(transcript_of[row.passage_id], [])
        return inside(spans, row.head_char_start, row.head_char_end) or inside(spans, row.tail_char_start, row.tail_char_end)

    check = {row.candidate_id: row for row in evidence}
    decision_of = {row.assertion_id: row for row in decisions}
    by_triple = defaultdict(list)
    for row in candidates:
        if row.decision == "accepted":
            by_triple[(row.head_corpus_entity_id, row.canonical_predicate, row.tail_corpus_entity_id)].append(row)
    result = []
    for assertion in assertions:
        triple = (assertion.subject_corpus_entity_id, assertion.predicate, assertion.object_corpus_entity_id)
        rows = sorted(by_triple[triple], key=lambda row: row.candidate_id)
        items = [{"candidate_id": row.candidate_id, "episode_id": row.episode_id, "passage_id": row.passage_id,
                  "context_start": row.context_char_start, "context_end": row.context_char_end,
                  "subject_text": row.head_text, "object_text": row.tail_text,
                  "subject_kind": row.head_endpoint_kind, "object_kind": row.tail_endpoint_kind,
                  "score": row.score, "nli": check[row.candidate_id].decision,
                  "nli_reason": check[row.candidate_id].reason, "boilerplate": in_boilerplate(row)} for row in rows]
        supported = [item["score"] for item in items if item["nli"] == "supported"]
        decision = decision_of[assertion.assertion_id]
        result.append({"id": assertion.assertion_id, "source": triple[0], "predicate": triple[1],
                       "target": triple[2], "decision": decision.decision, "reason": decision.reason,
                       "confidence": max(item["score"] for item in items),
                       "supported_confidence": max(supported) if supported else None, "evidence": items})
    return result


def timing(passage, segment_by_id) -> list[list[float]]:
    """The transcript segments of a passage as ``[from, to, start_s, end_s]``: their character
    range relative to the passage and their times. Whisper only gives times per segment (a few
    seconds of speech), not per word, so this is as fine as the source goes. A segment can
    begin in the previous passage or go on into the next one, then ``from`` is below 0 or
    ``to`` is past the end. Keeping the full range lets the app split the segment's time at
    the passage boundary in the right place."""
    rows = []
    for segment_id in passage.source_segment_ids:
        segment = segment_by_id.get(segment_id)
        if segment is None or segment.start_s is None or segment.end_s is None:
            continue
        rows.append([segment.char_start - passage.char_start, segment.char_end - passage.char_start,
                     round(segment.start_s, 2), round(segment.end_s, 2)])
    return rows


def node(entity, found, facts, skipped, generic) -> dict:
    row = {"id": entity.corpus_entity_id, "type": entity.type, "label": entity.canonical_label,
           "aliases": sorted(set(entity.observed_surfaces) - {entity.canonical_label}),
           "qid": facts.qid if facts else entity.wikidata_qid, "link_status": entity.link_status,
           "link_confidence": entity.link_confidence, "resolution_method": entity.resolution_method,
           "resolution_status": entity.resolution_status, "mention_count": entity.mention_count,
           "episode_ids": sorted(entity.episode_ids), "mentions": found.get(entity.corpus_entity_id, []),
           "boilerplate_mentions": skipped.get(entity.corpus_entity_id, 0), "generic": entity.corpus_entity_id in generic}
    if facts is not None:
        row["facts"] = {"qid_source": facts.qid_source, "is_human": facts.is_human,
                        "life": facts.life.model_dump() if facts.life else None,
                        "occupation_qids": facts.occupation_qids, "image": facts.image}
    return row


def build_graph(build_id, corpus_id, episodes, passages, mentions, entities, assignments, relation_candidates,
                assertions, verification_evidence, verification_decisions, facts, connections,
                boilerplate=None, generic=None, segments=None) -> dict:
    boilerplate, generic = boilerplate or {}, generic or set()
    segment_by_id = {segment.segment_id: segment for segment in segments or []}
    skipped = defaultdict(int)
    found = occurrences(passages, mentions, assignments, boilerplate, skipped)
    count = defaultdict(int)
    for p in passages:
        count[p.episode_id] += 1
    facts_of = {row.corpus_entity_id: row for row in facts}
    episode_rows = [{"id": e.episode_id, "title": e.title, "source_key": e.source_key,
                     "publication_date": e.publication_date, "audio_url": e.audio_url, "source_url": e.source_url,
                     "facets": e.facets, "passage_count": count[e.episode_id]}
                    for e in sorted(episodes, key=lambda e: e.episode_id)]
    passage_rows = [{"id": p.passage_id, "episode_id": p.episode_id, "index": p.index, "char_start": p.char_start,
                     "char_end": p.char_end, "start_s": p.start_s, "end_s": p.end_s, "word_count": p.word_count,
                     "text": p.text, "timing": timing(p, segment_by_id)} for p in sorted(passages, key=lambda p: (p.episode_id, p.index))]
    nodes = [node(e, found, facts_of.get(e.corpus_entity_id), skipped, generic)
             for e in sorted(entities, key=lambda e: e.corpus_entity_id)]
    episode_of = {p.transcript_version_id: p.episode_id for p in passages}
    related = sorted(({"source": c["source_passage_id"], "target": c["target_passage_id"], "method": c["method"],
                       "score": c["score"], "rank": c["rank"], "shared_entity_ids": sorted(c["shared_entity_ids"])}
                      for c in connections if c["decision"] == "selected"),
                     key=lambda r: (r["method"], r["source"], r["rank"], r["target"]))
    graph = {"graph_version": GRAPH_VERSION, "build_id": build_id, "corpus_id": corpus_id,
             "episodes": episode_rows, "passages": passage_rows, "nodes": nodes,
             "relations": relations(relation_candidates, assertions, verification_evidence, verification_decisions,
                                    passages, boilerplate),
             "co_occurrence": co_occurrence(found, len(passages)), "related": related,
             "boilerplate": [{"episode_id": episode_of[tv], "char_start": start, "char_end": end}
                             for tv, spans in sorted(boilerplate.items()) for start, end in spans]}
    graph["counts"] = {"episodes": len(episode_rows), "passages": len(passage_rows), "nodes": len(nodes),
                       "relations": len(graph["relations"]),
                       "nli_supported_relations": sum(r["decision"] == "supported" for r in graph["relations"]),
                       "co_occurrence_pairs": len(graph["co_occurrence"]), "related_passages": len(related),
                       "generic_nodes": sum(n["generic"] for n in nodes),
                       "boilerplate_sentences": len(graph["boilerplate"]),
                       "boilerplate_mentions": sum(skipped.values())}
    return graph


def encode(graph: dict) -> bytes:
    """Written the same way every time, so a rebuild can be compared byte for byte."""
    return (json.dumps(graph, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
