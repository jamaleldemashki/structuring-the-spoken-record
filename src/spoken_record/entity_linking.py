"""Links corpus entities to Wikidata: search, ranking with passage context, and abstention.

Local identity resolution stays in charge. This stage only adds a link decision to each
eligible CorpusEntity. Two local entities are never merged just because the search
returned the same item for both.
"""

import importlib.metadata
import json
import math
import platform
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from time import perf_counter
from typing import Callable
from uuid import uuid4

from spoken_record import __version__
from spoken_record.backends import CandidateRetriever, TextEncoder
from spoken_record.adapters.base import sha256, stable_id
from spoken_record.extraction_models import ExtractedMention
from spoken_record.ingestion import dump_json, dump_jsonl, read_jsonl, workspace_path
from spoken_record.linking_models import EntityLinkingConfig, LinkDecision, WikidataCandidate
from spoken_record.passage_models import Passage
from spoken_record.resolution import normalize_surface, validate_run as validate_resolution
from spoken_record.resolution_models import (AliasRule, CorpusEntity, EpisodeEntity,
                                             MentionAssignment, ResolutionDecision,
                                             ReviewCandidate)


# P31 (instance of) QIDs are used when present. For subclasses we fall back on words in
# the description, loading the whole Wikidata class tree would be too much here.
DIRECT_INSTANCE_TYPES = {
    "Person": {"Q5"},
    "Language": {"Q34770"},
    "Place": {"Q515", "Q6256", "Q82794", "Q486972", "Q123705", "Q618123",
              "Q1549591", "Q200250", "Q51929311", "Q747074", "Q902814"},
    "Work": {"Q571", "Q7725634", "Q47461344", "Q17537576"},
    "School": {"Q11862829", "Q43229", "Q2385804", "Q9174"},
    "Group": {"Q16334295", "Q41710", "Q2472587"},
    "Concept": {"Q151885", "Q7184903"},
    "Period": {"Q11514315", "Q186081"},
}

# English description words per type, used for every podcast. A podcast about musicians,
# for example, can add its own words with `extra_type_keywords`.
TYPE_KEYWORDS = {
    "Person": {"person", "philosopher", "theologian", "writer", "historian", "scholar", "poet",
               "politician", "cleric", "priest", "professor", "activist", "leader", "saint"},
    "Language": {"language", "dialect", "linguistic", "creole"},
    "Place": {"city", "country", "state", "region", "town", "village", "place", "island", "capital",
              "district", "municipality", "continent", "kingdom", "empire"},
    "Work": {"book", "work", "treatise", "text", "dialogue", "manuscript", "publication", "writing"},
    "School": {"school", "movement", "tradition", "religion", "church", "university", "theology",
               "philosophy", "doctrine", "denomination"},
    "Group": {"people", "group", "ethnic", "community", "population", "organization", "tribe"},
    "Concept": {"concept", "theory", "idea", "principle", "doctrine", "term", "notion"},
    "Period": {"period", "era", "century", "age", "year", "epoch"},
}

PERSON_CONTRADICTIONS = {"album", "book", "city", "country", "film", "language", "river", "song",
                         "species", "television series", "village"}


class WikidataRetriever:
    """Small client for the Wikidata (MediaWiki) API, with retries."""

    def __init__(self, config: EntityLinkingConfig):
        import requests

        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            # Wikimedia asks bots for a descriptive User-Agent with a contact page.
            # The repository URL is our contact page.
            "User-Agent": (f"structuring-the-spoken-record/{__version__} "
                           "(https://github.com/jamaleldemashki/structuring-the-spoken-record; "
                           "academic entity-linking prototype)"),
            "Api-User-Agent": (f"structuring-the-spoken-record/{__version__} "
                               "(https://github.com/jamaleldemashki/structuring-the-spoken-record)"),
        })

    def _get(self, params: dict) -> dict:
        """GET one JSON response, retrying on temporary API or network errors."""
        last_error: Exception | None = None
        for attempt in range(self.config.retries):
            wait = min(2 ** attempt, 4)
            try:
                response = self.session.get(self.config.api_url, params=params,
                                            timeout=self.config.request_timeout_seconds)
                response.raise_for_status()
                payload = response.json()
                if "error" in payload:
                    # With maxlag set, Wikimedia returns an error while its replicas lag behind
                    # and says in Retry-After how long to wait.
                    if payload["error"].get("code") == "maxlag":
                        wait = max(wait, float(response.headers.get("Retry-After", 5)))
                    raise RuntimeError(f"Wikidata API error: {payload['error']}")
                if self.config.request_delay_seconds:
                    time.sleep(self.config.request_delay_seconds)
                return payload
            except Exception as exc:  # requests has several network error classes
                last_error = exc
                if attempt + 1 < self.config.retries:
                    time.sleep(wait)
        raise RuntimeError(f"Wikidata request failed after {self.config.retries} attempts") from last_error

    def search(self, query: str) -> list[dict]:
        """Top search matches for one local entity label."""
        payload = self._get({"action": "wbsearchentities", "search": query,
            "language": self.config.language, "uselang": self.config.language,
            "type": "item", "limit": self.config.top_k, "format": "json", "maxlag": 5})
        return [{"qid": row["id"], "label": row.get("label", ""),
                 "description": row.get("description", ""),
                 "aliases": list(row.get("aliases", [])), "retrieval_rank": rank}
                for rank, row in enumerate(payload.get("search", []), 1)
                if re.fullmatch(r"Q[1-9][0-9]*", row.get("id", ""))]

    def details(self, qids: list[str]) -> dict[str, dict]:
        """Fetch aliases and instance-of (P31) claims, in batches the API accepts."""
        result: dict[str, dict] = {}
        for start in range(0, len(qids), 50):
            batch = qids[start:start + 50]
            payload = self._get({"action": "wbgetentities", "ids": "|".join(batch),
                "props": "labels|descriptions|aliases|claims", "languages": self.config.language,
                "languagefallback": 1, "format": "json", "maxlag": 5})
            for qid, row in payload.get("entities", {}).items():
                aliases = [value.get("value", "") for value in row.get("aliases", {}).get(
                    self.config.language, [])]
                instance_of = []
                for claim in row.get("claims", {}).get("P31", []):
                    value = claim.get("mainsnak", {}).get("datavalue", {}).get("value", {})
                    if isinstance(value, dict) and re.fullmatch(r"Q[1-9][0-9]*", value.get("id", "")):
                        instance_of.append(value["id"])
                result[qid] = {"aliases": aliases, "instance_of_qids": sorted(set(instance_of))}
        return result

    def retrieve(self, entities: list[CorpusEntity]) -> dict[str, list[dict]]:
        """Search every eligible entity, then fetch all distinct candidates in batches."""
        raw = {entity.corpus_entity_id: self.search(entity.canonical_label) for entity in entities}
        qids = sorted({row["qid"] for rows in raw.values() for row in rows}, key=lambda q: int(q[1:]))
        details = self.details(qids) if qids else {}
        for rows in raw.values():
            for row in rows:
                detail = details.get(row["qid"], {})
                row["aliases"] = list(dict.fromkeys([*row["aliases"], *detail.get("aliases", [])]))[:50]
                row["instance_of_qids"] = detail.get("instance_of_qids", [])
        return raw


class MpnetEncoder:
    """Text encoder: all-mpnet-base-v2 at a pinned revision, with its usual mean pooling."""

    def __init__(self, config: EntityLinkingConfig):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = torch.device(config.embedding_device)
        self.batch_size = config.embedding_batch_size
        self.max_tokens = config.embedding_max_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(config.embedding_repo_id,
                                                       revision=config.embedding_revision)
        self.model = AutoModel.from_pretrained(config.embedding_repo_id,
                                               revision=config.embedding_revision)
        self.model.to(self.device)
        self.model.eval()

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Normalized embeddings, so the dot product is the cosine similarity."""
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            encoded = self.tokenizer(batch, padding=True, truncation=True,
                                     max_length=self.max_tokens, return_tensors="pt")
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with self.torch.no_grad():
                output = self.model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).expand(output.size()).float()
            pooled = (output * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            pooled = self.torch.nn.functional.normalize(pooled, p=2, dim=1)
            vectors.extend(pooled.cpu().tolist())
        return vectors


def name_similarity(entity: CorpusEntity, candidate: dict) -> float:
    """Compare each name seen in the transcripts with the candidate's label and aliases."""
    local = [entity.canonical_label, *entity.reviewed_aliases, *entity.observed_surfaces]
    external = [candidate["label"], *candidate.get("aliases", [])]
    best = 0.0
    for left in local:
        for right in external:
            a, b = normalize_surface(left), normalize_surface(right)
            if not a or not b:
                continue
            best = max(best, 1.0 if a == b else SequenceMatcher(None, a, b).ratio())
    return best


def type_compatibility(entity_type: str, candidate: dict, keyword_score: float = 1.0,
                       extra_keywords: dict[str, list[str]] | None = None) -> float:
    """Soft type score from P31 and description words. The full class tree is not walked."""
    qids = set(candidate.get("instance_of_qids", []))
    if qids & DIRECT_INSTANCE_TYPES.get(entity_type, set()):
        return 1.0
    description = normalize_surface(candidate.get("description", ""))
    if any(re.search(rf"\b{re.escape(keyword)}\b", description)
           for keyword in TYPE_KEYWORDS.get(entity_type, set()) | set((extra_keywords or {}).get(entity_type, []))):
        return keyword_score
    if entity_type == "Person" and any(word in description for word in PERSON_CONTRADICTIONS):
        return 0.0
    # No type evidence is neutral, not a rejection. P31 often names a subclass (for example
    # "ancient Greek philosopher") that is not in our small map.
    return .5


def build_contexts(entities: list[CorpusEntity], mentions: list[ExtractedMention],
                   assignments: list[MentionAssignment], passages: list[Passage],
                   maximum_characters: int) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Collect a limited amount of passage context for each local entity."""
    assignment_by_mention = {row.mention_id: row.corpus_entity_id for row in assignments}
    passage_by_id = {row.passage_id: row for row in passages}
    entity_passages: dict[str, set[str]] = defaultdict(set)
    for mention in mentions:
        entity_id = assignment_by_mention[mention.mention_id]
        entity_passages[entity_id].update(mention.passage_ids)
    contexts, evidence = {}, {}
    for entity in entities:
        ordered = sorted(entity_passages[entity.corpus_entity_id],
                         key=lambda pid: (passage_by_id[pid].episode_id, passage_by_id[pid].index, pid))
        pieces, used, size = [], [], 0
        for passage_id in ordered:
            text = " ".join(passage_by_id[passage_id].text.split())
            remaining = maximum_characters - size
            if remaining <= 0:
                break
            pieces.append(text[:remaining])
            used.append(passage_id)
            size += min(len(text), remaining)
        contexts[entity.corpus_entity_id] = (
            f"Entity: {entity.canonical_label}. Expected type: {entity.type}. Podcast evidence: "
            + " ".join(pieces))
        evidence[entity.corpus_entity_id] = used
    return contexts, evidence


def candidate_document(candidate: dict) -> str:
    """Short text for a candidate (label and description) to compare with the transcript."""
    aliases = ", ".join(candidate.get("aliases", [])[:12])
    result = f"{candidate['label']}. {candidate.get('description', '')}."
    return result + (f" Also known as {aliases}." if aliases else "")


def dot(left: list[float], right: list[float]) -> float:
    """Cosine of two normalized vectors, clamped to the configured range."""
    return min(1.0, max(0.0, math.fsum(a * b for a, b in zip(left, right))))


def score_candidates(run_id: str, entities: list[CorpusEntity], raw: dict[str, list[dict]],
                     contexts: dict[str, str], config: EntityLinkingConfig,
                     encoder: TextEncoder) -> tuple[WikidataCandidate, ...]:
    """Rank the candidates by name match, type and MPNet context similarity."""
    ordered_entities = sorted(entities, key=lambda row: row.corpus_entity_id)
    pairs = [(entity, candidate) for entity in ordered_entities
             for candidate in raw.get(entity.corpus_entity_id, [])]
    texts = [contexts[entity.corpus_entity_id] for entity in ordered_entities]
    texts.extend(candidate_document(candidate) for _, candidate in pairs)
    vectors = encoder.encode(texts) if texts else []
    # zip() would quietly drop candidates if the adapter returned the wrong number of rows.
    if len(vectors) != len(texts) or (vectors and (not vectors[0] or any(
            len(vector) != len(vectors[0]) or not all(math.isfinite(value) for value in vector)
            for vector in vectors))):
        raise ValueError("encoder must return one finite, equal-dimensional vector per text")
    context_vectors = {entity.corpus_entity_id: vectors[index]
                       for index, entity in enumerate(ordered_entities)}
    candidate_vectors = vectors[len(ordered_entities):]
    rows = []
    for (entity, candidate), vector in zip(pairs, candidate_vectors):
        surface = name_similarity(entity, candidate)
        kind = type_compatibility(entity.type, candidate, config.keyword_type_score, config.extra_type_keywords)
        context = dot(context_vectors[entity.corpus_entity_id], vector)
        score = min(1.0, max(0.0, config.name_weight * surface +
                            config.type_weight * kind + config.context_weight * context))
        rows.append(WikidataCandidate(created_by_run_id=run_id,
            candidate_id=stable_id("wc", entity.corpus_entity_id, candidate["qid"]),
            corpus_entity_id=entity.corpus_entity_id, query=entity.canonical_label,
            qid=candidate["qid"], label=candidate["label"],
            description=candidate.get("description", ""), aliases=candidate.get("aliases", []),
            instance_of_qids=candidate.get("instance_of_qids", []),
            retrieval_rank=candidate["retrieval_rank"], name_similarity=surface,
            type_compatibility=kind, context_similarity=context, rank_score=score))
    return tuple(sorted(rows, key=lambda row: (row.corpus_entity_id, -row.rank_score,
                                               row.retrieval_rank, row.qid)))


def rerank_cached_candidates(run_id: str, entities: list[CorpusEntity],
                             cached: list[WikidataCandidate],
                             config: EntityLinkingConfig) -> tuple[WikidataCandidate, ...]:
    """Apply a (new) decision policy to saved search and embedding results."""
    entity_by_id = {row.corpus_entity_id: row for row in entities}
    rows = []
    for previous in cached:
        entity = entity_by_id.get(previous.corpus_entity_id)
        if entity is None:
            raise ValueError("candidate cache refers to a different resolution snapshot")
        raw = {"label": previous.label, "aliases": previous.aliases,
               "description": previous.description, "instance_of_qids": previous.instance_of_qids}
        surface = name_similarity(entity, raw)
        kind = type_compatibility(entity.type, raw, config.keyword_type_score, config.extra_type_keywords)
        score = min(1.0, max(0.0, config.name_weight * surface + config.type_weight * kind +
                            config.context_weight * previous.context_similarity))
        rows.append(previous.model_copy(update={"created_by_run_id": run_id,
            "name_similarity": surface, "type_compatibility": kind, "rank_score": score}))
    return tuple(sorted(rows, key=lambda row: (row.corpus_entity_id, -row.rank_score,
                                               row.retrieval_rank, row.qid)))


def make_decisions(run_id: str, entities: list[CorpusEntity], candidates: list[WikidataCandidate],
                   config: EntityLinkingConfig) -> tuple[LinkDecision, ...]:
    """Link, or abstain when the score or the lead over the runner-up is too small."""
    by_entity: dict[str, list[WikidataCandidate]] = defaultdict(list)
    for row in candidates:
        by_entity[row.corpus_entity_id].append(row)
    rows = []
    new_policy_fields = {"keyword_type_score", "nil_threshold", "single_candidate_context_floor",
                         "single_token_person_context_floor"}
    legacy_policy = (config.keyword_type_score == 1 and config.nil_threshold == 0 and
                     config.single_candidate_context_floor == 0 and
                     config.single_token_person_context_floor == 0)
    # Newer settings only go into the fingerprint when they are used, so older runs still validate.
    unused = set() if config.extra_type_keywords else {"extra_type_keywords"}
    policy_json = config.model_dump_json(exclude=(new_policy_fields if legacy_policy else set()) | unused or None)
    for entity in sorted(entities, key=lambda row: row.corpus_entity_id):
        ranked = sorted(by_entity[entity.corpus_entity_id],
                        key=lambda row: (-row.rank_score, row.retrieval_rank, row.qid))
        ids = [row.candidate_id for row in ranked]
        if not ranked:
            decision, selected, confidence, margin = "nil", None, None, None
            reason = "Wikidata search returned no item candidates"
        else:
            best = ranked[0]
            margin = best.rank_score - ranked[1].rank_score if len(ranked) > 1 else best.rank_score
            local_single_token_person = (entity.type == "Person" and
                                         len(normalize_surface(entity.canonical_label).split()) == 1 and
                                         len(normalize_surface(best.label).split()) == 1)
            context_floor = (config.single_token_person_context_floor
                             if local_single_token_person else 0)
            if len(ranked) == 1:
                context_floor = max(context_floor, config.single_candidate_context_floor)
            if best.rank_score < config.nil_threshold:
                decision, selected, confidence = "nil", None, best.rank_score
                reason = "all retrieved candidates scored below the configured NIL threshold"
            elif best.rank_score >= config.linked_threshold and margin >= config.minimum_margin \
                    and best.context_similarity >= context_floor:
                decision, selected, confidence = "linked", best.qid, best.rank_score
                reason = "top candidate passed the score and separation-margin thresholds"
            else:
                decision, selected, confidence = "ambiguous", None, best.rank_score
                if config.nil_threshold == 0 and config.single_candidate_context_floor == 0 \
                        and config.single_token_person_context_floor == 0:
                    # Runs made with the first score-and-margin policy (before the extra abstention
                    # checks) must still validate byte for byte.
                    reason = "retrieved candidates did not pass both score and separation-margin thresholds"
                else:
                    reason = "retrieved candidates did not pass all score, margin, and context-floor thresholds"
        rows.append(LinkDecision(created_by_run_id=run_id,
            decision_id=stable_id("ld", entity.corpus_entity_id, policy_json),
            corpus_entity_id=entity.corpus_entity_id, entity_type=entity.type,
            canonical_label=entity.canonical_label, candidate_ids=ids, selected_qid=selected,
            decision=decision, confidence=confidence, margin=margin, reason=reason))
    return tuple(rows)


def apply_decisions(run_id: str, entities: list[CorpusEntity],
                    decisions: list[LinkDecision]) -> tuple[CorpusEntity, ...]:
    """Copy the local identities and add only the link result."""
    decision_by_entity = {row.corpus_entity_id: row for row in decisions}
    enriched = []
    for entity in entities:
        decision = decision_by_entity.get(entity.corpus_entity_id)
        updates = {"created_by_run_id": run_id}
        if decision:
            updates.update({"wikidata_qid": decision.selected_qid,
                            "link_status": decision.decision,
                            "link_confidence": decision.confidence})
        enriched.append(entity.model_copy(update=updates))
    return tuple(enriched)


def review_view(decisions: list[LinkDecision], candidates: list[WikidataCandidate]) -> str:
    """Short Markdown table for review. All the numbers stay in the JSONL."""
    candidate_by_id = {row.candidate_id: row for row in candidates}
    counts = Counter(row.decision for row in decisions)
    lines = ["# Wikidata entity linking", "",
        "> [!warning] Provisional external links",
        "> A linked QID is a model-assisted decision grounded in transcript passages; ambiguous and NIL cases abstain.", "",
        f"- Linked: {counts['linked']}", f"- Ambiguous: {counts['ambiguous']}",
        f"- NIL: {counts['nil']}", ""]
    for decision in sorted(decisions, key=lambda row: (row.decision, row.entity_type,
                                                        row.canonical_label.casefold())):
        lines.extend([f"## {decision.canonical_label} ({decision.entity_type})", "",
                      f"- Decision: `{decision.decision}`", f"- Reason: {decision.reason}"])
        for candidate_id in decision.candidate_ids[:3]:
            row = candidate_by_id[candidate_id]
            lines.append(f"- [{row.label}](https://www.wikidata.org/wiki/{row.qid}) — "
                         f"{row.rank_score:.3f} (name {row.name_similarity:.3f}, type "
                         f"{row.type_compatibility:.3f}, context {row.context_similarity:.3f}) — "
                         f"{row.description or 'no description'}")
        lines.append("")
    return "\n".join(lines)


def load_parent(parent: Path):
    """Load a validated resolution-type run and its extraction evidence."""
    manifest = validate_resolution(parent)
    extraction = parent.parent / manifest.get("extraction_run_id", manifest["parent_run_id"])
    extraction_manifest = json.loads((extraction / "manifest.json").read_bytes())
    passage_run = extraction.parent / extraction_manifest["parent_run_id"]
    return (manifest, extraction,
        [EpisodeEntity.model_validate(row) for row in read_jsonl(parent / "episode_entities.jsonl")],
        [CorpusEntity.model_validate(row) for row in read_jsonl(parent / "corpus_entities.jsonl")],
        [MentionAssignment.model_validate(row) for row in read_jsonl(parent / "assignments.jsonl")],
        [ResolutionDecision.model_validate(row) for row in read_jsonl(parent / "decisions.jsonl")],
        [ReviewCandidate.model_validate(row) for row in read_jsonl(parent / "review_queue.jsonl")],
        [AliasRule.model_validate(row) for row in read_jsonl(parent / "aliases.jsonl")],
        [ExtractedMention.model_validate(row) for row in read_jsonl(extraction / "mentions.jsonl")],
        [Passage.model_validate(row) for row in read_jsonl(passage_run / "passages.jsonl")])


def counts_for(entities: list[CorpusEntity], eligible: list[CorpusEntity],
               candidates: list[WikidataCandidate], decisions: list[LinkDecision]) -> dict:
    """Counts for the manifest and the CLI output."""
    return {"corpus_entities": len(entities), "attempted_entities": len(eligible),
        "skipped_entities": len(entities) - len(eligible), "wikidata_candidates": len(candidates),
        "linked": sum(row.decision == "linked" for row in decisions),
        "ambiguous": sum(row.decision == "ambiguous" for row in decisions),
        "nil": sum(row.decision == "nil" for row in decisions),
        "by_type": dict(sorted(Counter(row.entity_type for row in decisions).items()))}


def link_run(root: Path, input_name: str, config_name: str,
             run_id: str | None = None, candidate_name: str | None = None,
             *, retriever_factory: Callable[[EntityLinkingConfig], CandidateRetriever] | None = None,
             encoder_factory: Callable[[EntityLinkingConfig], TextEncoder] | None = None) -> Path:
    """Search and rank Wikidata candidates for a validated resolution run."""
    root, started, now = root.resolve(), perf_counter(), datetime.now(timezone.utc)
    run_id = run_id or f"link-{now:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
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
    config = EntityLinkingConfig.model_validate_json(config_bytes)
    (parent_manifest, extraction, episode_entities, entities, assignments, identity_decisions,
     queue, aliases, mentions, passages) = load_parent(parent)
    eligible = [row for row in entities if row.type in config.entity_types]
    contexts, evidence = build_contexts(eligible, mentions, assignments, passages,
                                        config.max_context_characters)
    candidate_source_digest = None
    candidate_source = None
    if candidate_name:
        candidate_source = workspace_path(root, candidate_name)
        if candidate_source.parent != root / "data/prototype1/runs":
            raise ValueError("candidate cache must be a sibling run snapshot")
        candidate_manifest = validate_run(candidate_source)
        if candidate_manifest["parent_run_id"] != parent.name:
            raise ValueError("candidate cache was built from a different resolution snapshot")
        candidate_source_digest = sha256((candidate_source / "manifest.json").read_bytes())
        cached = [WikidataCandidate.model_validate(row)
                  for row in read_jsonl(candidate_source / "wikidata_candidates.jsonl")]
        candidates = rerank_cached_candidates(run_id, eligible, cached, config)
    else:
        raw = (retriever_factory or WikidataRetriever)(config).retrieve(eligible)
        candidates = score_candidates(run_id, eligible, raw, contexts, config,
                                      (encoder_factory or MpnetEncoder)(config))
    decisions = make_decisions(run_id, eligible, list(candidates), config)
    enriched = apply_decisions(run_id, entities, list(decisions))
    if sha256((parent / "manifest.json").read_bytes()) != parent_digest or config_path.read_bytes() != config_bytes:
        raise ValueError("parent or entity-linking configuration changed")
    output.mkdir(parents=True, exist_ok=False)
    inherited = {"episode_entities.jsonl": episode_entities, "corpus_entities.jsonl": enriched,
        "assignments.jsonl": assignments, "decisions.jsonl": identity_decisions,
        "review_queue.jsonl": queue, "aliases.jsonl": aliases}
    for name, rows in inherited.items():
        dump_jsonl(output / name, list(rows))
    dump_jsonl(output / "wikidata_candidates.jsonl", list(candidates))
    dump_jsonl(output / "link_decisions.jsonl", list(decisions))
    dump_jsonl(output / "link_contexts.jsonl", [{"corpus_entity_id": entity.corpus_entity_id,
        "canonical_label": entity.canonical_label, "entity_type": entity.type,
        "evidence_passage_ids": evidence[entity.corpus_entity_id],
        "context": contexts[entity.corpus_entity_id]} for entity in eligible])
    dump_json(output / "config.json", config.model_dump(mode="json"))
    summary = counts_for(list(enriched), eligible, list(candidates), list(decisions))
    dump_json(output / "summary.json", summary)
    (output / "review.md").write_text(review_view(list(decisions), list(candidates)), encoding="utf-8")
    names = [*inherited, "wikidata_candidates.jsonl", "link_decisions.jsonl",
             "link_contexts.jsonl", "config.json", "summary.json", "review.md"]
    code_root = Path(__file__).parent
    manifest = {"schema_version": 1, "stage": "entity_linking", "run_id": run_id,
        "status": "complete", "package_version": __version__, "started_utc": now.isoformat(),
        "elapsed_seconds": round(perf_counter() - started, 4), "python": platform.python_version(),
        "dependencies": {name: importlib.metadata.version(name)
                         for name in ("pydantic", "requests", "torch", "transformers")},
        "parent_run_id": parent.name, "parent_manifest_sha256": parent_digest,
        "extraction_run_id": extraction.name, "input_config": config_name,
        "input_config_sha256": sha256(config_bytes),
        "candidate_source_run_id": candidate_source.name if candidate_source else None,
        "candidate_source_manifest_sha256": candidate_source_digest,
        "model": {"repo_id": config.embedding_repo_id, "revision": config.embedding_revision,
                  "framework": "transformers-mean-pooling", "device": config.embedding_device},
        "counts": summary,
        "code_sha256": {str(path.relative_to(code_root)): sha256(path.read_bytes())
                        for path in sorted(code_root.rglob("*.py"))},
        "artifact_sha256": {name: sha256((output / name).read_bytes()) for name in names}}
    dump_json(output / "manifest.json", manifest)
    validate_run(output)
    return output


def validate_run(output: Path) -> dict:
    """Validate provenance and link rules, and check the decisions can be reproduced."""
    output = output.resolve()
    manifest = json.loads((output / "manifest.json").read_bytes())
    if manifest.get("schema_version") != 1 or manifest.get("stage") != "entity_linking" \
            or manifest.get("status") != "complete":
        raise ValueError("unsupported entity-linking snapshot")
    root = output.parents[3]
    parent = output.parent / manifest["parent_run_id"]
    if sha256((parent / "manifest.json").read_bytes()) != manifest["parent_manifest_sha256"]:
        raise ValueError("parent resolution manifest changed")
    config_path = workspace_path(root, manifest["input_config"])
    if sha256(config_path.read_bytes()) != manifest["input_config_sha256"]:
        raise ValueError("entity-linking configuration changed")
    if manifest.get("candidate_source_run_id"):
        candidate_source = output.parent / manifest["candidate_source_run_id"]
        if sha256((candidate_source / "manifest.json").read_bytes()) != \
                manifest["candidate_source_manifest_sha256"]:
            raise ValueError("entity-linking candidate cache changed")
    expected_names = {"episode_entities.jsonl", "corpus_entities.jsonl", "assignments.jsonl",
        "decisions.jsonl", "review_queue.jsonl", "aliases.jsonl", "wikidata_candidates.jsonl",
        "link_decisions.jsonl", "link_contexts.jsonl", "config.json", "summary.json", "review.md"}
    if set(manifest["artifact_sha256"]) != expected_names:
        raise ValueError("unexpected entity-linking artifact set")
    for name, digest in manifest["artifact_sha256"].items():
        if sha256((output / name).read_bytes()) != digest:
            raise ValueError(f"entity-linking artifact changed: {name}")
    config = EntityLinkingConfig.model_validate_json((output / "config.json").read_bytes())
    (_, _, parent_episode_entities, parent_entities, parent_assignments, parent_identity_decisions,
     parent_queue, parent_aliases, _, _) = load_parent(parent)
    # This stage may only add link fields to CorpusEntities. All other identity records must
    # match the parent exactly, so Wikidata can't change the local clustering.
    inherited = ((EpisodeEntity, "episode_entities.jsonl", parent_episode_entities),
        (MentionAssignment, "assignments.jsonl", parent_assignments),
        (ResolutionDecision, "decisions.jsonl", parent_identity_decisions),
        (ReviewCandidate, "review_queue.jsonl", parent_queue),
        (AliasRule, "aliases.jsonl", parent_aliases))
    for model, name, expected in inherited:
        actual = [model.model_validate(row) for row in read_jsonl(output / name)]
        if actual != list(expected):
            raise ValueError(f"entity linking changed local identity artifact: {name}")
    candidates = [WikidataCandidate.model_validate(row)
                  for row in read_jsonl(output / "wikidata_candidates.jsonl")]
    decisions = [LinkDecision.model_validate(row) for row in read_jsonl(output / "link_decisions.jsonl")]
    if len({row.candidate_id for row in candidates}) != len(candidates):
        raise ValueError("duplicate Wikidata candidate ID")
    if len({row.corpus_entity_id for row in decisions}) != len(decisions):
        raise ValueError("duplicate entity-linking decision")
    eligible = [row for row in parent_entities if row.type in config.entity_types]
    eligible_ids = {row.corpus_entity_id for row in eligible}
    if {row.corpus_entity_id for row in decisions} != eligible_ids:
        raise ValueError("every eligible entity must receive one linking decision")
    if any(row.corpus_entity_id not in eligible_ids for row in candidates):
        raise ValueError("candidate refers to an ineligible entity")
    for row in candidates:
        expected_score = config.name_weight * row.name_similarity + config.type_weight * \
            row.type_compatibility + config.context_weight * row.context_similarity
        if abs(row.rank_score - expected_score) > 1e-9:
            raise ValueError("candidate score does not match configured weights")
    if make_decisions(manifest["run_id"], eligible, candidates, config) != tuple(decisions):
        raise ValueError("entity-linking decisions are not reproducible from candidate scores")
    expected_entities = apply_decisions(manifest["run_id"], parent_entities, decisions)
    actual_entities = tuple(CorpusEntity.model_validate(row)
                            for row in read_jsonl(output / "corpus_entities.jsonl"))
    if actual_entities != expected_entities:
        raise ValueError("linked CorpusEntities do not match link decisions")
    summary = counts_for(list(actual_entities), eligible, candidates, decisions)
    if summary != manifest["counts"] or summary != json.loads((output / "summary.json").read_bytes()):
        raise ValueError("entity-linking counts differ")
    return manifest
