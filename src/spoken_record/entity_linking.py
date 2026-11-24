"""Links corpus entities to Wikidata: search, ranking with passage context, and abstention.

This step only adds a link decision to each eligible corpus entity. Entities that end up
with the same confident item are merged afterwards, in consolidation.py, where the rules for
that live.
"""

import math
import re
import time
from collections import defaultdict
from difflib import SequenceMatcher

from spoken_record import __version__
from spoken_record.backends import CandidateRetriever, TextEncoder
from spoken_record.adapters.base import stable_id
from spoken_record.extraction_models import ExtractedMention
from spoken_record.linking_models import EntityLinkingConfig, LinkDecision, WikidataCandidate
from spoken_record.passage_models import Passage
from spoken_record.resolution import normalize_surface
from spoken_record.resolution_models import (CorpusEntity, MentionAssignment)


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

# English description words per type, used for every podcast, so none of them belongs to one
# subject. A podcast adds the words of its subject with `extra_type_keywords` in its linking
# settings: "composer" for music, "philosopher" for philosophy.
TYPE_KEYWORDS = {
    "Person": {"person", "writer", "historian", "scholar", "poet", "politician", "professor",
               "activist", "leader"},
    "Language": {"language", "dialect", "linguistic", "creole"},
    "Place": {"city", "country", "state", "region", "town", "village", "place", "island", "capital",
              "district", "municipality", "continent", "kingdom", "empire"},
    "Work": {"book", "work", "text", "publication", "writing"},
    "School": {"school", "movement", "tradition", "religion", "church", "university"},
    "Group": {"people", "group", "ethnic", "community", "population", "organization", "tribe"},
    "Concept": {"concept", "theory", "idea", "principle", "term", "notion"},
    "Period": {"period", "era", "century", "age", "year", "epoch"},
}

# Sitelinks that end in "wiki" but are not a Wikipedia language version.
NON_WIKIPEDIA_SITES = {"commonswiki", "specieswiki", "metawiki", "mediawikiwiki", "wikidatawiki",
                       "sourceswiki", "incubatorwiki", "outreachwiki", "wikimaniawiki"}

PERSON_CONTRADICTIONS = {"album", "book", "city", "country", "film", "language", "river", "song",
                         "species", "television series", "village"}


class WikidataRetriever:
    """Small client for the Wikidata (MediaWiki) API, with retries."""

    def __init__(self, config: EntityLinkingConfig, cache=None):
        import requests

        self.config = config
        # Optional request cache (see cache.py). Wikidata answers then stay frozen until the
        # cache is cleared on purpose.
        self.cache = cache
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

    def _get(self, params: dict, keep=None) -> dict:
        """GET one JSON response, retrying on temporary API or network errors.

        ``keep`` reduces the answer to the fields the caller reads. A fresh answer goes
        through it too, so it doesn't matter whether an answer came from the cache.
        """
        keep = keep or (lambda payload: payload)
        if self.cache is not None:
            key = self.cache.key("wikidata", self.config.api_url, params)
            cached = self.cache.get(key)
            if cached is not None:
                return cached
            payload = keep(self._fetch(params))
            self.cache.put(key, payload)
            return payload
        return keep(self._fetch(params))

    def _fetch(self, params: dict) -> dict:
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

    def _with_maxlag(self, params: dict) -> dict:
        return params if self.config.maxlag is None else {**params, "maxlag": self.config.maxlag}

    def search(self, query: str) -> list[dict]:
        """Top label and alias matches for one name."""
        payload = self._get(self._with_maxlag({"action": "wbsearchentities", "search": query,
            "language": self.config.language, "uselang": self.config.language,
            "type": "item", "limit": self.config.top_k, "format": "json"}))
        return [{"qid": row["id"], "label": row.get("label", ""),
                 "description": row.get("description", ""),
                 "aliases": list(row.get("aliases", [])), "retrieval_rank": rank}
                for rank, row in enumerate(payload.get("search", []), 1)
                if re.fullmatch(r"Q[1-9][0-9]*", row.get("id", ""))]

    def fulltext_search(self, query: str, instance_types: list[str]) -> list[dict]:
        """Full-text matches limited to the given P31 values. Label and description come later."""
        filters = "|".join(f"P31={qid}" for qid in instance_types)
        payload = self._get(self._with_maxlag({"action": "query", "list": "search",
            "srsearch": f"{query} haswbstatement:{filters}", "srnamespace": 0,
            "srlimit": self.config.top_k, "format": "json"}))
        return [{"qid": row["title"], "label": "", "description": "", "aliases": [],
                 "retrieval_rank": rank}
                for rank, row in enumerate(payload.get("query", {}).get("search", []), 1)
                if re.fullmatch(r"Q[1-9][0-9]*", row.get("title", ""))]

    def entities(self, qids: list[str], params: dict, keep) -> dict[str, dict]:
        """The answer for every item, fetched in batches of 50 the API accepts.

        The cache holds one entry per item, not per batch. Otherwise a slightly different
        set of items would shift the batches and make every later batch look new. An item
        the API does not return at all gets ``{"absent": True}``.
        """
        found, missing, keys = {}, [], {}
        for qid in dict.fromkeys(qids):
            if self.cache is not None:
                keys[qid] = self.cache.key("wikidata-item", self.config.api_url, params, qid)
                cached = self.cache.get(keys[qid])
                if cached is not None:
                    found[qid] = cached
                    continue
            missing.append(qid)
        for start in range(0, len(missing), 50):
            batch = missing[start:start + 50]
            payload = keep(self._fetch({**params, "ids": "|".join(batch)}))
            for qid in batch:
                found[qid] = payload.get("entities", {}).get(qid, {"absent": True})
                if self.cache is not None:
                    self.cache.put(keys[qid], found[qid])
        return found

    def details(self, qids: list[str]) -> dict[str, dict]:
        """Labels, aliases, P31 claims and sitelinks of every item."""
        result: dict[str, dict] = {}
        rows = self.entities(qids, self._with_maxlag({"action": "wbgetentities",
            "props": "labels|descriptions|aliases|claims|sitelinks",
            "languages": self.config.language, "languagefallback": 1, "format": "json"}), linking_fields)
        for qid, row in rows.items():
            if row.get("absent"):
                continue
            aliases = [value.get("value", "") for value in row.get("aliases", {}).get(
                self.config.language, [])]
            instance_of = []
            for claim in row.get("claims", {}).get("P31", []):
                value = claim.get("mainsnak", {}).get("datavalue", {}).get("value", {})
                if isinstance(value, dict) and re.fullmatch(r"Q[1-9][0-9]*", value.get("id", "")):
                    instance_of.append(value["id"])
            result[qid] = {"aliases": aliases, "instance_of_qids": sorted(set(instance_of)),
                "label": row.get("labels", {}).get(self.config.language, {}).get("value", ""),
                "description": row.get("descriptions", {}).get(self.config.language, {}).get("value", ""),
                # Only Wikipedia language versions, not Wikiquote, Commons and so on.
                "sitelinks": sum(key.endswith("wiki") and key not in NON_WIKIPEDIA_SITES
                                 for key in row.get("sitelinks", {}))}
        return result

    def retrieve(self, entities: list[CorpusEntity]) -> dict[str, list[dict]]:
        """Run every search for every eligible entity, then fetch all distinct candidates in batches."""
        raw = {}
        for entity in entities:
            results = []
            for kind, query in search_queries(entity, self.config):
                rows = (self.search(query) if kind == "label" else
                        self.fulltext_search(query, self.config.fulltext_instance_types[entity.type]))
                results.append((query, rows))
            raw[entity.corpus_entity_id] = merge_results(results)
        qids = sorted({row["qid"] for rows in raw.values() for row in rows}, key=lambda q: int(q[1:]))
        details = self.details(qids) if qids else {}
        for entity_id, rows in raw.items():
            for row in rows:
                detail = details.get(row["qid"], {})
                row["label"] = row["label"] or detail.get("label", "")
                row["description"] = row["description"] or detail.get("description", "")
                row["aliases"] = list(dict.fromkeys([*row["aliases"], *detail.get("aliases", [])]))[:50]
                row["instance_of_qids"] = detail.get("instance_of_qids", [])
                if self.config.popularity_weight:
                    row["sitelinks"] = detail.get("sitelinks", 0)
        return {entity.corpus_entity_id: allowed_candidates(entity, raw[entity.corpus_entity_id], self.config)
                for entity in entities}


def linking_fields(payload: dict) -> dict:
    """Of an entity answer, keep labels, descriptions, aliases, the P31 values and the names
    of the sitelinks. The full claims of 50 items can be many megabytes."""
    entities = {}
    for qid, row in payload.get("entities", {}).items():
        instance_of = [{"mainsnak": {"datavalue": {"value": {"id": value["id"]}}}}
                       for claim in row.get("claims", {}).get("P31", [])
                       if isinstance(value := claim.get("mainsnak", {}).get("datavalue", {}).get("value"), dict)
                       and "id" in value]
        entities[qid] = {**{name: row[name] for name in ("labels", "descriptions", "aliases") if name in row},
                         "claims": {"P31": instance_of}, "sitelinks": {name: {} for name in row.get("sitelinks", {})}}
    return {"entities": entities}


def fullest_surface(entity: CorpusEntity) -> str | None:
    """The observed name with more full words than the canonical label, if there is one."""
    def full_words(value: str) -> int:
        return sum(len(token) > 1 for token in re.findall(r"[^\W\d_]+", value))

    best = max(entity.observed_surfaces, key=lambda value: (full_words(value), len(value), value),
               default=None)
    if best is None or full_words(best) <= full_words(entity.canonical_label):
        return None
    return best


def search_queries(entity: CorpusEntity, config: EntityLinkingConfig) -> list[tuple[str, str]]:
    """The searches for one entity, in order: label searches first, then full-text ones."""
    names = [entity.canonical_label]
    fullest = fullest_surface(entity) if config.search_fullest_surface else None
    if fullest and normalize_surface(fullest) != normalize_surface(entity.canonical_label):
        names.append(fullest)
    queries = [("label", name) for name in names]
    if entity.type in config.fulltext_instance_types:
        queries.extend(("fulltext", name) for name in names)
    return queries


def merge_results(results: list[tuple[str, list[dict]]]) -> list[dict]:
    """Pool the results of several searches. A QID found by several searches is kept once,
    with its best rank, and the merged order takes the first hit of every search before the
    second hit of any search."""
    best: dict[str, tuple[int, int, str, dict]] = {}
    for order, (query, rows) in enumerate(results):
        for row in rows:
            key = (row["retrieval_rank"], order)
            if row["qid"] not in best or key < best[row["qid"]][:2]:
                best[row["qid"]] = (*key, query, row)
    merged = []
    for rank, (_, _, query, row) in enumerate(sorted(best.values(), key=lambda item: item[:2]), 1):
        merged.append({**row, "query": query, "retrieval_rank": rank})
    return merged


def allowed_candidates(entity: CorpusEntity, rows: list[dict], config: EntityLinkingConfig) -> list[dict]:
    """Drop candidates whose P31 is not one of the required types for this entity type."""
    required = set(config.required_instance_types.get(entity.type, []))
    if not required:
        return rows
    kept = [row for row in rows if required & set(row.get("instance_of_qids", []))]
    return [{**row, "retrieval_rank": rank} for rank, row in enumerate(kept, 1)]


def is_namesake(entity: CorpusEntity, label: str, aliases: list[str]) -> bool:
    """True when the candidate carries the transcript name: every word of a local name is in
    its label or one alias ("Locke" and "John Locke"), or the two are almost the same."""
    local = [normalize_surface(value) for value in [entity.canonical_label, *entity.observed_surfaces]]
    external = [normalize_surface(value) for value in [label, *aliases]]
    for left in filter(None, local):
        for right in filter(None, external):
            if set(left.split()) <= set(right.split()) or SequenceMatcher(None, left, right).ratio() >= .9:
                return True
    return False


def popularity_scores(entity: CorpusEntity, candidates: list[tuple[str, list[str], int | None]]) -> list[float]:
    """Sitelinks of each namesake divided by the most sitelinks among the namesakes of one
    entity. Candidates with another name get 0. When no namesake has a sitelink, the score
    says nothing, so all namesakes get 1.0."""
    namesake = [is_namesake(entity, label, aliases) for label, aliases, _ in candidates]
    most = max((sitelinks or 0 for (_, _, sitelinks), same in zip(candidates, namesake) if same),
               default=0)
    return [0.0 if not same else (sitelinks or 0) / most if most else 1.0
            for (_, _, sitelinks), same in zip(candidates, namesake)]


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
    popularity_by_pair = {}
    for entity in ordered_entities:
        found = raw.get(entity.corpus_entity_id, [])
        scores = popularity_scores(entity, [(row["label"], row.get("aliases", []), row.get("sitelinks"))
                                            for row in found])
        popularity_by_pair.update({(entity.corpus_entity_id, row["qid"]): score
                                   for row, score in zip(found, scores)})
    rows = []
    for (entity, candidate), vector in zip(pairs, candidate_vectors):
        surface = name_similarity(entity, candidate)
        kind = type_compatibility(entity.type, candidate, config.keyword_type_score, config.extra_type_keywords)
        context = dot(context_vectors[entity.corpus_entity_id], vector)
        sitelinks = candidate.get("sitelinks")
        popularity = (popularity_by_pair[(entity.corpus_entity_id, candidate["qid"])]
                      if config.popularity_weight else 0.0)
        score = combined_score(config, surface, kind, context, popularity)
        rows.append(WikidataCandidate(created_by_run_id=run_id,
            candidate_id=stable_id("wc", entity.corpus_entity_id, candidate["qid"]),
            corpus_entity_id=entity.corpus_entity_id,
            query=candidate.get("query", entity.canonical_label),
            qid=candidate["qid"], label=candidate["label"],
            description=candidate.get("description", ""), aliases=candidate.get("aliases", []),
            instance_of_qids=candidate.get("instance_of_qids", []),
            retrieval_rank=candidate["retrieval_rank"], name_similarity=surface,
            type_compatibility=kind, context_similarity=context, sitelinks=sitelinks,
            popularity=popularity, rank_score=score))
    return tuple(sorted(rows, key=lambda row: (row.corpus_entity_id, -row.rank_score,
                                               row.retrieval_rank, row.qid)))


def combined_score(config: EntityLinkingConfig, name: float, kind: float, context: float,
                   popularity: float) -> float:
    """Weighted sum of the four scores, kept between 0 and 1."""
    score = (config.name_weight * name + config.type_weight * kind +
             config.context_weight * context)
    # Older configs have no popularity weight. Adding 0.0 would still be exact, but leaving
    # the term out keeps their float sums exactly as they were.
    if config.popularity_weight:
        score += config.popularity_weight * popularity
    return min(1.0, max(0.0, score))


def make_decisions(run_id: str, entities: list[CorpusEntity], candidates: list[WikidataCandidate],
                   config: EntityLinkingConfig) -> tuple[LinkDecision, ...]:
    """Link, or abstain when the score or the lead over the runner-up is too small."""
    by_entity: dict[str, list[WikidataCandidate]] = defaultdict(list)
    for row in candidates:
        by_entity[row.corpus_entity_id].append(row)
    rows = []
    policy_json = config.model_dump_json()
    for entity in sorted(entities, key=lambda row: row.corpus_entity_id):
        ranked = sorted(by_entity[entity.corpus_entity_id],
                        key=lambda row: (-row.rank_score, row.retrieval_rank, row.qid))
        ids = [row.candidate_id for row in ranked]
        if not ranked:
            decision, selected, confidence, margin = "nil", None, None, None
            reason = ("Wikidata search returned no candidates of an allowed type"
                      if entity.type in config.required_instance_types
                      else "Wikidata search returned no item candidates")
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


def linking_inputs(entities: list[CorpusEntity], mentions: list[ExtractedMention],
                   assignments: list[MentionAssignment], passages: list[Passage],
                   config: EntityLinkingConfig):
    """The entities of the configured types and the passage context of each."""
    eligible = [row for row in entities if row.type in config.entity_types]
    contexts, evidence = build_contexts(eligible, mentions, assignments, passages,
                                        config.max_context_characters)
    return eligible, contexts, evidence


def retrieve_and_score(run_id: str, eligible: list[CorpusEntity], contexts: dict[str, str],
                       config: EntityLinkingConfig, retriever: CandidateRetriever,
                       encoder: TextEncoder) -> tuple[WikidataCandidate, ...]:
    """Search Wikidata for every entity and rank the candidates."""
    raw = retriever.retrieve(eligible)
    return score_candidates(run_id, eligible, raw, contexts, config, encoder)
