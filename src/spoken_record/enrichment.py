"""Adds Wikidata facts (lifespans, occupations, image) to entities that already have a QID.

Only entities linked by the linking stage, or given a QID in a reviewed file, are enriched.
This step never searches for an identity itself. The claims it used are stored with the
build, so the lifespans can be derived again offline. The date rules keep the widest
plausible range, so a filter doesn't wrongly drop someone:

* Only best-rank claims: the preferred ones if there are any, otherwise the normal ones.
  Deprecated and "unknown/no value" claims are skipped.
* Low precision widens the range: decade = 10 years, century = 100, millennium = 1000.
  Earliest/latest date qualifiers (P1319/P1326) widen it further.
* Start is the earliest birth (P569), end is the latest death (P570). Without those we use
  floruit (P1317) and work period start/end (P2031/P2032) and mark the range approximate.
* No death date and born less than ``living_max_age_years`` before the retrieval year: living.
* Negative years are BCE, the same as in Wikidata.
"""

import math
import re

from spoken_record.enrichment_models import EnrichmentConfig, EntityFacts, Lifespan, ReviewedQid
from spoken_record.resolution_models import CorpusEntity

PROPERTIES = ("P18", "P31", "P106", "P569", "P570", "P1317", "P2031", "P2032")
DATE_PROPERTIES = ("P569", "P570", "P1317", "P2031", "P2032")
HUMAN, CIRCA = "Q5", "Q5727902"
SCHEMA_VERSION = 2  # Version 2 stores occupations as QIDs instead of a built-in "philosopher" flag.
def make_retriever(config: EnrichmentConfig):
    """Tests replace this with an offline retriever."""
    from spoken_record.entity_linking import WikidataRetriever

    class WikidataFacts(WikidataRetriever):
        """Fetch claims in batches with the linker's API client (User-Agent, retries)."""

        def fetch(self, qids: list[str]) -> dict[str, dict]:
            params = {"action": "wbgetentities", "props": "info|claims", "format": "json"}
            if self.config.maxlag is not None:
                params["maxlag"] = self.config.maxlag
            rows = self.entities(qids, params, fact_fields)
            return {qid: trim_entity(qid, {"missing": ""} if rows[qid].get("absent") else rows[qid]) for qid in qids}

    return WikidataFacts(config)


def fact_fields(payload: dict) -> dict:
    """Of an entity answer, keep only the claims this stage reads."""
    entities = {}
    for qid, row in payload.get("entities", {}).items():
        entities[qid] = {"missing": ""} if "missing" in row else {"lastrevid": row.get("lastrevid"),
            "claims": {pid: row["claims"][pid] for pid in PROPERTIES if pid in row.get("claims", {})}}
    return {"entities": entities}


def _time(snak: dict) -> tuple[str, int] | None:
    value = snak.get("datavalue", {}).get("value") if snak.get("snaktype") == "value" else None
    return (value["time"], value["precision"]) if isinstance(value, dict) and "time" in value else None


def trim_entity(qid: str, entity: dict) -> dict:
    """Keep only the claims this stage uses, in a stable shape that is easy to review."""
    if "missing" in entity:
        return {"qid": qid, "missing": True, "lastrevid": None, "claims": {}}
    claims = {}
    for pid in PROPERTIES:
        rows = []
        for claim in entity.get("claims", {}).get(pid, []):
            snak = claim.get("mainsnak", {})
            row = {"rank": claim.get("rank", "normal")}
            if pid in DATE_PROPERTIES:
                value = _time(snak)
                if value is None:
                    continue
                row["time"], row["precision"] = value
                qualifiers = claim.get("qualifiers", {})
                for q in ("P1319", "P1326"):
                    bounds = [t for t in (_time(s) for s in qualifiers.get(q, [])) if t]
                    if bounds:
                        row[q] = [list(b) for b in bounds]
                circa = [s.get("datavalue", {}).get("value", {}).get("id") for s in qualifiers.get("P1480", [])]
                row["circa"] = CIRCA in circa
            elif pid == "P18":
                # An image is a Commons file name, not an item.
                value = snak.get("datavalue", {}).get("value")
                if snak.get("snaktype") != "value" or not isinstance(value, str) or not value.strip():
                    continue
                row["file"] = value
            else:
                target = snak.get("datavalue", {}).get("value", {})
                if snak.get("snaktype") != "value" or not isinstance(target, dict) or "id" not in target:
                    continue
                row["id"] = target["id"]
            rows.append(row)
        if rows:
            claims[pid] = rows
    return {"qid": qid, "missing": False, "lastrevid": entity.get("lastrevid"), "claims": claims}


def year_interval(time: str, precision: int) -> tuple[int, int] | None:
    """Inclusive year range for a Wikidata time value, widened by its precision."""
    match = re.match(r"([+-])(\d+)-", time)
    if not match or precision < 6:
        return None
    year = int(match.group(2)) * (-1 if match.group(1) == "-" else 1)
    if precision >= 9:
        return year, year
    if precision == 8:
        low = math.floor(year / 10) * 10
        return low, low + 9
    size = 100 if precision == 7 else 1000
    unit = math.ceil(abs(year) / size)
    low, high = (unit - 1) * size + 1, unit * size
    return (low, high) if year > 0 else (-high, -low)


def best_rank(rows: list[dict]) -> list[dict]:
    preferred = [r for r in rows if r["rank"] == "preferred"]
    return preferred or [r for r in rows if r["rank"] == "normal"]


def date_range(rows: list[dict]) -> tuple[int, int, bool] | None:
    """Combine all best-rank values into (earliest, latest, approximate)."""
    intervals, approx = [], False
    for row in best_rank(rows):
        span = year_interval(row["time"], row["precision"])
        if span is None:
            continue
        lows, highs = [span[0]], [span[1]]
        for key, target in (("P1319", lows), ("P1326", highs)):
            for time, precision in row.get(key, []):
                bound = year_interval(time, precision)
                if bound:
                    target.append(bound[0] if key == "P1319" else bound[1])
                    approx = True
        intervals.append((min(lows), max(highs)))
        approx = approx or row["precision"] < 9 or row["circa"]
    if not intervals:
        return None
    if len({i for i in intervals}) > 1:
        approx = True
    return min(i[0] for i in intervals), max(i[1] for i in intervals), approx


def derive_life(raw: dict, reference_year: int, config: EnrichmentConfig) -> Lifespan | None:
    claims = raw["claims"]
    birth, death = date_range(claims.get("P569", [])), date_range(claims.get("P570", []))
    floruit = [r for r in (date_range(claims.get(p, [])) for p in ("P1317", "P2031", "P2032")) if r]
    sources = set()
    start = end = None
    if birth:
        start = birth[0]; sources.add("birth_death")
    elif floruit:
        start = min(f[0] for f in floruit); sources.add("floruit")
    living = False
    if death:
        end = death[1]; sources.add("birth_death")
    elif birth and reference_year - birth[1] <= config.living_max_age_years:
        living = True; sources.add("birth_death")
    elif floruit:
        end = max(f[1] for f in floruit); sources.add("floruit")
    if start is None and end is None:
        return None
    used = [r for r in (birth, death) if r] + (floruit if "floruit" in sources else [])
    basis = next(iter(sources)) if len(sources) == 1 else "mixed"
    approx = any(r[2] for r in used) or basis != "birth_death"
    return Lifespan(start=start, end=end, living=living, approximate=approx, basis=basis)


def build_facts(run_id: str, entities: list[CorpusEntity], reviewed: list[ReviewedQid], raw: dict[str, dict],
                reference_year: int, config: EnrichmentConfig) -> list[EntityFacts]:
    rows = []
    for cid, (qid, source) in sorted(targets(entities, reviewed, config).items()):
        entry = raw[qid]
        if entry["missing"]:
            continue
        ids = lambda pid: {r["id"] for r in best_rank(entry["claims"].get(pid, []))}
        rows.append(EntityFacts(created_by_run_id=run_id, corpus_entity_id=cid, qid=qid, qid_source=source,
            life=derive_life(entry, reference_year, config), is_human=HUMAN in ids("P31"),
            occupation_qids=sorted(ids("P106")), image=first_image(entry)))
    return rows


def first_image(entry: dict) -> str | None:
    """The first best-rank image. Wikidata keeps the order editors gave the claims."""
    images = best_rank(entry["claims"].get("P18", []))
    return images[0]["file"] if images else None


def targets(entities: list[CorpusEntity], reviewed: list[ReviewedQid], config: EnrichmentConfig) -> dict[str, tuple[str, str]]:
    """Map each entity to (QID, source). A reviewed QID wins over an automatic link."""
    wanted = {e.corpus_entity_id: e for e in entities if e.type in config.entity_types}
    out = {cid: (e.wikidata_qid, "linked") for cid, e in wanted.items() if e.link_status == "linked" and e.wikidata_qid}
    for row in reviewed:
        if row.corpus_entity_id not in wanted:
            raise ValueError(f"reviewed QID refers to an unknown or non-{config.entity_types} entity: {row.corpus_entity_id}")
        out[row.corpus_entity_id] = (row.qid, "reviewed")
    return out


def load_reviewed(data: bytes) -> list[ReviewedQid]:
    rows = [ReviewedQid.model_validate_json(line) for line in data.decode("utf-8").splitlines() if line.strip()]
    if len({r.corpus_entity_id for r in rows}) != len(rows):
        raise ValueError("reviewed QID file assigns an entity more than once")
    return rows
