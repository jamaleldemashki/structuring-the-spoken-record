"""Adds Wikidata facts (lifespans, occupations) to entities that already have a QID.

Only entities linked by the linking stage, or given a QID in a reviewed file, are enriched.
This stage never searches for an identity itself. The raw claims are saved with the run, so
validation can derive the lifespans again offline. The date rules keep the widest plausible
range, so a filter doesn't wrongly drop someone:

* Only best-rank claims: the preferred ones if there are any, otherwise the normal ones.
  Deprecated and "unknown/no value" claims are skipped.
* Low precision widens the range: decade = 10 years, century = 100, millennium = 1000.
  Earliest/latest date qualifiers (P1319/P1326) widen it further.
* Start is the earliest birth (P569), end is the latest death (P570). Without those we use
  floruit (P1317) and work period start/end (P2031/P2032) and mark the range approximate.
* No death date and born less than ``living_max_age_years`` before the retrieval year: living.
* Negative years are BCE, the same as in Wikidata.
"""

import importlib.metadata
import json
import math
import platform
import re
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from spoken_record import __version__
from spoken_record.adapters.base import sha256
from spoken_record.enrichment_models import EnrichmentConfig, EntityFacts, Lifespan, ReviewedQid
from spoken_record.ingestion import dump_json, dump_jsonl, read_jsonl, workspace_path
from spoken_record.resolution_models import CorpusEntity
from spoken_record.vault import vault_input

PROPERTIES = ("P31", "P106", "P569", "P570", "P1317", "P2031", "P2032")
DATE_PROPERTIES = ("P569", "P570", "P1317", "P2031", "P2032")
HUMAN, CIRCA = "Q5", "Q5727902"
SCHEMA_VERSION = 2  # Version 2 stores occupations as QIDs instead of a built-in "philosopher" flag.
ARTIFACTS = ("config.json", "reviewed_qids.jsonl", "wikidata_entities.jsonl", "entity_facts.jsonl", "summary.json")


def make_retriever(config: EnrichmentConfig):
    """Tests replace this with an offline retriever."""
    from spoken_record.entity_linking import WikidataRetriever

    class WikidataFacts(WikidataRetriever):
        """Fetch claims in batches with the linker's API client (User-Agent, retries)."""

        def fetch(self, qids: list[str]) -> dict[str, dict]:
            out = {}
            for start in range(0, len(qids), 50):
                batch = qids[start:start + 50]
                params = {"action": "wbgetentities", "ids": "|".join(batch), "props": "info|claims", "format": "json"}
                if self.config.maxlag is not None:
                    params["maxlag"] = self.config.maxlag
                payload = self._get(params)
                for qid in batch:
                    out[qid] = trim_entity(qid, payload.get("entities", {}).get(qid, {"missing": ""}))
            return out

    return WikidataFacts(config)


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
            occupation_qids=sorted(ids("P106"))))
    return rows


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


def load_entities(root: Path, parent: Path) -> list[CorpusEntity]:
    resolution, _, _ = vault_input(root, parent)
    return [CorpusEntity.model_validate(r) for r in read_jsonl(resolution / "corpus_entities.jsonl")]


def summarize(facts: list[EntityFacts], raw: dict[str, dict], n_targets: int) -> dict:
    lives = [f.life for f in facts if f.life]
    return {"targets": n_targets, "qids": len(raw), "missing_items": sum(r["missing"] for r in raw.values()),
            "enriched": len(facts), "with_lifespan": len(lives), "living": sum(l.living for l in lives),
            "approximate": sum(l.approximate for l in lives), "with_occupation": sum(bool(f.occupation_qids) for f in facts),
            "reviewed": sum(f.qid_source == "reviewed" for f in facts), "failed": 0}


def enrich_run(root: Path, input_name: str, config_name: str, run_id: str | None = None) -> Path:
    """Fetch claims for linked or reviewed QIDs and write the derived lifespans."""
    root, started, now = root.resolve(), perf_counter(), datetime.now(timezone.utc)
    run_id = run_id or f"enrich-{now:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]{0,95}", run_id):
        raise ValueError("invalid run ID")
    parent = workspace_path(root, input_name)
    if parent.parent != root / "data/prototype1/runs":
        raise ValueError("input must be a local run snapshot")
    output = parent.parent / run_id
    if output.exists():
        raise FileExistsError(output)
    parent_digest = sha256((parent / "manifest.json").read_bytes())
    config_bytes = workspace_path(root, config_name).read_bytes()
    config = EnrichmentConfig.model_validate_json(config_bytes)
    reviewed_bytes = workspace_path(root, config.reviewed_qids).read_bytes() if config.reviewed_qids else b""
    reviewed = load_reviewed(reviewed_bytes)
    entities = load_entities(root, parent)
    wanted = targets(entities, reviewed, config)
    qids = sorted({qid for qid, _ in wanted.values()})
    raw = make_retriever(config).fetch(qids) if qids else {}
    if sorted(raw) != qids:
        raise ValueError("retriever did not return exactly the requested QIDs")
    facts = build_facts(run_id, entities, reviewed, raw, now.year, config)
    counts = summarize(facts, raw, len(wanted))
    output.mkdir()
    (output / "config.json").write_bytes(config_bytes)
    (output / "reviewed_qids.jsonl").write_bytes(reviewed_bytes)
    dump_jsonl(output / "wikidata_entities.jsonl", [raw[q] for q in qids])
    dump_jsonl(output / "entity_facts.jsonl", facts)
    dump_json(output / "summary.json", counts)
    code_root = Path(__file__).parent
    manifest = {"schema_version": SCHEMA_VERSION, "stage": "entity_enrichment", "run_id": run_id, "status": "complete",
        "package_version": __version__, "started_utc": now.isoformat(), "reference_year": now.year,
        "elapsed_seconds": round(perf_counter() - started, 4), "python": platform.python_version(),
        "dependencies": {"pydantic": importlib.metadata.version("pydantic")},
        "parent_run_id": parent.name, "parent_manifest_sha256": parent_digest,
        "input_config": config_name, "counts": counts, "source": config.api_url,
        "code_sha256": {str(p.relative_to(code_root)): sha256(p.read_bytes()) for p in sorted(code_root.rglob("*.py"))},
        "artifact_sha256": {name: sha256((output / name).read_bytes()) for name in ARTIFACTS}}
    dump_json(output / "manifest.json", manifest)
    validate_run(output)
    return output


def validate_run(output: Path) -> dict:
    """Derive every fact again offline from the saved claims and inputs."""
    output = output.resolve()
    manifest = json.loads((output / "manifest.json").read_bytes())
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("stage") != "entity_enrichment" or manifest.get("status") != "complete":
        raise ValueError("unsupported entity enrichment snapshot")
    if set(manifest["artifact_sha256"]) != set(ARTIFACTS):
        raise ValueError("unexpected enrichment artifact set")
    for name, digest in manifest["artifact_sha256"].items():
        if sha256((output / name).read_bytes()) != digest:
            raise ValueError(f"artifact checksum mismatch: {name}")
    parent = output.parent / manifest["parent_run_id"]
    if sha256((parent / "manifest.json").read_bytes()) != manifest["parent_manifest_sha256"]:
        raise ValueError("parent snapshot manifest changed")
    config = EnrichmentConfig.model_validate_json((output / "config.json").read_bytes())
    reviewed = load_reviewed((output / "reviewed_qids.jsonl").read_bytes())
    raw = {r["qid"]: r for r in read_jsonl(output / "wikidata_entities.jsonl")}
    entities = load_entities(output.parents[3], parent)
    wanted = targets(entities, reviewed, config)
    if sorted(raw) != sorted({qid for qid, _ in wanted.values()}):
        raise ValueError("stored Wikidata items do not match the enrichment targets")
    expected = build_facts(manifest["run_id"], entities, reviewed, raw, manifest["reference_year"], config)
    actual = [EntityFacts.model_validate(r) for r in read_jsonl(output / "entity_facts.jsonl")]
    if actual != expected:
        raise ValueError("entity facts are not reproducible from the stored claims")
    if manifest["counts"] != summarize(expected, raw, len(wanted)) or json.loads((output / "summary.json").read_bytes()) != manifest["counts"]:
        raise ValueError("enrichment counts differ")
    return manifest


def enrichment_view(parent: Path) -> dict[str, EntityFacts] | None:
    """Facts from the closest enrichment run above this input, if there is one."""
    manifest = json.loads((parent / "manifest.json").read_bytes())
    if manifest.get("stage") != "entity_enrichment":
        return None
    return {r["corpus_entity_id"]: EntityFacts.model_validate(r) for r in read_jsonl(parent / "entity_facts.jsonl")}
