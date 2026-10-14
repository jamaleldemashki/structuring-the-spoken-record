"""JSON data for the exploration web app.

Like the Obsidian vault, this is only a projection of validated runs. It makes no identity,
relation or similarity decisions of its own. The edge kinds stay separate so the app can
show what each kind of connection does and doesn't mean.
"""

import importlib.metadata
import json
import math
import platform
import re
from collections import defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from spoken_record import __version__
from spoken_record.adapters.base import sha256, stable_id
from spoken_record.ingestion import dump_json, workspace_path
from spoken_record.vault import discovery_view, load_resolution_chain, vault_input, verification_view
from spoken_record.web_models import SiteConfig, WebExportConfig

DATA_FILES = ("site.json", "episodes.json", "passages.json", "entities.json", "edges.json", "related.json", "summary.json")
SCHEMA_VERSION = 2


def encode(value) -> bytes:
    """JSON written the same way every time, so validation can compare a rebuild byte for byte."""
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def facet(episode, name: str | None) -> str | None:
    return episode.facets.get(name) if name else None


def episode_code(episode, site: SiteConfig) -> str:
    """Short code like "HoP 243" from the series facet and the episode number in the title."""
    number = re.match(site.episode_number_pattern, episode.title) if site.episode_number_pattern else None
    series = facet(episode, site.series_facet) or ""
    return f"{series} {number.group(1)}".strip() if number else (series or episode.source_key)


def episode_title(episode, site: SiteConfig) -> str:
    if site.episode_number_pattern and (number := re.match(site.episode_number_pattern, episode.title)):
        return episode.title[number.end():].strip() or episode.title
    return episode.title


def site_row(site: SiteConfig, present_types: set[str]) -> dict:
    """The site description the app is built from.

    Types that appear in the data but not in the config are added at the end, hidden by
    default, so nothing becomes unreachable.
    """
    types = [t.model_dump() for t in site.entity_types]
    listed = {t["type"] for t in types}
    types += [{"type": t, "label": t, "plural": t, "visible_by_default": False}
              for t in sorted(present_types - listed)]
    return {"schema_version": SCHEMA_VERSION, "archive_id": site.archive_id, "title": site.title,
        "subtitle": site.subtitle, "series_label": site.series_label if site.series_facet else None,
        "era_label": site.era_label if site.era_facet else None, "entity_types": types,
        "lifespan_type": site.lifespan_type, "lived_presets": [p.model_dump() for p in site.lived_presets],
        "search_placeholder": site.search_placeholder}


def primary_passage(mention, passages_by_id: dict) -> str:
    """The passage that contains the first character of the mention."""
    for pid in mention.passage_ids:
        p = passages_by_id[pid]
        if p.char_start <= mention.char_start < p.char_end:
            return pid
    raise ValueError(f"mention {mention.mention_id} starts outside its recorded passages")


def entity_occurrences(passages, mentions, assignments) -> dict[str, list[dict]]:
    """All mentions of each entity, as offsets relative to their passage (clamped to it)."""
    by_id = {p.passage_id: p for p in passages}
    entity_of = {a.mention_id: a.corpus_entity_id for a in assignments}
    occurrences = defaultdict(list)
    for m in mentions:
        pid = primary_passage(m, by_id)
        p = by_id[pid]
        occurrences[entity_of[m.mention_id]].append(
            {"passage_id": pid, "start": m.char_start - p.char_start, "end": min(m.char_end, p.char_end) - p.char_start})
    for rows in occurrences.values():
        rows.sort(key=lambda r: (r["passage_id"], r["start"], r["end"]))
    return occurrences


def relation_edges(assertions, status: str, config: WebExportConfig) -> list[dict]:
    """Directed, named edges from the relation assertions, with their evidence passages."""
    edges = []
    for a in assertions or []:
        edges.append({"id": stable_id("we", a.subject_corpus_entity_id, a.predicate, a.object_corpus_entity_id),
            "kind": "relation", "source": a.subject_corpus_entity_id, "target": a.object_corpus_entity_id,
            "predicate": a.predicate, "label": config.predicate_labels.get(a.predicate, a.predicate.replace("_", " ")),
            "status": status, "confidence": round(a.confidence, 4),
            "evidence": sorted({e.passage_id for e in a.evidence})})
    return edges


def co_discussed_edges(occurrences: dict, types: dict, passage_count: int, config: WebExportConfig) -> list[dict]:
    """Undirected "discussed together" edges, weighted by shared passages and PMI.

    PMI compares how often two entities share a passage with what chance would give, so
    generic pairs rank below specific ones. It only describes co-occurrence, not a relation.
    """
    wanted = set(config.co_discussed_types)
    passages_of = {cid: {o["passage_id"] for o in rows} for cid, rows in occurrences.items() if types[cid] in wanted}
    members = defaultdict(set)
    for cid, pids in passages_of.items():
        for pid in pids:
            members[pid].add(cid)
    shared = defaultdict(set)
    for pid, cids in members.items():
        for a, b in combinations(sorted(cids), 2):
            shared[(a, b)].add(pid)
    edges = []
    for (a, b), pids in sorted(shared.items()):
        if len(pids) < config.min_shared_passages:
            continue
        pmi = math.log(len(pids) * passage_count / (len(passages_of[a]) * len(passages_of[b])))
        edges.append({"id": stable_id("wc", a, b), "kind": "co_discussed", "source": a, "target": b,
                      "count": len(pids), "pmi": round(pmi, 4), "evidence": sorted(pids)})
    return edges


def entity_row(e, occurrences: dict, facts) -> dict:
    row = {"id": e.corpus_entity_id, "label": e.canonical_label, "type": e.type,
        "aliases": sorted(set(e.observed_surfaces) - {e.canonical_label}), "qid": e.wikidata_qid,
        "link_status": e.link_status, "mention_count": e.mention_count, "episode_ids": sorted(e.episode_ids),
        "resolution_status": e.resolution_status, "occurrences": occurrences.get(e.corpus_entity_id, [])}
    if facts is not None:
        row["qid"] = facts.qid
        if facts.life is not None:
            row["life"] = facts.life.model_dump()
    return row


def build_web_data(config: WebExportConfig, episodes, transcripts, passages, mentions, episode_entities,
                   corpus_entities, assignments, review_queue, coreference_links, relation_assertions,
                   verification, discovery, enrichment=None) -> dict[str, bytes]:
    """Turn the validated records into the app's data files."""
    occurrences = entity_occurrences(passages, mentions, assignments)
    passages_per_episode = defaultdict(int)
    for p in passages:
        passages_per_episode[p.episode_id] += 1
    site = config.site
    episode_rows = [{"id": e.episode_id, "code": episode_code(e, site), "title": episode_title(e, site),
        "date": e.publication_date[:10] if e.publication_date else None, "series": facet(e, site.series_facet),
        "era": facet(e, site.era_facet), "audio_url": e.audio_url, "source_url": e.source_url,
        "passage_count": passages_per_episode[e.episode_id]} for e in sorted(episodes, key=lambda e: e.episode_id)]
    passage_rows = [{"id": p.passage_id, "episode_id": p.episode_id, "index": p.index, "start_s": p.start_s,
        "end_s": p.end_s, "word_count": p.word_count, **({"text": p.text} if config.include_passage_text else {})}
        for p in sorted(passages, key=lambda p: (p.episode_id, p.index))]
    entity_rows = [entity_row(e, occurrences, (enrichment or {}).get(e.corpus_entity_id))
                   for e in sorted(corpus_entities, key=lambda e: e.corpus_entity_id)]
    status = "nli_supported" if verification is not None else "unverified"
    types = {e.corpus_entity_id: e.type for e in corpus_entities}
    edges = relation_edges(relation_assertions, status, config) + \
        co_discussed_edges(occurrences, types, len(passages), config)
    related = []
    if discovery is not None:
        for c in discovery["connections"]:
            if c["method"] == config.discovery_method and c["decision"] == "selected":
                related.append({"source": c["source_passage_id"], "target": c["target_passage_id"],
                    "score": round(c["score"], 4), "rank": c["rank"], "shared_entity_ids": sorted(c["shared_entity_ids"])})
        related.sort(key=lambda r: (r["source"], r["rank"], r["target"]))
    summary = {"schema_version": SCHEMA_VERSION, "episodes": len(episode_rows), "passages": len(passage_rows),
        "entities": len(entity_rows), "relation_edges": sum(e["kind"] == "relation" for e in edges),
        "co_discussed_edges": sum(e["kind"] == "co_discussed" for e in edges), "related_passages": len(related),
        "relation_status": status if relation_assertions is not None else "none",
        "discovery_method": config.discovery_method if discovery is not None else None,
        "lifespans": "not_enriched"}
    if enrichment is not None:
        summary["lifespans"] = "enriched"
        summary["dated_entities"] = sum("life" in r for r in entity_rows)
    payload = {"site.json": site_row(site, {r["type"] for r in entity_rows}), "episodes.json": episode_rows, "passages.json": passage_rows, "entities.json": entity_rows,
               "edges.json": edges, "related.json": related, "summary.json": summary}
    return {name: encode(payload[name]) for name in DATA_FILES}


def load_inputs(root: Path, parent: Path):
    """Use the vault's chain readers so both exports read exactly the same records."""
    from spoken_record.enrichment import enrichment_view
    resolution_parent, coreference_links, relation_assertions = vault_input(root, parent)
    records = load_resolution_chain(root, resolution_parent)
    return (records, coreference_links, relation_assertions, verification_view(parent), discovery_view(parent),
            enrichment_view(parent))


def export_web(root: Path, input_name: str, config_name: str, run_id: str | None = None) -> Path:
    """Write the app data for a validated identity, relation, verification or discovery run."""
    root, started, now = root.resolve(), perf_counter(), datetime.now(timezone.utc)
    run_id = run_id or f"web-{now:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]{0,95}", run_id):
        raise ValueError("invalid run ID")
    parent = workspace_path(root, input_name)
    if parent.parent != root / "data/prototype1/runs":
        raise ValueError("input must be a local run snapshot")
    output = root / "export/prototype1" / run_id
    if output.exists():
        raise FileExistsError(output)
    parent_digest = sha256((parent / "manifest.json").read_bytes())
    config_path = workspace_path(root, config_name)
    config_bytes = config_path.read_bytes()
    config = WebExportConfig.model_validate_json(config_bytes)
    records, links, assertions, verification, discovery, enrichment = load_inputs(root, parent)
    files = build_web_data(config, *records, links, assertions, verification, discovery, enrichment)
    if sha256((parent / "manifest.json").read_bytes()) != parent_digest or config_path.read_bytes() != config_bytes:
        raise ValueError("parent or web configuration changed")
    for name, data in files.items():
        path = output / "data" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    code_root = Path(__file__).parent
    manifest = {"schema_version": SCHEMA_VERSION, "stage": "export_web", "run_id": run_id, "status": "complete",
        "package_version": __version__, "started_utc": now.isoformat(),
        "elapsed_seconds": round(perf_counter() - started, 4), "python": platform.python_version(),
        "dependencies": {"pydantic": importlib.metadata.version("pydantic")},
        "parent_run": input_name, "parent_manifest_sha256": parent_digest,
        "input_config": config_name, "input_config_sha256": sha256(config_bytes),
        "counts": json.loads(files["summary.json"]),
        "code_sha256": {str(p.relative_to(code_root)): sha256(p.read_bytes()) for p in sorted(code_root.rglob("*.py"))},
        "artifact_sha256": {f"data/{name}": sha256(data) for name, data in sorted(files.items())}}
    dump_json(output / "manifest.json", manifest)
    validate_run(output)
    return output


def validate_run(output: Path) -> dict:
    """Rebuild the data from the parent run and require identical files."""
    output = output.resolve()
    manifest = json.loads((output / "manifest.json").read_bytes())
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("stage") != "export_web" or manifest.get("status") != "complete":
        raise ValueError("unsupported web export")
    root = output.parents[2]
    parent = workspace_path(root, manifest["parent_run"])
    if sha256((parent / "manifest.json").read_bytes()) != manifest["parent_manifest_sha256"]:
        raise ValueError("parent manifest changed")
    config_path = workspace_path(root, manifest["input_config"])
    if sha256(config_path.read_bytes()) != manifest["input_config_sha256"]:
        raise ValueError("web configuration changed")
    config = WebExportConfig.model_validate_json(config_path.read_bytes())
    records, links, assertions, verification, discovery, enrichment = load_inputs(root, parent)
    expected = build_web_data(config, *records, links, assertions, verification, discovery, enrichment)
    if set(manifest["artifact_sha256"]) != {f"data/{name}" for name in expected}:
        raise ValueError("unexpected web artifact set")
    for name, data in expected.items():
        actual = (output / "data" / name).read_bytes()
        if actual != data or sha256(actual) != manifest["artifact_sha256"][f"data/{name}"]:
            raise ValueError(f"web artifact is changed or not reproducible: data/{name}")
    if manifest["counts"] != json.loads(expected["summary.json"]):
        raise ValueError("web manifest counts differ")
    return manifest
