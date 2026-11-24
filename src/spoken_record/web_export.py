"""JSON data for the exploration web app, made from the knowledge graph (graph.py).

This only decides how the graph is shown: episode codes and titles, which relations (the
NLI-supported ones) and which co-occurrence pairs become edges, and the image links. It
makes no identity, relation or similarity decisions of its own. The edge kinds stay
separate so the app can show what each kind of connection does and doesn't mean.
"""

import hashlib
import json
import re
from pathlib import Path
from urllib.parse import quote

from spoken_record.adapters.base import stable_id
from spoken_record.web_models import SiteConfig, WebExportConfig

DATA_FILES = ("site.json", "episodes.json", "passages.json", "entities.json", "edges.json", "related.json", "summary.json")
SCHEMA_VERSION = 2


def encode(value) -> bytes:
    """JSON written the same way every time, so validation can compare a rebuild byte for byte."""
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def facet(episode: dict, name: str | None) -> str | None:
    return episode["facets"].get(name) if name else None


def episode_code(episode: dict, site: SiteConfig) -> str:
    """Short code like "HoP 243" from the series facet and the episode number in the title."""
    number = re.match(site.episode_number_pattern, episode["title"]) if site.episode_number_pattern else None
    series = facet(episode, site.series_facet) or ""
    return f"{series} {number.group(1)}".strip() if number else (series or episode["source_key"])


def episode_title(episode: dict, site: SiteConfig) -> str:
    if site.episode_number_pattern and (number := re.match(site.episode_number_pattern, episode["title"])):
        return episode["title"][number.end():].strip() or episode["title"]
    return episode["title"]


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
        "search_placeholder": site.search_placeholder, "co_discussed_share": site.co_discussed_share}


def relation_edges(relations: list[dict], config: WebExportConfig, shown: set[str] | None = None) -> list[dict]:
    """The relations the NLI check supported, as directed and named edges. The evidence is
    the supported pieces outside boilerplate. ``shown`` limits the edges to entities that
    get edges at all."""
    edges = []
    for r in relations:
        pieces = [e for e in r["evidence"] if e["nli"] == "supported" and not e.get("boilerplate")]
        if r["decision"] != "supported" or not pieces or r["predicate"] in config.hidden_predicates \
                or (shown is not None and not {r["source"], r["target"]} <= shown):
            continue
        edges.append({"id": stable_id("we", r["source"], r["predicate"], r["target"]),
            "kind": "relation", "source": r["source"], "target": r["target"], "predicate": r["predicate"],
            "label": config.predicate_labels.get(r["predicate"], r["predicate"].replace("_", " ")),
            "inverse_label": config.inverse_labels.get(
                r["predicate"], f'{config.predicate_labels.get(r["predicate"], r["predicate"].replace("_", " "))} (by)'),
            **({"group": config.predicate_groups[r["predicate"]]} if r["predicate"] in config.predicate_groups else {}),
            "status": "nli_supported", "confidence": round(max(e["score"] for e in pieces), 4),
            "evidence": sorted({e["passage_id"] for e in pieces})})
    return edges


def co_discussed_edges(pairs: list[dict], types: dict[str, str], config: WebExportConfig,
                       shown: set[str] | None = None) -> list[dict]:
    """"Discussed together" edges between the configured types, from a minimum number of
    shared passages on. All of them are exported: which ones are drawn depends on the
    filters, so the app picks them (each item's strongest share, see web/src/lib/strength.ts)."""
    wanted = set(config.co_discussed_types)
    return [{"id": row["id"], "kind": "co_discussed", "source": row["source"], "target": row["target"],
             "count": row["count"], "pmi": row["pmi"], "evidence": row["evidence"]}
            for row in pairs if types[row["source"]] in wanted and types[row["target"]] in wanted
            and row["count"] >= config.min_shared_passages
            and (shown is None or {row["source"], row["target"]} <= shown)]


def only_in_boilerplate(node: dict) -> bool:
    return not node["mentions"] and node.get("boilerplate_mentions", 0) > 0


def entity_row(node: dict, episode_of: dict[str, str]) -> dict:
    """The entity as the app shows it. Its episodes and mention count only come from mentions
    outside boilerplate, so the host is not "in every episode" because of the intro."""
    row = {"id": node["id"], "label": node["label"], "type": node["type"], "aliases": node["aliases"],
           "qid": node["qid"], "link_status": node["link_status"], "mention_count": len(node["mentions"]),
           "episode_ids": sorted({episode_of[m["passage_id"]] for m in node["mentions"]}),
           "resolution_status": node["resolution_status"], "occurrences": node["mentions"]}
    if node.get("generic"):
        row["generic"] = True
    facts = node.get("facts")
    if facts:
        if facts["life"] is not None:
            row["life"] = facts["life"]
        if facts["image"] and commons_image(facts["image"]):
            row["image"] = commons_image(facts["image"])
    return row


# Browsers can show these directly. Commons renders SVG thumbnails as PNG.
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg"}
THUMB_WIDTH = 250  # one of the thumbnail widths Wikimedia keeps ready


def commons_image(name: str) -> dict | None:
    """Links for a Commons file: a thumbnail, the original and the file page (author, licence).
    Commons stores a file under the first one and two hex digits of the MD5 of its name."""
    name = name.strip().replace(" ", "_")
    suffix = Path(name).suffix.lower()
    if suffix not in IMAGE_SUFFIXES:
        return None
    digest = hashlib.md5(name.encode("utf-8")).hexdigest()
    quoted = quote(name, safe="()',!-._~")
    base = "https://upload.wikimedia.org/wikipedia/commons"
    thumb = f"{base}/thumb/{digest[0]}/{digest[:2]}/{quoted}/{THUMB_WIDTH}px-{quoted}" + (".png" if suffix == ".svg" else "")
    return {"file": name.replace("_", " "), "thumb": thumb,
            "original": f"{base}/{digest[0]}/{digest[:2]}/{quoted}",
            "page": f"https://commons.wikimedia.org/wiki/File:{quoted}"}


def build_web_data(graph: dict, config: WebExportConfig) -> dict[str, bytes]:
    """Turn the knowledge graph into the app's data files."""
    site = config.site
    episode_rows = [{"id": e["id"], "code": episode_code(e, site), "title": episode_title(e, site),
        "date": e["publication_date"][:10] if e["publication_date"] else None, "series": facet(e, site.series_facet),
        "era": facet(e, site.era_facet), "audio_url": e["audio_url"], "source_url": e["source_url"],
        "passage_count": e["passage_count"]} for e in graph["episodes"]]
    passage_rows = [{"id": p["id"], "episode_id": p["episode_id"], "index": p["index"], "start_s": p["start_s"],
        "end_s": p["end_s"], "word_count": p["word_count"], **({"text": p["text"], "timing": p.get("timing", [])} if config.include_passage_text else {})}
        for p in graph["passages"]]
    # An entity only heard in boilerplate (the host in the intro) is left out. Generic terms
    # stay searchable but get no edges.
    visible = [n for n in graph["nodes"] if not only_in_boilerplate(n)]
    episode_of = {p["id"]: p["episode_id"] for p in graph["passages"]}
    entity_rows = [entity_row(n, episode_of) for n in visible]
    types = {n["id"]: n["type"] for n in graph["nodes"]}
    shown = {n["id"] for n in visible if not n.get("generic")}
    edges = relation_edges(graph["relations"], config, shown) + \
        co_discussed_edges(graph["co_occurrence"], types, config, shown)
    related = sorted(({"source": r["source"], "target": r["target"], "score": round(r["score"], 4), "rank": r["rank"],
                       "shared_entity_ids": r["shared_entity_ids"]}
                      for r in graph["related"] if r["method"] == config.discovery_method),
                     key=lambda r: (r["source"], r["rank"], r["target"]))
    summary = {"schema_version": SCHEMA_VERSION, "episodes": len(episode_rows), "passages": len(passage_rows),
        "entities": len(entity_rows), "relation_edges": sum(e["kind"] == "relation" for e in edges),
        "co_discussed_edges": sum(e["kind"] == "co_discussed" for e in edges), "related_passages": len(related),
        "relation_status": "nli_supported", "discovery_method": config.discovery_method,
        "lifespans": "enriched", "dated_entities": sum("life" in r for r in entity_rows)}
    # The headings for relations, in the order the config first names them.
    groups = list(dict.fromkeys(config.predicate_groups.values()))
    payload = {"site.json": {**site_row(site, {r["type"] for r in entity_rows}), **({"relation_groups": groups} if groups else {})},
               "episodes.json": episode_rows,
               "passages.json": passage_rows, "entities.json": entity_rows, "edges.json": edges,
               "related.json": related, "summary.json": summary}
    return {name: encode(payload[name]) for name in DATA_FILES}
