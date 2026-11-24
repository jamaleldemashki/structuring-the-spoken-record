"""The corpus build: joins the episode graphs into one knowledge graph and the app data.

``update`` first brings the selected episode folders up to date (see episodes.py). Each one
already holds the graph of its episode: entities, pronoun links, relations and their NLI
check. The build then does what needs several episodes, always in this order:

1. merge: episode entities into corpus entities, by name and name variants, keeping every
   merge made inside an episode
2. link: Wikidata candidates and decision per corpus entity, then entities of the same type
   with the same Wikidata item become one (the types are set in the corpus config)
3. lift: pronoun links and relations from episode entities to corpus entities, and the
   NLI results of all episodes combined per relation
4. discover: related passages in other episodes, from the stored passage vectors
5. enrich: lifespans, occupations and images from Wikidata
6. graph: the knowledge graph of the build, ``graph.json`` (see graph.py)
7. web: the files the app reads, made from the graph

Wikidata requests go through the cache in ``data/cache/``. A build is written once to
``data/builds/<id>/`` and never changed.
"""

import importlib.metadata
import json
import platform
import re
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from spoken_record import __version__
from spoken_record.adapters.base import sha256
from spoken_record.cache import Cache, CachedEncoder
from spoken_record.corpus import load_corpus, select_episodes
from spoken_record.episodes import Models, episode_dir, load_episode, update_episode
from spoken_record.graph import encode as encode_graph
from spoken_record.ingestion import dump_json, dump_jsonl, read_jsonl, workspace_path

BUILD_ROOT = "data/builds"
EXPORT_ROOT = "export"


def config_of(root: Path, model, name: str):
    data = workspace_path(root, name).read_bytes()
    return model.model_validate_json(data), data


def update(root: Path, corpus_name: str, selection: str, build_id: str | None = None, *,
           refresh_wikidata: bool = False, loaders: dict | None = None, progress=print) -> Path:
    """Process the selected episodes that are new or changed, then build the corpus."""
    root, started, now = root.resolve(), perf_counter(), datetime.now(timezone.utc)
    loaders = loaders or {}
    corpus, setup = load_corpus(root, corpus_name)
    build_id = build_id or f"{setup.corpus_id}-{now:%Y%m%dT%H%M%SZ}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]{0,95}", build_id):
        raise ValueError("build ID must contain only letters, numbers and dashes")
    output = root / BUILD_ROOT / build_id
    export = root / EXPORT_ROOT / f"{build_id}-web"
    if output.exists() or export.exists():
        raise FileExistsError(output if output.exists() else export)
    chosen = select_episodes(setup, corpus.episode_codes, selection)
    progress(f"{len(chosen)} episodes selected")

    wikidata = Cache(root / "data/cache/wikidata.sqlite")
    model_cache = Cache(root / "data/cache/models.sqlite")
    if refresh_wikidata:
        wikidata.clear()
    models = Models(root, corpus, model_cache, loaders)
    reports = []
    for number, selected in enumerate(chosen, 1):
        report = update_episode(root, corpus, setup, selected, models)
        reports.append(report)
        progress(f"[{number}/{len(chosen)}] {selected.code or selected.source_path}: "
                 + (", ".join(report["redone"]) if report["redone"] else "up to date"))

    try:
        episodes = sorted((load_episode(episode_dir(root, setup.corpus_id, setup, s.source_path)) for s in chosen),
                          key=lambda row: row["bundle"].episode.episode_id)
        records = build_records(root, corpus, setup, episodes, build_id, wikidata, model_cache, models,
                                loaders, now, progress)
    finally:
        cache_counts = {"wikidata_hits": wikidata.hits, "wikidata_misses": wikidata.misses,
                        "model_hits": model_cache.hits, "model_misses": model_cache.misses}
        wikidata.close()
        model_cache.close()
    progress("writing the build")
    write_build(root, output, export, corpus_name, corpus, build_id, chosen, episodes, reports, records,
                cache_counts, now, started)
    return output


def build_records(root, corpus, setup, episodes, run_id, wikidata, model_cache, models, loaders, now, progress):
    from spoken_record.consolidation import (corpus_identities, lift_coreference, lift_relations,
                                             merge_by_wikidata, supported_by_namesake, unit_map)
    from spoken_record.discovery import PassageEncoder, compute_discovery
    from spoken_record.discovery_models import DiscoveryConfig
    from spoken_record.enrichment import build_facts, load_reviewed, make_retriever, targets
    from spoken_record.enrichment_models import EnrichmentConfig
    from spoken_record.entity_linking import (MpnetEncoder, WikidataRetriever, apply_decisions,
                                              linking_inputs, make_decisions, retrieve_and_score)
    from spoken_record.identity_models import IdentityRefinementConfig
    from spoken_record.linking_models import EntityLinkingConfig
    from spoken_record.relation_verification import aggregate
    from spoken_record.relations import build_cooccurrence
    from spoken_record.resolution import name_mentions, resolution_fingerprint, rules_for, validate_aliases
    from spoken_record.resolution_models import AliasRule, ResolutionConfig
    from spoken_record.graph import build_graph
    from spoken_record.web_export import build_web_data
    from spoken_record.web_models import WebExportConfig

    steps = corpus.steps
    episode_records = [row["bundle"].episode for row in episodes]
    transcripts = [row["bundle"].transcript for row in episodes]
    passages = [p for row in episodes for p in sorted(row["passages"], key=lambda p: p.index)]
    mentions = sorted((m for row in episodes for m in row["mentions"]),
                      key=lambda m: (m.transcript_version_id, m.char_start, m.char_end, m.type))
    unit_assignments = [a for row in episodes for a in row["assignments"]]

    progress("merge")
    resolution, _ = config_of(root, ResolutionConfig, steps.resolve)
    mentions = name_mentions(mentions, resolution)
    rules = [AliasRule.model_validate(row) for row in read_jsonl(workspace_path(root, resolution.aliases))]
    rules = rules_for(rules, mentions, episode_records)
    validate_aliases(rules, mentions, episode_records)
    refinement, _ = config_of(root, IdentityRefinementConfig, steps.refine)
    entities, assignments, resolution_decisions, queue, redirects, identity_candidates = corpus_identities(
        run_id, setup.corpus_id, mentions, passages, unit_assignments, rules, resolution,
        resolution_fingerprint(resolution, rules), refinement)
    from spoken_record.resolution import resolve_records
    episode_entities = resolve_records(run_id, setup.corpus_id, mentions, passages, rules, resolution,
                                       resolution_fingerprint(resolution, rules))[0]

    progress("link")
    config, _ = config_of(root, EntityLinkingConfig, steps.link)
    eligible, contexts, _ = linking_inputs(list(entities), mentions, list(assignments), passages, config)
    retriever = loaders.get("wikidata", WikidataRetriever)(config)
    retriever.cache = wikidata
    encoder = CachedEncoder(model_cache, "mpnet-link",
        {"repo_id": config.embedding_repo_id, "revision": config.embedding_revision,
         "max_tokens": config.embedding_max_tokens, "device": config.embedding_device},
        lambda: loaders.get("link_encoder", MpnetEncoder)(config))
    wikidata_candidates = retrieve_and_score(run_id, eligible, contexts, config, retriever, encoder)
    link_decisions = make_decisions(run_id, eligible, list(wikidata_candidates), config)
    entities = apply_decisions(run_id, list(entities), list(link_decisions))
    if corpus.merge_by_wikidata:
        progress("merge by Wikidata item")
        supported = supported_by_namesake(list(entities), link_decisions, wikidata_candidates,
                                          config.linked_threshold, corpus.merge_by_wikidata, config.type_weight)
        entities, assignments, resolution_decisions, redirects = merge_by_wikidata(
            run_id, list(entities), list(assignments), list(resolution_decisions), list(redirects),
            identity_candidates, link_decisions, corpus.merge_by_wikidata, supported)

    progress("lift")
    units = unit_map(unit_assignments, assignments)
    links = lift_coreference([link for row in episodes for link in row["coreference_links"]], units)
    relation_candidates, assertions = lift_relations(
        run_id, [c for row in episodes for c in row["relation_candidates"]], units)
    evidence = [e for row in episodes for e in row["verification_evidence"]]
    accepted = [row for row in relation_candidates if row.decision == "accepted"]
    verification_decisions, supported = aggregate(run_id, accepted, list(assertions), evidence)
    relation_contexts = [c for row in episodes for c in row["relation_contexts"]]
    cooccurrence = build_cooccurrence(run_id, passages, mentions, list(assignments))

    progress("filters")
    boilerplate, generic = filter_records(root, corpus, [row["bundle"] for row in episodes], list(entities))

    progress("discover")
    config, _ = config_of(root, DiscoveryConfig, steps.discover)
    stored = {item["input_sha256"]: {k: v for k, v in item.items() if k not in ("passage_id", "input_sha256")}
              for row in episodes for item in row["embeddings"]}
    # Passages with boilerplate get a shorter text, which the episode folder has no vector for.
    fresh = CachedEncoder(model_cache, "mpnet-passage", {"repo_id": config.repo_id, "revision": config.revision,
                          "max_tokens": config.max_tokens, "batch_size": config.batch_size},
                          lambda: loaders.get("passage_encoder", PassageEncoder)(root, config))

    def encode(texts):
        missing = [text for text in texts if sha256(text.encode()) not in stored]
        if missing:
            stored.update({sha256(text.encode()): item for text, item in zip(missing, fresh.encode(missing), strict=True)})
        return [stored[sha256(text.encode())] for text in texts]

    identity = (episode_records, transcripts, passages, mentions, list(episode_entities), list(entities),
                list(assignments), list(queue))
    exclusions = [{"transcript_version_id": tv, "char_start": start, "char_end": end, "review_status": "automatic"}
                  for tv, spans in sorted(boilerplate.items()) for start, end in spans]
    _, _, _, connections, _ = compute_discovery(identity, exclusions, config, encode)

    progress("enrich")
    config, _ = config_of(root, EnrichmentConfig, steps.enrich)
    reviewed = load_reviewed(workspace_path(root, config.reviewed_qids).read_bytes() if config.reviewed_qids else b"")
    qids = sorted({qid for qid, _ in targets(list(entities), reviewed, config).values()})
    facts_retriever = loaders.get("facts", make_retriever)(config)
    facts_retriever.cache = wikidata
    raw_facts = facts_retriever.fetch(qids) if qids else {}
    if sorted(raw_facts) != qids:
        raise ValueError("retriever did not return exactly the requested QIDs")
    facts = build_facts(run_id, list(entities), reviewed, raw_facts, now.year, config)

    progress("graph")
    graph = build_graph(run_id, setup.corpus_id, episode_records, passages, mentions, list(entities),
                        list(assignments), list(relation_candidates), list(assertions), evidence,
                        verification_decisions, facts, connections, boilerplate, generic,
                        [segment for row in episodes for segment in row["bundle"].segments])

    progress("web")
    config, _ = config_of(root, WebExportConfig, steps.web)
    web_files = build_web_data(graph, config)

    if steps.search:
        # Vectors for asking questions in the app. They come from a model, so validation can
        # only check their hash, not redo them from graph.json.
        progress("search")
        from spoken_record.search_index import SearchConfig, search_payload
        from spoken_record.web_export import encode as encode_web
        config, _ = config_of(root, SearchConfig, steps.search)
        search_encoder = CachedEncoder(model_cache, "minilm-search", {"repo_id": config.repo_id, "revision": config.revision,
                                       "max_tokens": config.max_tokens, "batch_size": config.batch_size},
                                       lambda: loaders.get("search_encoder", PassageEncoder)(root, config))
        web_files["search.json"] = encode_web(search_payload(config, graph["passages"], graph["nodes"], search_encoder.encode))

    return {
        "jsonl": {
            "episode_entities.jsonl": episode_entities, "corpus_entities.jsonl": entities,
            "assignments.jsonl": assignments, "resolution_decisions.jsonl": resolution_decisions,
            "review_queue.jsonl": queue, "identity_candidates.jsonl": identity_candidates,
            "redirects.jsonl": redirects, "episode_entity_map.jsonl": [
                {"episode_entity_id": unit, "corpus_entity_id": target} for unit, target in sorted(units.items())],
            "wikidata_candidates.jsonl": wikidata_candidates, "link_decisions.jsonl": link_decisions,
            "coreference_links.jsonl": links, "relation_contexts.jsonl": relation_contexts,
            "relation_candidates.jsonl": relation_candidates, "relation_assertions.jsonl": assertions,
            "cooccurrence_edges.jsonl": cooccurrence, "verification_evidence.jsonl": evidence,
            "verification_decisions.jsonl": verification_decisions, "verified_assertions.jsonl": supported,
            "passage_connections.jsonl": connections, "wikidata_entities.jsonl": [raw_facts[q] for q in qids],
            "entity_facts.jsonl": facts},
        "graph": graph,
        "web": web_files,
    }


def filter_records(root, corpus, bundles, entities):
    """Boilerplate spans per transcript version and the IDs of generic entities."""
    from spoken_record.filters import FiltersConfig, boilerplate_spans, generic_entities
    if not corpus.steps.filters:
        return {}, set()
    settings, _ = config_of(root, FiltersConfig, corpus.steps.filters)
    return (boilerplate_spans(bundles, settings.boilerplate),
            generic_entities(entities, settings.generic))


def settings_files(root, corpus, corpus_name) -> list[str]:
    """The corpus config, the settings of every step and the files they name."""
    from spoken_record.enrichment_models import EnrichmentConfig
    from spoken_record.extraction_models import ExtractionConfig
    from spoken_record.resolution_models import ResolutionConfig

    steps = corpus.steps
    paths = {corpus_name, corpus.profile, *(path for path in steps.model_dump().values() if path)}
    paths.add(config_of(root, ExtractionConfig, steps.extract)[0].labels_config)
    paths.add(config_of(root, ResolutionConfig, steps.resolve)[0].aliases)
    reviewed = config_of(root, EnrichmentConfig, steps.enrich)[0].reviewed_qids
    if reviewed:
        paths.add(reviewed)
    return sorted(paths)


def write_build(root, output, export, corpus_name, corpus, build_id, chosen, episodes, reports, records,
                cache_counts, now, started):
    output.mkdir(parents=True)
    for name, rows in records["jsonl"].items():
        dump_jsonl(output / name, list(rows))
    (output / "graph.json").write_bytes(encode_graph(records["graph"]))
    selection = [{"code": s.code, "source_path": s.source_path,
                  "episode_id": row["bundle"].episode.episode_id,
                  "episode_folder_sha256": sha256(json.dumps(row["manifest"], sort_keys=True).encode())}
                 for s, row in zip(chosen, sorted(episodes, key=lambda row: row["manifest"]["source_path"]))]
    dump_json(output / "selection.json", {"episodes": selection})
    settings = {"corpus": corpus_name, **{name: path for name, path in corpus.steps.model_dump().items() if path}}
    # A copy of every settings file goes into the build, so it can be checked later even
    # after the settings were edited.
    copies = settings_files(root, corpus, corpus_name)
    for path in copies:
        target = output / "settings" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(workspace_path(root, path).read_bytes())
    support = {path: sha256(workspace_path(root, path).read_bytes()) for path in copies}
    names = sorted([*records["jsonl"], "selection.json", "graph.json", *(f"settings/{path}" for path in copies)])
    code_root = Path(__file__).parent
    manifest = {"schema_version": 1, "stage": "corpus_build", "build_id": build_id, "status": "complete",
        "package_version": __version__, "started_utc": now.isoformat(),
        "elapsed_seconds": round(perf_counter() - started, 4), "python": platform.python_version(),
        "export": str(export.relative_to(root)), "settings": settings, "settings_sha256": support, "episodes": reports, "cache": cache_counts,
        "counts": json.loads(records["web"]["summary.json"]),
        "code_sha256": {str(p.relative_to(code_root)): sha256(p.read_bytes()) for p in sorted(code_root.rglob("*.py"))},
        "artifact_sha256": {name: sha256((output / name).read_bytes()) for name in names}}
    dump_json(output / "manifest.json", manifest)

    for name, data in records["web"].items():
        path = export / "data" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    from spoken_record.web_export import SCHEMA_VERSION
    dump_json(export / "manifest.json", {"schema_version": SCHEMA_VERSION, "stage": "export_web",
        "run_id": f"{build_id}-web", "status": "complete", "package_version": __version__,
        "started_utc": now.isoformat(), "build": str(output.relative_to(root)),
        "build_manifest_sha256": sha256((output / "manifest.json").read_bytes()),
        "dependencies": {"pydantic": importlib.metadata.version("pydantic")},
        "counts": json.loads(records["web"]["summary.json"]),
        "artifact_sha256": {f"data/{name}": sha256(data) for name, data in sorted(records["web"].items())}})
