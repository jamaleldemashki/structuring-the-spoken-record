"""Checks a build without running any model or asking Wikidata.

    spoken-record validate --build data/builds/hop-230-254

1. Every file of the build and of its web export still has the hash written at build time.
2. The decisions are redone from the stored scores, with the copy of the settings the build
   made: the Wikidata link decisions from the candidate scores, the NLI verdict of every
   relation from its evidence, and the lifespans, occupations and images from the stored
   Wikidata claims.
3. ``graph.json`` is rebuilt from the records and must be the same byte for byte, and so must
   the app data made from it.

Step 3 also needs the episode folders the build was made from. If a later build updated
one of them (for example after a settings change), that part is skipped and the report says
so. Anything else that does not match raises a ValueError.
"""

import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from spoken_record.adapters.base import sha256
from spoken_record.ingestion import read_jsonl, workspace_path


def rows(folder: Path, name: str, model):
    return [model.model_validate(row) for row in read_jsonl(folder / name)]


def check_hashes(folder: Path, hashes: dict[str, str], what: str) -> None:
    for name, digest in hashes.items():
        path = folder / name
        if not path.is_file() or sha256(path.read_bytes()) != digest:
            raise ValueError(f"{what} file is missing or was changed: {name}")


def validate_build(root: Path, build: Path) -> dict:
    from spoken_record.corpus import CorpusConfig
    from spoken_record.enrichment import build_facts, load_reviewed
    from spoken_record.enrichment_models import EnrichmentConfig, EntityFacts
    from spoken_record.entity_linking import make_decisions
    from spoken_record.graph import build_graph, encode
    from spoken_record.linking_models import EntityLinkingConfig, LinkDecision, WikidataCandidate
    from spoken_record.relation_models import RelationAssertion, RelationCandidate
    from spoken_record.relation_verification import aggregate
    from spoken_record.resolution_models import CorpusEntity, MentionAssignment
    from spoken_record.verification_models import VerificationDecision, VerificationEvidence
    from spoken_record.web_export import build_web_data
    from spoken_record.web_models import WebExportConfig

    root, build = root.resolve(), build.resolve()
    manifest = json.loads((build / "manifest.json").read_bytes())
    if manifest.get("stage") != "corpus_build" or manifest.get("status") != "complete":
        raise ValueError("not a complete corpus build")
    report = {"build": manifest["build_id"], "checks": [], "skipped": []}
    check_hashes(build, manifest["artifact_sha256"], "build")
    report["checks"].append(f"{len(manifest['artifact_sha256'])} build files unchanged")

    export = workspace_path(root, manifest["export"])
    export_manifest = json.loads((export / "manifest.json").read_bytes())
    if export_manifest["build_manifest_sha256"] != sha256((build / "manifest.json").read_bytes()):
        raise ValueError("the web export does not belong to this build")
    check_hashes(export, export_manifest["artifact_sha256"], "web export")
    report["checks"].append(f"{len(export_manifest['artifact_sha256'])} web export files unchanged")

    def settings(model, step):
        return model.model_validate_json((build / "settings" / manifest["settings"][step]).read_bytes())

    def setting_file(path):
        return (build / "settings" / path).read_bytes()

    run_id = manifest["build_id"]
    entities = rows(build, "corpus_entities.jsonl", CorpusEntity)

    link_decisions = rows(build, "link_decisions.jsonl", LinkDecision)
    linked = [SimpleNamespace(corpus_entity_id=row.corpus_entity_id, type=row.entity_type,
                              canonical_label=row.canonical_label) for row in link_decisions]
    redone = make_decisions(run_id, linked, rows(build, "wikidata_candidates.jsonl", WikidataCandidate),
                            settings(EntityLinkingConfig, "link"))
    if list(redone) != link_decisions:
        raise ValueError("the Wikidata link decisions do not follow from the stored candidate scores")
    report["checks"].append(f"{len(link_decisions)} Wikidata link decisions redone from their scores")

    candidates = rows(build, "relation_candidates.jsonl", RelationCandidate)
    assertions = rows(build, "relation_assertions.jsonl", RelationAssertion)
    evidence = rows(build, "verification_evidence.jsonl", VerificationEvidence)
    decisions = rows(build, "verification_decisions.jsonl", VerificationDecision)
    accepted = [row for row in candidates if row.decision == "accepted"]
    redone_decisions, supported = aggregate(run_id, accepted, assertions, evidence)
    if redone_decisions != decisions or supported != rows(build, "verified_assertions.jsonl", RelationAssertion):
        raise ValueError("the relation verdicts do not follow from the stored NLI evidence")
    report["checks"].append(f"{len(decisions)} relation verdicts redone from the NLI evidence")

    enrichment = settings(EnrichmentConfig, "enrich")
    reviewed = load_reviewed(setting_file(enrichment.reviewed_qids) if enrichment.reviewed_qids else b"")
    raw = {row["qid"]: row for row in read_jsonl(build / "wikidata_entities.jsonl")}
    year = datetime.fromisoformat(manifest["started_utc"]).year
    facts = rows(build, "entity_facts.jsonl", EntityFacts)
    if build_facts(run_id, entities, reviewed, raw, year, enrichment) != facts:
        raise ValueError("the lifespans and images do not follow from the stored Wikidata claims")
    report["checks"].append(f"{len(facts)} lifespans and images redone from the stored claims")

    graph_bytes = (build / "graph.json").read_bytes()
    graph = json.loads(graph_bytes)
    web = build_web_data(graph, settings(WebExportConfig, "web"))
    for name, data in web.items():
        if (export / "data" / name).read_bytes() != data:
            raise ValueError(f"the web data does not follow from graph.json: {name}")
    report["checks"].append("web data rebuilt from graph.json, byte for byte")
    if "search.json" in {name.removeprefix("data/") for name in export_manifest["artifact_sha256"]}:
        report["skipped"].append("search.json holds model vectors, only its hash is checked")

    from spoken_record.episodes import episode_dir, load_episode
    from spoken_record.ingestion import IngestSetup
    corpus_path = build / "settings" / manifest["settings"]["corpus"]
    corpus = CorpusConfig.model_validate_json(corpus_path.read_bytes())
    setup = IngestSetup(root, corpus.model_dump(), corpus_path)
    selection = json.loads((build / "selection.json").read_bytes())["episodes"]
    folders = [episode_dir(root, corpus.corpus_id, setup, row["source_path"]) for row in selection]
    changed = [row["code"] or row["source_path"] for row, folder in zip(selection, folders)
               if not (folder / "manifest.json").is_file() or sha256(json.dumps(json.loads(
                   (folder / "manifest.json").read_bytes()), sort_keys=True).encode()) != row["episode_folder_sha256"]]
    if changed:
        report["skipped"].append("graph.json not rebuilt: a later build updated the episode folders of "
                                 + ", ".join(changed[:5]) + (" and more" if len(changed) > 5 else ""))
        return report
    episodes = sorted((load_episode(folder) for folder in folders), key=lambda row: row["bundle"].episode.episode_id)
    passages = [p for row in episodes for p in sorted(row["passages"], key=lambda p: p.index)]
    from spoken_record.resolution import name_mentions
    from spoken_record.resolution_models import ResolutionConfig
    mentions = name_mentions(sorted((m for row in episodes for m in row["mentions"]),
                                    key=lambda m: (m.transcript_version_id, m.char_start, m.char_end, m.type)),
                             settings(ResolutionConfig, "resolve"))
    boilerplate, generic = {}, set()
    if "filters" in manifest["settings"]:
        from spoken_record.filters import FiltersConfig, boilerplate_spans, generic_entities
        filters = settings(FiltersConfig, "filters")
        boilerplate = boilerplate_spans([row["bundle"] for row in episodes], filters.boilerplate)
        generic = generic_entities(entities, filters.generic)
    rebuilt = build_graph(run_id, corpus.corpus_id, [row["bundle"].episode for row in episodes], passages, mentions,
                          entities, rows(build, "assignments.jsonl", MentionAssignment), candidates, assertions,
                          evidence, decisions, facts, read_jsonl(build / "passage_connections.jsonl"),
                          boilerplate, generic, [segment for row in episodes for segment in row["bundle"].segments])
    if encode(rebuilt) != graph_bytes:
        raise ValueError("graph.json does not follow from the records of the build")
    report["checks"].append("graph.json rebuilt from the records and the episode folders, byte for byte")
    return report
