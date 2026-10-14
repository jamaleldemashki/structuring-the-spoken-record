"""Command line entry point: one command per stage, `pipeline` for recipes, and a few
evaluation helpers.
"""

import argparse
import json
import sys
from pathlib import Path

from spoken_record.ingestion import ingest
from spoken_record.evaluation import prepare_packet, score
from spoken_record.extraction import extract_run
from spoken_record.context_resolution import analyze_run
from spoken_record.identity_refinement import refine_run
from spoken_record.entity_linking import link_run
from spoken_record.reference_evaluation import prepare_reference_packet, score_reference_packet
from spoken_record.resolution import resolve_run
from spoken_record.segmentation import segment_run, validate_run
from spoken_record.vault import export_vault


def main() -> int:
    """Run one command and validate what it wrote."""
    parser = argparse.ArgumentParser(description="Prototype 1: transcript ingestion and passage segmentation")
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("pipeline", help="plan, run or resume an explicit stage recipe")
    command.add_argument("--root", type=Path, default=Path.cwd())
    command.add_argument("--config", required=True)
    command.add_argument("--run-id", required=True)
    mode = command.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="validate inputs and print the plan without inference or writes")
    mode.add_argument("--resume", action="store_true", help="reuse only validated stages from this exact pipeline journal")
    command = commands.add_parser("ingest", help="ingest only the frozen pilot into a new snapshot")
    command.add_argument("--root", type=Path, default=Path.cwd())
    command.add_argument("--config", default="config/prototype1.json")
    command.add_argument("--pilot", default="config/pilots/development-three.json")
    command.add_argument("--run-id", default=None)
    command = commands.add_parser("export-obsidian", help="project a resolution snapshot into an Obsidian vault")
    command.add_argument("--root", type=Path, default=Path.cwd())
    command.add_argument("--input-run", required=True)
    command.add_argument("--config", default="config/vault.concept-navigation-v1.json")
    command.add_argument("--run-id", default=None)
    command = commands.add_parser("enrich-entities", help="add Wikidata lifespans for linked or reviewed QIDs")
    command.add_argument("--root", type=Path, default=Path.cwd())
    command.add_argument("--input-run", required=True)
    command.add_argument("--config", default="config/enrichment.wikidata-v1.json")
    command.add_argument("--run-id", default=None)
    command = commands.add_parser("export-web", help="write the exploration web app's JSON data from a validated snapshot")
    command.add_argument("--root", type=Path, default=Path.cwd())
    command.add_argument("--input-run", required=True)
    command.add_argument("--config", default="config/web.hop-v3.json")
    command.add_argument("--run-id", default=None)
    command = commands.add_parser("segment", help="build passages and provisional windows from a saved ingestion run")
    command.add_argument("--root", type=Path, default=Path.cwd())
    command.add_argument("--input-run", required=True)
    command.add_argument("--config", default="config/segmentation.json")
    command.add_argument("--exclusions", default=None, help="optional reviewed production-credit spans (JSONL)")
    command.add_argument("--run-id", default=None)
    command = commands.add_parser("validate", help="check snapshot hashes, spans and references")
    command.add_argument("--run", type=Path, required=True)
    command = commands.add_parser("prepare-annotations", help="create a new, unreviewed annotation packet")
    command.add_argument("--root", type=Path, default=Path.cwd())
    command.add_argument("--input-run", required=True)
    command.add_argument("--pilot", default="config/pilots/evaluation-development.json")
    command.add_argument("--packet-id", required=True)
    command = commands.add_parser("score-ner", help="score complete predictions against reviewed annotations")
    command.add_argument("--packet", type=Path, required=True)
    command.add_argument("--gold", type=Path, required=True)
    command.add_argument("--predictions", type=Path, required=True)
    command.add_argument("--split", choices=("development", "test"), required=True)
    command = commands.add_parser("prepare-reference-evaluation",
                                  help="create an identity and third-person coreference review packet")
    command.add_argument("--root", type=Path, default=Path.cwd())
    command.add_argument("--input-run", required=True)
    command.add_argument("--config", default="config/reference_evaluation.json")
    command.add_argument("--packet-id", required=True)
    command = commands.add_parser("score-reference-evaluation",
                                  help="score a completely reviewed identity/coreference packet")
    command.add_argument("--packet", type=Path, required=True)
    command = commands.add_parser("extract", help="run pinned GLiNER extraction into a new snapshot")
    command.add_argument("--root", type=Path, default=Path.cwd())
    command.add_argument("--input-run", required=True)
    command.add_argument("--model", default="config/model.development-calibrated.json")
    command.add_argument("--packet", default=None, help="optional reviewed annotation packet to score against")
    command.add_argument("--run-id", default=None)
    command = commands.add_parser("resolve", help="build conservative episode and corpus identities")
    command.add_argument("--root", type=Path, default=Path.cwd())
    command.add_argument("--input-run", required=True)
    command.add_argument("--config", default="config/resolution.reviewed-m54-v2.json")
    command.add_argument("--packet", default=None, help="optional reviewed annotation packet to score against")
    command.add_argument("--run-id", default=None)
    command = commands.add_parser("analyze-resolution", help="retrieve and score identity pairs for human review")
    command.add_argument("--root", type=Path, default=Path.cwd())
    command.add_argument("--input-run", required=True)
    command.add_argument("--config", default="config/resolution_context.json")
    command.add_argument("--packet", required=True, help="reviewed annotation packet to score against")
    command.add_argument("--run-id", default=None)
    command = commands.add_parser("refine-resolution", help="provisionally refine corpus identities with name and context evidence")
    command.add_argument("--root", type=Path, default=Path.cwd())
    command.add_argument("--input-run", required=True)
    command.add_argument("--config", default="config/identity_refinement.surname-variant.json")
    command.add_argument("--run-id", default=None)
    command = commands.add_parser("coreference", help="link third-person references to named Person entities")
    command.add_argument("--root", type=Path, default=Path.cwd())
    command.add_argument("--input-run", required=True)
    command.add_argument("--config", default="config/coreference.fcoref.json")
    command.add_argument("--run-id", default=None)
    command = commands.add_parser("link-entities",
                                  help="retrieve and context-rank Wikidata candidates with abstention")
    command.add_argument("--root", type=Path, default=Path.cwd())
    command.add_argument("--input-run", required=True)
    command.add_argument("--config", default="config/linking.wikidata-mpnet-v4.json")
    command.add_argument("--run-id", default=None)
    command.add_argument("--candidate-run", default=None,
                         help="optional completed linking run whose retrieval/embedding evidence is reused")
    command = commands.add_parser("relations", help="extract grounded passage relations with GLiREL")
    command.add_argument("--root", type=Path, default=Path.cwd())
    command.add_argument("--input-run", required=True)
    command.add_argument("--config", default="config/relations.glirel-antecedent-expanded-v1.json")
    command.add_argument("--run-id", default=None)
    command = commands.add_parser("verify-relations", help="verify fixed relation candidates against transcript evidence with NLI")
    command.add_argument("--root", type=Path, default=Path.cwd())
    command.add_argument("--input-run", required=True)
    command.add_argument("--config", required=True)
    command.add_argument("--run-id", default=None)
    command = commands.add_parser("discover-passages", help="rank related cross-episode passages with MPNet and entity overlap")
    command.add_argument("--root", type=Path, default=Path.cwd())
    command.add_argument("--input-run", required=True)
    command.add_argument("--config", required=True)
    command.add_argument("--run-id", default=None)
    args = parser.parse_args()
    try:
        if args.command == "discover-passages":
            from spoken_record.discovery import discover_run
            output = discover_run(args.root, args.input_run, args.config, args.run_id)
            manifest = validate_run(output)
            print(json.dumps({"run": str(output), "status": manifest["status"], "counts": manifest["counts"]}, indent=2))
            return 0
        if args.command == "verify-relations":
            from spoken_record.relation_verification import verify_run
            output = verify_run(args.root, args.input_run, args.config, args.run_id)
            manifest = validate_run(output)
            print(json.dumps({"run": str(output), "status": manifest["status"], "counts": manifest["counts"]}, indent=2))
            return 0
        if args.command == "pipeline":
            from spoken_record.pipeline import plan_pipeline, run_pipeline

            if args.dry_run:
                result = plan_pipeline(args.root, args.config, args.run_id)
            else:
                result = run_pipeline(args.root, args.config, args.run_id, resume=args.resume,
                    progress=lambda event: print(json.dumps(event), file=sys.stderr, flush=True))
            print(json.dumps(result, indent=2))
            return 0
        if args.command == "prepare-annotations":
            output = prepare_packet(args.root, args.input_run, args.pilot, args.packet_id)
            print(json.dumps({"packet": str(output), "status": "unreviewed"}, indent=2))
            return 0
        if args.command == "score-ner":
            print(json.dumps(score(args.packet, args.gold, args.predictions, args.split), indent=2))
            return 0
        if args.command == "prepare-reference-evaluation":
            output = prepare_reference_packet(args.root, args.input_run, args.config, args.packet_id)
            manifest = validate_run(output)
            print(json.dumps({"packet": str(output), "status": manifest["status"],
                              "counts": manifest["counts"]}, indent=2))
            return 0
        if args.command == "score-reference-evaluation":
            print(json.dumps(score_reference_packet(args.packet), indent=2))
            return 0
        if args.command == "extract":
            output = extract_run(args.root, args.input_run, args.model, args.packet, args.run_id)
            manifest = validate_run(output)
            print(json.dumps({"run": str(output), "status": manifest["status"], "counts": manifest["counts"]}, indent=2))
            return 1 if manifest["status"] == "failed" else 0
        if args.command == "resolve":
            output = resolve_run(args.root, args.input_run, args.config, args.packet, args.run_id)
            manifest = validate_run(output)
            print(json.dumps({"run": str(output), "status": manifest["status"], "counts": manifest["counts"]}, indent=2))
            return 0
        if args.command == "analyze-resolution":
            output = analyze_run(args.root, args.input_run, args.config, args.packet, args.run_id)
            manifest = validate_run(output)
            print(json.dumps({"run": str(output), "status": manifest["status"], "counts": manifest["counts"]}, indent=2))
            return 0
        if args.command == "refine-resolution":
            output = refine_run(args.root, args.input_run, args.config, args.run_id)
            manifest = validate_run(output)
            print(json.dumps({"run": str(output), "status": manifest["status"], "counts": manifest["counts"]}, indent=2))
            return 0
        if args.command == "coreference":
            from spoken_record.coreference import coreference_run

            output = coreference_run(args.root, args.input_run, args.config, args.run_id)
            manifest = validate_run(output)
            print(json.dumps({"run": str(output), "status": manifest["status"], "counts": manifest["counts"]}, indent=2))
            return 0
        if args.command == "link-entities":
            output = link_run(args.root, args.input_run, args.config, args.run_id, args.candidate_run)
            manifest = validate_run(output)
            print(json.dumps({"run": str(output), "status": manifest["status"],
                              "counts": manifest["counts"]}, indent=2))
            return 0
        if args.command == "relations":
            from spoken_record.relations import relation_run

            output = relation_run(args.root, args.input_run, args.config, args.run_id)
            manifest = validate_run(output)
            print(json.dumps({"run": str(output), "status": manifest["status"], "counts": manifest["counts"]}, indent=2))
            return 0
        if args.command == "export-obsidian":
            output = export_vault(args.root, args.input_run, args.config, args.run_id)
            manifest = validate_run(output)
            print(json.dumps({"export": str(output), "vault": str(output / "obsidian"),
                              "status": manifest["status"], "counts": manifest["counts"]}, indent=2))
            return 0
        if args.command == "enrich-entities":
            from spoken_record.enrichment import enrich_run
            output = enrich_run(args.root, args.input_run, args.config, args.run_id)
            manifest = validate_run(output)
            print(json.dumps({"run": str(output), "status": manifest["status"], "counts": manifest["counts"]}, indent=2))
            return 0
        if args.command == "export-web":
            from spoken_record.web_export import export_web
            output = export_web(args.root, args.input_run, args.config, args.run_id)
            manifest = validate_run(output)
            print(json.dumps({"export": str(output), "data": str(output / "data"),
                              "status": manifest["status"], "counts": manifest["counts"]}, indent=2))
            return 0
        if args.command == "ingest":
            output = ingest(args.root, args.config, args.pilot, args.run_id)
        elif args.command == "segment":
            output = segment_run(args.root, args.input_run, args.config, args.exclusions, args.run_id)
        else:
            output = args.run
        manifest = validate_run(output)
    except (ValueError, OSError, KeyError, TypeError, RuntimeError, ImportError) as exc:
        parser.exit(2, f"{type(exc).__name__}: {exc}\n")
    print(json.dumps({"run": str(output), "status": manifest["status"], "counts": manifest["counts"]}, indent=2))
    return 1 if manifest["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
