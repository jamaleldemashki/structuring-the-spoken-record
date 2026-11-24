"""Command line entry point.

    spoken-record update --corpus config/hop.json --episodes "HoP 230-254"
    spoken-record validate --build data/builds/hop-230-254

``update`` processes the chosen episodes that are new or changed and then builds the corpus
and the app data (see build.py). ``validate`` checks a build without any model (see
validate.py).
"""

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(prog="spoken-record",
                                     description="Turn podcast transcripts into a linked, browsable archive.")
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("update", help="process new or changed episodes, then build the corpus and the app data")
    command.add_argument("--root", type=Path, default=Path.cwd())
    command.add_argument("--corpus", required=True, help="corpus config, for example config/hop.json")
    command.add_argument("--episodes", required=True,
                         help='episode codes, ranges or transcript paths, for example "HoP 230-254, HAP 54" or "all"')
    command.add_argument("--build-id", default=None, help="name of the build (default: corpus and time)")
    command.add_argument("--refresh-wikidata", action="store_true",
                         help="ask Wikidata again instead of using the stored answers")
    command = commands.add_parser("validate", help="check a build: hashes, decisions from stored scores, graph and app data")
    command.add_argument("--root", type=Path, default=Path.cwd())
    command.add_argument("--build", type=Path, required=True, help="build folder, for example data/builds/hop-230-254")
    args = parser.parse_args()
    if args.command == "validate":
        from spoken_record.validate import validate_build
        try:
            report = validate_build(args.root, args.root / args.build if not args.build.is_absolute() else args.build)
        except (ValueError, OSError, KeyError) as exc:
            parser.exit(2, f"{type(exc).__name__}: {exc}\n")
        print(json.dumps(report, indent=2))
        return 0
    try:
        from spoken_record.build import update

        output = update(args.root, args.corpus, args.episodes, args.build_id,
                        refresh_wikidata=args.refresh_wikidata,
                        progress=lambda message: print(message, file=sys.stderr, flush=True))
    except (ValueError, OSError, KeyError, TypeError, RuntimeError, ImportError) as exc:
        parser.exit(2, f"{type(exc).__name__}: {exc}\n")
    manifest = json.loads((output / "manifest.json").read_bytes())
    print(json.dumps({"build": str(output.relative_to(args.root.resolve())), "export": manifest["export"],
                      "counts": manifest["counts"], "cache": manifest["cache"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
