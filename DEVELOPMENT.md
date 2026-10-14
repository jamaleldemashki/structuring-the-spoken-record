# Development

How to set up the project, run the pipeline and the web app, and add another podcast.
For how the parts fit together see [ARCHITECTURE.md](ARCHITECTURE.md).

## Setup

The pipeline needs Python 3.11 (tested with 3.11.8 on macOS arm64). The model packages are
split into lock files so you only install what the stages you run need. The Wikidata client
(requests) is already in `requirements-dev.lock`.

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.lock
.venv/bin/python -m pip install -r requirements-model.lock        # GLiNER, torch, transformers
.venv/bin/python -m pip install -r requirements-coreference.lock  # FCoref
.venv/bin/python -m pip install seqeval==1.2.2 --use-pep517
.venv/bin/python -m pip install -r requirements-relations.lock    # GLiREL
.venv/bin/python -m pip install --no-deps --no-build-isolation .
```

The model versions in `requirements-model.lock` matter: newer transformers and
huggingface_hub versions break GLiNER 0.2.21 (see the comment in that file).

All models are pinned to a commit in their config and loaded offline. On a new machine,
download each one once at the revision named in `config/model.development-calibrated.json`,
`config/coreference.fcoref.json`, `config/relations.glirel-antecedent-expanded-v1.json`,
`config/linking.wikidata-mpnet-v4.json` and `config/discovery.mpnet-v1.json` (into the
normal Hugging Face cache, `~/.cache/huggingface/hub`). The NLI model of
`config/verification.nli-v1.json` goes into `.cache/huggingface/hub` inside the repository,
as set by its `cache_dir`.

The History of Philosophy transcripts go into `HoP/` (not in git). `scripts/corpus_audit.py`
produced `data/audit/`, which the HoP adapter uses to match transcripts to metadata.

The web app needs Node 22 or newer.

## Run the pipeline

`config/pipeline.hop-dev-three.json` runs every stage on the three development episodes
(HoP 243 and 244 on Aquinas, and the HAP 54 interview with Wilson Moses) and ends with the
web export. Check it first with `--dry-run`, which validates configs and inputs without
running a model or writing anything:

```sh
PYTHONPATH=src .venv/bin/python -m spoken_record.cli pipeline \
  --config config/pipeline.hop-dev-three.json --run-id my-run --dry-run

PYTHONPATH=src .venv/bin/python -m spoken_record.cli pipeline \
  --config config/pipeline.hop-dev-three.json --run-id my-run
```

Each stage writes `data/prototype1/runs/my-run-<stage>`, and the web data ends up in
`export/prototype1/my-run-web/data/`. The linking and enrichment stages call the Wikidata API,
everything else runs offline. If a stage fails, fix the cause and run the same command with
`--resume`. A run ID can't be reused, so pick a new one for a new run.

To run only part of the chain, write a recipe with `input_run` pointing at an existing run
and list only the stages after it, for example an Obsidian export of an existing run:

```json
{
  "schema_version": 1,
  "input_run": "data/prototype1/runs/my-run-enrich",
  "stages": [{"id": "vault", "kind": "export-obsidian", "config": "config/vault.concept-navigation-v1.json"}]
}
```

The vault is then in `export/prototype1/<run-id>-vault/obsidian/`.

Every stage also has its own command (`ingest`, `segment`, `extract`, `resolve`,
`refine-resolution`, `link-entities`, `coreference`, `relations`, `verify-relations`,
`discover-passages`, `enrich-entities`, `export-obsidian`, `export-web`). They take
`--input-run` and default to the configs used in the recipe. See
`python -m spoken_record.cli <command> --help`.

To check any run or export again:

```sh
PYTHONPATH=src .venv/bin/python -m spoken_record.cli validate --run data/prototype1/runs/my-run-enrich
```

The CLI exits with 0 when everything worked, 1 when some items failed and 2 for config or
validation errors.

## Reviewed QIDs for ambiguous people

The linker leaves an entity `ambiguous` or `NIL` when the evidence is not clear enough
(Aristotle and Augustine in the development episodes, for example). These people get no
lifespan unless you give them a QID by hand. Put one decision per line in a JSONL file and
name it in `reviewed_qids` in a copy of `config/enrichment.wikidata-v1.json`:

```json
{"corpus_entity_id": "ce-...", "qid": "Q868", "reviewer": "JE", "note": "Aristotle, checked against the passage"}
```

A reviewed QID wins over an automatic link. The file is copied into the run and hashed.

## Run the web app

```sh
cd web
npm install          # once, versions are pinned in package.json
npm run data         # copy the newest web export into public/data (checks the manifest hashes)
npm run dev          # development server on http://localhost:5173
npm test             # filter, graph, layout and formatting logic
npm run check        # type check
npm run build        # static site in web/dist, works from any web server
```

`npm run data -- <export run folder>` copies a specific export. `npm run data -- <export run folder> <id>`
puts it in `public/data/<id>/`, and the app opens it with `?archive=<id>`. That way several
podcasts can sit side by side in one build.

`web/public/data` and `web/dist` contain transcript text and are ignored by git.

## Add another podcast

For a podcast transcribed with Whisper:

1. Put the Whisper JSON files in one folder, for example `Podcasts/Longitude/whisper/`.
2. Write a profile, `config/profiles/longitude.json`:
   `{"schema_version": 1, "adapter": "whisper_json", "transcripts_dir": "Podcasts/Longitude/whisper"}`
3. Write the episode metadata as JSONL, one row per transcript, matched by its path inside
   `transcripts_dir` (the format is described at the top of `src/spoken_record/adapters/whisper_json.py`):
   `{"file": "season-2/ep07.json", "title": "Ep. 7: Clocks at Sea", "date": "2022-05-17", "audio_url": "https://...", "facets": {"season": "Season 2"}}`
4. Write the ingest config:
   `{"schema_version": 1, "corpus_id": "longitude", "profile": "config/profiles/longitude.json", "metadata": "Podcasts/Longitude/episodes.jsonl"}`
   `audit` and `quarantine` are optional. Set `"require_metadata": false` only if episodes
   without title, date or audio are fine.
5. Write a pilot that lists the transcripts with their SHA-256 (`shasum -a 256 <file>`), like
   `config/pilots/development-three.json`.
6. Copy `config/pipeline.hop-dev-three.json` and point it at the new ingest config and pilot.
   Where the podcast differs, give it its own aliases, `extra_type_keywords`, vault naming and
   web `site` block (copy `config/web.hop-v3.json`).
7. Run the recipe, then `npm run data -- export/prototype1/<run-id>-web longitude` and open
   the app with `?archive=longitude`.

## Tests

```sh
PYTHONPATH=src .venv/bin/python -m pytest -q
cd web && npm test
```

The Python tests use small synthetic transcripts in temporary folders and offline fakes for
every model and for Wikidata, so they need neither the HoP data nor the model downloads.
They cover the stage records and validators, the runner (resume, failures, changed inputs),
both exports and the second-podcast run.

## Evaluation helpers

`prepare-annotations` and `score-ner` build a packet of passages for manual NER annotation
and score GLiNER against it. The GLiNER threshold in `config/entity_labels.development-calibrated.json`
was picked this way. `prepare-reference-evaluation` and `score-reference-evaluation` do the
same for identity merges and pronoun links, and `analyze-resolution` lists possible duplicate
entities for review. None of them are needed to run the pipeline.
