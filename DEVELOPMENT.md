# Development

How to set up the project, build the archive, run the web app, and add episodes or another
podcast. For how the parts fit together see [ARCHITECTURE.md](ARCHITECTURE.md).

## Setup

The pipeline needs Python 3.11 (tested with 3.11.8 on macOS arm64). The model packages are
split into lock files.

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.lock
.venv/bin/python -m pip install -r requirements-model.lock        # GLiNER, torch, transformers
.venv/bin/python -m pip install -r requirements-coreference.lock  # FCoref
.venv/bin/python -m pip install seqeval==1.2.2 --use-pep517
.venv/bin/python -m pip install -r requirements-relations.lock    # GLiREL
.venv/bin/python -m pip install --no-deps --no-build-isolation .
```

`requirements-dev.lock` also installs spaCy's English model (`en_core_web_sm` 3.8.0), whose
lemma tables build the matching keys of terms.

The model versions in `requirements-model.lock` matter: newer transformers and
huggingface_hub versions break GLiNER 0.2.21 (see the comment in that file).

All models are pinned to a commit in their settings and loaded offline. On a new machine,
download each one once at the revision named in `config/model.development-calibrated.json`,
`config/coreference.fcoref.json`, `config/relations.glirel-proposals-v2.json`,
`config/verification.nli-large-v2.json`, `config/linking.wikidata-mpnet-v5.json` and
`config/discovery.mpnet-v1.json` (into the normal Hugging Face cache, `~/.cache/huggingface/hub`).
The NLI model (870 MB) has `cache_dir` set to null for that reason: inside a synced folder it
would be uploaded and could be evicted.

The History of Philosophy transcripts go into `HoP/` (not in git). The web app needs Node 22
or newer.

## Build the archive

```sh
PYTHONPATH=src .venv/bin/python -m spoken_record.cli update \
  --corpus config/hop.json --episodes "HoP 230-254" --build-id hop-230-254
```

`--episodes` takes episode codes, ranges, transcript paths and `all`, separated by commas,
for example `"HoP 243-244, HAP 54"`. Upper and lower case don't matter. The codes come from
the corpus audit (series and episode number), and each transcript is hashed and compared
with the audit before it is read. An unknown code, a code that fits two transcripts, or a
transcript that changed since the audit stops the run with a message.

`update` first brings the folder of every chosen episode in `data/episodes/hop/` up to date,
then builds the corpus into `data/builds/<build-id>/` and the app data into
`export/<build-id>-web/`. The progress is printed per episode and step. Without
`--build-id` the build is named after the corpus and the time. A build is never
overwritten, so use a new name for every build.

Wikidata is asked during linking and enrichment, and only about names and items that are
not in `data/cache/wikidata.sqlite` yet. Everything else runs offline. The stored answers
stay as they are until you run with `--refresh-wikidata`.

The first build of 3 episodes takes about 6 minutes, mostly for loading the models and
asking Wikidata. Building the same episodes again takes a few seconds.

The knowledge graph of the build is `data/builds/<build-id>/graph.json`. To check a build
without running any model:

```sh
PYTHONPATH=src .venv/bin/python -m spoken_record.cli validate --build data/builds/hop-230-254
```

It confirms that no file was changed, redoes the decisions from the stored scores, and
rebuilds the graph and the app data byte for byte (see ARCHITECTURE.md).

## Add a new episode

1. Put the transcript into `HoP/01_whisper/` and its row into the metadata.
2. Run the audit again, so the new transcript gets its checksum and metadata join:
   `python3 scripts/corpus_audit.py`
3. Run `update` with the new episode in the selection, for example `--episodes "HoP 230-255"`.

Only the new episode goes through the models. The other episode folders are reused and the
corpus build runs over all of them.

## Filters and edges

`config/filters.json` sets the boilerplate and generic-term filters (see ARCHITECTURE.md).
Leaving `filters` out of the corpus config switches both off.

In the web settings (`config/web.hop-v3.json`), `co_discussed_share` is the default share of
each entity's links that the app draws at the start (10% for HoP). The slider on the toolbar then
changes one selected item at a time. The export holds all pairs and the app makes the choice
(`web/src/lib/strength.ts`), so it follows the filters. `graph.json` always keeps every pair.
After changing a web setting, run `update` again with a new build name; nothing but the app
data is redone, which takes seconds.

## Corrections

The build needs no review. Two optional files can still correct it:

- `config/aliases/hop.jsonl` (named in `config/resolution.json`): names that refer to the
  same entity, for example transcription errors like "Kilward B" for Robert Kilwardby.
- A reviewed QID file named in `reviewed_qids` in a copy of `config/enrichment.wikidata-v1.json`,
  for people the linker left ambiguous or NIL. One decision per line:

  ```json
  {"corpus_entity_id": "ce-...", "qid": "Q60045", "reviewer": "JE", "note": "Albert the Great, checked against the passage"}
  ```

## Run the web app

```sh
cd web
npm install          # once, versions are pinned in package.json
npm run data         # copy the newest export into public/data (checks the manifest hashes)
npm run models       # once per model: the sentence model for questions, into public/models
npm run dev          # development server on http://localhost:5173
npm test             # filter, graph, layout and formatting logic
npm run check        # type check
npm run build        # static site in web/dist, works from any web server
```

`npm run data -- ../export/<build-id>-web` copies a specific export.
`npm run data -- ../export/<build-id>-web <id>` puts it in `public/data/<id>/`, and the app
opens it with `?archive=<id>`. That way several podcasts can sit side by side in one build.

`web/public/data` and `web/dist` contain transcript text and are ignored by git.

Questions in the search field ("Is there any relation between Thomas Aquinas and Aristotle
when it comes to the soul?") use `search.json` from the export, made by the `search` step
(`config/search.minilm-v1.json`). The model for the question itself runs in the browser.
`npm run models` downloads it at the revision `search.json` names into `public/models/`
(23 MB, ignored by git). Without it, questions still work by the names in them, only the
passages are not ordered by meaning.

## Host the web app

The study version is a static site on Cloudflare Pages behind a password. The password check
is `web/functions/_middleware.js`, it covers every file, the transcripts included.

Once:

```sh
cd web
npx wrangler@4.150.0 login
npx wrangler@4.150.0 pages project create structuring-the-spoken-record --production-branch main
npx wrangler@4.150.0 pages secret put SITE_PASSWORD --project-name structuring-the-spoken-record
```

After every data update (`update`, then `npm run data`):

```sh
npm run deploy       # builds web/dist and uploads it with the password check
```

The site is at `https://structuring-the-spoken-record.pages.dev`. To try it locally first:
`npx wrangler@4.150.0 pages dev dist --binding SITE_PASSWORD=test`.

## Add another podcast

For a podcast transcribed with Whisper:

1. Put the Whisper JSON files in one folder, for example `Podcasts/Longitude/whisper/`.
2. Write a profile, `config/profiles/longitude.json`:
   `{"schema_version": 1, "adapter": "whisper_json", "transcripts_dir": "Podcasts/Longitude/whisper"}`
3. Write the episode metadata as JSONL, one row per transcript, matched by its path inside
   `transcripts_dir` (the format is described at the top of `src/spoken_record/adapters/whisper_json.py`):
   `{"file": "season-2/ep07.json", "title": "Ep. 7: Clocks at Sea", "date": "2022-05-17", "audio_url": "https://...", "facets": {"season": "Season 2"}}`
4. Copy `config/hop.json` to `config/longitude.json` and change:
   - `corpus_id`, `profile` and `metadata`; leave out `audit` and `quarantine`
   - `episode_codes`, so episodes can be chosen by code:
     `{"series_field": "facets.season", "number_fields": ["title"], "number_pattern": "Ep\\. (\\d+)", "path_field": "file", "path_in_transcripts_dir": true}`
   - where the podcast differs, its own aliases, `extra_type_keywords` and web `site`
     block (copy `config/web.hop-v3.json`)
5. Run `update --corpus config/longitude.json --episodes "Season 2 1-10"`, then
   `npm run data -- ../export/<build-id>-web longitude` and open the app with `?archive=longitude`.

The pipeline knows eight categories (person, work, school, concept, place, period, language,
group). The `site` block gives them the names of the podcast, for example "Musician" and
"Album" for a music podcast, and says which are shown at first. The words that tell the linker
what a Wikidata description of each category looks like are general. Words of one subject go
into `extra_type_keywords`, as the philosophy words do for History of Philosophy. Nothing in the
app needs changing: what the data lacks (lifespans, dates, audio, transcript text) is left out
of the pages.

Set `"require_metadata": false` in the corpus config only if episodes without title, date or
audio are fine.

## Tests

```sh
PYTHONPATH=src .venv/bin/python -m pytest -q
cd web && npm test
```

The Python tests use small synthetic transcripts in temporary folders and offline fakes for
every model and for Wikidata, so they need neither the HoP data nor the model downloads.
They cover the rules of every step, the episode folders and their keys, the corpus build,
the merge rules, the caches and a second podcast.
