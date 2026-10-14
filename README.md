# Structuring the Spoken Record

**Graph-Based Exploration of Podcast Transcripts, a History of Philosophy Case Study**

MA thesis by Jamal Eldemashki (Leibniz Universität Hannover / TIB), Aug 2026 – Feb 2027.

## What this is about

Long-running podcasts accumulate huge archives of spoken content that stay siloed episode by episode, once a transcript exists it's searchable on its own, but nothing connects a mention of a person, concept, or work to related discussions elsewhere in the archive, or shows how a topic was treated at different points over time.

This thesis designs and implements a pipeline that takes a podcast archive and produces a structured, interconnected representation of the entities and relationships it contains, plus a navigation/exploration layer on top of the recordings. Applied to the _History of Philosophy_ podcast, the pipeline:

1. **Prepares the corpus**, cleans and linguistically normalizes transcripts and episode metadata.
2. **Extracts and resolves entities**, philosophers, concepts, works, schools, speakers, places, and historical periods, mentioned across hundreds of episodes, resolving different mentions of the same entity and linking to external identifiers where appropriate.
3. **Constructs relationships and represents them as a graph**, an attributed property graph plus a complementary RDF/OWL representation (an "Open Knowledge Format" graph), so the result is both a visual, interactively explorable interface and a machine-readable interface for formal querying and reuse.
4. **Builds an interactive exploration interface**, search, filter, and focused navigation across entities and episodes.
5. **Evaluates** extraction quality, usability, and analytical value, and tests whether the pipeline transfers to a second, differently structured podcast or video series.

The corpus itself, around 700 Whisper-transcribed episodes spanning the main _History of Philosophy_ show and its Africana, Indian, Chinese, and German-philosophy sub-series, plus scraped episode metadata, was provided directly by the supervisor, who did the transcription himself. This thesis's own contribution starts at corpus validation and cleaning and runs through extraction, resolution, representation, and interface.

## What is in this repository

```
src/spoken_record/   the pipeline (Python), one module per stage
config/              recipes, stage configs and the HoP profile, pilots and reviewed aliases
web/                 the exploration web app (Svelte, TypeScript, Cytoscape)
tests/               pipeline tests on synthetic transcripts with offline model fakes
scripts/             the HoP corpus audit that the HoP adapter relies on
data/audit/          output of that audit (metadata joins, no transcript text)
schema/              JSON schema of the source records
```

The pipeline reads Whisper transcripts and episode metadata and runs these stages:

ingest, segment into passages, extract mentions (GLiNER), resolve identities, refine Person
identities, link to Wikidata, pronoun coreference (FCoref), relations (GLiREL), check relations
with NLI, find related passages (MPNet), add Wikidata lifespans, and export to Obsidian or to
the web app.

Every stage writes a new snapshot (JSONL plus a `manifest.json` with hashes) and never changes
its input, so each result can be traced back to the transcript passage it comes from. The web
app shows the archive as a graph of people, works, concepts and so on, where every connection
opens the passages behind it and plays the audio from there.

Nothing in the pipeline or the app is specific to History of Philosophy. A podcast is described
by config files, see [DEVELOPMENT.md](DEVELOPMENT.md#add-another-podcast).

- [DEVELOPMENT.md](DEVELOPMENT.md): setup, running the pipeline and the web app, adding a podcast
- [ARCHITECTURE.md](ARCHITECTURE.md): how the stages, records and exports fit together

The transcripts were provided by the supervisor and are not in this repository. Everything
that contains transcript text (`data/prototype1/`, `export/prototype1/`, `web/public/data/`)
stays local and is ignored by git.
