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
src/spoken_record/   the pipeline (Python)
config/              the corpus config of each podcast and the settings of every step
web/                 the exploration web app (Svelte, TypeScript, Cytoscape)
tests/               pipeline tests on synthetic transcripts with offline model fakes
scripts/             the HoP corpus audit that episode selection and metadata rely on
data/audit/          output of that audit (metadata joins and checksums, no transcript text)
schema/              JSON schema of the source records
```

One command processes the chosen episodes and builds the archive:

```sh
spoken-record update --corpus config/hop.json --episodes "HoP 230-254"
```

The pipeline works from the bottom up. Each episode is processed on its own and stored in its
own folder: passages, mentions (GLiNER), the entities of the episode, pronoun links (FCoref),
relations (GLiREL) and their check against the transcript (NLI), and passage vectors (MPNet).
An episode is processed once and only again when its transcript or the settings change. The
corpus build then joins all episodes: it merges the same person, work or concept across
episodes, links them to Wikidata, finds related passages in other episodes, adds lifespans
and portraits, and writes the knowledge graph of the archive (`graph.json`), from which the
data for the web app is made. Adding a new episode only runs the models on that episode.

Every connection in the app opens the passages behind it and plays the audio from there.
Nothing in the pipeline or the app is specific to History of Philosophy. A podcast is
described by one corpus config, see [DEVELOPMENT.md](DEVELOPMENT.md#add-another-podcast).

- [DEVELOPMENT.md](DEVELOPMENT.md): setup, running the pipeline and the web app, adding episodes and podcasts
- [ARCHITECTURE.md](ARCHITECTURE.md): the two layers, the merge rules, the records and the cache

The transcripts were provided by the supervisor and are not in this repository. Everything
that contains transcript text (`data/episodes/`, `data/builds/`, `data/cache/`, `export/`,
`web/public/data/`) stays local and is ignored by git.
