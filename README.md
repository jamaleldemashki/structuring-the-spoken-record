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

## Repo contents

This repository holds the LaTeX thesis source, the working plan and literature tracking, and (once development starts) the pipeline code itself. The raw dataset is supervisor-provided and kept out of version control due to size.
