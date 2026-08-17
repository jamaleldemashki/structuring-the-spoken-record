# Thesis Roadmap — Structuring the Spoken Record

**Graph-Based Exploration of Podcast Transcripts, a History of Philosophy Case Study**

- Start: 2026-08-07
- Duration: 6 months
- Target submission: **~2027-02-07**
- Weekly supervisor/self check-ins (see template at bottom)
- Reference manager: Zotero · Writing: Overleaf

---

## 0. Open items to resolve immediately (Week 2, by 2026-08-24)

1. **Define "Open Knowledge Format" precisely.** No canonical, citable spec matching the assignment's description (graph-based, visual + RDF/OWL dual interface) turned up in the literature — it doesn't match the established Open Research Knowledge Graph (ORKG) work, and an unrelated same-named concept surfaced from June 2026 (an LLM-wiki markdown spec for AI agents) that's almost certainly not it. **I need to ask my supervisor directly for the paper/spec/tool they mean**, since RQ2 and the whole representation layer (Approach step 3) depend on this. Bring this to my first meeting.
2. **Pick the second podcast/video series candidate for RQ3** (generalizability). The "GPW" sub-series in my supervisor-provided dataset turned out to be "German Philosophy and the World" — a Peter Adamson interview mini-series that's part of the same HoP show, not a separate podcast, so it doesn't count. Start an actual shortlist so corpus/legal access can be checked early rather than in month 5.
3. **Decide early whether Community Validation (Approach step 6) is realistic.** It's explicitly contingent — ask my supervisor if there's an actual contact into the podcast's volunteer contributor group. If not, drop it now rather than let it linger as scope.

~~Confirm transcript access~~ — **resolved 2026-08-17.** My supervisor transcribed the audio himself and gave me the dataset directly, with confirmed rights to use it — see item 1 below for what's actually in it.

---

## 0.5 What I already have: the supervisor-provided `HoP/` dataset

This changes the shape of the early roadmap, so it's worth spelling out before the phases below. `HoP/` contains:

- **Episode metadata scrape** (`00_infos/hop_episodes_cache.jsonl` + 545 cached pages from historyofphilosophy.net): link, title, series, pub date, audio URL per episode.
- **724 Whisper ASR transcripts** (`01_whisper/`), covering effectively the whole show family: HoP main (486), HAP/Africana Philosophy (141), HPI/Indian Philosophy (60), HPC/Chinese Philosophy (13), GPW/German Philosophy and the World (10).
- **724 files of rough NLP output** (`02_nlp/`): per-episode bag-of-words + named-entity frequency counts from what looks like a generic/default NER pass — noisy (e.g. "Groucho" as an entity) and with no surface-form merging ("Aquinas" vs. "Thomas Aquinas" are separate). This is a baseline to build on, not a finished extraction — the real RQ1 work (resolution, domain-tuned extraction) is still fully open.
- **856 Obsidian notes** (`03_notes/`), one per episode with transcript embedded — 132 don't match anything in `01_whisper`/`02_nlp` (older 2023-vintage duplicates + at least one corrupted filename), needs a dedup pass.
- **Corpus-level aggregates**: corpus-wide word/entity frequencies, wordclouds, an episode similarity matrix.

**What this means practically:** corpus _acquisition_ (scraping + transcription of 700+ episodes) is essentially done — that would otherwise have eaten weeks of Phase 1. What's left for corpus prep is **validation**, not construction: spot-checking ASR quality (especially on proper nouns like Avicenna, Arabic/Sanskrit terms — exactly where Whisper is most likely to err and exactly what my entity extraction most needs to get right), deciding how to handle the missing speaker-diarization (transcripts are continuous text with no per-utterance speaker labels, but "speakers" is a required entity type — episode-level guest identity is available from metadata, within-episode attribution is not), and deduping the notes folder. Audio is still archived elsewhere, so re-transcription/diarization stays possible later if needed.

The time saved here has been reallocated below: Phase 1 is compressed, and that time moved into Phase 2 (the hardest, most novel part) and into end-of-thesis buffer.

**For my methodology chapter:** I should be explicit that the transcription was done by my supervisor and provided to me — say so plainly rather than presenting it as something I built. My own contribution starts at corpus validation/cleaning and runs through extraction, resolution, representation, and interface.

---

## Phase overview (revised 2026-08-17)

| Phase                                      | Weeks | Dates (approx)  | Maps to Approach step  |
| ------------------------------------------ | ----- | --------------- | ---------------------- |
| 0. Orientation & Literature                | 1–3   | Aug 7 – Aug 27  | (foundation for all)   |
| 1. Corpus Validation & Cleaning            | 4–5   | Aug 28 – Sep 10 | Step 1                 |
| 2. Information Extraction & Resolution     | 6–11  | Sep 11 – Oct 22 | Step 2 → RQ1           |
| 3. Relationship & Graph/RDF Representation | 12–15 | Oct 23 – Nov 19 | Step 3                 |
| 4. Interactive Exploration Interface       | 16–18 | Nov 20 – Dec 10 | Step 4                 |
| 5. Evaluation & Generalizability           | 19–21 | Dec 11 – Dec 31 | Step 5 → RQ2, RQ3      |
| 6. Writing Consolidation & Buffer          | 22–26 | Jan 1 – Feb 7   | write-up, defense prep |

Compared to the original plan: Phases 0–1 are compressed by 2 weeks (raw corpus acquisition is already done), Phase 2 (extraction & resolution — the core methodological contribution) gets one extra week, and the final buffer gets one extra week too.

**Cross-cutting rule: write continuously, not at the end.** Each phase below ends with "what gets drafted in the thesis" — do that drafting in the same week the work happens, while it's fresh. Month 6 is for revision and integration, not first-draft writing from scratch.

Given I'm "solid coding, new to some of the NLP/KG/frontend stack," phases 2–4 include explicit ramp-up time rather than assuming I hit the ground running.

---

## Phase 0 — Orientation & Literature Foundation (Weeks 1–3, through Aug 27)

I'm already ~1.5 weeks in (thesis-writing guide, Overleaf setup done).

**Goals**

- Resolve the open items above (OKF definition, RQ3 podcast candidate, community-validation feasibility).
- Build a working literature map in Zotero across the core areas, tagged by theme.
- Draft the Background chapter's general sections (ASR/Whisper transcription basics, NER, coreference and entity resolution, entity linking, property graphs, RDF and the Semantic Web, OWL and ontologies) — these don't depend on my own results, so they can be written alongside the literature review. The Open Knowledge Format subsection stays blocked until the supervisor clarifies it (applied properly in Phase 3).
- Finalize scope: confirm which entity types (philosophers, concepts, works, schools, speakers, places, periods) are must-have vs. stretch — the speaker-diarization gap in the dataset makes "speakers" worth an explicit scoping decision now.
- Decide the remaining technical stack: graph store (Neo4j vs. RDF triple store like GraphDB/Fuseki, or both), entity linking targets (Wikidata and/or VIAF — see [LITERATURE.md](LITERATURE.md)), frontend approach for the exploration interface. **NLP approach decided (2026-08-17): a classical pipeline, no LLM** — generic pretrained NER (spaCy) + a curated gazetteer (built from Wikidata/SEP/VIAF entries) for the domain-specific entity types generic NER won't recognize, a dedicated coreference model for cross-mention resolution, and a separate entity-linking step. Fine-tuning a pretrained NER model on the gold-standard sample stays an optional stretch if off-the-shelf + gazetteer underperforms.
- Set up the project repo and environment; do a first read-through of the `HoP/` dataset structure (I now have a head start on this from the folder review).
- Draft a one-page internal problem statement + refined RQs.

**Deliverable by end of phase:** Related Work outline (bullet-level, in Overleaf) + Background chapter first draft (general sections) + confirmed tooling decisions + cleaned reading list in Zotero.

---

## Phase 1 — Corpus Validation & Cleaning (Weeks 4–5, Aug 28 – Sep 10)

Scope shifted from _building_ the corpus to _validating and normalizing_ the one I already have.

**Goals**

- Spot-check Whisper transcript quality across a sample spanning sub-series (HoP/HAP/HPI/HPC/GPW) and eras — pay particular attention to proper-noun accuracy (philosopher names, Arabic/Sanskrit/Latin terms).
- Dedup `03_notes/` against `01_whisper/`+`02_nlp/`; resolve the 132 mismatched files.
- Decide and implement the speaker-attribution approach (episode-level metadata only, vs. investing in diarization from the archived audio).
- Build the normalization pipeline on top of the existing transcripts: tokenization, lemmatization, spelling normalization, acronym expansion, multi-word expression detection (e.g. "categorical imperative", "Neoplatonism").
- Produce corpus statistics (episode count, word count, sub-series/speaker distribution, time span covered) — real thesis content for the Data chapter.

**Deliverable:** Validated, cleaned, versioned corpus + corpus description section drafted.

---

## Phase 2 — Information Extraction & Resolution (Weeks 6–11, Sep 11 – Oct 22)

This is the methodological core answering **RQ1** — it now has 6 weeks instead of 5.

**Goals**

- Weeks 6–7: ramp-up + first real NER pass, building on (not just reusing) the existing rough `02_nlp/` baseline. Build the gazetteer (from Wikidata/SEP/VIAF) for domain-specific entity types, combine with off-the-shelf NER, evaluate against a small gold-standard annotated sample; fine-tune only if that combination underperforms.
- Weeks 8–10: entity resolution — coreference within and across episodes (the existing baseline shows "Aquinas"/"Thomas Aquinas" as unmerged, so this is genuinely unsolved), linking to external identifiers (Wikidata/SEP/VIAF as appropriate).
- Week 11: run precision/recall evaluation against the gold sample, error analysis, iterate.

**Deliverable:** Extraction + resolution pipeline, evaluated with precision/recall → directly answers RQ1. Methods + Results sections drafted.

---

## Phase 3 — Relationship Construction & Graph/RDF Representation (Weeks 12–15, Oct 23 – Nov 19)

**Goals**

- Model relationships derived from transcript content + metadata (e.g. "discussed-in-episode", "influenced-by", "member-of-school", "contemporary-of") as an attributed property graph.
- Design the RDF/OWL schema/ontology in parallel (reuse existing vocabularies where sensible — schema.org, CIDOC-CRM for historical entities, FOAF, or Wikidata's own ontology — rather than inventing one from scratch).
- Populate both representations from the same underlying data; keep them in sync programmatically, not by hand.
- **This is where the resolved "Open Knowledge Format" definition from Phase 0 gets applied** — don't start schema design before that's settled.

**Deliverable:** Populated property graph + RDF/OWL export, both queryable. Representation chapter drafted.

---

## Phase 4 — Interactive Exploration Interface (Weeks 16–18, Nov 20 – Dec 10)

**Goals**

- Build search, filter, and focused-navigation UI over the graph (entity pages, timeline view, cross-episode navigation, relationship browsing).
- Keep scope realistic: a usable, evaluable prototype beats a polished but incomplete product. Favor an existing graph-viz framework (e.g. a lightweight web app over Neo4j Bloom / a custom D3 or Cytoscape.js view / a SPARQL-backed search UI) over building visualization tooling from scratch.

**Deliverable:** Working interactive prototype, screenshots/demo captured for the thesis.

---

## Phase 5 — Evaluation & Generalizability (Weeks 19–21, Dec 11 – Dec 31)

Note: this spans the holidays — treat it as a lighter-intensity phase and shift buffer accordingly if needed.

**Goals**

- Usability + analytical-value evaluation of the representation and interface (user testing if feasible, or structured self/expert walkthroughs against realistic tasks) → **RQ2**.
- Run the pipeline against the second podcast/video series shortlisted in Phase 0 → **RQ3**.
- If community validation (Approach step 6) was confirmed feasible in Phase 0, this is a natural place for it — but treat it as bonus, not blocking.

**Deliverable:** Evaluation results answering RQ2 and RQ3. Evaluation chapter drafted.

---

## Phase 6 — Writing Consolidation & Buffer (Weeks 22–26, Jan 1 – Feb 7)

**Goals**

- Integrate all previously-drafted sections into a coherent whole; write Introduction and Conclusion last.
- Finalize the Implementation chapter's Reproducibility section — repo structure, how to (re)run the pipeline, config/data locations — describing the project as it actually ended up, not as planned.
- Full read-through for argument coherence, not just prose polish.
- Reserve the **last ~1.5–2 weeks purely as buffer** — formatting, references, printing/binding/submission logistics, supervisor's final read, unexpected fixes. Do not schedule real new work here.

**Deliverable:** Submitted thesis.

---

## Risk notes

- **Biggest scope risk:** phases 2–4 (extraction, representation, interface) are each nontrivial engineering efforts on top of being research questions. If Phase 2 overruns, compress Phase 4 (interface) first — a simpler interface with a solid, well-evaluated extraction/representation layer is a stronger thesis than a polished UI over weak data.
- **RQ3 (generalizability) is explicitly secondary/exploratory** per the assignment — don't let it compete for time with RQ1/RQ2 until Phase 5.
- **Community validation is explicitly contingent** — treat as opportunistic, not planned-critical-path, unless Phase 0 confirms real access.
- **Attribution:** the transcribed dataset came from my supervisor, not from my own scraping/transcription work — state this plainly in the methodology chapter rather than implying it as my own pipeline output.
