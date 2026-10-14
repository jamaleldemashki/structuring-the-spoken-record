"""Obsidian vault export. Every link points back to the passage it comes from."""

import importlib.metadata
import json
import platform
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from spoken_record import __version__
from spoken_record.adapters.base import sha256
from spoken_record.extraction_models import ExtractedMention
from spoken_record.coreference_models import CoreferenceLink
from spoken_record.ingestion import dump_json, read_jsonl, workspace_path
from spoken_record.models import Episode, TranscriptVersion
from spoken_record.navigation import build_concept_navigation
from spoken_record.passage_models import Passage
from spoken_record.relation_models import RelationAssertion
from spoken_record.resolution import validate_run as validate_resolution
from spoken_record.resolution_models import (CorpusEntity, EpisodeEntity, MentionAssignment,
                                             ReviewCandidate)
from spoken_record.vault_models import VaultConfig


WIKILINK = re.compile(r"\[\[([^|\]#]+)(?:#\^([A-Za-z0-9-]+))?(?:\|[^\]]+)?\]\]")
ANCHOR = re.compile(r"(?m)^\^([A-Za-z0-9-]+)$")


def yaml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def markdown_text(value: str) -> str:
    """Escape transcript text so Obsidian shows it as plain text, without links or formatting."""
    value = value.replace("\\", "\\\\")
    value = re.sub(r"([\[\]*_`$<>|])", r"\\\1", value)
    value = re.sub(r"(?m)^( {0,3})(#{1,6}|>|[-+])(?=\s)", r"\1\\\2", value)
    value = re.sub(r"(?m)^( {0,3})(\d+)(\.)(?=\s)", r"\1\2\\\3", value)
    return value


def link_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace("|", "\\|").replace("]", "\\]")


def timecode(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def passage_time(passage: Passage) -> str:
    if passage.start_s is None:
        return "timestamp unavailable"
    return f"{timecode(passage.start_s)}–{timecode(passage.end_s)}"


def frontmatter(kind: str, fields: dict) -> str:
    lines = ["---", f"kind: {yaml_string(kind)}"]
    for key, value in fields.items():
        lines.append(f"{key}: {json.dumps(value, ensure_ascii=False)}")
    return "\n".join([*lines, "---", ""])


def safe_filename(value: str, maximum: int = 100) -> str:
    value = unicodedata.normalize("NFC", value)
    value = re.sub(r'[\\/:*?"<>|#^\[\]\x00-\x1f]', " - ", value)
    value = " ".join(value.split()).strip(" .-")
    return (value[:maximum].rstrip(" .-") or "Untitled")


def series_code(config: VaultConfig, value: str | None) -> str | None:
    if not value:
        return None
    if not config.series_codes:
        return value
    return {k.casefold(): v for k, v in config.series_codes.items()}.get(value.casefold())


def episode_reference(episode: Episode, config: VaultConfig) -> str:
    """Short reference like "HoP 243", built from the naming rules in the config."""
    show = series_code(config, episode.facets.get(config.series_facet)) if config.series_facet else None
    number = re.match(config.episode_number_pattern, episode.title) if config.episode_number_pattern else None
    if show and number:
        return f"{show} {number.group(1)}"
    source = re.search(config.source_key_pattern, episode.source_key, re.I) if config.source_key_pattern else None
    if source and (source_show := series_code(config, source.group(1))):
        return f"{source_show} {source.group(2)}"
    return safe_filename(episode.title, 48)


def note_paths(config: VaultConfig, episodes: list[Episode], entities: list[CorpusEntity]):
    episode_names = {e.episode_id: safe_filename(episode_reference(e, config)) for e in episodes}
    by_name = defaultdict(list)
    for episode_id, name in episode_names.items():
        by_name[name.casefold()].append(episode_id)
    for ids in by_name.values():
        if len(ids) > 1:
            for episode_id in ids:
                episode_names[episode_id] = f"{episode_names[episode_id]} — {episode_id[-8:]}"

    entity_names = {e.corpus_entity_id: safe_filename(e.canonical_label) for e in entities}
    by_type_name = defaultdict(list)
    for entity in entities:
        by_type_name[(entity.type, entity_names[entity.corpus_entity_id].casefold())].append(entity)
    episode_by_id = {e.episode_id: e for e in episodes}
    for group in by_type_name.values():
        if len(group) > 1:
            for entity in group:
                context = " & ".join(episode_reference(episode_by_id[eid], config) for eid in entity.episode_ids)
                entity_names[entity.corpus_entity_id] = safe_filename(f"{entity.canonical_label} — {context}")
    final_groups = defaultdict(list)
    for entity in entities:
        final_groups[(entity.type, entity_names[entity.corpus_entity_id].casefold())].append(entity.corpus_entity_id)
    for ids in final_groups.values():
        if len(ids) > 1:
            for corpus_entity_id in ids:
                entity_names[corpus_entity_id] += f" — {corpus_entity_id[-8:]}"
    episode_paths = {eid: f"{config.episode_folder}/{name}" for eid, name in episode_names.items()}
    entity_paths = {entity.corpus_entity_id: f"{config.entity_folder}/{entity.type}/{entity_names[entity.corpus_entity_id]}"
                    for entity in entities}
    if len(set(episode_paths.values())) != len(episode_paths) or len(set(entity_paths.values())) != len(entity_paths):
        raise ValueError("human-readable note filenames are not unique")
    return episode_paths, entity_paths


def entity_link(entity: CorpusEntity, entity_paths: dict[str, str]) -> str:
    return f"[[{entity_paths[entity.corpus_entity_id]}|{link_label(entity.canonical_label)}]]"


def relation_note_paths(config: VaultConfig, assertions: list[RelationAssertion],
                        entities: dict[str, CorpusEntity]) -> dict[str, str]:
    """Readable, unique paths for the provisional relation notes."""
    paths: dict[str, str] = {}
    used: set[str] = set()
    for assertion in sorted(assertions, key=lambda row: row.assertion_id):
        subject = entities[assertion.subject_corpus_entity_id].canonical_label
        object_label = entities[assertion.object_corpus_entity_id].canonical_label
        predicate = assertion.predicate.replace("_", " ")
        stem = safe_filename(f"{subject} — {predicate} — {object_label}", 140)
        candidate = f"{config.relation_folder}/{stem}"
        if candidate.casefold() in used:
            candidate = f"{candidate} — {assertion.assertion_id[-8:]}"
        used.add(candidate.casefold())
        paths[assertion.assertion_id] = candidate
    return paths


def episode_link(episode: Episode, episode_paths: dict[str, str], passage: Passage | None = None,
                 naming: VaultConfig | None = None) -> str:
    """Link labelled with the episode title, or its short reference when naming rules are given."""
    anchor = f"#^{passage.passage_id}" if passage else ""
    label = episode_reference(episode, naming) if naming else episode.title
    if passage is not None:
        label = f"{label} · {passage_time(passage)}"
    return f"[[{episode_paths[episode.episode_id]}{anchor}|{link_label(label)}]]"


def build_vault_files(config: VaultConfig, episodes: list[Episode], transcripts: list[TranscriptVersion],
                      passages: list[Passage], mentions: list[ExtractedMention],
                      episode_entities: list[EpisodeEntity], corpus_entities: list[CorpusEntity],
                      assignments: list[MentionAssignment], queue: list[ReviewCandidate],
                      coreference_links: list[CoreferenceLink] | None = None,
                      relation_assertions: list[RelationAssertion] | None = None,
                      verification: dict | None = None,
                      discovery: dict | None = None) -> dict[str, bytes]:
    episode_by_id = {e.episode_id: e for e in episodes}
    transcript_by_episode = {t.episode_id: t for t in transcripts}
    passage_by_id = {p.passage_id: p for p in passages}
    mention_by_id = {m.mention_id: m for m in mentions}
    corpus_by_id = {e.corpus_entity_id: e for e in corpus_entities}
    assignment_by_mention = {a.mention_id: a for a in assignments}
    episode_paths, entity_paths = note_paths(config, episodes, corpus_entities)
    relation_paths = relation_note_paths(config, relation_assertions or [], corpus_by_id)
    corpus_mentions = defaultdict(list)
    passage_corpus_ids = defaultdict(set)
    episode_corpus_ids = defaultdict(set)
    linked_coreference = [row for row in (coreference_links or []) if row.decision == "linked"]
    abstained_coreference = [row for row in (coreference_links or []) if row.decision == "abstain"]
    coreference_by_id = {row.reference_id: row for row in linked_coreference}
    coreference_by_corpus = defaultdict(list)
    coreference_by_passage = defaultdict(list)
    for row in linked_coreference:
        coreference_by_corpus[row.corpus_entity_id].append(row)
        for passage_id in row.passage_ids:
            coreference_by_passage[passage_id].append(row)
    for mention in mentions:
        assignment = assignment_by_mention[mention.mention_id]
        corpus_mentions[assignment.corpus_entity_id].append(mention)
        episode_corpus_ids[mention.episode_id].add(assignment.corpus_entity_id)
        for passage_id in mention.passage_ids:
            passage_corpus_ids[passage_id].add(assignment.corpus_entity_id)

    files: dict[str, bytes] = {}
    sorted_episodes = sorted(episodes, key=lambda e: (e.title.casefold(), e.episode_id))
    sorted_entities = sorted(corpus_entities, key=lambda e: (e.type, e.canonical_label.casefold(), e.corpus_entity_id))

    for episode in sorted_episodes:
        episode_passages = sorted((p for p in passages if p.episode_id == episode.episode_id), key=lambda p: p.index)
        if "".join(p.text for p in episode_passages) != transcript_by_episode[episode.episode_id].text:
            raise ValueError("passages do not reconstruct the episode transcript")
        entity_ids = sorted(episode_corpus_ids[episode.episode_id],
                            key=lambda cid: (corpus_by_id[cid].type, corpus_by_id[cid].canonical_label.casefold(), cid))
        fields = {"episode_id": episode.episode_id, "title": episode.title, "aliases": [episode.title],
                  "language": episode.language, "publication_date": episode.publication_date,
                  "source_url": episode.source_url, "audio_url": episode.audio_url,
                  "entity_count": len(entity_ids), "passage_count": len(episode_passages)}
        lines = [frontmatter("episode", fields), f"# {markdown_text(episode.title)}", "",
                 "## Source", "", f"- Publication date: {episode.publication_date or 'unknown'}",
                 f"- Source page: {episode.source_url or 'unavailable'}",
                 f"- Audio: {episode.audio_url or 'unavailable'}", "", "## Entities", ""]
        if entity_ids:
            lines.extend(f"- **{corpus_by_id[cid].type}:** {entity_link(corpus_by_id[cid], entity_paths)}"
                         for cid in entity_ids)
        else:
            lines.append("No accepted entity mentions.")
        lines.extend(["", "## Transcript", "",
                      "> [!info] Evidence",
                      "> Passage text is rendered from the immutable transcript. Times are source-segment approximations.", ""])
        for passage in episode_passages:
            lines.extend([f"### Passage {passage.index + 1} · {passage_time(passage)}", "",
                          f"^{passage.passage_id}", "", markdown_text(passage.text), ""])
            ids = sorted(passage_corpus_ids[passage.passage_id],
                         key=lambda cid: (corpus_by_id[cid].type, corpus_by_id[cid].canonical_label.casefold(), cid))
            lines.append("**Entities:** " + (", ".join(entity_link(corpus_by_id[cid], entity_paths) for cid in ids)
                                               if ids else "none detected"))
            references = sorted(coreference_by_passage[passage.passage_id],
                                key=lambda row: (row.char_start, row.reference_id))
            if references:
                by_target = defaultdict(list)
                for row in references:
                    by_target[row.corpus_entity_id].append(row)
                rendered = []
                for corpus_entity_id, rows in sorted(by_target.items(),
                        key=lambda item: (corpus_by_id[item[0]].canonical_label.casefold(), item[0])):
                    forms = Counter(row.text.casefold() for row in rows)
                    form_text = ", ".join(f"{markdown_text(form)} × {count}"
                                          for form, count in sorted(forms.items()))
                    rendered.append(f"{entity_link(corpus_by_id[corpus_entity_id], entity_paths)} ({form_text})")
                lines.append("**Provisional coreference:** " + ", ".join(rendered))
            lines.append("")
        files[f"{episode_paths[episode.episode_id]}.md"] = "\n".join(lines).encode()

    for entity in sorted_entities:
        member_mentions = sorted(corpus_mentions[entity.corpus_entity_id],
                                 key=lambda m: (episode_by_id[m.episode_id].title.casefold(), m.char_start, m.mention_id))
        aliases = list(dict.fromkeys([entity.canonical_label, *entity.reviewed_aliases, *entity.observed_surfaces]))
        entity_coreference = sorted(coreference_by_corpus[entity.corpus_entity_id],
                                    key=lambda row: (row.episode_id, row.char_start, row.reference_id))
        coreference_status = ("provisional_fcoref" if entity_coreference else
                              "attempted_no_link" if coreference_links is not None else "not_attempted")
        fields = {"corpus_entity_id": entity.corpus_entity_id, "entity_type": entity.type,
                  "canonical_label": entity.canonical_label, "aliases": aliases,
                  "resolution_status": entity.resolution_status, "resolution_method": entity.resolution_method,
                  "identity_scope": "cross_episode" if entity.episode_count > 1 else "episode_local",
                  "coreference_status": coreference_status, "semantic_linking_status": entity.link_status,
                  "mention_count": entity.mention_count, "episode_count": entity.episode_count,
                  "wikidata_qid": entity.wikidata_qid, "link_status": entity.link_status}
        if entity.link_status == "linked":
            external_link = (f"**External semantic linking:** "
                             f"[{entity.wikidata_qid}](https://www.wikidata.org/wiki/{entity.wikidata_qid}) — "
                             f"provisional score {entity.link_confidence:.3f}  ")
        elif entity.link_status == "ambiguous":
            external_link = (f"**External semantic linking:** ambiguous; no QID published"
                             + (f" (top score {entity.link_confidence:.3f})  "
                                if entity.link_confidence is not None else "  "))
        elif entity.link_status == "nil":
            external_link = "**External semantic linking:** NIL; no suitable Wikidata item published  "
        else:
            external_link = "**External semantic linking:** not attempted  "
        lines = [frontmatter("entity", fields), f"# {markdown_text(entity.canonical_label)}", "",
                 f"**Type:** {entity.type}  ", f"**Resolution:** {entity.resolution_status} ({entity.resolution_method})  ",
                 f"**Scope:** {'cross-episode' if entity.episode_count > 1 else 'episode-local'}  ",
                 (f"**Pronoun coreference:** provisional FCoref ({len(entity_coreference)} linked reference(s))  "
                  if entity_coreference else
                  "**Pronoun coreference:** attempted; no linked reference  " if coreference_links is not None else
                  "**Pronoun/nominal coreference:** not attempted  "),
                 external_link,
                 f"**Mentions:** {entity.mention_count} across {entity.episode_count} episode(s)", ""]
        if entity.resolution_status == "unresolved":
            lines.extend(["> [!warning] Unresolved identity",
                          f"> This episode-local node is kept separate pending a reviewed cross-episode decision ({episode_reference(episode_by_id[entity.episode_ids[0]], config)}).", ""])
        lines.extend(["## Observed surfaces", ""])
        lines.extend(f"- {markdown_text(surface)}" for surface in entity.observed_surfaces)
        lines.extend(["", "## Evidence", ""])
        by_episode_passage = defaultdict(lambda: defaultdict(list))
        for mention in member_mentions:
            for passage_id in mention.passage_ids:
                by_episode_passage[mention.episode_id][passage_id].append(mention)
        for episode_id in sorted(by_episode_passage, key=lambda eid: (episode_by_id[eid].title.casefold(), eid)):
            ep = episode_by_id[episode_id]
            lines.extend([f"### {episode_link(ep, episode_paths, naming=config)}", ""])
            for passage_id in sorted(by_episode_passage[episode_id], key=lambda pid: (passage_by_id[pid].index, pid)):
                p = passage_by_id[passage_id]
                surfaces = sorted({m.text for m in by_episode_passage[episode_id][passage_id]}, key=lambda s: (s.casefold(), s))
                lines.append(f"- {episode_link(ep, episode_paths, p, naming=config)} — " +
                             ", ".join(f"“{markdown_text(surface)}”" for surface in surfaces))
            lines.append("")
        if entity_coreference:
            lines.extend(["## Provisional coreferential references", "",
                          "> [!warning] Model output",
                          "> These pronoun links have not yet been evaluated against reviewed coreference annotations.", ""])
            by_coref_passage = defaultdict(list)
            for row in entity_coreference:
                for passage_id in row.passage_ids:
                    by_coref_passage[(row.episode_id, passage_id)].append(row)
            for (episode_id, passage_id), rows in sorted(by_coref_passage.items(),
                    key=lambda item: (episode_by_id[item[0][0]].title.casefold(),
                                      passage_by_id[item[0][1]].index, item[0][1])):
                ep, passage = episode_by_id[episode_id], passage_by_id[passage_id]
                forms = Counter(row.text.casefold() for row in rows)
                surfaces = ", ".join(f"“{markdown_text(form)}” × {count}"
                                     for form, count in sorted(forms.items()))
                lines.append(f"- {episode_link(ep, episode_paths, passage, naming=config)} — {surfaces}")
            lines.append("")
        files[f"{entity_paths[entity.corpus_entity_id]}.md"] = "\n".join(lines).encode()

    # Obsidian's graph has no edge labels, so each relation gets a note in between. That
    # shows the predicate and makes clear it is a model suggestion, not a fact.
    for assertion in sorted(relation_assertions or [], key=lambda row: row.assertion_id):
        subject = corpus_by_id[assertion.subject_corpus_entity_id]
        object_entity = corpus_by_id[assertion.object_corpus_entity_id]
        predicate_label = assertion.predicate.replace("_", " ")
        uses_coreference = any(
            evidence.head_endpoint_kind == "coreference_reference"
            or evidence.tail_endpoint_kind == "coreference_reference"
            for evidence in assertion.evidence)
        lines = [frontmatter("provisional_relation", {
                    "assertion_id": assertion.assertion_id,
                    "predicate": assertion.predicate,
                    "confidence": assertion.confidence,
                    "review_status": assertion.review_status,
                    "derivation": assertion.derivation,
                    "coreference_assisted": uses_coreference,
                }),
                f"# {markdown_text(subject.canonical_label)} — {markdown_text(predicate_label)} — "
                f"{markdown_text(object_entity.canonical_label)}", "",
                "> [!warning] Provisional GLiREL prediction",
                "> This relation is model output with passage evidence, not a reviewed fact.", "",
                f"- Subject: {entity_link(subject, entity_paths)}",
                f"- Predicate: `{assertion.predicate}`",
                f"- Object: {entity_link(object_entity, entity_paths)}",
                f"- Confidence: {assertion.confidence:.3f}",
                f"- Coreference-assisted: {'yes' if uses_coreference else 'no'}",
                f"- Status: `{assertion.review_status}`", "", "## Evidence", ""]
        for evidence in assertion.evidence:
            passage = passage_by_id[evidence.passage_id]
            episode = episode_by_id[passage.episode_id]
            resolved = []
            if evidence.head_endpoint_kind == "coreference_reference":
                reference = coreference_by_id[evidence.head_mention_id]
                resolved.append(f"subject “{markdown_text(reference.text)}” → {markdown_text(subject.canonical_label)}")
            if evidence.tail_endpoint_kind == "coreference_reference":
                reference = coreference_by_id[evidence.tail_mention_id]
                resolved.append(f"object “{markdown_text(reference.text)}” → {markdown_text(object_entity.canonical_label)}")
            suffix = f"; {'; '.join(resolved)}" if resolved else ""
            lines.append(f"- {episode_link(episode, episode_paths, passage, naming=config)} — "
                         f"confidence {evidence.score:.3f}{suffix}")
        lines.append("")
        files[f"{relation_paths[assertion.assertion_id]}.md"] = "\n".join(lines).encode()

    by_type = defaultdict(list)
    for entity in sorted_entities:
        by_type[entity.type].append(entity)
    home = [frontmatter("index", {"vault_name": config.vault_name}), f"# {markdown_text(config.vault_name)}", "",
            "This vault is a generated projection of the Prototype 1 records. Entity links open evidence-backed notes; episode links open the full transcript at a passage anchor.", "",
            "## Coverage", "", f"- Episodes: {len(episodes)}", f"- Passages: {len(passages)}",
            f"- Accepted mentions: {len(mentions)}", f"- Corpus entities: {len(corpus_entities)}",
            f"- Cross-episode entities: {sum(e.episode_count > 1 for e in corpus_entities)}",
            f"- Resolution review candidates: {len(queue)}",
            f"- Provisional linked pronouns: {len(linked_coreference)}" if coreference_links is not None else
            "- Pronoun coreference: not attempted",
            f"- Provisional semantic relations: {len(relation_assertions)}" if relation_assertions is not None else
            "- Semantic relations: not attempted", "", "## Episodes", ""]
    home.extend(f"- {episode_link(episode, episode_paths)}" for episode in sorted_episodes)
    home.extend(["", "## Entity types", ""])
    for entity_type in sorted(by_type):
        home.append(f"- [[{config.index_folder}/Entities - {entity_type}|{entity_type}]] ({len(by_type[entity_type])})")
    home.extend(["", f"See [[{config.review_folder}/Resolution queue|Resolution queue]] and "
                 f"[[{config.review_folder}/Limitations|Limitations]] before interpreting graph structure.", ""])
    files[f"{config.index_folder}/Home.md"] = "\n".join(home).encode()
    for entity_type, typed in sorted(by_type.items()):
        lines = [frontmatter("index", {"entity_type": entity_type}), f"# {entity_type} entities", ""]
        lines.extend(f"- {entity_link(entity, entity_paths)} — {entity.resolution_status}; "
                     f"{entity.mention_count} mention(s), {entity.episode_count} episode(s)" for entity in typed)
        lines.append("")
        files[f"{config.index_folder}/Entities - {entity_type}.md"] = "\n".join(lines).encode()

    review = [frontmatter("review", {"candidate_count": len(queue)}), "# Resolution queue", "",
              "These repeated normalized forms remain separate because the configured policy requires a reviewed identity decision.", ""]
    episode_entity_to_corpus = {a.episode_entity_id: a.corpus_entity_id for a in assignments}
    for candidate in sorted(queue, key=lambda q: (q.type, q.normalized_key, q.candidate_id)):
        links = []
        for eid in candidate.episode_entity_ids:
            cid = episode_entity_to_corpus[eid]
            links.append(entity_link(corpus_by_id[cid], entity_paths))
        review.extend([f"## {markdown_text(candidate.observed_surfaces[0])} ({candidate.type})", "",
                       f"- Reason: `{candidate.reason}`", f"- Mentions: {candidate.mention_count}",
                       f"- Episode-local entities: {', '.join(links)}", ""])
    files[f"{config.review_folder}/Resolution queue.md"] = "\n".join(review).encode()
    if verification is not None:
        # This list keeps every proposal, also the ones left out of the graph.
        evidence_by_id = {row["candidate_id"]: row for row in verification["evidence"]}
        decisions_by_id = {row["assertion_id"]: row for row in verification["decisions"]}
        lines = [frontmatter("review", {"verification_method": "nli", "review_status": "provisional"}),
            "# Relation verification", "",
            "The graph shows only model-supported relations. All decisions remain provisional; NLI scores are uncalibrated.",
            "Original GLiREL proposals are retained below, including withheld relations.", ""]
        for assertion in verification["assertions"]:
            decision = decisions_by_id[assertion.assertion_id]
            subject = entity_link(corpus_by_id[assertion.subject_corpus_entity_id], entity_paths)
            target = entity_link(corpus_by_id[assertion.object_corpus_entity_id], entity_paths)
            lines.extend([f"## {subject} → `{assertion.predicate}` → {target}", "",
                          f"- Verification: **{decision['decision']}** ({decision['reason']})"])
            for candidate_id in decision["candidate_ids"]:
                row = evidence_by_id[candidate_id]
                passage = passage_by_id[row["passage_id"]]
                score_text = (f"entailment {row['scores']['entailment']:.3f}, "
                    f"contradiction {row['scores']['contradiction']:.3f}, neutral {row['scores']['neutral']:.3f}"
                    if row["scores"] else "not scored")
                lines.append(f"- {episode_link(episode_by_id[passage.episode_id], episode_paths, passage, naming=config)}: "
                             f"{row['decision']} ({row['reason']}); {score_text}")
            lines.append("")
        files[f"{config.review_folder}/Relation verification.md"] = "\n".join(lines).encode()
        # Verification results go into per-relation properties, so older vault layouts stay the same.
        for assertion in relation_assertions or []:
            path = f"{relation_paths[assertion.assertion_id]}.md"
            data = files[path].decode()
            files[path] = data.replace("---\n", '---\nverification_status: "model_supported"\n', 1).encode()
        home_path = f"{config.index_folder}/Home.md"
        files[home_path] += (f"\nOnly NLI-supported provisional relations appear in the graph. "
            f"See [[{config.review_folder}/Relation verification|all verification decisions]].\n").encode()
    if relation_assertions is not None:
        # Candidate links go into the review folder, which is excluded from the graph, so unreviewed
        # relations don't show up there. Each row still links to its entities and passage.
        relation_review = [frontmatter("review", {"candidate_count": len(relation_assertions)}),
            "# Provisional relation candidates", "",
            "> [!warning] GLiREL model output",
            "> These directed relations are grounded to named mentions or resolved third-person references but are not reviewed facts.", ""]
        for assertion in sorted(relation_assertions, key=lambda row: (-row.confidence, row.assertion_id)):
            subject = entity_link(corpus_by_id[assertion.subject_corpus_entity_id], entity_paths)
            object_link = entity_link(corpus_by_id[assertion.object_corpus_entity_id], entity_paths)
            uses_coreference = any(
                evidence.head_endpoint_kind == "coreference_reference"
                or evidence.tail_endpoint_kind == "coreference_reference"
                for evidence in assertion.evidence)
            relation_review.extend([
                f"## {subject} → `{assertion.predicate}` → {object_link}", "",
                f"- Confidence: {assertion.confidence:.3f}",
                f"- Coreference-assisted: {'yes' if uses_coreference else 'no'}",
                f"- Status: `{assertion.review_status}`", "- Evidence:",
            ])
            for evidence in assertion.evidence:
                passage = passage_by_id[evidence.passage_id]
                episode = episode_by_id[passage.episode_id]
                relation_review.append(
                    f"  - {episode_link(episode, episode_paths, passage, naming=config)} — {evidence.score:.3f}")
            relation_review.append("")
        files[f"{config.review_folder}/Relation candidates.md"] = "\n".join(relation_review).encode()
    has_automatic_resolution = any(entity.resolution_method == "automatic_context" for entity in corpus_entities)
    identity_limitation = ("- Automatic name/context resolution is provisional; ambiguous candidates still abstain and need evaluation."
                           if has_automatic_resolution else
                           "- Contextual/semantic identity comparison has not been implemented; current resolution uses exact forms and reviewed aliases only.")
    coreference_limitation = (f"- FCoref pronoun links are provisional model output; {len(abstained_coreference)} ambiguous or unanchored references abstain and no reviewed coreference evaluation exists yet."
                              if coreference_links is not None else
                              "- Pronoun and nominal coreference have not been attempted; `he` or `the author` is not attached to a person entity.")
    limitations = [frontmatter("review", {}), "# Limitations", "",
        "- Entity notes come from GLiNER detections and therefore include false positives and miss some gold mentions.",
        "- `provisional_exact` cross-episode links are navigation hypotheses, not verified identity claims.",
        "- `unresolved` repeated forms remain separate until a reviewed alias or block decision is supplied.",
        coreference_limitation,
        identity_limitation,
        "- A shared episode does not assert a semantic relation between two entities.",
        "- Passage times approximate the intersecting Whisper source segments; they are not word-level timings.",
        (f"- External linking is provisional: {sum(e.link_status == 'linked' for e in corpus_entities)} "
         f"entities publish a Wikidata QID, {sum(e.link_status == 'ambiguous' for e in corpus_entities)} "
         f"abstain as ambiguous, and {sum(e.link_status == 'nil' for e in corpus_entities)} are explicit NIL."
         if any(e.link_status != "not_attempted" for e in corpus_entities) else
         "- External entity linking has not been attempted; a null Wikidata ID does not mean NIL."), ""]
    if relation_assertions is not None:
        limitations.insert(-3,
            f"- GLiREL produced {len(relation_assertions)} provisional typed relation candidates; they appear as visually distinct relation nodes and remain unreviewed model output.")
    files[f"{config.review_folder}/Limitations.md"] = "\n".join(limitations).encode()
    corpus = [frontmatter("corpus", {"corpus_id": episodes[0].corpus_id, "episode_count": len(episodes)}),
              f"# {config.corpus_title}", "",
              "This structural hub connects the pilot episodes without asserting a semantic relation between their entities.", "",
              "## Episodes", ""]
    corpus.extend(f"- {episode_link(episode, episode_paths)}" for episode in sorted_episodes)
    corpus.append("")
    files[f"{config.corpus_folder}/Corpus.md"] = "\n".join(corpus).encode()
    start = [frontmatter("index", {"vault_name": config.vault_name}), "# Start here", "",
             "This file is at the vault root. If you can see `05_Corpus`, `10_Episodes`, and `20_Entities` in the file explorer, the correct folder is open.", "",
             f"- [[{config.corpus_folder}/Corpus|Open the connected corpus graph hub]]",
             f"- [[{config.index_folder}/Home|Open the indexes]]",
             f"- [[{config.review_folder}/Resolution queue|Review unresolved identities]]", ""]
    if relation_assertions is not None:
        start.insert(-1, f"- [[{config.review_folder}/Relation candidates|Inspect provisional typed relations]]")
        start.insert(-1, f"- Open `{config.relation_folder}` to inspect relation nodes in the main graph")
    start.extend(["## Find one entity", "",
                  "Use Quick Switcher (`Cmd+O`) to open an entity by filename, then open its Local Graph.",
                  "For an exact Graph filter, query a property such as:", "",
                  "```text", '[kind:entity] [canonical_label:"Thomas Aquinas"]', "```", ""])
    files["START HERE.md"] = "\n".join(start).encode()
    if config.concept_navigation:
        add_concept_navigation(config, files, episodes, passages, mentions, corpus_entities,
                               assignments, episode_paths, entity_paths)
    graph = {"collapse-filter": False, "search": " ".join(f'-path:\"{p}\"' for p in config.graph_exclude_paths),
             "showTags": False, "showAttachments": False, "hideUnresolved": True,
             "showOrphans": False, "collapse-color-groups": False,
             "colorGroups": [
                 {"query": f'path:\"{config.corpus_folder}\"', "color": {"a": 1, "rgb": 10066329}},
                 {"query": f'path:\"{config.episode_folder}\"', "color": {"a": 1, "rgb": 4360181}},
                 {"query": f'path:\"{config.entity_folder}\"', "color": {"a": 1, "rgb": 16750848}},
                 {"query": f'path:\"{config.relation_folder}\"', "color": {"a": 1, "rgb": 11240191}}],
             "collapse-display": False, "showArrow": False, "textFadeMultiplier": 0,
             "nodeSizeMultiplier": 1, "lineSizeMultiplier": 1, "collapse-forces": False,
             "centerStrength": 0.518713248970312, "repelStrength": 10, "linkStrength": 1,
             "linkDistance": 250, "scale": 1, "close": True}
    if config.concept_navigation:
        # Obsidian uses the first matching color group, so the type colors come before the general one.
        graph["colorGroups"] = [
            {"query": '[entity_type:Concept]', "color": {"a": 1, "rgb": 4897144}},
            {"query": '[entity_type:School]', "color": {"a": 1, "rgb": 15641186}},
            *graph["colorGroups"]]
    files[".obsidian/graph.json"] = (json.dumps(graph, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    files[".obsidian/core-plugins.json"] = (json.dumps(["backlink", "file-explorer", "graph", "search"], indent=2) + "\n").encode()
    if discovery is not None:
        # Imported here so the export never loads any model packages.
        from spoken_record.vault_discovery import add_passage_discovery
        add_passage_discovery(config, files, episodes, passages, corpus_entities,
                              episode_paths, entity_paths, discovery)
    return files


def add_concept_navigation(config, files, episodes, passages, mentions, entities, assignments,
                           episode_paths, entity_paths):
    """Add a limited number of shared-passage links in both directions. Identities don't change.

    The full list is saved with the export for checking. The visible links are labelled as
    co-mentions, not as school membership or any factual relation.
    """
    report = build_concept_navigation(passages, mentions, entities, assignments)
    entity_by_id = {row.corpus_entity_id: row for row in entities}
    episode_by_id = {row.episode_id: row for row in episodes}
    passage_by_id = {row.passage_id: row for row in passages}
    mention_by_id = {row.mention_id: row for row in mentions}
    by_entity = defaultdict(list)
    for row in report["connections"]:
        by_entity[row["topic_entity_id"]].append((row["other_entity_id"], row))
        by_entity[row["other_entity_id"]].append((row["topic_entity_id"], row))
    topic_ids = {row["corpus_entity_id"] for row in report["topics"]}
    inventory_path = f"{config.review_folder}/Shared passage inventory"
    inventory = [frontmatter("review", {"navigation_method": report["method"]}),
        "# All shared-passage connections", "",
        "Complete named co-mention navigation, including neighbors omitted from compact entity notes. "
        "These links do not assert influence, membership or agreement. This review note is excluded "
        "from the default graph.", ""]
    for entity_id in sorted(topic_ids | set(by_entity)):
        heading = ("People and works in shared passages" if entity_id in topic_ids
                   else "Concepts and schools in shared passages")
        lines = ["", f"## {heading}", "",
            "These entities have named mentions in the same passage. This does not establish "
            "membership, influence, agreement or the passage's main topic. Types and identities "
            "retain their upstream review status.", ""]
        neighbors = sorted(by_entity[entity_id], key=lambda pair: (-pair[1]["passage_count"],
            entity_by_id[pair[0]].canonical_label.casefold(), pair[0]))
        inventory.extend([f"## {entity_link(entity_by_id[entity_id], entity_paths)}", "",
                          f"^nav-{entity_id}", ""])
        for other_id, row in neighbors:
            inventory.append(f"- {entity_link(entity_by_id[other_id], entity_paths)} — "
                             f"{row['passage_count']} shared passage(s)")
            for item in row["evidence"]:
                passage = passage_by_id[item["passage_id"]]
                inventory.append(f"  - {episode_link(episode_by_id[passage.episode_id], episode_paths, passage, naming=config)}")
        if not neighbors:
            inventory.append("No named Person or Work co-mentions found.")
        inventory.append("")
        lines.append(f"Showing {min(len(neighbors), config.navigation_max_neighbors)} of {len(neighbors)} "
                     "neighbors, ranked by distinct shared passages.")
        for other_id, row in neighbors[:config.navigation_max_neighbors]:
            other = entity_by_id[other_id]
            lines.extend(["", f"- {entity_link(other, entity_paths)} ({other.type}) — "
                f"{row['passage_count']} shared passage(s) across {row['episode_count']} episode(s)."])
            evidence = sorted(row["evidence"], key=lambda e: (
                episode_by_id[e["episode_id"]].title.casefold(),
                passage_by_id[e["passage_id"]].index, e["passage_id"]))
            for item in evidence[:config.navigation_max_evidence]:
                passage = passage_by_id[item["passage_id"]]
                lines.append(f"  - {episode_link(episode_by_id[passage.episode_id], episode_paths, passage, naming=config)}")
            if len(evidence) > config.navigation_max_evidence:
                lines.append(f"  - {len(evidence) - config.navigation_max_evidence} additional shared passage(s) "
                             "in the complete navigation inventory.")
        if not neighbors:
            lines.append("No named Person or Work co-mentions were found in this entity's passages.")
        lines.extend(["", f"[[{inventory_path}#^nav-{entity_id}|All neighbors and shared-passage evidence]]", "",
                      f"[[{config.index_folder}/Explore concepts and schools|Explore concepts and schools]]", ""])
        files[f"{entity_paths[entity_id]}.md"] += "\n".join(lines).encode()

    index = [frontmatter("index", {"navigation_method": report["method"]}),
        "# Explore concepts and schools", "",
        "Start with an entity below, follow its shared-passage links to people and works, "
        "then inspect the transcript evidence. These are detected labels, not a curated topic taxonomy.", "",
        "Equal names can represent unresolved episode-local identities; this view does not merge them. "
        "Cross-episode exploration can proceed through shared people or works even when a concept remains local.", "",
        "## Coverage", ""]
    index.extend(f"- {key.replace('_', ' ')}: {value}" for key, value in report["counts"].items())
    if not report["counts"]["cross_episode_topics"]:
        index.extend(["", "No Concept or School identity currently has named evidence in multiple episodes. "
                      "This is a coverage limit, not evidence that the discussions are unrelated."])
    for kind in ("Concept", "School"):
        index.extend(["", f"## {kind}", ""])
        entries = sorted((e for e in report["topics"] if e["entity_type"] == kind),
            key=lambda e: (-len(e["passage_ids"]), entity_by_id[e["corpus_entity_id"]].canonical_label.casefold(),
                           e["corpus_entity_id"]))
        if not entries:
            index.append(f"No {kind} entities detected.")
        for entry in entries:
            entity = entity_by_id[entry["corpus_entity_id"]]
            episode_labels = ", ".join(episode_reference(episode_by_id[eid], config) for eid in entry["episode_ids"])
            index.append(f"- {entity_link(entity, entity_paths)} — {len(entry['passage_ids'])} passage(s); "
                         f"{markdown_text(episode_labels)}; {entity.resolution_status}")
            # A short excerpt from the source shows when a label is noisy, without changing the text.
            if entry["mention_ids"]:
                mention = mention_by_id[entry["mention_ids"][0]]
                passage = passage_by_id[next(pid for pid in entry["passage_ids"] if pid in mention.passage_ids)]
                start = max(0, mention.char_start - passage.char_start - 70)
                end = min(len(passage.text), mention.char_end - passage.char_start + 100)
                excerpt = ("…" if start else "") + passage.text[start:end].strip() + ("…" if end < len(passage.text) else "")
                index.append(f"  - {episode_link(episode_by_id[passage.episode_id], episode_paths, passage, naming=config)}: "
                             f"“{markdown_text(excerpt)}”")
    index.extend(["", "## Read the graph", "",
        "Open a concept or school note and its Local Graph. Direct entity links in this export "
        "mean shared passage context. Nodes in `30_Relations` represent separate provisional typed relations.",
        "For a global concept/person/work view, use:", "", "```text",
        'path:"20_Entities" ([entity_type:Concept] OR [entity_type:School] OR [entity_type:Person] OR [entity_type:Work])',
        "```", "", "Native graph edges have no predicate labels. Read the note's shared-passage section "
        "before interpreting a connection. Home, type indexes and the corpus hub are navigation aids, "
        "excluded from this export's default graph.", "",
        "## Complete inventory", "", f"[[{inventory_path}|Browse all neighbors and supporting passages]], "
        "including those outside the compact view's display limits. Original entity evidence sections remain complete.", ""])
    files[f"{config.index_folder}/Explore concepts and schools.md"] = "\n".join(index).encode()
    files[f"{inventory_path}.md"] = "\n".join(inventory).encode()
    files["_meta/concept_navigation.json"] = (json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    for path in ("START HERE.md", f"{config.index_folder}/Home.md"):
        files[path] += (f"\n## Explore discussions\n\n"
            f"- [[{config.index_folder}/Explore concepts and schools|Explore concepts and schools through shared passages]]\n"
            "- Shared-passage links are navigation aids, not factual relations or school membership.\n").encode()


def navigation_counts(files: dict[str, bytes]) -> dict[str, int]:
    """Add navigation counts to the manifest only when the navigation view is on."""
    report = files.get("_meta/concept_navigation.json")
    if report is None:
        return {}
    return {f"navigation_{key}": value for key, value in json.loads(report)["counts"].items()}


def validate_links(files: dict[str, bytes], config: VaultConfig, passages: list[Passage]) -> None:
    markdown = {name[:-3]: data.decode() for name, data in files.items() if name.endswith(".md")}
    anchors = {name: set(ANCHOR.findall(text)) for name, text in markdown.items()}
    adjacency = defaultdict(set)
    for source, text in markdown.items():
        for target, anchor in WIKILINK.findall(text):
            if target not in markdown:
                raise ValueError(f"broken vault link from {source}: {target}")
            if anchor and anchor not in anchors[target]:
                raise ValueError(f"broken vault anchor from {source}: {target}#^{anchor}")
            adjacency[source].add(target)
            adjacency[target].add(source)
    expected = {p.passage_id for p in passages}
    episode_text = "\n".join(text for name, text in markdown.items() if name.startswith(config.episode_folder + "/"))
    observed_list = ANCHOR.findall(episode_text)
    if set(observed_list) != expected or len(observed_list) != len(expected):
        raise ValueError("episode-note anchors do not match passages")
    exploration = {name for name in markdown if not name.startswith(
        (config.index_folder + "/", config.review_folder + "/"))}
    reached, pending = set(), ["START HERE"]
    while pending:
        current = pending.pop()
        if current in reached:
            continue
        reached.add(current)
        pending.extend(adjacency[current] & exploration - reached)
    if reached != exploration:
        raise ValueError("episode/entity exploration graph is disconnected")


def load_resolution_chain(root: Path, parent: Path):
    resolution_manifest = validate_resolution(parent)
    extraction = parent.parent / resolution_manifest.get("extraction_run_id", resolution_manifest["parent_run_id"])
    extraction_manifest = json.loads((extraction / "manifest.json").read_bytes())
    passage_run = extraction.parent / extraction_manifest["parent_run_id"]
    passage_manifest = json.loads((passage_run / "manifest.json").read_bytes())
    source_run = passage_run.parent / passage_manifest["parent_run_id"]
    return (
        [Episode.model_validate(r) for r in read_jsonl(source_run / "episodes.jsonl")],
        [TranscriptVersion.model_validate(r) for r in read_jsonl(source_run / "transcripts.jsonl")],
        [Passage.model_validate(r) for r in read_jsonl(passage_run / "passages.jsonl")],
        [ExtractedMention.model_validate(r) for r in read_jsonl(extraction / "mentions.jsonl")],
        [EpisodeEntity.model_validate(r) for r in read_jsonl(parent / "episode_entities.jsonl")],
        [CorpusEntity.model_validate(r) for r in read_jsonl(parent / "corpus_entities.jsonl")],
        [MentionAssignment.model_validate(r) for r in read_jsonl(parent / "assignments.jsonl")],
        [ReviewCandidate.model_validate(r) for r in read_jsonl(parent / "review_queue.jsonl")],
    )


def vault_input(root: Path, parent: Path):
    manifest = json.loads((parent / "manifest.json").read_bytes())
    if manifest.get("stage") == "entity_enrichment":
        from spoken_record.enrichment import validate_run as validate_enrichment
        validate_enrichment(parent)
        return vault_input(root, parent.parent / manifest["parent_run_id"])
    if manifest.get("stage") == "passage_discovery":
        from spoken_record.discovery import validate_run as validate_discovery
        validate_discovery(parent)
        return vault_input(root, parent.parent / manifest["parent_run_id"])
    if manifest.get("stage") == "relation_verification":
        from spoken_record.relation_verification import validate_run as validate_verification
        validate_verification(parent)
        resolution, links, _ = vault_input(root, parent.parent / manifest["parent_run_id"])
        assertions = [RelationAssertion.model_validate(row) for row in read_jsonl(parent / "relation_assertions.jsonl")]
        return resolution, links, assertions
    if manifest.get("stage") == "relations":
        from spoken_record.relations import validate_run as validate_relations
        validate_relations(parent)
        wrapped_parent = parent.parent / manifest["parent_run_id"]
        resolution, links, _ = vault_input(root, wrapped_parent)
        assertions = [RelationAssertion.model_validate(row)
                      for row in read_jsonl(parent / "relation_assertions.jsonl")]
        return resolution, links, assertions
    if manifest.get("stage") == "coreference":
        from spoken_record.coreference import validate_run as validate_coreference
        validate_coreference(parent)
        resolution = parent.parent / manifest["parent_run_id"]
        links = [CoreferenceLink.model_validate(row) for row in read_jsonl(parent / "coreference_links.jsonl")]
        return resolution, links, None
    validate_resolution(parent)
    return parent, None, None


def verification_view(parent: Path) -> dict | None:
    """Load all decisions of a validated verification run."""
    manifest = json.loads((parent / "manifest.json").read_bytes())
    if manifest.get("stage") in ("passage_discovery", "entity_enrichment"):
        return verification_view(parent.parent / manifest["parent_run_id"])
    if manifest.get("stage") != "relation_verification":
        return None
    original = parent.parent / manifest["parent_run_id"]
    return {"decisions": read_jsonl(parent / "verification_decisions.jsonl"),
            "evidence": read_jsonl(parent / "verification_evidence.jsonl"),
            "assertions": [RelationAssertion.model_validate(row) for row in read_jsonl(original / "relation_assertions.jsonl")]}


def discovery_view(parent: Path) -> dict | None:
    """Load the chosen passage links from a validated discovery run."""
    manifest = json.loads((parent / "manifest.json").read_bytes())
    if manifest.get("stage") == "entity_enrichment":
        return discovery_view(parent.parent / manifest["parent_run_id"])
    if manifest.get("stage") != "passage_discovery":
        return None
    return {"connections": read_jsonl(parent / "passage_connections.jsonl"),
            "inputs": read_jsonl(parent / "passage_inputs.jsonl"),
            "counts": manifest["counts"], "run_id": manifest["run_id"]}


def export_vault(root: Path, input_name: str, config_name: str, run_id: str | None = None) -> Path:
    """Write an Obsidian vault for a validated identity, coreference or relation run."""
    root, started, now = root.resolve(), perf_counter(), datetime.now(timezone.utc)
    run_id = run_id or f"vault-{now:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]{0,95}", run_id):
        raise ValueError("invalid run ID")
    parent = workspace_path(root, input_name)
    if parent.parent != root / "data/prototype1/runs":
        raise ValueError("input must be a resolution snapshot")
    output = root / "export/prototype1" / run_id
    if output.exists():
        raise FileExistsError(output)
    resolution_parent, coreference_links, relation_assertions = vault_input(root, parent)
    parent_digest = sha256((parent / "manifest.json").read_bytes())
    config_path = workspace_path(root, config_name)
    config_bytes = config_path.read_bytes()
    config = VaultConfig.model_validate_json(config_bytes)
    records = load_resolution_chain(root, resolution_parent)
    files = build_vault_files(config, *records, coreference_links, relation_assertions,
                             verification_view(parent), discovery_view(parent))
    validate_links(files, config, records[2])
    vault_input(root, parent)
    if sha256((parent / "manifest.json").read_bytes()) != parent_digest or config_path.read_bytes() != config_bytes:
        raise ValueError("parent or vault configuration changed")
    vault = output / "obsidian"
    for name, data in files.items():
        path = vault / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    counts = {"episodes": len(records[0]), "passages": len(records[2]), "mentions": len(records[3]),
              "episode_entities": len(records[4]), "corpus_entities": len(records[5]),
              "review_candidates": len(records[7]), "markdown_notes": sum(n.endswith('.md') for n in files),
              "passage_anchors": len(records[2]), "cross_episode_entities": sum(e.episode_count > 1 for e in records[5]),
              "linked_coreference_references": sum(row.decision == "linked" for row in (coreference_links or []))}
    if any(row.link_status != "not_attempted" for row in records[5]):
        counts.update({"wikidata_linked_entities": sum(row.link_status == "linked" for row in records[5]),
                       "wikidata_ambiguous_entities": sum(row.link_status == "ambiguous" for row in records[5]),
                       "wikidata_nil_entities": sum(row.link_status == "nil" for row in records[5])})
    if relation_assertions is not None:
        counts["provisional_relation_assertions"] = len(relation_assertions)
    counts.update(navigation_counts(files))
    if (view := discovery_view(parent)) is not None:
        counts.update({f"discovery_{k}": v for k, v in view["counts"].items()})
        counts["discovery_passage_notes"] = len(records[2])
    code_root = Path(__file__).parent
    manifest = {"schema_version": 2, "stage": "export_obsidian", "run_id": run_id, "status": "complete",
        "package_version": __version__, "started_utc": now.isoformat(),
        "elapsed_seconds": round(perf_counter() - started, 4), "python": platform.python_version(),
        "dependencies": {"pydantic": importlib.metadata.version("pydantic")},
        "parent_run": input_name, "parent_manifest_sha256": parent_digest,
        "input_config": config_name, "input_config_sha256": sha256(config_bytes), "counts": counts,
        "code_sha256": {str(p.relative_to(code_root)): sha256(p.read_bytes()) for p in sorted(code_root.rglob("*.py"))},
        "artifact_sha256": {f"obsidian/{name}": sha256(data) for name, data in sorted(files.items())}}
    output.mkdir(parents=True, exist_ok=True)
    dump_json(output / "manifest.json", manifest)
    validate_run(output)
    return output


def validate_run(output: Path) -> dict:
    output = output.resolve()
    manifest = json.loads((output / "manifest.json").read_bytes())
    if manifest.get("schema_version") != 2 or manifest.get("stage") != "export_obsidian" or manifest.get("status") != "complete":
        raise ValueError("unsupported Obsidian export")
    root = output.parents[2]
    parent = workspace_path(root, manifest["parent_run"])
    if sha256((parent / "manifest.json").read_bytes()) != manifest["parent_manifest_sha256"]:
        raise ValueError("parent resolution manifest changed")
    resolution_parent, coreference_links, relation_assertions = vault_input(root, parent)
    config_path = workspace_path(root, manifest["input_config"])
    if sha256(config_path.read_bytes()) != manifest["input_config_sha256"]:
        raise ValueError("vault configuration changed")
    config = VaultConfig.model_validate_json(config_path.read_bytes())
    records = load_resolution_chain(root, resolution_parent)
    expected = build_vault_files(config, *records, coreference_links, relation_assertions,
                                verification_view(parent), discovery_view(parent))
    expected_names = {f"obsidian/{name}" for name in expected}
    if set(manifest["artifact_sha256"]) != expected_names:
        raise ValueError("unexpected vault artifact set")
    for relative, digest in manifest["artifact_sha256"].items():
        data = (output / relative).read_bytes()
        if relative.startswith("obsidian/.obsidian/"):
            json.loads(data)
            continue
        if sha256(data) != digest or data != expected[relative.removeprefix("obsidian/")]:
            raise ValueError(f"vault artifact is changed or not reproducible: {relative}")
    validate_links(expected, config, records[2])
    counts = {"episodes": len(records[0]), "passages": len(records[2]), "mentions": len(records[3]),
              "episode_entities": len(records[4]), "corpus_entities": len(records[5]),
              "review_candidates": len(records[7]), "markdown_notes": sum(n.endswith('.md') for n in expected),
              "passage_anchors": len(records[2]), "cross_episode_entities": sum(e.episode_count > 1 for e in records[5]),
              "linked_coreference_references": sum(row.decision == "linked" for row in (coreference_links or []))}
    if any(row.link_status != "not_attempted" for row in records[5]):
        counts.update({"wikidata_linked_entities": sum(row.link_status == "linked" for row in records[5]),
                       "wikidata_ambiguous_entities": sum(row.link_status == "ambiguous" for row in records[5]),
                       "wikidata_nil_entities": sum(row.link_status == "nil" for row in records[5])})
    if relation_assertions is not None:
        counts["provisional_relation_assertions"] = len(relation_assertions)
    counts.update(navigation_counts(expected))
    if (view := discovery_view(parent)) is not None:
        counts.update({f"discovery_{k}": v for k, v in view["counts"].items()})
        counts["discovery_passage_notes"] = len(records[2])
    if counts != manifest["counts"]:
        raise ValueError("vault manifest counts differ")
    return manifest
