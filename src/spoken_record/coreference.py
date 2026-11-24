"""Pronoun coreference with FCoref. Offsets point into the original transcript, and a
pronoun is only linked to an entity when the match is clear.
"""

import json
import re

import spacy

from spoken_record.backends import CoreferenceBackend
from spoken_record.adapters.base import stable_id
from spoken_record.coreference_models import (CoreferenceConfig, CoreferenceLink,
                                              CoreferenceSpan, ModelCoreferenceCluster)
from spoken_record.extraction_models import ExtractedMention
from spoken_record.models import TranscriptVersion
from spoken_record.passage_models import Passage
from spoken_record.resolution_models import MentionAssignment


def intersecting_passages(passage_rows: list[Passage], start: int, end: int) -> list[str]:
    return [passage.passage_id for passage in passage_rows
            if passage.char_start < end and passage.char_end > start]


def span_matches_mention(span: CoreferenceSpan, mention: ExtractedMention) -> bool:
    return ((span.char_start <= mention.char_start and span.char_end >= mention.char_end) or
            (mention.char_start <= span.char_start and mention.char_end >= span.char_end))


def capitalized_name(surface: str) -> bool:
    return any(token[:1].isupper() for token in re.findall(r"[^\W_]+", surface, re.UNICODE))


def project_clusters(run_id: str, episode_id: str, transcript: TranscriptVersion,
                     raw_clusters: list[list[tuple[int, int]]], passages: list[Passage],
                     mentions: list[ExtractedMention], assignments: list[MentionAssignment],
                     config: CoreferenceConfig, fingerprint: str):
    passage_rows = sorted((passage for passage in passages if passage.episode_id == episode_id),
                          key=lambda passage: passage.index)
    person_mentions = [mention for mention in mentions
                       if mention.episode_id == episode_id and mention.type == config.anchor_type]
    assignment_by_mention = {row.mention_id: row for row in assignments}
    forms = set(config.reference_forms)
    clusters, links = [], []
    occupied_references = set()
    for raw_spans in raw_clusters:
        spans = [CoreferenceSpan(char_start=start, char_end=end, text=transcript.text[start:end])
                 for start, end in raw_spans]
        if len(spans) < 2:
            continue
        cluster_id = stable_id("fc", transcript.transcript_version_id,
                               json.dumps(raw_spans, separators=(",", ":")), fingerprint)
        cluster = ModelCoreferenceCluster(created_by_run_id=run_id, cluster_id=cluster_id,
            episode_id=episode_id, transcript_version_id=transcript.transcript_version_id,
            spans=spans, model_fingerprint=fingerprint)
        clusters.append(cluster)
        reference_spans = [span for span in spans if span.text.casefold().strip() in forms]
        anchor_spans = [span for span in spans if span.text.casefold().strip() not in forms]
        anchors = sorted({mention.mention_id for span in anchor_spans for mention in person_mentions
                          if span_matches_mention(span, mention) and
                          (not config.require_capitalized_anchor or capitalized_name(mention.text))})
        episode_entities = sorted({assignment_by_mention[mid].episode_entity_id for mid in anchors})
        corpus_entities = sorted({assignment_by_mention[mid].corpus_entity_id for mid in anchors})
        for span in reference_spans:
            reference_key = (span.char_start, span.char_end)
            if reference_key in occupied_references:
                raise ValueError("one reference span occurs in multiple model clusters")
            occupied_references.add(reference_key)
            passage_ids = intersecting_passages(passage_rows, span.char_start, span.char_end)
            if not passage_ids:
                raise ValueError("coreference span has no evidence passage")
            if len(corpus_entities) == 1:
                decision, reason, corpus_entity_id = "linked", "single_corpus_entity", corpus_entities[0]
            elif not corpus_entities:
                decision, reason, corpus_entity_id = "abstain", "no_named_person_anchor", None
            else:
                decision, reason, corpus_entity_id = "abstain", "ambiguous_person_anchors", None
            links.append(CoreferenceLink(created_by_run_id=run_id,
                reference_id=stable_id("cr", transcript.transcript_version_id,
                                       str(span.char_start), str(span.char_end), fingerprint),
                episode_id=episode_id, transcript_version_id=transcript.transcript_version_id,
                cluster_id=cluster_id, char_start=span.char_start, char_end=span.char_end,
                text=span.text, reference_kind="third_person_pronoun", passage_ids=passage_ids,
                anchor_mention_ids=anchors, episode_entity_ids=episode_entities,
                corpus_entity_id=corpus_entity_id, decision=decision, reason=reason))
    return (tuple(sorted(clusters, key=lambda row: row.cluster_id)),
            tuple(sorted(links, key=lambda row: (row.episode_id, row.char_start, row.char_end))))


def predict_clusters(model: CoreferenceBackend, text: str, config: CoreferenceConfig) -> list[list[tuple[int, int]]]:
    """FCoref clusters for one transcript, as character spans. This only depends on the text."""
    prediction = model.predict(texts=[text], max_tokens_in_batch=config.max_tokens_in_batch)[0]
    raw = prediction.get_clusters(as_strings=False)
    if hasattr(prediction, "release_logits"):
        prediction.release_logits()
    return [[(int(start), int(end)) for start, end in cluster] for cluster in raw]


def load_model(config: CoreferenceConfig) -> CoreferenceBackend:
    """Import and load the model only for inference. Validation doesn't need it."""
    from fastcoref import FCoref
    from huggingface_hub import snapshot_download

    local_model = snapshot_download(config.repo_id, revision=config.revision,
                                    local_files_only=config.offline_only)
    return FCoref(model_name_or_path=local_model, device=config.device,
                  nlp=spacy.blank("en"), enable_progress_bar=False)
