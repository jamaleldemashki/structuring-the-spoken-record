"""Context windows for relation extraction. Earlier named mentions are added as context
only, they don't create extra candidate pairs.
"""

from spoken_record.adapters.base import sha256, stable_id
from spoken_record.relation_models import RelationContext


class GLiRELBudget:
    """Count the real encoder input for the pinned model, prompts and entity markers included."""

    def __init__(self, model, config):
        """Raise if the configured word or subword limits are larger than the model allows."""
        self.model = model
        self.config = config
        self.tokenizer = model.token_rep_layer.bert_layer.tokenizer
        if config.max_context_tokens > model.base_config.max_len:
            raise ValueError("relation context word budget exceeds checkpoint max_len")
        limit = model.token_rep_layer.bert_layer.model.config.max_position_embeddings
        if config.encoder_max_tokens > limit:
            raise ValueError("relation encoder budget exceeds checkpoint positions")

    def __call__(self, tokens, ner):
        """Run the model's own preprocessing (no truncation, no forward pass) to count tokens."""
        labels = self.config.prompts()
        words = list(tokens)
        if getattr(self.model.base_config, "add_entity_markers", False):
            from glirel.modules.base import insert_entity_markers
            words, _, _ = insert_entity_markers(words, [(row[0], row[1]) for row in ner],
                [], self.model.base_config.entity_start_token, self.model.base_config.entity_end_token)
        prompt = [part for label in labels for part in (self.model.rel_token, label)]
        prompt.append(self.model.sep_token)
        encoded = self.tokenizer([prompt + words], is_split_into_words=True,
                                 truncation=False, padding=False)
        return len(encoded["input_ids"][0])


def antecedent_contexts(run_id, passage, endpoints, doc, mentions, references,
                        assignments, passages, config, budget):
    """Pick the closest earlier anchors of the cluster and fit whole sentences into both budgets.

    ``doc`` is the full transcript. Names are not substituted, sentences are not spliced and
    nothing is guessed from nearby names. Only endpoints in the focal sentence become relation
    candidates, so the extra context can't create pairs across sentences.
    """
    span_by_id = {}
    for endpoint in endpoints:
        span = doc.char_span(endpoint.char_start, endpoint.char_end, alignment_mode="strict")
        if span is not None and span.text == endpoint.text:
            span_by_id[endpoint.source_id] = span
    sentences = list(doc.sents)
    mention_by_id = {row.mention_id: row for row in mentions}
    reference_by_id = {row.reference_id: row for row in references}
    contexts, records = [], []
    for index, sentence in enumerate(sentences):
        if not passage.char_start <= sentence.start_char < sentence.end_char <= passage.char_end:
            continue
        focal = [row for row in endpoints if row.source_id in span_by_id
                 and sentence.start <= span_by_id[row.source_id].start
                 and span_by_id[row.source_id].end <= sentence.end]
        if len(focal) < 2:
            continue
        selected, missing = [], []
        for endpoint in focal:
            if endpoint.kind != "coreference_reference":
                continue
            link = reference_by_id[endpoint.source_id]
            anchors = [mention_by_id[mid] for mid in link.anchor_mention_ids if mid in mention_by_id
                and mention_by_id[mid].transcript_version_id == passage.transcript_version_id
                and mention_by_id[mid].char_end <= endpoint.char_start
                and assignments[mid].corpus_entity_id == endpoint.corpus_entity_id]
            if anchors:
                selected.append(max(anchors, key=lambda row: (row.char_end, row.mention_id)))
            else:
                missing.append(endpoint.source_id)
        earliest = max(0, index - config.max_context_sentences + 1)
        anchor_sentences = [i for anchor in selected for i in range(earliest, index + 1)
                            if sentences[i].start_char <= anchor.char_start
                            and anchor.char_end <= sentences[i].end_char]
        has_reference = any(row.kind == "coreference_reference" for row in focal)
        first = min(anchor_sentences) if anchor_sentences else max(earliest, index - 1) if has_reference else index
        # Drop the oldest context sentences until it fits. The focal sentence always stays whole.
        while True:
            first_sentence = sentences[first]
            tokens = [token.text for token in doc[first_sentence.start:sentence.end]]
            ner, by_span = [], {}
            for endpoint in focal:
                span = span_by_id[endpoint.source_id]
                start, end = span.start - first_sentence.start, span.end - first_sentence.start
                ner.append([start, end - 1, endpoint.type, endpoint.text])
                by_span[(start, end)] = endpoint
            encoded_count = budget(tokens, ner)
            fits = len(tokens) <= config.max_context_tokens and encoded_count <= config.encoder_max_tokens
            if fits or first == index:
                break
            first += 1
        start, end = first_sentence.start_char, sentence.end_char
        included = sorted({anchor.mention_id for anchor in selected
                           if start <= anchor.char_start and anchor.char_end <= end})
        record = RelationContext(created_by_run_id=run_id,
            context_id=stable_id("rc", passage.passage_id, str(sentence.start_char)),
            episode_id=passage.episode_id, transcript_version_id=passage.transcript_version_id,
            passage_id=passage.passage_id,
            evidence_passage_ids=[p.passage_id for p in passages
                if p.transcript_version_id == passage.transcript_version_id
                and p.char_start < end and p.char_end > start],
            char_start=start, char_end=end, focal_start=sentence.start_char, focal_end=end,
            text_sha256=sha256(doc.text[start:end].encode("utf-8")),
            endpoint_ids=[row.source_id for row in focal],
            selected_anchor_ids=sorted({anchor.mention_id for anchor in selected}),
            included_anchor_ids=included, missing_anchor_reference_ids=sorted(missing),
            token_count=len(tokens), encoder_token_count=encoded_count, sentence_count=index-first+1,
            status="ready" if fits else "budget_exceeded")
        records.append(record)
        if fits:
            contexts.append((tokens, ner, by_span, start, end))
    skipped = sorted(row.source_id for row in endpoints if row.source_id not in span_by_id)
    return contexts, set(span_by_id), skipped, records
