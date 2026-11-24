"""Regression tests for relation contexts: offsets, distant antecedents and budgets."""

from types import SimpleNamespace

import pytest
import spacy

from test_relations import relation_config
from spoken_record.coreference_models import CoreferenceLink
from spoken_record.extraction_models import ExtractedMention
from spoken_record.passage_models import Passage
from spoken_record.relation_contexts import antecedent_contexts, GLiRELBudget
from spoken_record.relations import passage_endpoints, project_predictions, build_assertions
from spoken_record.resolution_models import MentionAssignment


def example():
    """Put another name between Augustine's distant anchor and his resolved pronoun."""
    text = 'Augustine spoke about café life. Zeno disagreed. The debate continued. He wrote Book of Reason.'
    split = text.index('He wrote')
    passages = [Passage(created_by_run_id='p', passage_id=f'p{i}', episode_id='e',
        transcript_version_id='t', index=i, char_start=start, char_end=end, text=text[start:end],
        word_count=len(text[start:end].split()), timing_precision='unavailable', source_segment_ids=[],
        method='sentence_packing', segmentation_fingerprint='seg')
        for i,(start,end) in enumerate([(0,split),(split,len(text))])]
    mentions = [ExtractedMention(created_by_run_id='ner', mention_id=mid, episode_id='e',
        transcript_version_id='t', char_start=text.index(surface), char_end=text.index(surface)+len(surface),
        text=surface, type=kind, score=.9, source_window_ids=['w'], passage_ids=[pid], extraction_fingerprint='ner')
        for mid,surface,kind,pid in [('a','Augustine','Person','p0'), ('z','Zeno','Person','p0'),
                                   ('b','Book of Reason','Work','p1')]]
    assignments = [MentionAssignment(created_by_run_id='resolve', mention_id=m.mention_id,
        episode_entity_id=f'ee-{m.mention_id}', corpus_entity_id=f'ce-{m.mention_id}') for m in mentions]
    reference = CoreferenceLink(created_by_run_id='coref', reference_id='he', episode_id='e',
        transcript_version_id='t', cluster_id='c', char_start=split, char_end=split+2, text='He',
        reference_kind='third_person_pronoun', passage_ids=['p1'], anchor_mention_ids=['a'],
        episode_entity_ids=['ee-a'], corpus_entity_id='ce-a', decision='linked', reason='single_corpus_entity')
    return text, passages, mentions, assignments, reference


def contexts(**updates):
    """Build contexts with a fake subword counter, without loading GLiREL."""
    text, passages, mentions, assignments, reference = example()
    config = relation_config().model_copy(update=dict(context_unit='coreference_antecedent_window',
        include_linked_coreference=True, **updates))
    nlp = spacy.blank('en')
    nlp.add_pipe('sentencizer')
    mapping = {row.mention_id: row for row in assignments}
    endpoints = passage_endpoints(passages[1], mentions, mapping, [reference])
    result = antecedent_contexts('run', passages[1], endpoints, nlp(text), mentions, [reference],
        mapping, passages, config, lambda tokens, ner: len(tokens)*2+10)
    return result, (text, passages, mentions, assignments, reference, mapping, config)


def test_distant_anchor_crosses_passage_without_changing_triple_subject():
    """The named anchor is only context, the pronoun stays the relation endpoint."""
    (inputs, _, _, records), args = contexts()
    text, passages, mentions, assignments, reference, mapping, config = args
    tokens, ner, spans, start, end = inputs[0]
    assert start == 0 and end == len(text)
    assert 'Augustine' in tokens and 'Zeno' in tokens
    assert {row.source_id for row in spans.values()} == {'he','b'}
    assert records[0].included_anchor_ids == ['a']
    assert records[0].evidence_passage_ids == ['p0','p1']
    by_id = {row.source_id: list(span) for span,row in spans.items()}
    candidates = project_predictions('run', passages[1], [dict(head_pos=by_id['he'],
        tail_pos=by_id['b'], label='author of', score=.9)], spans, mapping, config, 'fp', start,end)
    assertion = build_assertions('run', candidates)[0]
    assert assertion.subject_corpus_entity_id == 'ce-a'
    assert assertion.predicate == 'authored'


@pytest.mark.parametrize('limits', [dict(max_context_tokens=12), dict(encoder_max_tokens=36),
                                    dict(max_context_sentences=2)])
def test_antecedent_omissions_are_explicit_and_pronoun_is_preserved(limits):
    """A limited context never rewrites or drops the focal reference."""
    (inputs, _, _, records), _ = contexts(**limits)
    assert len(inputs) == 1
    assert records[0].selected_anchor_ids == ['a'] and records[0].included_anchor_ids == []
    assert records[0].status == 'ready'
    assert 'Augustine' not in inputs[0][0] and 'He' in inputs[0][0]


def test_overlong_focal_sentence_abstains_without_truncation():
    """A focal sentence that doesn't fit the budget never reaches the model."""
    (inputs, _, _, records), _ = contexts(max_context_tokens=3)
    assert inputs == []
    assert records[0].status == 'budget_exceeded'
    assert records[0].char_start == records[0].focal_start


def test_encoder_budget_counts_relation_prompt_and_special_tokens():
    """Count the real word-list input, all prompt labels included."""
    calls = []

    def tokenizer(batch, **kwargs):
        """Fake special tokens, and keep the prepared encoder input visible to the test."""
        calls.append((batch, kwargs))
        return {'input_ids': [list(range(len(batch[0])+2))]}

    model = SimpleNamespace(base_config=SimpleNamespace(max_len=256), rel_token='[REL]', sep_token='[SEP]',
        token_rep_layer=SimpleNamespace(bert_layer=SimpleNamespace(tokenizer=tokenizer,
            model=SimpleNamespace(config=SimpleNamespace(max_position_embeddings=512)))))
    budget = GLiRELBudget(model, relation_config())
    assert budget(['He', 'wrote'], []) == 9
    assert calls[0][0][0] == ['[REL]','student of','[REL]','author of','[SEP]','He','wrote']
    assert calls[0][1]['truncation'] is False
