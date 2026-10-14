from spoken_record.extraction_models import ExtractedMention
from spoken_record.identity_models import IdentityRefinementConfig
from spoken_record.identity_refinement import build_decisions, refine_records
from spoken_record.passage_models import Passage
from spoken_record.resolution_models import (CorpusEntity, EpisodeEntity, MentionAssignment,
                                             ResolutionDecision)


def config():
    return IdentityRefinementConfig(candidate_types=["Person"],
        merge_methods=["exact_key", "person_name_containment", "person_initial_alignment"],
        fuzzy_similarity_floor=.88, person_surname_similarity_floor=.9,
        merge_threshold=.55, minimum_context_similarity=.05,
        surface_weight=.45, lexical_context_weight=.4, neighbor_weight=.15,
        maximum_component_entities=8)


def entity(entity_id, label, episode_id):
    return CorpusEntity(created_by_run_id="base", corpus_entity_id=entity_id, corpus_id="hop",
        type="Person", canonical_label=label, normalized_keys=[label.casefold()],
        observed_surfaces=[label], reviewed_aliases=[], episode_entity_ids=[f"ee-{entity_id}"],
        episode_ids=[episode_id], mention_count=1, episode_count=1,
        resolution_method="episode_local", resolution_status="provisional",
        registry_key=f"episode:{entity_id}")


def records(entities, contexts):
    passages, mentions, assignments, episode_entities, decisions = [], [], [], [], []
    for index, (entity_row, context) in enumerate(zip(entities, contexts)):
        text = f"{entity_row.canonical_label} {context}"
        passage = Passage(created_by_run_id="passages", passage_id=f"p{index}",
            episode_id=entity_row.episode_ids[0], transcript_version_id=f"t{index}", index=0,
            char_start=0, char_end=len(text), text=text, word_count=len(text.split()),
            timing_precision="unavailable", source_segment_ids=[], method="token_fallback",
            segmentation_fingerprint="segments-v1")
        mention = ExtractedMention(created_by_run_id="extract", mention_id=f"m{index}",
            episode_id=passage.episode_id, transcript_version_id=passage.transcript_version_id,
            char_start=0, char_end=len(entity_row.canonical_label), text=entity_row.canonical_label,
            type="Person", score=.9, source_window_ids=[f"w{index}"],
            passage_ids=[passage.passage_id], extraction_fingerprint="extract-v1")
        episode_entity = EpisodeEntity(created_by_run_id="base",
            episode_entity_id=entity_row.episode_entity_ids[0], episode_id=passage.episode_id,
            transcript_version_id=passage.transcript_version_id, type="Person",
            canonical_surface=entity_row.canonical_label, normalized_keys=entity_row.normalized_keys,
            mention_ids=[mention.mention_id], passage_ids=[passage.passage_id], mention_count=1,
            is_singleton=True, mean_score=.9, max_score=.9, resolution_method="exact_normalized",
            resolution_status="provisional")
        assignment = MentionAssignment(created_by_run_id="base", mention_id=mention.mention_id,
            episode_entity_id=episode_entity.episode_entity_id,
            corpus_entity_id=entity_row.corpus_entity_id)
        decision = ResolutionDecision(created_by_run_id="base", decision_id=f"d{index}",
            episode_entity_id=episode_entity.episode_entity_id,
            corpus_entity_id=entity_row.corpus_entity_id, method="episode_local",
            status="provisional", reason="synthetic base")
        passages.append(passage)
        mentions.append(mention)
        assignments.append(assignment)
        episode_entities.append(episode_entity)
        decisions.append(decision)
    return passages, mentions, assignments, episode_entities, decisions


def test_name_variants_form_one_provisional_component_with_readable_canonical_label():
    entities = [entity("a", "Aquinas", "e1"), entity("t", "Thomas Aquinas", "e2"),
                entity("i", "T. Aquinas", "e3")]
    context = "discusses essence being soul and medieval philosophy"
    passages, mentions, assignments, episode_entities, base_decisions = records(entities, [context] * 3)
    candidates = build_decisions("refine", entities, mentions, assignments, passages, config())
    assert any(row.decision == "merge" and "person_name_containment" in row.retrieval_methods
               for row in candidates)
    assert any(row.decision == "merge" and "person_initial_alignment" in row.retrieval_methods
               for row in candidates)
    refined, refined_assignments, decisions, queue, redirects, final_candidates = refine_records(
        "refine", entities, episode_entities, assignments, base_decisions, [], candidates, config())
    assert len(refined) == 1
    assert refined[0].canonical_label == "Thomas Aquinas"
    assert refined[0].resolution_method == "automatic_context"
    assert len({row.corpus_entity_id for row in refined_assignments}) == 1
    assert len(redirects) == 2 and not queue
    assert all(row.method == "automatic_context" for row in decisions)
    assert sum(row.decision == "merge" for row in final_candidates) >= 2


def test_qualified_names_are_kept_separate_despite_similar_context():
    entities = [entity("elea", "Zeno of Elea", "e1"), entity("citium", "Zeno of Citium", "e2")]
    context = "ancient Greek philosopher discusses nature logic and argument"
    passages, mentions, assignments, _, _ = records(entities, [context, context])
    candidates = build_decisions("refine", entities, mentions, assignments, passages, config())
    assert len(candidates) == 1
    assert candidates[0].decision == "separate"
    assert candidates[0].contradictions == ["conflicting_person_qualifier"]


def test_asr_surname_spelling_variant_is_retrieved_without_automatic_merge():
    """A one-letter surname difference goes to review but is not merged automatically."""
    entities = [entity("short", "Douglas", "e1"),
                entity("full", "Frederick Douglass", "e1")]
    context = "abolition slavery nationalism autobiography public political thought"
    passages, mentions, assignments, _, _ = records(entities, [context, context])
    candidates = build_decisions("refine", entities, mentions, assignments, passages, config())
    assert len(candidates) == 1
    assert candidates[0].retrieval_methods == ["person_surname_variant"]
    assert candidates[0].decision == "abstain"
