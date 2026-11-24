"""Tests for building entities from the bottom up: inside an episode, then across episodes."""

from test_identity_refinement import config as refinement
from spoken_record.consolidation import consolidate_episode, corpus_identities, lift_relations, unit_map
from spoken_record.extraction_models import ExtractedMention
from spoken_record.passage_models import Passage
from spoken_record.relation_models import RelationCandidate
from spoken_record.resolution_models import MentionAssignment, ResolutionConfig

RESOLUTION = ResolutionConfig(normalization="unicode-nfc-casefold-whitespace-v1", aliases="aliases.jsonl",
                              cross_episode_exact_types=["Place", "Language"])
TEXTS = {"ep-1": "Thomas Aquinas taught in Paris. Aquinas wrote about the soul and the intellect.",
         "ep-2": "Thomas Aquinas wrote about the soul and the intellect in Paris."}


def records():
    passages, mentions = [], []
    for episode_id, text in TEXTS.items():
        passages.append(Passage(created_by_run_id="test", passage_id=f"p-{episode_id}", episode_id=episode_id,
            transcript_version_id=f"t-{episode_id}", index=0, char_start=0, char_end=len(text), text=text,
            word_count=len(text.split()), timing_precision="unavailable", source_segment_ids=[],
            method="token_fallback", segmentation_fingerprint="segments-v1"))
        for name in ("Thomas Aquinas", "Aquinas"):
            start = text.find(name)
            while start >= 0:
                # "Aquinas" inside "Thomas Aquinas" is not a mention of its own.
                if not (name == "Aquinas" and text[max(0, start - 7):start] == "Thomas "):
                    mentions.append(ExtractedMention(created_by_run_id="test",
                        mention_id=f"m-{episode_id}-{start}", episode_id=episode_id,
                        transcript_version_id=f"t-{episode_id}", char_start=start, char_end=start + len(name),
                        text=name, type="Person", score=.9, source_window_ids=["w"],
                        passage_ids=[f"p-{episode_id}"], extraction_fingerprint="extract-v1"))
                start = text.find(name, start + 1)
    return passages, sorted(mentions, key=lambda m: (m.transcript_version_id, m.char_start))


def units_of(passages, mentions, settings):
    rows = []
    for episode_id in TEXTS:
        local = [m for m in mentions if m.episode_id == episode_id]
        rows.extend(consolidate_episode("test", "hop", episode_id, local,
            [p for p in passages if p.episode_id == episode_id], [], RESOLUTION, "fp", settings)[2])
    return rows


def test_names_of_one_person_become_one_entity_inside_the_episode():
    passages, mentions = records()
    local = [m for m in mentions if m.episode_id == "ep-1"]
    _, entities, assignments = consolidate_episode("test", "hop", "ep-1", local, passages[:1], [],
                                                   RESOLUTION, "fp", refinement())
    assert [e.canonical_label for e in entities] == ["Thomas Aquinas"]
    assert entities[0].corpus_entity_id.startswith("eu-")
    assert {a.corpus_entity_id for a in assignments} == {entities[0].corpus_entity_id}


def test_a_merge_made_inside_an_episode_is_kept_across_episodes():
    passages, mentions = records()
    # Name-variant merging is switched off for the corpus, so only the episode merge can join
    # "Aquinas" with "Thomas Aquinas".
    corpus_rules = refinement().model_copy(update={"merge_methods": ["exact_key"]})
    unit_assignments = units_of(passages, mentions, refinement())
    entities, assignments, *_ = corpus_identities("test", "hop", mentions, passages, unit_assignments, [],
                                                  RESOLUTION, "fp", corpus_rules)
    [aquinas] = [e for e in entities if "aquinas" in e.normalized_keys]
    assert aquinas.normalized_keys == ["aquinas", "thomas aquinas"] and aquinas.episode_ids == ["ep-1", "ep-2"]
    assert {a.corpus_entity_id for a in assignments} == {aquinas.corpus_entity_id}
    assert set(unit_map(unit_assignments, assignments).values()) == {aquinas.corpus_entity_id}


def test_lifting_turns_a_relation_between_merged_entities_into_a_self_relation():
    def candidate(head, tail):
        return RelationCandidate(created_by_run_id="test", candidate_id=f"rel-{head}-{tail}", episode_id="ep-1",
            passage_id="p-ep-1", context_char_start=0, context_char_end=20, head_mention_id="m1",
            tail_mention_id="m2", head_corpus_entity_id=head, tail_corpus_entity_id=tail,
            head_text="Aquinas", tail_text="Paris", head_char_start=0, head_char_end=7, tail_char_start=8,
            tail_char_end=13, prompt_label="lived in", canonical_predicate="lived_in", score=.9,
            decision="accepted", reason="test")

    lifted, assertions = lift_relations("test", [candidate("eu-a", "eu-b"), candidate("eu-a", "eu-c")],
                                        {"eu-a": "ce-1", "eu-b": "ce-1", "eu-c": "ce-2"})
    assert [row.decision for row in lifted] == ["self_relation", "accepted"]
    assert [(a.subject_corpus_entity_id, a.object_corpus_entity_id) for a in assertions] == [("ce-1", "ce-2")]


def test_lifting_keeps_both_directions_of_a_proposal_for_the_nli_check():
    def candidate(head, tail):
        return RelationCandidate(created_by_run_id="test", candidate_id=f"rel-{head}-{tail}", episode_id="ep-1",
            passage_id="p-ep-1", context_char_start=0, context_char_end=30, head_mention_id=f"m-{head}",
            tail_mention_id=f"m-{tail}", head_corpus_entity_id=head, tail_corpus_entity_id=tail,
            head_text="Albert", tail_text="Proclus", head_char_start=0, head_char_end=6, tail_char_start=20,
            tail_char_end=27, prompt_label="influenced by", canonical_predicate="influenced_by", score=.4,
            decision="accepted", reason="test")

    units = {"eu-a": "ce-1", "eu-b": "ce-2"}
    lifted, _ = lift_relations("test", [candidate("eu-a", "eu-b"), candidate("eu-b", "eu-a")], units)
    assert [row.decision for row in lifted] == ["accepted", "accepted"]


def within_episode():
    from spoken_record.identity_models import IdentityRefinementConfig
    return IdentityRefinementConfig.model_validate_json(
        open("config/identity_refinement.within-episode.json", "rb").read())


def episode(text, names):
    passages, mentions = [], []
    for index, sentence in enumerate(text):
        passages.append(Passage(created_by_run_id="test", passage_id=f"p{index}", episode_id="ep", transcript_version_id="t",
            index=index, char_start=sum(len(s) for s in text[:index]), char_end=sum(len(s) for s in text[:index + 1]),
            text=sentence, word_count=len(sentence.split()), timing_precision="unavailable", source_segment_ids=[],
            method="token_fallback", segmentation_fingerprint="segments-v1"))
    full = "".join(text)
    for name in names:
        start = full.find(name)
        while start >= 0:
            inside = any(other != name and name in other and full[start - other.index(name):].startswith(other)
                         for other in names)
            if not inside:
                passage = next(p for p in passages if p.char_start <= start < p.char_end)
                mentions.append(ExtractedMention(created_by_run_id="test", mention_id=f"m{start}", episode_id="ep",
                    transcript_version_id="t", char_start=start, char_end=start + len(name), text=name,
                    type="Person", score=.9, source_window_ids=["w"], passage_ids=[passage.passage_id],
                    extraction_fingerprint="extract-v1"))
            start = full.find(name, start + 1)
    return passages, sorted(mentions, key=lambda m: m.char_start)


def test_a_surname_joins_the_full_name_inside_an_episode_even_with_little_shared_context():
    text = ["Thomas Aquinas was born near Naples. "] + [f"Aquinas argued about topic {n} at length. " for n in range(8)]
    passages, mentions = episode(text, ["Thomas Aquinas", "Aquinas"])
    _, strict, _ = consolidate_episode("test", "hop", "ep", mentions, passages, [], RESOLUTION, "fp", refinement())
    _, entities, _ = consolidate_episode("test", "hop", "ep", mentions, passages, [], RESOLUTION, "fp", within_episode())
    assert len(strict) == 2 and [e.canonical_label for e in entities] == ["Thomas Aquinas"]


def test_a_surname_that_fits_two_people_joins_neither():
    text = ["Roger Bacon studied optics. ", "Francis Bacon wrote essays. ", "Bacon was clever. "]
    passages, mentions = episode(text, ["Roger Bacon", "Francis Bacon", "Bacon"])
    _, entities, _ = consolidate_episode("test", "hop", "ep", mentions, passages, [], RESOLUTION, "fp", within_episode())
    assert sorted(e.canonical_label for e in entities) == ["Bacon", "Francis Bacon", "Roger Bacon"]


def linked(entity_id, label, episode_id, qid, kind="Person", status="linked"):
    from test_identity_refinement import entity
    return entity(entity_id, label, episode_id).model_copy(update={
        "type": kind, "wikidata_qid": qid if status == "linked" else None, "link_status": status,
        "link_confidence": .9 if status == "linked" else None})


def link_decision(entity_id):
    from types import SimpleNamespace
    return SimpleNamespace(corpus_entity_id=entity_id, decision_id=f"ld-{entity_id}")


def test_entities_with_the_same_wikidata_item_become_one():
    from spoken_record.consolidation import merge_by_wikidata
    entities = [linked("ce-1", "Plato", "ep-1", "Q859"), linked("ce-2", "Plato", "ep-2", "Q859"),
                linked("ce-3", "Summa", "ep-1", "Q653871", "Work"), linked("ce-4", "Summa", "ep-1", "Q653871", "School"),
                linked("ce-5", "Albert", "ep-1", None, status="ambiguous"),
                linked("ce-6", "Albert", "ep-2", None, status="ambiguous")]
    assignments = [MentionAssignment(created_by_run_id="t", mention_id=f"m{i}", episode_entity_id=f"ee-{i}",
                                     corpus_entity_id=e.corpus_entity_id) for i, e in enumerate(entities)]
    merged, moved, _, redirects = merge_by_wikidata("t", entities, assignments, [], [], (),
        [link_decision(e.corpus_entity_id) for e in entities], ["Person", "Work", "School"])
    plato = [e for e in merged if e.canonical_label == "Plato"]
    assert len(plato) == 1 and plato[0].episode_ids == ["ep-1", "ep-2"]
    assert plato[0].resolution_method == "same_wikidata_item" and plato[0].wikidata_qid == "Q859"
    # The same item under another type, and abstained links, stay apart.
    assert sum(e.canonical_label == "Summa" for e in merged) == 2
    assert sum(e.canonical_label == "Albert" for e in merged) == 2
    assert {a.corpus_entity_id for a in moved[:2]} == {plato[0].corpus_entity_id}
    assert [(r.from_corpus_entity_id, r.to_corpus_entity_id, r.reason) for r in redirects] == [
        ("ce-2", "ce-1", "same_wikidata_item")]


def test_two_people_the_name_rules_kept_apart_are_not_joined_by_a_shared_link():
    from types import SimpleNamespace
    from spoken_record.consolidation import merge_by_wikidata
    entities = [linked("ce-1", "John Locke", "ep-1", "Q9353"), linked("ce-2", "Gary Locke", "ep-2", "Q9353")]
    separate = (SimpleNamespace(decision="separate", left_corpus_entity_id="ce-1", right_corpus_entity_id="ce-2"),)
    merged, _, _, redirects = merge_by_wikidata("t", entities, [], [], [], separate,
        [link_decision(e.corpus_entity_id) for e in entities], ["Person"])
    assert len(merged) == 2 and redirects == []


def test_a_first_name_joins_the_one_full_name_with_an_epithet_in_the_episode():
    text = ["Alexander of Hales taught in Paris. ", "Alexander was a Franciscan. ",
            "William of Ockham and William of Auvergne disagreed. ", "William was right. "]
    passages, mentions = episode(text, ["Alexander of Hales", "Alexander", "William of Ockham",
                                        "William of Auvergne", "William"])
    _, entities, _ = consolidate_episode("test", "hop", "ep", mentions, passages, [], RESOLUTION, "fp", refinement())
    names = sorted(sorted(e.observed_surfaces) for e in entities)
    assert ["Alexander", "Alexander of Hales"] in names
    # Two full names start with "William", so the plain "William" joins neither.
    assert ["William"] in names


def test_a_first_name_that_comes_before_the_full_name_stays_apart():
    text = ["Aquinas read Boethius on the Trinity. ", "Later Boethius of Dacia wrote on eternity. "]
    passages, mentions = episode(text, ["Boethius of Dacia", "Boethius"])
    _, entities, _ = consolidate_episode("test", "hop", "ep", mentions, passages, [], RESOLUTION, "fp", refinement())
    assert sorted(sorted(e.observed_surfaces) for e in entities) == [["Boethius"], ["Boethius of Dacia"]]


def test_an_entity_that_just_missed_its_link_joins_a_namesake_with_the_same_item():
    from types import SimpleNamespace
    from spoken_record.consolidation import merge_by_wikidata, supported_by_namesake
    entities = [linked("ce-1", "Bonaventure", "ep-1", "Q43746"),
                linked("ce-2", "Bonaventure", "ep-2", None, status="ambiguous"),
                linked("ce-3", "Alexander", "ep-3", "Q8409"),
                linked("ce-4", "Alexander", "ep-4", None, status="ambiguous"),
                linked("ce-5", "Bonaventure", "ep-5", None, status="ambiguous")]
    candidates = [SimpleNamespace(candidate_id="c-2", qid="Q43746", label="Bonaventure", aliases=["Saint Bonaventure"]),
                  SimpleNamespace(candidate_id="c-4", qid="Q8409", label="Alexander the Great",
                                  aliases=["Alexander III of Macedon", "Alexandros"]),
                  SimpleNamespace(candidate_id="c-5", qid="Q43746", label="Bonaventure", aliases=[])]
    decisions = [SimpleNamespace(corpus_entity_id="ce-2", decision="ambiguous", confidence=.72, candidate_ids=["c-2"],
                                 decision_id="ld-2"),
                 SimpleNamespace(corpus_entity_id="ce-4", decision="ambiguous", confidence=.73, candidate_ids=["c-4"],
                                 decision_id="ld-4"),
                 # Too weak on its own score, so it does not join.
                 SimpleNamespace(corpus_entity_id="ce-5", decision="ambiguous", confidence=.65, candidate_ids=["c-5"],
                                 decision_id="ld-5"),
                 SimpleNamespace(corpus_entity_id="ce-1", decision_id="ld-1"),
                 SimpleNamespace(corpus_entity_id="ce-3", decision_id="ld-3")]
    supported = supported_by_namesake(entities, decisions, candidates, 0.70, ["Person"])
    assert supported == {"ce-2": ("Person", "Q43746")}
    merged, _, _, redirects = merge_by_wikidata("t", entities, [], [], [], (), decisions, ["Person"], supported)
    assert [(r.from_corpus_entity_id, r.to_corpus_entity_id) for r in redirects] == [("ce-2", "ce-1")]
    bonaventure = next(e for e in merged if e.corpus_entity_id == "ce-1")
    assert bonaventure.episode_ids == ["ep-1", "ep-2"] and bonaventure.wikidata_qid == "Q43746"
    assert sum(e.canonical_label == "Alexander" for e in merged) == 2


def test_a_mention_typed_wrongly_joins_its_linked_namesake_of_another_type():
    from types import SimpleNamespace
    from spoken_record.consolidation import merge_by_wikidata, supported_by_namesake
    person = linked("ce-1", "Averroes", "ep-1", "Q39837").model_copy(update={"mention_count": 22})
    school = linked("ce-2", "Averroes", "ep-2", None, kind="School", status="ambiguous")
    idea = linked("ce-3", "republic", "ep-3", None, kind="Concept", status="ambiguous")
    work = linked("ce-4", "Republic", "ep-4", "Q123397", kind="Work").model_copy(update={"mention_count": 9})
    weak = linked("ce-5", "Averroes", "ep-5", None, kind="School", status="ambiguous")
    entities = [person, school, idea, work, weak]
    candidates = [SimpleNamespace(candidate_id="c-2", qid="Q39837", label="Averroes", aliases=["Ibn Rushd"], type_compatibility=.5),
                  SimpleNamespace(candidate_id="c-3", qid="Q123397", label="Republic", aliases=[], type_compatibility=.5),
                  SimpleNamespace(candidate_id="c-5", qid="Q39837", label="Averroes", aliases=[], type_compatibility=.5)]
    decisions = [SimpleNamespace(corpus_entity_id="ce-2", decision="ambiguous", confidence=.68, candidate_ids=["c-2"], decision_id="ld-2"),
                 SimpleNamespace(corpus_entity_id="ce-3", decision="ambiguous", confidence=.68, candidate_ids=["c-3"], decision_id="ld-3"),
                 # Under the threshold even without the type penalty.
                 SimpleNamespace(corpus_entity_id="ce-5", decision="ambiguous", confidence=.60, candidate_ids=["c-5"], decision_id="ld-5"),
                 SimpleNamespace(corpus_entity_id="ce-1", decision_id="ld-1"), SimpleNamespace(corpus_entity_id="ce-4", decision_id="ld-4")]
    types = ["Person", "Work", "School", "Concept"]
    supported = supported_by_namesake(entities, decisions, candidates, 0.70, types, type_weight=.15)
    # The lowercase idea was never heard as "Republic", so it stays apart from the book.
    assert supported == {"ce-2": ("Person", "Q39837")}
    merged, _, _, redirects = merge_by_wikidata("t", entities, [], [], [], (), decisions, types, supported)
    assert [(r.from_corpus_entity_id, r.to_corpus_entity_id) for r in redirects] == [("ce-2", "ce-1")]
    averroes = next(e for e in merged if e.corpus_entity_id == "ce-1")
    assert averroes.type == "Person" and averroes.episode_ids == ["ep-1", "ep-2"]


def test_a_first_name_joined_to_a_full_name_does_not_pull_in_namesakes_of_other_episodes():
    texts = {"ep-1": "Boethius of Dacia taught in Paris on eternity and the world. Boethius said philosophy is free.",
             "ep-2": "Boethius taught on eternity and the world and wrote that philosophy is free."}
    passages, mentions, unit_assignments = [], [], []
    for episode_id, text in texts.items():
        passages.append(Passage(created_by_run_id="test", passage_id=f"p-{episode_id}", episode_id=episode_id,
            transcript_version_id=f"t-{episode_id}", index=0, char_start=0, char_end=len(text), text=text,
            word_count=len(text.split()), timing_precision="unavailable", source_segment_ids=[],
            method="token_fallback", segmentation_fingerprint="segments-v1"))
        local = []
        for name in ("Boethius of Dacia", "Boethius"):
            start = text.find(name)
            while start >= 0:
                if not (name == "Boethius" and text.startswith("Boethius of Dacia", start)):
                    local.append(ExtractedMention(created_by_run_id="test", mention_id=f"m-{episode_id}-{start}",
                        episode_id=episode_id, transcript_version_id=f"t-{episode_id}", char_start=start,
                        char_end=start + len(name), text=name, type="Person", score=.9, source_window_ids=["w"],
                        passage_ids=[f"p-{episode_id}"], extraction_fingerprint="extract-v1"))
                start = text.find(name, start + 1)
        mentions.extend(local)
        unit_assignments.extend(consolidate_episode("test", "hop", episode_id, local, passages[-1:], [],
                                                    RESOLUTION, "fp", within_episode())[2])
    entities, assignments, *_ = corpus_identities("test", "hop", sorted(mentions, key=lambda m: m.mention_id),
                                                  passages, unit_assignments, [], RESOLUTION, "fp", within_episode())
    dacia = next(e for e in entities if "boethius of dacia" in e.normalized_keys)
    # The Boethius of episode 2 is someone else and stays out of Boethius of Dacia.
    assert dacia.episode_ids == ["ep-1"]


def test_merges_made_inside_episodes_are_kept_even_beyond_the_cluster_limit():
    from spoken_record.identity_refinement import consistent_components
    from types import SimpleNamespace
    entities = [SimpleNamespace(corpus_entity_id=f"ce-{i}") for i in range(5)]
    def merge(left, right, method):
        return SimpleNamespace(decision="merge", left_corpus_entity_id=left, right_corpus_entity_id=right,
                               retrieval_methods=[method], model_copy=None)
    fixed = tuple(merge("ce-0", f"ce-{i}", "same_episode_entity") for i in range(1, 5))
    components, decisions = consistent_components(entities, fixed, 2)
    assert len(components.members[components.find("ce-0")]) == 5
    assert all(row.decision == "merge" for row in decisions)
