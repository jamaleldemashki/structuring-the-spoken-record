"""Conservative entity resolution (mention to episode entity to corpus entity), plus
pairwise scoring against annotations.
"""

import importlib.metadata
import itertools
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
from spoken_record.adapters.base import sha256, stable_id
from spoken_record.evaluation import Annotation, TYPES, load_sample
from spoken_record.extraction import validate_run as validate_extraction
from spoken_record.extraction_models import ExtractedMention
from spoken_record.ingestion import dump_json, dump_jsonl, read_jsonl, workspace_path
from spoken_record.models import Episode
from spoken_record.passage_models import Passage
from spoken_record.resolution_models import (AliasRule, CorpusEntity, EpisodeEntity, MentionAssignment,
                                             ResolutionConfig, ResolutionDecision, ReviewCandidate)


def normalize_surface(text: str) -> str:
    """Identity key: NFC, casefold, single spaces. Kept narrow on purpose."""
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def rule_keys(rule: AliasRule) -> set[str]:
    return {normalize_surface(value) for value in [rule.canonical_label, *rule.aliases]}


def rule_applies(rule: AliasRule, mention: ExtractedMention) -> bool:
    if normalize_surface(mention.text) not in rule_keys(rule):
        return False
    if rule.scope == "corpus":
        return True
    if rule.scope == "episode":
        return mention.episode_id in rule.episode_ids
    return mention.mention_id in rule.mention_ids


def select_surface(mentions: list[ExtractedMention]) -> str:
    counts = Counter(m.text for m in mentions)
    max_scores = {surface: max(m.score for m in mentions if m.text == surface) for surface in counts}
    return sorted(counts, key=lambda s: (-counts[s], -max_scores[s], -len(s), normalize_surface(s), s))[0]


def validate_aliases(rules: list[AliasRule], mentions: list[ExtractedMention], passages: list[Passage],
                     episodes: list[Episode]) -> None:
    if len({r.alias_id for r in rules}) != len(rules):
        raise ValueError("duplicate alias rule ID")
    mention_ids, passage_ids, episode_ids = ({m.mention_id for m in mentions}, {p.passage_id for p in passages},
                                              {e.episode_id for e in episodes})
    for rule in rules:
        if rule.corpus_id not in {e.corpus_id for e in episodes}:
            raise ValueError("alias rule uses a different corpus namespace")
        if rule.scope == "mentions" and not set(rule.mention_ids) <= mention_ids:
            raise ValueError("alias rule references an unknown mention")
        if rule.scope == "episode" and not set(rule.episode_ids) <= episode_ids:
            raise ValueError("alias rule references an unknown episode")
        if not set(rule.evidence_passage_ids) <= passage_ids:
            raise ValueError("alias rule references unknown evidence")
        normalized_aliases = [normalize_surface(value) for value in rule.aliases]
        if any(not value for value in normalized_aliases):
            raise ValueError("alias normalizes to an empty key")
        if len(set(normalized_aliases)) != len(normalized_aliases):
            raise ValueError("alias rule has duplicate normalized aliases")
    # A mention can't be covered by two reviewed rules that disagree (merge and block).
    for mention in mentions:
        applicable = [r for r in rules if r.type == mention.type and rule_applies(r, mention)]
        merge_ids = {r.identity_id for r in applicable if r.decision == "merge"}
        if len(merge_ids) > 1 or (merge_ids and any(r.decision == "block" for r in applicable)):
            raise ValueError("conflicting reviewed alias decisions")


def resolve_records(run_id: str, corpus_id: str, mentions: list[ExtractedMention], passages: list[Passage],
                    rules: list[AliasRule], config: ResolutionConfig, fingerprint: str):
    passage_by_id = {p.passage_id: p for p in passages}
    by_episode_group = defaultdict(list)
    mention_merge_rule, mention_block_rules = {}, defaultdict(list)
    for mention in mentions:
        applicable = [r for r in rules if r.type == mention.type and rule_applies(r, mention)]
        merges = [r for r in applicable if r.decision == "merge"]
        blocks = [r for r in applicable if r.decision == "block"]
        if merges:
            mention_merge_rule[mention.mention_id] = merges[0]
            group = (mention.episode_id, mention.type, "alias", merges[0].identity_id)
        elif any(r.scope == "mentions" for r in blocks):
            group = (mention.episode_id, mention.type, "blocked", mention.mention_id)
        else:
            group = (mention.episode_id, mention.type, "exact", normalize_surface(mention.text))
        mention_block_rules[mention.mention_id] = blocks
        by_episode_group[group].append(mention)

    episode_entities, internal = [], {}
    for group, members in sorted(by_episode_group.items()):
        episode_id, entity_type, method, key = group
        aliases = sorted({mention_merge_rule[m.mention_id].alias_id for m in members
                          if m.mention_id in mention_merge_rule})
        normalized = sorted({normalize_surface(m.text) for m in members})
        canonical = mention_merge_rule[members[0].mention_id].canonical_label if method == "alias" else select_surface(members)
        entity_id = stable_id("ee", episode_id, entity_type, method, key)
        passage_ids = sorted({pid for m in members for pid in m.passage_ids},
                             key=lambda pid: (passage_by_id[pid].index, pid))
        status = "reviewed" if method == "alias" else "unresolved" if method == "blocked" else "provisional"
        episode_entities.append(EpisodeEntity(
            created_by_run_id=run_id, episode_entity_id=entity_id, episode_id=episode_id,
            transcript_version_id=members[0].transcript_version_id, type=entity_type,
            canonical_surface=canonical, normalized_keys=normalized,
            mention_ids=sorted(m.mention_id for m in members), passage_ids=passage_ids,
            mention_count=len(members), is_singleton=len(members) == 1,
            mean_score=sum(m.score for m in members) / len(members), max_score=max(m.score for m in members),
            resolution_method={"alias": "reviewed_alias", "blocked": "mention_block"}.get(method, "exact_normalized"),
            resolution_status=status, alias_rule_ids=aliases))
        internal[entity_id] = {"members": members, "method": method, "key": key,
                               "merge_rule": mention_merge_rule.get(members[0].mention_id),
                               "blocks": [r for m in members for r in mention_block_rules[m.mention_id]]}

    by_corpus_group = defaultdict(list)
    for entity in episode_entities:
        info = internal[entity.episode_entity_id]
        if info["method"] == "alias":
            corpus_key = (entity.type, "reviewed", info["key"])
        elif info["blocks"]:
            corpus_key = (entity.type, "blocked", entity.episode_entity_id)
        elif entity.type in config.cross_episode_exact_types:
            corpus_key = (entity.type, "exact", entity.normalized_keys[0])
        elif entity.type == "Person" and ((config.cross_episode_multitoken_person and
                len(entity.normalized_keys) == 1 and len(entity.normalized_keys[0].split()) > 1) or
                not config.single_token_person_requires_review):
            corpus_key = (entity.type, "exact", entity.normalized_keys[0])
        elif entity.type != "Person" and not config.all_other_types_require_review:
            corpus_key = (entity.type, "exact", entity.normalized_keys[0])
        else:
            corpus_key = (entity.type, "local", entity.episode_entity_id)
        by_corpus_group[corpus_key].append(entity)

    corpus_entities, decisions, episode_to_corpus = [], [], {}
    # Note keys whose surface shows up in several episodes. That separates a normal
    # singleton from a case where we chose not to merge across episodes.
    shared_keys = Counter((e.type, key) for e in episode_entities for key in e.normalized_keys)
    for group, members in sorted(by_corpus_group.items()):
        entity_type, policy, key = group
        episode_ids = sorted({e.episode_id for e in members})
        mention_ids = [mid for e in members for mid in e.mention_ids]
        mention_id_set = set(mention_ids)
        member_mentions = [m for m in mentions if m.mention_id in mention_id_set]
        alias_rule = internal[members[0].episode_entity_id]["merge_rule"] if policy == "reviewed" else None
        if policy == "reviewed":
            method, status, registry = "reviewed_alias", "reviewed", f"reviewed:{alias_rule.identity_id}"
            canonical, reviewed_aliases = alias_rule.canonical_label, sorted(set(alias_rule.aliases))
        elif policy == "exact" and len(episode_ids) > 1:
            method, status, registry = "provisional_exact", "provisional", f"exact:{entity_type}:{key}"
            canonical, reviewed_aliases = select_surface(member_mentions), []
        elif policy == "blocked":
            method, status, registry = "blocked", "unresolved", f"blocked:{members[0].episode_entity_id}"
            canonical, reviewed_aliases = members[0].canonical_surface, []
        else:
            method = "episode_local"
            unresolved = any(shared_keys[(entity_type, norm)] > 1 for norm in members[0].normalized_keys)
            status, registry = ("unresolved" if unresolved else "provisional"), f"episode:{members[0].episode_entity_id}"
            canonical, reviewed_aliases = members[0].canonical_surface, []
        corpus_id_value = stable_id("ce", corpus_id, entity_type, registry)
        corpus_entities.append(CorpusEntity(
            created_by_run_id=run_id, corpus_entity_id=corpus_id_value, corpus_id=corpus_id,
            type=entity_type, canonical_label=canonical,
            normalized_keys=sorted({k for e in members for k in e.normalized_keys}),
            observed_surfaces=sorted({m.text for m in member_mentions}, key=lambda s: (normalize_surface(s), s)),
            reviewed_aliases=reviewed_aliases,
            episode_entity_ids=sorted(e.episode_entity_id for e in members), episode_ids=episode_ids,
            mention_count=len(mention_ids), episode_count=len(episode_ids),
            resolution_method=method, resolution_status=status, registry_key=registry))
        for entity in members:
            episode_to_corpus[entity.episode_entity_id] = corpus_id_value
            rule_ids = entity.alias_rule_ids or sorted({r.alias_id for r in internal[entity.episode_entity_id]["blocks"]})
            reason = {"reviewed_alias": "reviewed alias identity",
                      "provisional_exact": "type-allowed exact normalized key across episodes",
                      "blocked": "reviewed rule blocks automatic grouping",
                      "episode_local": "cross-episode merge requires review"}[method]
            decisions.append(ResolutionDecision(
                created_by_run_id=run_id,
                decision_id=stable_id("rd", entity.episode_entity_id, corpus_id_value, fingerprint),
                episode_entity_id=entity.episode_entity_id, corpus_entity_id=corpus_id_value,
                method=method, status=status, alias_rule_ids=rule_ids, reason=reason))

    mention_to_episode = {mid: e.episode_entity_id for e in episode_entities for mid in e.mention_ids}
    assignments = [MentionAssignment(created_by_run_id=run_id, mention_id=m.mention_id,
        episode_entity_id=mention_to_episode[m.mention_id],
        corpus_entity_id=episode_to_corpus[mention_to_episode[m.mention_id]]) for m in mentions]

    candidate_groups = defaultdict(list)
    for entity in episode_entities:
        for key in entity.normalized_keys:
            candidate_groups[(entity.type, key)].append(entity)
    queue = []
    for (entity_type, key), members in sorted(candidate_groups.items()):
        episode_ids = sorted({e.episode_id for e in members})
        corpus_ids = {episode_to_corpus[e.episode_entity_id] for e in members}
        if len(episode_ids) < 2 or len(corpus_ids) == 1:
            continue
        blocks = [r for e in members for r in internal[e.episode_entity_id]["blocks"]]
        reason = "blocked_by_rule" if blocks else "single_token_person" if entity_type == "Person" else "type_requires_review"
        queue.append(ReviewCandidate(
            created_by_run_id=run_id, candidate_id=stable_id("rc", corpus_id, entity_type, key),
            type=entity_type, normalized_key=key,
            observed_surfaces=sorted({m.text for e in members for m in internal[e.episode_entity_id]["members"]}),
            episode_entity_ids=sorted(e.episode_entity_id for e in members), episode_ids=episode_ids,
            mention_count=sum(e.mention_count for e in members), reason=reason))
    return (tuple(sorted(episode_entities, key=lambda r: r.episode_entity_id)),
            tuple(sorted(corpus_entities, key=lambda r: r.corpus_entity_id)),
            tuple(sorted(assignments, key=lambda r: r.mention_id)),
            tuple(sorted(decisions, key=lambda r: r.decision_id)),
            tuple(sorted(queue, key=lambda r: r.candidate_id)))


def pairwise(items: list[dict], cluster_field: str, cross_episode: bool = False) -> dict:
    tp = fp = fn = predicted_pairs = gold_pairs = pair_count = 0
    for left, right in itertools.combinations(items, 2):
        if cross_episode and left["episode_id"] == right["episode_id"]:
            continue
        pair_count += 1
        gold_same = left["gold_identity"] == right["gold_identity"]
        predicted_same = left[cluster_field] == right[cluster_field]
        gold_pairs += gold_same
        predicted_pairs += predicted_same
        tp += gold_same and predicted_same
        fp += not gold_same and predicted_same
        fn += gold_same and not predicted_same
    return {"pairs_considered": pair_count, "gold_same_pairs": gold_pairs, "predicted_same_pairs": predicted_pairs,
            "tp": tp, "fp": fp, "fn": fn, "false_merges": fp, "false_splits": fn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None}


def baseline_keys(item: dict, rules: list[AliasRule], config: ResolutionConfig) -> None:
    norm, entity_type = normalize_surface(item["text"]), item["type"]
    applicable = [r for r in rules if r.type == entity_type and norm in rule_keys(r) and
                  (r.scope == "corpus" or r.scope == "episode" and item["episode_id"] in r.episode_ids or
                   r.scope == "mentions" and item.get("mention_id") in r.mention_ids)]
    merges = [r for r in applicable if r.decision == "merge"]
    blocks = [r for r in applicable if r.decision == "block"]
    item["singleton"] = item.get("mention_id") or item["item_id"]
    item["exact"] = f"{entity_type}:{norm}"
    item["exact_alias"] = (f"alias:{entity_type}:{merges[0].identity_id}" if merges else
                           f"blocked:{item['singleton']}" if blocks else item["exact"])
    if merges:
        item["conservative"] = item["exact_alias"]
    elif blocks:
        item["conservative"] = f"blocked:{item['singleton']}"
    elif entity_type in config.cross_episode_exact_types or \
            entity_type == "Person" and ((config.cross_episode_multitoken_person and len(norm.split()) > 1) or
                                         not config.single_token_person_requires_review) or \
            entity_type != "Person" and not config.all_other_types_require_review:
        item["conservative"] = item["exact"]
    else:
        item["conservative"] = f"episode:{item['episode_id']}:{item['exact']}"


def evaluate_resolution(packet: Path, mentions: list[ExtractedMention], assignments: list[MentionAssignment],
                        rules: list[AliasRule], config: ResolutionConfig) -> dict:
    _, samples = load_sample(packet)
    gold = [Annotation.model_validate(r) for r in read_jsonl(packet / "annotations.jsonl")]
    mention_map = {(m.episode_id, m.char_start, m.char_end, m.type): m for m in mentions}
    assignment_map = {a.mention_id: a.corpus_entity_id for a in assignments}
    gold_keys, items, gold_items = set(), [], []
    unresolved_gold = aligned = 0
    for annotation in gold:
        sample = samples[annotation.passage_id]
        for index, mention in enumerate(annotation.mentions):
            key = (sample["episode_id"], sample["char_start"] + mention.start,
                   sample["char_start"] + mention.end, mention.type)
            gold_keys.add(key)
            gold_item = {"item_id": f"{annotation.passage_id}:{index}", "episode_id": sample["episode_id"],
                         "text": mention.text, "type": mention.type, "gold_identity": mention.identity_id}
            if mention.identity_id is not None:
                baseline_keys(gold_item, rules, config)
                gold_items.append(gold_item)
            else:
                unresolved_gold += 1
            predicted = mention_map.get(key)
            if predicted is not None:
                aligned += 1
                if mention.identity_id is not None:
                    item = {**gold_item, "mention_id": predicted.mention_id,
                            "resolved": assignment_map[predicted.mention_id]}
                    baseline_keys(item, rules, config)
                    items.append(item)
    predicted_sample = [m for m in mentions if any(m.episode_id == s["episode_id"] and
        m.char_start >= s["char_start"] and m.char_end <= s["char_end"] for s in samples.values())]
    spurious = sum((m.episode_id, m.char_start, m.char_end, m.type) not in gold_keys for m in predicted_sample)

    def tracks(rows, resolved=False):
        keys = [("all_singletons", "singleton"), ("exact_normalized", "exact"),
                ("exact_plus_reviewed_aliases", "exact_alias"), ("conservative_pipeline", "resolved" if resolved else "conservative")]
        return {name: {"all": pairwise(rows, field), "cross_episode": pairwise(rows, field, True)}
                for name, field in keys}

    return {"schema_version": 1, "metric": "pairwise-same-identity-v1",
            "alignment": {"gold_mentions": sum(len(a.mentions) for a in gold), "exactly_aligned": aligned,
                          "missed_or_mistyped": sum(len(a.mentions) for a in gold) - aligned,
                          "spurious_or_wrong_type": spurious, "gold_identity_known": len(gold_items),
                          "aligned_identity_known": len(items), "gold_identity_unresolved": unresolved_gold},
            "predicted_mention_track": tracks(items, True),
            "gold_mention_diagnostic_track": tracks(gold_items),
            "notes": ["Pairwise scores use only exact span/type aligned predicted mentions with known gold identity.",
                      "The gold-mention track is diagnostic and is not the end-to-end result.",
                      "Singleton mentions remain in every track; unresolved gold identities are reported and excluded."]}


def resolution_fingerprint(config: ResolutionConfig, rules: list[AliasRule]) -> str:
    payload = {"algorithm": "conservative-resolution-1", "config": config.model_dump(mode="json"),
               "aliases": [r.model_dump(mode="json") for r in rules]}
    return sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode())


def resolve_run(root: Path, input_name: str, config_name: str, packet_name: str | None = None,
                run_id: str | None = None) -> Path:
    """Build episode and corpus identities from the accepted mentions."""
    root, started, now = root.resolve(), perf_counter(), datetime.now(timezone.utc)
    run_id = run_id or f"resolve-{now:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]{0,95}", run_id):
        raise ValueError("invalid run ID")
    run_root, parent = root / "data/prototype1/runs", workspace_path(root, input_name)
    if parent.parent != run_root or parent.name == run_id:
        raise ValueError("input must be a sibling extraction snapshot")
    output = run_root / run_id
    if output.exists():
        raise FileExistsError(output)
    parent_manifest = validate_extraction(parent)
    parent_digest = sha256((parent / "manifest.json").read_bytes())
    config_path = workspace_path(root, config_name)
    config_bytes = config_path.read_bytes()
    config = ResolutionConfig.model_validate_json(config_bytes)
    alias_path = workspace_path(root, config.aliases)
    alias_bytes = alias_path.read_bytes()
    rules = [AliasRule.model_validate(r) for r in read_jsonl(alias_path)]
    mentions = [ExtractedMention.model_validate(r) for r in read_jsonl(parent / "mentions.jsonl")]
    passage_run = parent.parent / parent_manifest["parent_run_id"]
    passages = [Passage.model_validate(r) for r in read_jsonl(passage_run / "passages.jsonl")]
    passage_manifest = json.loads((passage_run / "manifest.json").read_bytes())
    source_run = passage_run.parent / passage_manifest["parent_run_id"]
    episodes = [Episode.model_validate(r) for r in read_jsonl(source_run / "episodes.jsonl")]
    corpus_ids = {e.corpus_id for e in episodes}
    if len(corpus_ids) != 1:
        raise ValueError("one resolution run must use one corpus namespace")
    validate_aliases(rules, mentions, passages, episodes)
    fingerprint = resolution_fingerprint(config, rules)
    episode_entities, corpus_entities, assignments, decisions, queue = resolve_records(
        run_id, next(iter(corpus_ids)), mentions, passages, rules, config, fingerprint)
    # The annotation packet is optional, resolution runs without it.
    evaluation = {"status": "not_evaluated", "reason": "no evaluation packet supplied"}
    support = {str(config_path.relative_to(root)): sha256(config_bytes),
               str(alias_path.relative_to(root)): sha256(alias_bytes)}
    if packet_name:
        packet = workspace_path(root, packet_name)
        evaluation = evaluate_resolution(packet, mentions, assignments, rules, config)
        support.update({str((packet / name).relative_to(root)): sha256((packet / name).read_bytes())
                        for name in ("manifest.json", "sample.jsonl", "annotations.jsonl")})
    validate_extraction(parent)
    if sha256((parent / "manifest.json").read_bytes()) != parent_digest or config_path.read_bytes() != config_bytes or \
            alias_path.read_bytes() != alias_bytes:
        raise ValueError("parent or resolution configuration changed")
    for name, digest in support.items():
        if sha256(workspace_path(root, name).read_bytes()) != digest:
            raise ValueError("evaluation input changed during resolution")
    output.mkdir(parents=True, exist_ok=False)
    artifacts = {"episode_entities.jsonl": episode_entities, "corpus_entities.jsonl": corpus_entities,
                 "assignments.jsonl": assignments, "decisions.jsonl": decisions,
                 "review_queue.jsonl": queue, "aliases.jsonl": rules}
    for name, rows in artifacts.items():
        dump_jsonl(output / name, list(rows))
    dump_json(output / "config.json", config.model_dump(mode="json"))
    dump_json(output / "evaluation.json", evaluation)
    names = [*artifacts, "config.json", "evaluation.json"]
    counts = {"mentions": len(mentions), "episode_entities": len(episode_entities),
              "corpus_entities": len(corpus_entities), "assignments": len(assignments),
              "decisions": len(decisions), "reviewed_alias_rules": len(rules),
              "review_candidates": len(queue),
              "cross_episode_entities": sum(e.episode_count > 1 for e in corpus_entities),
              "unresolved_entities": sum(e.resolution_status == "unresolved" for e in corpus_entities)}
    code_root = Path(__file__).parent
    manifest = {"schema_version": 1, "stage": "resolve", "run_id": run_id, "status": "complete",
        "package_version": __version__, "started_utc": now.isoformat(),
        "elapsed_seconds": round(perf_counter() - started, 4), "python": platform.python_version(),
        "dependencies": {"pydantic": importlib.metadata.version("pydantic")},
        "parent_run_id": parent.name, "parent_manifest_sha256": parent_digest,
        "input_config": config_name, "input_aliases": config.aliases, "evaluation_packet": packet_name,
        "support_sha256": support, "resolution_fingerprint": fingerprint, "counts": counts,
        "code_sha256": {str(p.relative_to(code_root)): sha256(p.read_bytes()) for p in sorted(code_root.rglob("*.py"))},
        "artifact_sha256": {name: sha256((output / name).read_bytes()) for name in names}}
    dump_json(output / "manifest.json", manifest)
    validate_run(output)
    return output


def validate_run(output: Path) -> dict:
    output = output.resolve()
    manifest = json.loads((output / "manifest.json").read_bytes())
    if manifest.get("stage") == "refine_resolution":
        from spoken_record.identity_refinement import validate_run as validate_refinement
        return validate_refinement(output)
    if manifest.get("stage") == "entity_linking":
        from spoken_record.entity_linking import validate_run as validate_linking
        return validate_linking(output)
    if manifest.get("schema_version") != 1 or manifest.get("stage") != "resolve" or manifest.get("status") != "complete":
        raise ValueError("unsupported resolution snapshot")
    parent = output.parent / manifest["parent_run_id"]
    if sha256((parent / "manifest.json").read_bytes()) != manifest["parent_manifest_sha256"]:
        raise ValueError("parent extraction manifest changed")
    parent_manifest = validate_extraction(parent)
    root = output.parents[3]
    expected_names = {"episode_entities.jsonl", "corpus_entities.jsonl", "assignments.jsonl",
                      "decisions.jsonl", "review_queue.jsonl", "aliases.jsonl", "config.json", "evaluation.json"}
    if set(manifest["artifact_sha256"]) != expected_names:
        raise ValueError("unexpected resolution artifact set")
    for name, digest in manifest["artifact_sha256"].items():
        if sha256((output / name).read_bytes()) != digest:
            raise ValueError(f"artifact checksum mismatch: {name}")
    for name, digest in manifest["support_sha256"].items():
        if sha256(workspace_path(root, name).read_bytes()) != digest:
            raise ValueError(f"resolution support changed: {name}")
    config = ResolutionConfig.model_validate_json((output / "config.json").read_bytes())
    rules = [AliasRule.model_validate(r) for r in read_jsonl(output / "aliases.jsonl")]
    mentions = [ExtractedMention.model_validate(r) for r in read_jsonl(parent / "mentions.jsonl")]
    passage_run = parent.parent / parent_manifest["parent_run_id"]
    passages = [Passage.model_validate(r) for r in read_jsonl(passage_run / "passages.jsonl")]
    passage_manifest = json.loads((passage_run / "manifest.json").read_bytes())
    source_run = passage_run.parent / passage_manifest["parent_run_id"]
    episodes = [Episode.model_validate(r) for r in read_jsonl(source_run / "episodes.jsonl")]
    validate_aliases(rules, mentions, passages, episodes)
    expected = resolve_records(manifest["run_id"], episodes[0].corpus_id, mentions, passages, rules, config,
                               manifest["resolution_fingerprint"])
    model_types = (EpisodeEntity, CorpusEntity, MentionAssignment, ResolutionDecision, ReviewCandidate)
    names = ("episode_entities.jsonl", "corpus_entities.jsonl", "assignments.jsonl", "decisions.jsonl", "review_queue.jsonl")
    actual = tuple([kind.model_validate(r) for r in read_jsonl(output / name)] for kind, name in zip(model_types, names))
    if tuple(map(list, expected)) != actual:
        raise ValueError("resolution records are not reproducible")
    episode_entities, corpus_entities, assignments, decisions, queue = actual
    if len(assignments) != len(mentions) or {a.mention_id for a in assignments} != {m.mention_id for m in mentions}:
        raise ValueError("every mention must have exactly one assignment")
    if len({a.mention_id for a in assignments}) != len(assignments):
        raise ValueError("mention has multiple assignments")
    episode_ids, corpus_ids = ({e.episode_entity_id for e in episode_entities},
                               {e.corpus_entity_id for e in corpus_entities})
    if any(a.episode_entity_id not in episode_ids or a.corpus_entity_id not in corpus_ids for a in assignments):
        raise ValueError("assignment refers to an unknown entity")
    counts = {"mentions": len(mentions), "episode_entities": len(episode_entities),
              "corpus_entities": len(corpus_entities), "assignments": len(assignments),
              "decisions": len(decisions), "reviewed_alias_rules": len(rules),
              "review_candidates": len(queue),
              "cross_episode_entities": sum(e.episode_count > 1 for e in corpus_entities),
              "unresolved_entities": sum(e.resolution_status == "unresolved" for e in corpus_entities)}
    if counts != manifest["counts"]:
        raise ValueError("resolution manifest counts differ")
    evaluation = {"status": "not_evaluated", "reason": "no evaluation packet supplied"}
    if manifest["evaluation_packet"]:
        packet = workspace_path(root, manifest["evaluation_packet"])
        evaluation = evaluate_resolution(packet, mentions, assignments, rules, config)
    if evaluation != json.loads((output / "evaluation.json").read_bytes()):
        raise ValueError("resolution evaluation is not reproducible")
    return manifest
