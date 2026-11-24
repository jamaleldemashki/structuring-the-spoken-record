"""The episode layer: everything that only needs one transcript, stored once per episode.

Each episode gets a folder in ``data/episodes/<corpus>/<transcript name>/``:

* ``episode.json``, ``transcript.json``, ``segments.jsonl``: the transcript and its metadata
* ``passages.jsonl``: sentences packed into passages
* ``candidates.jsonl``, ``mentions.jsonl``: GLiNER output above the floor, and the accepted mentions
* ``coreference_clusters.json``: FCoref clusters as character spans
* the episode graph: ``entities.jsonl`` (all names of one entity in this episode, like
  "Aquinas" and "Thomas Aquinas"), ``assignments.jsonl`` (mention to entity),
  ``coreference_links.jsonl`` (pronoun to entity), ``relation_*.jsonl`` (GLiREL) and
  ``verification_*.jsonl`` (the NLI check of every relation)
* ``passage_embeddings.jsonl``: one MPNet vector per passage, for related passages
* ``manifest.json``: for every part, the key it was made with and the hash of its files

A part is redone only when its key changes. The key holds everything the part depends on:
the transcript, the settings and the model revision. Changing the GLiREL settings, for
example, touches no episode folder at all. Nothing in here uses a corpus-wide ID, so adding
other episodes never changes an episode folder.
"""

import json
from pathlib import Path, PurePosixPath

from spoken_record.adapters.base import sha256
from spoken_record.cache import Cache
from spoken_record.corpus import CorpusConfig, SelectedEpisode
from spoken_record.extraction_models import ExtractionCandidate, ExtractedMention
from spoken_record.ingestion import IngestSetup, dump_json, dump_jsonl, read_jsonl, workspace_path
from spoken_record.models import Episode, SourceBundle, SourceSegment, TranscriptVersion
from spoken_record.passage_models import Passage, SegmentationConfig

# Raise when the code of a part changes in a way that changes its output.
VERSION = {"source": 1, "passages": 1, "mentions": 1, "coreference": 1, "graph": 4, "embeddings": 1}
RUN_ID = "episode-layer"


def episode_dir(root: Path, corpus_id: str, setup: IngestSetup, source_path: str) -> Path:
    """Folder name from the transcript file name, which is unique in the transcript folder."""
    relative = PurePosixPath(Path(source_path).relative_to(setup.profile["transcripts_dir"]).as_posix())
    return root / "data/episodes" / corpus_id / str(relative.with_suffix("")).replace("/", "__")


def config_hash(root: Path, name: str) -> str:
    return sha256(workspace_path(root, name).read_bytes())


def write_part(folder: Path, files: dict[str, object]) -> dict[str, str]:
    """Replace the files of one part and return their hashes."""
    hashes = {}
    for name, value in files.items():
        path = folder / name
        path.unlink(missing_ok=True)
        if name.endswith(".jsonl"):
            dump_jsonl(path, value)
        else:
            dump_json(path, value)
        hashes[name] = sha256(path.read_bytes())
    return hashes


def part_is_current(folder: Path, manifest: dict, part: str, key: str) -> bool:
    entry = manifest.get("parts", {}).get(part)
    return bool(entry) and entry["key"] == key and all(
        (folder / name).is_file() and sha256((folder / name).read_bytes()) == digest
        for name, digest in entry["files"].items())


class Models:
    """Loads each model once, and only when an episode needs it."""

    def __init__(self, root: Path, corpus: CorpusConfig, model_cache: Cache, loaders: dict | None = None):
        self.root, self.corpus, self.model_cache, self.loaders = root, corpus, model_cache, loaders or {}
        self._gliner = self._coreference = self._encoder = self._glirel = self._verifier = None

    def gliner(self):
        if self._gliner is None:
            from spoken_record.extraction import (GLiNERWhitespaceBudget, load_configuration,
                                                  load_gliner, verify_runtime)
            config, labels, _, _ = load_configuration(self.root, self.corpus.steps.extract)
            budget = GLiNERWhitespaceBudget(config.revision)
            model = self.loaders.get("gliner", load_gliner)(config)
            verify_runtime(model, config, budget)
            self._gliner = (model, config, labels, budget)
        return self._gliner

    def coreference(self):
        if self._coreference is None:
            from spoken_record.coreference import load_model
            from spoken_record.coreference_models import CoreferenceConfig
            config = CoreferenceConfig.model_validate_json(
                workspace_path(self.root, self.corpus.steps.coreference).read_bytes())
            self._coreference = (self.loaders.get("coreference", load_model)(config), config)
        return self._coreference

    def glirel(self):
        """GLiREL behind the model cache. It is only loaded for a context it has not seen."""
        if self._glirel is None:
            from spoken_record.cache import CachedRelationModel
            from spoken_record.relation_contexts import GLiRELBudget
            from spoken_record.relation_models import RelationExtractionConfig
            from spoken_record.relations import load_model
            config = RelationExtractionConfig.model_validate_json(
                workspace_path(self.root, self.corpus.steps.relations).read_bytes())
            self._glirel = CachedRelationModel(self.model_cache, config, self.loaders.get("glirel", load_model),
                                               self.loaders.get("glirel_budget", GLiRELBudget))
        return self._glirel

    def verifier(self):
        if self._verifier is None:
            from spoken_record.cache import CachedVerifier
            from spoken_record.relation_verification import NLIBackend
            from spoken_record.verification_models import VerificationConfig
            config = VerificationConfig.model_validate_json(
                workspace_path(self.root, self.corpus.steps.verify).read_bytes())
            self._verifier = CachedVerifier(self.model_cache, config,
                                            lambda: self.loaders.get("nli", NLIBackend)(self.root, config))
        return self._verifier

    def encoder(self):
        if self._encoder is None:
            from spoken_record.discovery import PassageEncoder
            from spoken_record.discovery_models import DiscoveryConfig
            config = DiscoveryConfig.model_validate_json(
                workspace_path(self.root, self.corpus.steps.discover).read_bytes())
            self._encoder = self.loaders.get("passage_encoder", PassageEncoder)(self.root, config)
        return self._encoder


def keys(root: Path, corpus: CorpusConfig, setup: IngestSetup, source_sha256: str,
         transcript_version_id: str | None) -> dict[str, str]:
    """The key of every part. The model parts are keyed by the transcript version, so a
    change in the metadata alone (a corrected title) does not rerun any model."""
    from spoken_record.coreference_models import CoreferenceConfig
    from spoken_record.discovery_models import DiscoveryConfig
    from spoken_record.extraction import load_configuration

    # Only what reading a transcript depends on: the ingest settings and their files.
    ingest = {name: getattr(corpus, name) for name in ("corpus_id", "profile", "metadata", "audit",
                                                       "quarantine", "require_metadata")}
    support = {str(path.relative_to(root)): sha256(path.read_bytes())
               for path in setup.support_files() if path != setup.config_path}
    result = {"source": Cache.key("source", VERSION["source"], source_sha256, ingest, support)}
    if transcript_version_id is None:
        return result
    passages = Cache.key("passages", VERSION["passages"], transcript_version_id,
                         config_hash(root, corpus.steps.segment))
    extraction, labels, _, _ = load_configuration(root, corpus.steps.extract)
    coreference = CoreferenceConfig.model_validate_json(workspace_path(root, corpus.steps.coreference).read_bytes())
    discovery = DiscoveryConfig.model_validate_json(workspace_path(root, corpus.steps.discover).read_bytes())
    from spoken_record.resolution_models import ResolutionConfig
    resolution = ResolutionConfig.model_validate_json(workspace_path(root, corpus.steps.resolve).read_bytes())
    mentions = Cache.key("mentions", VERSION["mentions"], passages, extraction.model_dump(mode="json"), labels)
    clusters = Cache.key("coreference", VERSION["coreference"], transcript_version_id,
                         {"repo_id": coreference.repo_id, "revision": coreference.revision,
                          "max_tokens_in_batch": coreference.max_tokens_in_batch})
    result.update({
        "passages": passages,
        "mentions": mentions,
        "coreference": clusters,
        # The episode graph depends on the mentions, the clusters and the rules of every step
        # that runs inside the episode. Its model calls are cached by their input.
        "graph": Cache.key("graph", VERSION["graph"], mentions, clusters,
                           config_hash(root, resolution.aliases),
                           *(config_hash(root, name) for name in (corpus.steps.resolve, corpus.steps.refine,
                             corpus.steps.refine_episode or corpus.steps.refine,
                             corpus.steps.coreference, corpus.steps.relations, corpus.steps.verify))),
        "embeddings": Cache.key("embeddings", VERSION["embeddings"], passages,
                                {"repo_id": discovery.repo_id, "revision": discovery.revision,
                                 "max_tokens": discovery.max_tokens, "batch_size": discovery.batch_size}),
    })
    return result


def update_episode(root: Path, corpus: CorpusConfig, setup: IngestSetup, selected: SelectedEpisode,
                   models: Models) -> dict:
    """Bring one episode folder up to date. Returns which parts were redone."""
    folder = episode_dir(root, setup.corpus_id, setup, selected.source_path)
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = folder / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes()) if manifest_path.is_file() else {}
    skipped = setup.skip_reason(selected.source_path)
    if skipped:
        raise ValueError(f"{selected.source_path}: {skipped[1]}")
    data = setup.source_path(selected.source_path).read_bytes()
    source_sha256 = sha256(data)
    redone = []
    part_keys = keys(root, corpus, setup, source_sha256, None)
    if not part_is_current(folder, manifest, "source", part_keys["source"]):
        bundle = setup.read(selected.source_path, data, selected.expected_sha256, RUN_ID)
        # The other parts stay. They are keyed by the transcript version, so they are only
        # redone below if the transcript text itself changed.
        manifest = {"schema_version": 1, "parts": manifest.get("parts", {})}
        manifest["parts"]["source"] = {"key": part_keys["source"], "files": write_part(folder, {
            "episode.json": bundle.episode.model_dump(mode="json"),
            "transcript.json": bundle.transcript.model_dump(mode="json"),
            "segments.jsonl": bundle.segments})}
        redone.append("source")
    bundle = load_bundle(folder)
    manifest.update({"episode_id": bundle.episode.episode_id, "source_path": selected.source_path,
                     "source_sha256": source_sha256,
                     "transcript_version_id": bundle.transcript.transcript_version_id})
    part_keys = keys(root, corpus, setup, source_sha256, bundle.transcript.transcript_version_id)

    if not part_is_current(folder, manifest, "passages", part_keys["passages"]):
        from spoken_record.passages import build_passages, check_passages
        config = SegmentationConfig.model_validate_json(workspace_path(root, corpus.steps.segment).read_bytes())
        built = build_passages(bundle, config, RUN_ID)
        check_passages(bundle, built)
        manifest["parts"]["passages"] = {"key": part_keys["passages"], "files": write_part(folder, {
            "passages.jsonl": built})}
        redone.append("passages")
    passages = [Passage.model_validate(row) for row in read_jsonl(folder / "passages.jsonl")]

    if not part_is_current(folder, manifest, "mentions", part_keys["mentions"]):
        candidates, mentions = extract_mentions(root, corpus, bundle, passages, models)
        manifest["parts"]["mentions"] = {"key": part_keys["mentions"], "files": write_part(folder, {
            "candidates.jsonl": candidates, "mentions.jsonl": mentions})}
        redone.append("mentions")

    if not part_is_current(folder, manifest, "coreference", part_keys["coreference"]):
        from spoken_record.coreference import predict_clusters
        model, config = models.coreference()
        clusters = predict_clusters(model, bundle.transcript.text, config)
        manifest["parts"]["coreference"] = {"key": part_keys["coreference"], "files": write_part(folder, {
            "coreference_clusters.json": {"transcript_version_id": bundle.transcript.transcript_version_id,
                                          "clusters": clusters}})}
        redone.append("coreference")

    if not part_is_current(folder, manifest, "graph", part_keys["graph"]):
        manifest["parts"]["graph"] = {"key": part_keys["graph"], "files": write_part(folder, episode_graph(
            root, corpus, setup, bundle, passages, folder, models))}
        redone.append("graph")

    if not part_is_current(folder, manifest, "embeddings", part_keys["embeddings"]):
        texts = [passage.text for passage in passages]
        encoded = models.encoder().encode(texts) if texts else []
        rows = [{**item, "passage_id": passage.passage_id, "input_sha256": sha256(passage.text.encode())}
                for passage, item in zip(passages, encoded, strict=True)]
        manifest["parts"]["embeddings"] = {"key": part_keys["embeddings"], "files": write_part(folder, {
            "passage_embeddings.jsonl": rows})}
        redone.append("embeddings")

    dump_json_replace(manifest_path, manifest)
    return {"source_path": selected.source_path, "folder": str(folder.relative_to(root)), "redone": redone}


def dump_json_replace(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    dump_json(temporary, value)
    temporary.replace(path)


def extract_mentions(root: Path, corpus: CorpusConfig, bundle: SourceBundle, passages: list[Passage],
                     models: Models) -> tuple[list[ExtractionCandidate], list[ExtractedMention]]:
    """GLiNER over one episode, with the same windows and rules as the extraction stage."""
    from spoken_record.models import TYPES
    from spoken_record.extraction import extraction_fingerprint, predict_episode, resolve_candidates
    from spoken_record.windows import build_windows, require_model_ready

    model, config, labels, budget = models.gliner()
    segmentation = SegmentationConfig.model_validate_json(
        workspace_path(root, corpus.steps.segment).read_bytes()).model_copy(update={
            "window_max_tokens": config.window_max_tokens, "window_overlap_tokens": config.window_overlap_tokens})
    windows = build_windows(bundle, passages, segmentation, RUN_ID, budget)
    require_model_ready(windows)
    descriptions = [labels["labels"][label] for label in TYPES]
    label_map = {description: canonical for canonical, description in labels["labels"].items()}
    raw, _ = predict_episode(model, bundle, windows, descriptions, label_map, config, budget, RUN_ID)
    candidates, mentions = resolve_candidates(raw, float(labels["provisional_threshold"]),
                                              extraction_fingerprint(config, labels), passages)
    return list(candidates), list(mentions)


def episode_graph(root: Path, corpus: CorpusConfig, setup: IngestSetup, bundle: SourceBundle,
                  passages: list[Passage], folder: Path, models: Models) -> dict[str, list]:
    """Entities, pronoun links, relations and their NLI check, using only this episode."""
    from spoken_record.consolidation import consolidate_episode
    from spoken_record.coreference import project_clusters
    from spoken_record.coreference_models import CoreferenceConfig
    from spoken_record.identity_models import IdentityRefinementConfig
    from spoken_record.relation_models import RelationExtractionConfig
    from spoken_record.relation_verification import compute_verification
    from spoken_record.relations import compute_relations
    from spoken_record.resolution import name_mentions, resolution_fingerprint
    from spoken_record.resolution_models import AliasRule, ResolutionConfig
    from spoken_record.verification_models import VerificationConfig

    def settings(model, name):
        data = workspace_path(root, name).read_bytes()
        return model.model_validate_json(data), sha256(data)

    episode_id, transcript = bundle.episode.episode_id, bundle.transcript
    mentions = [ExtractedMention.model_validate(row) for row in read_jsonl(folder / "mentions.jsonl")]
    clusters = json.loads((folder / "coreference_clusters.json").read_bytes())["clusters"]
    resolution, _ = settings(ResolutionConfig, corpus.steps.resolve)
    mentions = name_mentions(mentions, resolution)
    rules = [AliasRule.model_validate(row) for row in read_jsonl(workspace_path(root, resolution.aliases))]
    refinement, _ = settings(IdentityRefinementConfig, corpus.steps.refine_episode or corpus.steps.refine)
    episode_entities, entities, assignments = consolidate_episode(RUN_ID, setup.corpus_id, episode_id, mentions,
        passages, rules, resolution, resolution_fingerprint(resolution, rules), refinement)

    coreference, fingerprint = settings(CoreferenceConfig, corpus.steps.coreference)
    _, links = project_clusters(RUN_ID, episode_id, transcript, clusters, passages, mentions, assignments,
                                coreference, fingerprint)

    relations, fingerprint = settings(RelationExtractionConfig, corpus.steps.relations)
    if relations.context_unit != "coreference_antecedent_window":
        raise ValueError("the episode graph needs antecedent-window relation contexts for the NLI check")
    glirel = models.glirel()
    candidates, assertions, _, outcomes, contexts, _ = compute_relations(RUN_ID, passages, mentions, assignments,
        list(links), [transcript], relations, fingerprint, glirel, glirel.budget)
    failed = [row for row in outcomes if row.status == "failed"]
    if failed:
        raise ValueError(f"GLiREL failed on {len(failed)} passage(s), first: {failed[0].error}")

    verification, _ = settings(VerificationConfig, corpus.steps.verify)
    accepted = [row for row in candidates if row.decision == "accepted"]
    verifier = models.verifier()
    evidence, decisions, _ = compute_verification(RUN_ID, accepted, list(assertions), contexts, [transcript],
        mentions, assignments, list(links), [label.model_dump(mode="json") for label in relations.labels],
        verification, lambda: verifier)
    return {"episode_entities.jsonl": episode_entities, "entities.jsonl": entities, "assignments.jsonl": assignments,
            "coreference_links.jsonl": list(links), "relation_contexts.jsonl": contexts,
            "relation_candidates.jsonl": list(candidates), "relation_assertions.jsonl": list(assertions),
            "verification_evidence.jsonl": evidence, "verification_decisions.jsonl": decisions}


def load_bundle(folder: Path) -> SourceBundle:
    return SourceBundle(episode=Episode.model_validate_json((folder / "episode.json").read_bytes()),
        transcript=TranscriptVersion.model_validate_json((folder / "transcript.json").read_bytes()),
        segments=[SourceSegment.model_validate(row) for row in read_jsonl(folder / "segments.jsonl")])


def load_episode(folder: Path) -> dict:
    """All records of one episode folder, checked against its manifest."""
    manifest = json.loads((folder / "manifest.json").read_bytes())
    for part in manifest["parts"].values():
        for name, digest in part["files"].items():
            if sha256((folder / name).read_bytes()) != digest:
                raise ValueError(f"episode file changed after it was written: {folder.name}/{name}")
    from spoken_record.coreference_models import CoreferenceLink
    from spoken_record.relation_models import RelationCandidate, RelationContext
    from spoken_record.resolution_models import CorpusEntity, MentionAssignment
    from spoken_record.verification_models import VerificationEvidence

    def rows(name, model):
        return [model.model_validate(row) for row in read_jsonl(folder / name)]

    return {"manifest": manifest, "bundle": load_bundle(folder),
            "passages": rows("passages.jsonl", Passage), "mentions": rows("mentions.jsonl", ExtractedMention),
            "entities": rows("entities.jsonl", CorpusEntity), "assignments": rows("assignments.jsonl", MentionAssignment),
            "coreference_links": rows("coreference_links.jsonl", CoreferenceLink),
            "relation_contexts": rows("relation_contexts.jsonl", RelationContext),
            "relation_candidates": rows("relation_candidates.jsonl", RelationCandidate),
            "verification_evidence": rows("verification_evidence.jsonl", VerificationEvidence),
            "embeddings": read_jsonl(folder / "passage_embeddings.jsonl")}
