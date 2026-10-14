"""Tests for the pipeline runner: full runs, resume, and running without the optional models."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from test_ingestion import workspace
from test_extraction import configured_workspace, FakeModel
from spoken_record.ingestion import dump_json
from spoken_record.pipeline import plan_pipeline, run_pipeline, validate_snapshot


def foundation_recipe(workspace):
    """Real ingest and segment recipe over one synthetic transcript."""
    root, _ = workspace
    dump_json(root / "segment.json", {"schema_version": 1, "passage_min_words": 3,
        "passage_target_words": 6, "passage_max_words": 12, "window_max_tokens": 10,
        "window_overlap_tokens": 2})
    recipe = {"schema_version": 1, "stages": [
        {"id": "source", "kind": "ingest", "config": "config.json", "pilot": "pilot.json"},
        {"id": "passages", "kind": "segment", "config": "segment.json"}]}
    dump_json(root / "pipeline.json", recipe)
    return root, recipe


def test_plan_no_writes_and_foundation_resume(workspace):
    """Planning writes nothing, and finished stages are validated and reused on resume."""
    root, _ = foundation_recipe(workspace)
    before = set(root.rglob("*"))
    plan = plan_pipeline(root, "pipeline.json", "test")
    assert set(root.rglob("*")) == before
    assert plan["steps"][1]["input_run"] == "data/prototype1/runs/test-source"
    result = run_pipeline(root, "pipeline.json", "test")
    assert result["completed_stages"] == 2
    output = root / result["final_output"]
    artifacts = {p.name: p.read_bytes() for p in output.iterdir()}
    assert run_pipeline(root, "pipeline.json", "test", resume=True) == result
    assert artifacts == {p.name: p.read_bytes() for p in output.iterdir()}
    with pytest.raises(FileExistsError):
        run_pipeline(root, "pipeline.json", "test")


def test_resume_after_transient_failure(workspace, monkeypatch):
    """A failure before writing keeps the finished stages, and the run can be resumed."""
    root, _ = foundation_recipe(workspace)
    import spoken_record.pipeline as pipeline
    original = pipeline.execute_stage

    def fail_segment(root, step):
        """Pretend a dependency is missing before the stage writes its run."""
        if step["spec"]["kind"] == "segment":
            raise RuntimeError("temporary backend failure")
        return original(root, step)

    monkeypatch.setattr(pipeline, "execute_stage", fail_segment)
    with pytest.raises(RuntimeError, match="temporary"):
        run_pipeline(root, "pipeline.json", "retry")
    journal = root / "data/prototype1/pipelines/retry/state.json"
    assert json.loads(journal.read_bytes())["status"] == "failed"
    source = root / "data/prototype1/runs/retry-source/manifest.json"
    original_bytes = source.read_bytes()
    monkeypatch.setattr(pipeline, "execute_stage", original)
    assert run_pipeline(root, "pipeline.json", "retry", resume=True)["status"] == "complete"
    assert source.read_bytes() == original_bytes


@pytest.mark.parametrize("change", ["config", "source", "artifact", "code"])
def test_resume_refuses_changed_evidence(workspace, monkeypatch, change):
    """Resume refuses changed configs, sources, outputs or code."""
    root, _ = foundation_recipe(workspace)
    run_pipeline(root, "pipeline.json", "stale")
    if change == "code":
        import spoken_record.pipeline as pipeline
        original = pipeline.plan_pipeline

        def changed_plan(*args):
            """Pretend the code changed without touching the package files."""
            plan = original(*args)
            plan["code_sha256"]["pipeline.py"] = "changed"
            return plan

        monkeypatch.setattr(pipeline, "plan_pipeline", changed_plan)
    else:
        name = {"config": "segment.json", "source": workspace[1],
                "artifact": "data/prototype1/runs/stale-passages/passages.jsonl"}[change]
        with (root / name).open("a") as handle:
            handle.write(" ")
    with pytest.raises(ValueError):
        run_pipeline(root, "pipeline.json", "stale", resume=True)


def test_disabled_required_stage_and_unknown_options_fail_early(workspace):
    """Disabling a stage never feeds the next stage records it can't use."""
    root, recipe = foundation_recipe(workspace)
    recipe["stages"][0]["enabled"] = False
    dump_json(root / "pipeline.json", recipe)
    with pytest.raises(ValueError, match="cannot consume"):
        plan_pipeline(root, "pipeline.json", "invalid")
    recipe["stages"][0]["enabled"] = True
    recipe["stages"][1]["packet"] = "unrelated"
    dump_json(root / "pipeline.json", recipe)
    with pytest.raises(ValueError, match="not supported"):
        plan_pipeline(root, "pipeline.json", "invalid")
    assert not (root / "data/prototype1/pipelines").exists()


def test_resume_refuses_partial_stage_output(workspace, monkeypatch):
    """An output folder left by a failed stage is never overwritten or reused."""
    root, _ = foundation_recipe(workspace)
    import spoken_record.pipeline as pipeline
    original = pipeline.execute_stage

    def partial_write(root, step):
        """Leave a half-written stage folder, like an interrupted run would."""
        if step["spec"]["kind"] == "segment":
            output = root / step["output"]
            output.mkdir()
            (output / "partial.txt").write_text("retained evidence")
            raise RuntimeError("interrupted publication")
        return original(root, step)

    monkeypatch.setattr(pipeline, "execute_stage", partial_write)
    with pytest.raises(RuntimeError):
        run_pipeline(root, "pipeline.json", "partial")
    monkeypatch.setattr(pipeline, "execute_stage", original)
    with pytest.raises(FileExistsError, match="partial output"):
        run_pipeline(root, "pipeline.json", "partial", resume=True)
    assert (root / "data/prototype1/runs/partial-passages/partial.txt").read_text() == "retained evidence"


def test_item_failures_stop_downstream(workspace, monkeypatch):
    """A complete manifest with failed items must stop the next stage."""
    root, _ = foundation_recipe(workspace)
    import spoken_record.pipeline as pipeline
    original = pipeline.validate_snapshot

    def failed_count(path):
        """Run marked complete, but some items failed."""
        manifest = original(path)
        manifest["counts"]["failed"] = 1
        return manifest

    monkeypatch.setattr(pipeline, "validate_snapshot", failed_count)
    with pytest.raises(ValueError, match="contains failures"):
        run_pipeline(root, "pipeline.json", "failed")
    assert not (root / "data/prototype1/runs/failed-passages").exists()


def full_recipe(workspace, monkeypatch, include_discovery, export):
    """Write a full synthetic recipe ending in ``export`` and install the offline model fakes."""
    root, _, _, _ = configured_workspace(workspace)
    repository = Path(__file__).resolve().parents[1]
    for name in ("coreference.fcoref.json", "linking.wikidata-mpnet-v4.json",
                 "vault.concept-navigation-v1.json", "identity_refinement.surname-variant.json"):
        (root / name).write_bytes((repository / "config" / name).read_bytes())
    # The fake GLiREL has no tokenizer, so the tests use the simpler sentence-context relations config.
    name = "relations.glirel-coreference.json"
    (root / name).write_bytes((repository / "tests/fixtures" / name).read_bytes())
    (root / "aliases.jsonl").write_text("")
    dump_json(root / "resolve.json", {"schema_version": 1,
        "normalization": "unicode-nfc-casefold-whitespace-v1", "aliases": "aliases.jsonl",
        "cross_episode_exact_types": ["Place", "Language"]})
    recipe = {"schema_version": 1, "stages": [
        {"id": "source", "kind": "ingest", "config": "config.json", "pilot": "pilot.json"},
        {"id": "passages", "kind": "segment", "config": "segmentation.json"},
        {"id": "ner", "kind": "extract", "config": "model.test-calibrated.json"},
        {"id": "resolve", "kind": "resolve", "config": "resolve.json"},
        {"id": "refine", "kind": "refine-resolution", "config": "identity_refinement.surname-variant.json"},
        {"id": "link", "kind": "link-entities", "config": "linking.wikidata-mpnet-v4.json"},
        {"id": "coref", "kind": "coreference", "config": "coreference.fcoref.json"},
        {"id": "relations", "kind": "relations", "config": "relations.glirel-coreference.json"},
        export]}
    if include_discovery:
        from test_discovery import FakeEncoder
        (root / "discovery.json").write_bytes((repository / "config/discovery.mpnet-v1.json").read_bytes())
        recipe["stages"].insert(-1, {"id": "discovery", "kind": "discover-passages", "config": "discovery.json"})
        monkeypatch.setattr("spoken_record.discovery.PassageEncoder", FakeEncoder)
    dump_json(root / "pipeline.json", recipe)

    class EmptyCoreference:
        """Episode processed fine, just no pronoun references."""

        def predict(self, **kwargs):
            """One empty cluster result per transcript."""
            return [self for _ in kwargs["texts"]]

        def get_clusters(self, **kwargs):
            """Zero clusters is a valid result, not a failed stage."""
            return []

    class EmptyRelations:
        """No relation found, the co-occurrence edges stay."""

        def predict_relations(self, *args, **kwargs):
            """Abstain for every synthetic relation context."""
            return []

    class EmptyRetriever:
        """No network calls and no Wikidata matches."""

        def retrieve(self, entities):
            """Empty candidate lists."""
            return {entity.corpus_entity_id: [] for entity in entities}

    class Encoder:
        """Fixed normalized vectors, without importing any model library."""

        def encode(self, texts):
            """Same number and order of texts as the adapter contract requires."""
            return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr("spoken_record.extraction.load_gliner", lambda config: FakeModel())
    monkeypatch.setattr("spoken_record.coreference.load_model", lambda config: EmptyCoreference())
    monkeypatch.setattr("spoken_record.relations.load_model", lambda config: EmptyRelations())
    monkeypatch.setattr("spoken_record.entity_linking.WikidataRetriever", lambda config: EmptyRetriever())
    monkeypatch.setattr("spoken_record.entity_linking.MpnetEncoder", lambda config: Encoder())
    return root


@pytest.mark.parametrize("include_discovery", [False, True])
def test_pipeline_to_vault_without_annotation_packet(workspace, monkeypatch, include_discovery):
    """The real stages run through to the export with offline model fakes."""
    root = full_recipe(workspace, monkeypatch, include_discovery,
                       {"id": "vault", "kind": "export-obsidian", "config": "vault.concept-navigation-v1.json"})
    result = run_pipeline(root, "pipeline.json", "full")
    assert result["completed_stages"] == (10 if include_discovery else 9)
    if include_discovery:
        counts = validate_snapshot(root / "data/prototype1/runs/full-discovery")["counts"]
        assert counts["mpnet_cosine_connections"] == 0  # One episode; no forced matches.
        assert (root / result["final_output"] / "obsidian/00_Index/Explore related discussions.md").is_file()
    assert (root / result["final_output"] / "obsidian/START HERE.md").is_file()
    assert validate_snapshot(root / result["final_output"])["stage"] == "export_obsidian"
    evaluation = root / "data/prototype1/runs/full-resolve/evaluation.json"
    assert json.loads(evaluation.read_bytes())["status"] == "not_evaluated"
    assert run_pipeline(root, "pipeline.json", "full", resume=True) == result


def test_cli_import_does_not_require_model_stack():
    """Block the optional libraries in a fresh process, to catch imports that should be lazy."""
    code = '''
import importlib.abc
import sys
class BlockModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'fastcoref', 'glirel', 'gliner', 'transformers', 'huggingface_hub', 'torch'}:
            raise ImportError('optional library imported: ' + fullname)
sys.meta_path.insert(0, BlockModels())
import spoken_record.cli
sys.argv = ['spoken-record', 'pipeline', '--help']
spoken_record.cli.main()
'''
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    result = subprocess.run([sys.executable, "-c", code], env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert "--dry-run" in result.stdout and "--resume" in result.stdout
