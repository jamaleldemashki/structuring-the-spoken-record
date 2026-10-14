"""Runs a recipe: an ordered list of stages, each with its own config.

The runner handles the order, the checks before starting, progress and resume. Each stage
module still owns its records, model calls, decision rules and validation. Planning never
runs a model, and recipes don't pick up any CLI defaults.
"""

import importlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from spoken_record.adapters.base import sha256
from spoken_record.ingestion import workspace_path


class StageSpec(BaseModel):
    """One stage in a recipe, with its config and optional extra inputs."""

    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^[a-z][a-z0-9-]{0,23}$")
    kind: Literal["ingest", "segment", "extract", "resolve", "refine-resolution",
                  "link-entities", "coreference", "relations", "verify-relations", "discover-passages",
                  "enrich-entities", "export-obsidian", "export-web"]
    config: str
    enabled: bool = True
    pilot: str | None = None
    packet: str | None = None
    exclusions: str | None = None
    candidate_run: str | None = None


class PipelineConfig(BaseModel):
    """A recipe: stages in order, optionally starting from an existing run."""

    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1] = 1
    input_run: str | None = None
    stages: list[StageSpec] = Field(min_length=1)


@dataclass(frozen=True)
class StageDefinition:
    """What a built-in stage accepts and produces, and where its function lives."""

    module: str
    function: str
    accepts: tuple[str, ...]
    produces: str
    config_model: str | None = None
    options: tuple[str, ...] = ()


# One table says which stage can follow which and what to call. Stage modules
# (and their model libraries) are only imported when the stage runs.
IDENTITIES = ("resolve", "refine_resolution", "entity_linking")
# Runs that enrichment and the exports can follow back down to the identity records.
PROJECTABLE = (*IDENTITIES, "coreference", "relations", "relation_verification", "passage_discovery")
STAGES = {
    "ingest": StageDefinition("ingestion", "ingest", (), "ingest", options=("pilot",)),
    "segment": StageDefinition("segmentation", "segment_run", ("ingest",), "segment",
                               "passage_models.SegmentationConfig", ("exclusions",)),
    "extract": StageDefinition("extraction", "extract_run", ("segment",), "extract",
                               "extraction_models.ExtractionConfig", ("packet",)),
    "resolve": StageDefinition("resolution", "resolve_run", ("extract",), "resolve",
                               "resolution_models.ResolutionConfig", ("packet",)),
    "refine-resolution": StageDefinition("identity_refinement", "refine_run", ("resolve",),
                                         "refine_resolution", "identity_models.IdentityRefinementConfig"),
    "link-entities": StageDefinition("entity_linking", "link_run", IDENTITIES, "entity_linking",
                                     "linking_models.EntityLinkingConfig", ("candidate_run",)),
    "coreference": StageDefinition("coreference", "coreference_run", IDENTITIES, "coreference",
                                   "coreference_models.CoreferenceConfig"),
    "relations": StageDefinition("relations", "relation_run", (*IDENTITIES, "coreference"),
                                 "relations", "relation_models.RelationExtractionConfig"),
    "verify-relations": StageDefinition("relation_verification", "verify_run", ("relations",),
                                        "relation_verification", "verification_models.VerificationConfig"),
    "discover-passages": StageDefinition("discovery", "discover_run",
        (*IDENTITIES, "coreference", "relations", "relation_verification"),
        "passage_discovery", "discovery_models.DiscoveryConfig"),
    "enrich-entities": StageDefinition("enrichment", "enrich_run", PROJECTABLE, "entity_enrichment",
                                       "enrichment_models.EnrichmentConfig"),
    "export-obsidian": StageDefinition("vault", "export_vault", (*PROJECTABLE, "entity_enrichment"),
                                       "export_obsidian", "vault_models.VaultConfig"),
    "export-web": StageDefinition("web_export", "export_web", (*PROJECTABLE, "entity_enrichment"),
                                  "export_web", "web_models.WebExportConfig"),
}
EXPORT_KINDS = ("export-obsidian", "export-web")


def validate_snapshot(path: Path) -> dict:
    """Validate a run with the stage validators (no model needed)."""
    from spoken_record.segmentation import validate_run
    return validate_run(path)


def require_success(manifest: dict) -> None:
    """Stop if any item failed, even when the run itself is marked complete."""
    if manifest.get("status") != "complete" or manifest.get("counts", {}).get("failed", 0):
        raise ValueError(f"snapshot {manifest.get('run_id')} contains failures or is incomplete")


def plan_pipeline(root: Path, config_name: str, run_id: str) -> dict:
    """Check stage order and inputs and return the plan. Nothing is written."""
    root = root.resolve()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]{0,63}", run_id):
        raise ValueError("pipeline run ID must be 1–64 letters, numbers or dashes")
    dependencies: dict[str, str] = {}

    def capture(name: str) -> bytes:
        """Read a file and record its hash, so later edits are noticed."""
        path = workspace_path(root, name)
        data = path.read_bytes()
        dependencies[str(path.relative_to(root))] = sha256(data)
        return data

    recipe = PipelineConfig.model_validate_json(capture(config_name))
    if len({stage.id for stage in recipe.stages}) != len(recipe.stages):
        raise ValueError("pipeline stage IDs must be unique")
    parent = recipe.input_run
    previous = None
    if parent:
        parent_path = workspace_path(root, parent)
        if parent_path.parent != root / "data/prototype1/runs":
            raise ValueError("input_run must be a local run snapshot")
        manifest = validate_snapshot(parent_path)
        require_success(manifest)
        previous = manifest["stage"]
        capture(f"{parent}/manifest.json")
    steps = []
    for spec in recipe.stages:
        if not spec.enabled:
            continue
        definition = STAGES[spec.kind]
        if (spec.kind == "ingest" and previous is not None) or (
                spec.kind != "ingest" and previous not in definition.accepts):
            raise ValueError(f"{spec.id}: {spec.kind} cannot consume {previous!r}")
        for option in ("pilot", "packet", "exclusions", "candidate_run"):
            if getattr(spec, option) is not None and option not in definition.options:
                raise ValueError(f"{spec.id}: option {option} is not supported")
        payload = json.loads(capture(spec.config))
        if definition.config_model:
            module, name = definition.config_model.split(".")
            contract = getattr(importlib.import_module(f"spoken_record.{module}"), name)
            contract.model_validate(payload)
        if spec.kind == "ingest":
            if not spec.pilot:
                raise ValueError("ingest requires an explicit frozen pilot")
            capture(payload["profile"])
            # Metadata, audit and quarantine are optional, not every podcast has them.
            for key in ("metadata", "audit", "quarantine"):
                if payload.get(key):
                    capture(payload[key])
            # Only hash the transcripts in the pilot, not the whole archive.
            pilot = json.loads(capture(spec.pilot))
            if not pilot.get("episodes"):
                raise ValueError("ingest pilot is empty")
            for entry in pilot["episodes"]:
                source = capture(entry["source_path"])
                if sha256(source) != entry["source_sha256"]:
                    raise ValueError(f"pilot source changed: {entry['source_path']}")
        for key in ("labels_config", "aliases", "reviewed_qids"):
            if payload.get(key):
                data = capture(payload[key])
                if key == "aliases":
                    from spoken_record.resolution_models import AliasRule
                    for line in data.decode("utf-8").splitlines():
                        if line.strip():
                            AliasRule.model_validate_json(line)
        if spec.kind == "extract":
            from spoken_record.extraction import load_configuration
            load_configuration(root, spec.config)
        if spec.packet:
            for name in ("manifest.json", "sample.jsonl", "annotations.jsonl"):
                capture(f"{spec.packet}/{name}")
        if spec.exclusions:
            capture(spec.exclusions)
        if spec.candidate_run:
            cache = workspace_path(root, spec.candidate_run)
            manifest = validate_snapshot(cache)
            require_success(manifest)
            if manifest["stage"] != "entity_linking" or manifest["parent_run_id"] != Path(parent).name:
                raise ValueError("candidate cache must use the same resolution parent")
            capture(f"{spec.candidate_run}/manifest.json")
        if spec.kind == "relations" and payload.get("include_linked_coreference") and previous != "coreference":
            raise ValueError("coreference-aware relations require an enabled coreference stage or snapshot")
        if spec.kind == "verify-relations":
            workspace_path(root, payload.get("cache_dir", ".cache/huggingface/hub"))
            if steps:
                producer = json.loads(workspace_path(root, steps[-1]["spec"]["config"]).read_bytes())
                has_contexts = producer.get("context_unit") == "coreference_antecedent_window"
            else:
                producer = json.loads(capture(f"{parent}/manifest.json"))
                has_contexts = "relation_contexts.jsonl" in producer.get("artifact_sha256", {})
            if not has_contexts:
                raise ValueError("verification requires antecedent-window relation contexts")
        if spec.kind == "discover-passages" and payload.get("cache_dir"):
            workspace_path(root, payload["cache_dir"])
        stage_id = f"{run_id}-{spec.id}"
        base = "export/prototype1" if spec.kind in EXPORT_KINDS else "data/prototype1/runs"
        output = f"{base}/{stage_id}"
        workspace_path(root, output)
        steps.append({"spec": spec.model_dump(), "run_id": stage_id, "input_run": parent,
                      "output": output, "produces": definition.produces})
        parent, previous = output, definition.produces
    if not steps:
        raise ValueError("pipeline must have at least one enabled stage")
    code_root = Path(__file__).parent
    return {"schema_version": 1, "run_id": run_id, "recipe": config_name,
            "dependencies": dependencies,
            "code_sha256": {str(p.relative_to(code_root)): sha256(p.read_bytes())
                            for p in sorted(code_root.rglob("*.py"))},
            "steps": steps, "final_output": parent}


def execute_stage(root: Path, step: dict) -> Path:
    """Call the stage function with the arguments it expects."""
    spec = StageSpec.model_validate(step["spec"])
    definition = STAGES[spec.kind]
    function = getattr(importlib.import_module(f"spoken_record.{definition.module}"), definition.function)
    kwargs = {"root": root, "run_id": step["run_id"]}
    kwargs["model_name" if spec.kind == "extract" else "config_name"] = spec.config
    if spec.kind != "ingest":
        kwargs["input_name"] = step["input_run"]
    for option in definition.options:
        parameter = "candidate_name" if option == "candidate_run" else f"{option}_name"
        kwargs[parameter] = getattr(spec, option)
    return function(**kwargs)


def save_state(path: Path, state: dict) -> None:
    """Replace the progress journal in one step (write a temp file, then rename)."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_pipeline(root: Path, config_name: str, run_id: str, *, resume: bool = False,
                 progress=None) -> dict:
    """Run or resume a recipe. Stale, partial or unrelated outputs are refused."""
    root = root.resolve()
    plan = plan_pipeline(root, config_name, run_id)
    directory = workspace_path(root, f"data/prototype1/pipelines/{run_id}")
    if resume:
        if not directory.is_dir():
            raise ValueError("no pipeline journal to resume")
    else:
        # Make sure no output folder exists yet before running anything.
        for step in plan["steps"]:
            if workspace_path(root, step["output"]).exists():
                raise FileExistsError(step["output"])
        directory.mkdir(parents=True, exist_ok=False)
    lock = directory / "running.lock"
    # Mode "x" fails if the lock exists, so two runners can't write the same outputs.
    with lock.open("x", encoding="utf-8") as handle:
        handle.write("Pipeline is running. Remove only after confirming its process stopped.\n")
    state_path = directory / "state.json"
    try:
        if resume:
            state = json.loads(state_path.read_bytes())
            if state["plan"] != plan:
                raise ValueError("recipe, inputs or code changed; use a new pipeline run ID")
        else:
            state = {"plan": plan, "status": "running", "completed": []}
            save_state(state_path, state)
        completed = state["completed"]
        if len(completed) > len(plan["steps"]):
            raise ValueError("journal contains unexpected stages")
        # Validate every stage that already finished before going on.
        for step, saved in zip(plan["steps"], completed):
            output = workspace_path(root, step["output"])
            manifest = validate_snapshot(output)
            require_success(manifest)
            expected = {"id": step["spec"]["id"], "output": step["output"],
                        "manifest_sha256": sha256((output / "manifest.json").read_bytes())}
            if saved != expected:
                raise ValueError("completed pipeline output changed or was reordered")
        for step in plan["steps"][len(completed):]:
            if workspace_path(root, step["output"]).exists():
                raise FileExistsError(f"unrecorded/partial output {step['output']}; use a new run ID")
        state.pop("error", None)
        state["status"] = "running"
        save_state(state_path, state)
        try:
            for step in plan["steps"][len(completed):]:
                state["active_stage"] = step["spec"]["id"]
                save_state(state_path, state)
                if progress:
                    progress({"stage": state["active_stage"], "status": "running"})
                # Check that no config or input was edited while earlier stages ran.
                for name, digest in plan["dependencies"].items():
                    if sha256(workspace_path(root, name).read_bytes()) != digest:
                        raise ValueError(f"pipeline input changed: {name}")
                output = execute_stage(root, step).resolve()
                if output != workspace_path(root, step["output"]):
                    raise ValueError("stage returned an unexpected output path")
                manifest = validate_snapshot(output)
                require_success(manifest)
                if manifest["stage"] != step["produces"] or manifest["run_id"] != step["run_id"]:
                    raise ValueError("stage returned an incompatible snapshot")
                actual_parent = manifest.get("parent_run_id") or Path(manifest.get("parent_run", "")).name
                if step["input_run"] and actual_parent != Path(step["input_run"]).name:
                    raise ValueError("stage returned a different parent snapshot")
                completed.append({"id": state["active_stage"], "output": step["output"],
                                  "manifest_sha256": sha256((output / "manifest.json").read_bytes())})
                save_state(state_path, state)
            state["status"] = "complete"
            state.pop("active_stage", None)
            save_state(state_path, state)
        except Exception as exc:
            state["status"] = "failed"
            state["error"] = f"{type(exc).__name__}: {exc}"
            save_state(state_path, state)
            raise
        return {"run_id": run_id, "status": state["status"], "completed_stages": len(completed),
                "final_output": plan["final_output"], "journal": str(state_path.relative_to(root))}
    finally:
        lock.unlink()
