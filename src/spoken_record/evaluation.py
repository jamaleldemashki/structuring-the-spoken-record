"""Annotation packets for manual NER review, and exact typed-span scoring."""

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from spoken_record.adapters.base import sha256
from spoken_record.ingestion import dump_json, dump_jsonl, read_jsonl, workspace_path
from spoken_record.segmentation import validate_run

EntityType = Literal["Person", "Work", "School", "Concept", "Place", "Period",
                     "Language", "Group"]
TYPES = ("Person", "Work", "School", "Concept", "Place", "Period", "Language", "Group")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Mention(StrictModel):
    start: int = Field(ge=0, strict=True)
    end: int = Field(gt=0, strict=True)
    text: str = Field(min_length=1)
    type: EntityType
    identity_id: str | None = None

    @model_validator(mode="after")
    def valid_span(self):
        if self.end <= self.start or self.end - self.start != len(self.text):
            raise ValueError("invalid mention range or length")
        return self


class Annotation(StrictModel):
    passage_id: str
    text_sha256: str
    status: Literal["unreviewed", "in_progress", "reviewed"] = "unreviewed"
    exhaustive: bool = False
    reviewer: str | None = None
    annotation_seconds: float | None = Field(default=None, ge=0)
    unresolved_notes: list[str] = Field(default_factory=list)
    mentions: list[Mention] = Field(default_factory=list)


class Prediction(StrictModel):
    passage_id: str
    text_sha256: str
    status: Literal["complete", "failed"]
    mentions: list[Mention] = Field(default_factory=list)


def prepare_packet(root: Path, input_name: str, pilot_name: str, packet_id: str) -> Path:
    """Pick two passages per episode, not the first or last, without looking at model output."""
    root = root.resolve()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]{0,95}", packet_id):
        raise ValueError("invalid packet ID")
    parent = workspace_path(root, input_name)
    manifest = validate_run(parent)
    parent_digest = sha256((parent / "manifest.json").read_bytes())
    if manifest["stage"] != "segment":
        raise ValueError("annotation preparation requires a passage run")
    pilot_path = workspace_path(root, pilot_name)
    pilot_bytes = pilot_path.read_bytes()
    pilot = json.loads(pilot_bytes)
    if pilot.get("split") not in {"development", "test"}:
        raise ValueError("pilot must declare development or test split")
    entries = {e["source_path"]: e for e in pilot["episodes"]}
    if len(entries) != len(pilot["episodes"]):
        raise ValueError("duplicate pilot source paths")
    source = parent.parent / manifest["parent_run_id"]
    episodes = {e["episode_id"]: e for e in read_jsonl(source / "episodes.jsonl")}
    versions = {t["episode_id"]: t for t in read_jsonl(source / "transcripts.jsonl")}
    by_episode = defaultdict(list)
    for p in read_jsonl(parent / "passages.jsonl"):
        by_episode[p["episode_id"]].append(p)
    exclusions = read_jsonl(parent / "exclusions.jsonl")
    samples = []
    for eid, episode in sorted(episodes.items()):
        entry = entries.get(episode["source_path"])
        if entry is None or entry["source_sha256"] != versions[eid]["source_sha256"]:
            raise ValueError("input episode is absent from pilot or source hash differs")
        passages = sorted(by_episode[eid], key=lambda p: p["index"])
        # Skip the first and last passages, they are often intro and credits. The reviewer
        # still has to check that the chosen passages have real content.
        candidates = [p for p in passages[1:-1] if p["text"].strip() and not any(
            e["transcript_version_id"] == p["transcript_version_id"] and
            e["char_start"] < p["char_end"] and e["char_end"] > p["char_start"] for e in exclusions)]
        if len(candidates) < 2:
            raise ValueError("episode needs at least two eligible internal passages")
        for index in sorted({len(candidates) // 3, 2 * len(candidates) // 3}):
            p = candidates[index]
            samples.append({**p, "title": episode["title"], "source_path": episode["source_path"],
                            "split": pilot["split"], "text_sha256": sha256(p["text"].encode())})
    output = root / "data" / "prototype1" / "evaluation" / packet_id
    if any(p.is_symlink() for p in (output, *output.parents) if p != root and p.is_relative_to(root)):
        raise ValueError("annotation output must not use symlinks")
    validate_run(parent)
    if pilot_path.read_bytes() != pilot_bytes or sha256((parent / "manifest.json").read_bytes()) != parent_digest:
        raise ValueError("pilot or parent changed during preparation")
    output.mkdir(parents=True, exist_ok=False)
    dump_jsonl(output / "sample.jsonl", samples)
    dump_jsonl(output / "annotations.jsonl", [Annotation(passage_id=p["passage_id"], text_sha256=p["text_sha256"]) for p in samples])
    lines = ["# Annotation review packet", "", "Unreviewed drafts. Edit annotations.jsonl using ANNOTATION_GUIDELINES.md.",
             "Offsets are zero-based Unicode characters within each passage; end is exclusive.", ""]
    for p in samples:
        fence = "`" * max(3, 1 + max((len(m[0]) for m in re.finditer(r"`+", p["text"])), default=0))
        lines += [f"## {p['title']} · passage {p['index'] + 1}", "", f"ID: `{p['passage_id']}`",
                  f"Transcript offset: {p['char_start']}; split: {p['split']}", "", fence + "text", p["text"], fence, ""]
    (output / "review.md").write_text("\n".join(lines), encoding="utf-8")
    dump_json(output / "manifest.json", {"schema_version": 1, "stage": "annotation_packet", "packet_id": packet_id,
        "split": pilot["split"], "parent_run": input_name, "parent_manifest_sha256": parent_digest,
        "pilot": pilot_name, "pilot_sha256": sha256(pilot_bytes), "types": list(TYPES),
        "selection": "internal-passage-tertiles-v1; content and repeated identities await human review",
        "episode_count": len(episodes), "passage_count": len(samples),
        "sample_sha256": sha256((output / "sample.jsonl").read_bytes()),
        "code_sha256": sha256(Path(__file__).read_bytes())})
    return output


def load_sample(packet: Path) -> tuple[dict, dict]:
    manifest = json.loads((packet / "manifest.json").read_bytes())
    if manifest.get("stage") != "annotation_packet" or manifest.get("schema_version") != 1:
        raise ValueError("unsupported annotation packet")
    if manifest.get("split") not in {"development", "test"} or manifest.get("types") != list(TYPES):
        raise ValueError("invalid packet split or type inventory")
    if sha256((packet / "sample.jsonl").read_bytes()) != manifest["sample_sha256"]:
        raise ValueError("annotation sample checksum mismatch")
    rows = read_jsonl(packet / "sample.jsonl")
    samples = {r["passage_id"]: r for r in rows}
    if not rows or len(samples) != len(rows) or len(rows) != manifest["passage_count"]:
        raise ValueError("empty, duplicate or incomplete sample")
    if len({r["episode_id"] for r in rows}) != manifest["episode_count"]:
        raise ValueError("sample episode count differs")
    for r in rows:
        if r["split"] != manifest["split"] or sha256(r["text"].encode()) != r["text_sha256"]:
            raise ValueError("sample split or text hash differs")
    return manifest, samples


def checked_mentions(record: Annotation | Prediction, sample: dict) -> set[tuple]:
    if record.text_sha256 != sample["text_sha256"]:
        raise ValueError("annotation/prediction text hash differs from sample")
    keys = set()
    for m in record.mentions:
        if sample["text"][m.start:m.end] != m.text:
            raise ValueError("mention does not reproduce literal passage slice")
        key = (m.start, m.end, m.type)
        if key in keys:
            raise ValueError("duplicate mention; deduplicate predictions before scoring")
        if any(m.start < end and m.end > start for start, end, _ in keys):
            raise ValueError("overlapping mentions violate flat annotation contract")
        keys.add(key)
    return keys


def metrics(tp: int, fp: int, fn: int) -> dict:
    return {"tp": tp, "fp": fp, "fn": fn, "support": tp + fn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None}


def score_records(packet: Path, gold_rows: list[dict], prediction_rows: list[dict], split: str) -> dict:
    manifest, samples = load_sample(packet)
    if split != manifest["split"]:
        raise ValueError("requested split differs from packet")
    gold = [Annotation.model_validate(r) for r in gold_rows]
    predictions = [Prediction.model_validate(r) for r in prediction_rows]
    for rows in (gold, predictions):
        if len(rows) != len(samples) or {r.passage_id for r in rows} != set(samples):
            raise ValueError("require exactly one record for every sampled passage")
    if any(g.status != "reviewed" or not g.exhaustive or not (g.reviewer or "").strip() or g.unresolved_notes for g in gold):
        raise ValueError("gold requires exhaustive human review and resolved uncertainties")
    if any(p.status != "complete" for p in predictions):
        raise ValueError("failed predictions must not be treated as empty results")
    predicted = {p.passage_id: p for p in predictions}
    counts = {label: [0, 0, 0] for label in TYPES}
    errors = []
    for g in gold:
        expected = checked_mentions(g, samples[g.passage_id])
        actual = checked_mentions(predicted[g.passage_id], samples[g.passage_id])
        for i, keys in enumerate((expected & actual, actual - expected, expected - actual)):
            for start, end, label in sorted(keys):
                counts[label][i] += 1
                if i:
                    errors.append({"passage_id": g.passage_id, "kind": "fp" if i == 1 else "fn",
                                   "start": start, "end": end, "type": label})
    by_type = {label: metrics(*values) for label, values in counts.items()}
    supported = [m["f1"] for m in by_type.values() if m["support"]]
    return {"schema_version": 1, "metric": "exact-flat-typed-span-v1", "split": split,
            "passage_count": len(samples), "micro": metrics(*(sum(c[i] for c in counts.values()) for i in range(3))),
            "per_type": by_type, "macro_f1_gold_supported_types": sum(supported) / len(supported) if supported else None,
            "macro_types": [label for label, m in by_type.items() if m["support"]], "errors": errors,
            "sample_sha256": manifest["sample_sha256"],
            "identity_scoring": "not implemented; identity_id is ignored by NER scorer"}


def score(packet: Path, gold_path: Path, predictions_path: Path, split: str) -> dict:
    result = score_records(packet, read_jsonl(gold_path), read_jsonl(predictions_path), split)
    result.update({"gold_sha256": sha256(gold_path.read_bytes()),
                   "predictions_sha256": sha256(predictions_path.read_bytes())})
    return result
