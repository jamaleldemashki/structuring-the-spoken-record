"""CSV worksheets for reviewing a packet in a spreadsheet, without showing model decisions."""

import csv
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from spoken_record.reference_evaluation import load_packet, validate_reference_packet
from spoken_record.reference_evaluation_models import (
    CoreferenceEvaluationAnnotation,
    CoreferenceEvaluationSample,
    IdentityEvaluationAnnotation,
    IdentityEvaluationSample,
)


WORKSHEET_FIELDS = (
    "kind", "number", "item_id", "prompt", "episode", "candidate_targets",
    "allowed_decisions", "decision", "target_corpus_entity_id", "notes",
)
IMMUTABLE_WORKSHEET_FIELDS = WORKSHEET_FIELDS[:7]


def _replace_jsonl(path: Path, rows: list) -> None:
    """Write the JSONL to a temp file next to the target, then replace the target."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=path.parent,
                                         prefix=f".{path.name}.", delete=False) as handle:
            temporary = Path(handle.name)
            for row in rows:
                handle.write(json.dumps(row.model_dump(mode="json"), ensure_ascii=False,
                                        sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def worksheet_rows(identity_samples: list[IdentityEvaluationSample],
                   coreference_samples: list[CoreferenceEvaluationSample],
                   identity_annotations: list[IdentityEvaluationAnnotation],
                   coreference_annotations: list[CoreferenceEvaluationAnnotation]) -> list[dict[str, str]]:
    """Worksheet rows without model decisions or scores."""
    identity_gold = {row.candidate_id: row for row in identity_annotations}
    coreference_gold = {row.reference_id: row for row in coreference_annotations}
    rows = []
    for index, sample in enumerate(identity_samples, 1):
        annotation = identity_gold[sample.candidate_id]
        rows.append({"kind": "identity", "number": f"I{index}", "item_id": sample.candidate_id,
            "prompt": f"{sample.left_label} ↔ {sample.right_label}",
            "episode": " | ".join(sample.episode_ids), "candidate_targets": "",
            "allowed_decisions": "same | different | uncertain",
            "decision": annotation.gold_decision or "", "target_corpus_entity_id": "",
            "notes": annotation.notes})
    for index, sample in enumerate(coreference_samples, 1):
        annotation = coreference_gold[sample.reference_id]
        targets = " | ".join(f"{row.corpus_entity_id} — {row.canonical_label}"
                             for row in sample.candidate_targets)
        rows.append({"kind": "coreference", "number": f"C{index}", "item_id": sample.reference_id,
            "prompt": f"{sample.text!r} at [{sample.char_start}, {sample.char_end})",
            "episode": sample.episode_title, "candidate_targets": targets,
            "allowed_decisions": "link | no_link | ambiguous | out_of_scope",
            "decision": annotation.gold_decision or "",
            "target_corpus_entity_id": annotation.gold_corpus_entity_id or "",
            "notes": annotation.notes})
    return rows


def export_worksheet(packet: Path, worksheet: Path) -> Path:
    """Write a new worksheet. Existing review work is never overwritten."""
    validate_reference_packet(packet)
    identity, coreference, identity_gold, coreference_gold = load_packet(packet)
    if worksheet.exists():
        raise FileExistsError(worksheet)
    worksheet.parent.mkdir(parents=True, exist_ok=True)
    with worksheet.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=WORKSHEET_FIELDS)
        writer.writeheader()
        writer.writerows(worksheet_rows(identity, coreference, identity_gold, coreference_gold))
    return worksheet


def read_worksheet(path: Path) -> list[dict[str, str]]:
    """Read a worksheet. The header must match exactly and item IDs must be unique."""
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != WORKSHEET_FIELDS:
            raise ValueError("reference worksheet header changed")
        rows = list(reader)
    identifiers = [row["item_id"] for row in rows]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("reference worksheet contains duplicate item IDs")
    return rows


def reviewed_annotations(identity_samples: list[IdentityEvaluationSample],
                         coreference_samples: list[CoreferenceEvaluationSample],
                         identity_annotations: list[IdentityEvaluationAnnotation],
                         coreference_annotations: list[CoreferenceEvaluationAnnotation],
                         worksheet: list[dict[str, str]], reviewer: str,
                         reviewed_at: str) -> tuple[list[IdentityEvaluationAnnotation],
                                                   list[CoreferenceEvaluationAnnotation]]:
    """Check the worksheet decisions and copy filled-in rows into the annotations."""
    expected = worksheet_rows(identity_samples, coreference_samples,
                              identity_annotations, coreference_annotations)
    expected_by_id = {row["item_id"]: row for row in expected}
    if {row["item_id"] for row in worksheet} != set(expected_by_id) or len(worksheet) != len(expected):
        raise ValueError("reference worksheet does not cover the packet exactly once")
    identity_by_id = {row.candidate_id: row for row in identity_annotations}
    coreference_by_id = {row.reference_id: row for row in coreference_annotations}
    coreference_sample_by_id = {row.reference_id: row for row in coreference_samples}
    for row in worksheet:
        expected_row = expected_by_id[row["item_id"]]
        if any(row[field] != expected_row[field] for field in IMMUTABLE_WORKSHEET_FIELDS):
            raise ValueError(f"immutable worksheet columns changed for {row['item_id']}")
        decision, target, notes = (row["decision"].strip(),
                                   row["target_corpus_entity_id"].strip(), row["notes"])
        if not decision:
            if target:
                raise ValueError(f"blank decision carries a target for {row['item_id']}")
            continue
        if row["kind"] == "identity":
            if decision not in {"same", "different", "uncertain"} or target:
                raise ValueError(f"invalid identity judgment for {row['item_id']}")
            identity_by_id[row["item_id"]] = IdentityEvaluationAnnotation(
                candidate_id=row["item_id"], status="reviewed", gold_decision=decision,
                reviewer=reviewer, reviewed_at=reviewed_at, notes=notes)
        else:
            if decision not in {"link", "no_link", "ambiguous", "out_of_scope"}:
                raise ValueError(f"invalid coreference judgment for {row['item_id']}")
            available = {value.corpus_entity_id
                         for value in coreference_sample_by_id[row["item_id"]].candidate_targets}
            if (decision == "link") != bool(target) or (target and target not in available):
                raise ValueError(f"invalid coreference target for {row['item_id']}")
            coreference_by_id[row["item_id"]] = CoreferenceEvaluationAnnotation(
                reference_id=row["item_id"], status="reviewed", gold_decision=decision,
                gold_corpus_entity_id=target or None, reviewer=reviewer,
                reviewed_at=reviewed_at, notes=notes)
    # Keep the original sample order even if the spreadsheet was sorted.
    return ([identity_by_id[row.candidate_id] for row in identity_samples],
            [coreference_by_id[row.reference_id] for row in coreference_samples])


def import_worksheet(packet: Path, worksheet: Path, reviewer: str,
                     reviewed_at: str | None = None) -> dict:
    """Import worksheet decisions into both annotation JSONL files."""
    validate_reference_packet(packet)
    identity, coreference, identity_gold, coreference_gold = load_packet(packet)
    reviewer = reviewer.strip()
    if not reviewer:
        raise ValueError("reviewer must be nonempty")
    timestamp = reviewed_at or datetime.now(timezone.utc).isoformat()
    new_identity, new_coreference = reviewed_annotations(
        identity, coreference, identity_gold, coreference_gold, read_worksheet(worksheet),
        reviewer, timestamp)
    identity_path, coreference_path = (packet / "identity_annotations.jsonl",
                                       packet / "coreference_annotations.jsonl")
    old_identity, old_coreference = identity_path.read_bytes(), coreference_path.read_bytes()
    try:
        _replace_jsonl(identity_path, new_identity)
        _replace_jsonl(coreference_path, new_coreference)
        validate_reference_packet(packet)
    except Exception:
        # If writing or the final validation fails, put both files back as they were.
        identity_path.write_bytes(old_identity)
        coreference_path.write_bytes(old_coreference)
        raise
    return review_status(packet)


def review_status(packet: Path) -> dict:
    """Completion counts. Unreviewed rows are not counted as results."""
    validate_reference_packet(packet)
    _, _, identity, coreference = load_packet(packet)
    return {"identity_reviewed": sum(row.status == "reviewed" for row in identity),
        "identity_total": len(identity),
        "coreference_reviewed": sum(row.status == "reviewed" for row in coreference),
        "coreference_total": len(coreference),
        "complete": all(row.status == "reviewed" for row in identity + coreference)}
