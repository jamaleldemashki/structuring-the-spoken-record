#!/usr/bin/env python3
"""Corpus audit for the History of Philosophy transcripts.

Reads the supervisor-provided HoP/ folder (read only, nothing is written there) and writes:

  data/audit/episode_audit.jsonl    one row per usable transcript
  data/audit/quarantine.jsonl       transcripts that need a decision first
  data/audit/notes_assessment.json  summary of HoP/03_notes/
  data/audit/run_manifest.json      settings and counts of the run

The HoP adapter uses episode_audit.jsonl to join transcripts to their metadata:

  tier 1  transcript stem equals the metadata `filename` stem       (542)
  tier 2  (show, episode number, suffix) taken from `nav_title`     (171)
  tier 3  no match, goes to the quarantine with a reason code       ( 11)

Tier 2 is needed because only 542 of the 756 metadata records have a `filename`,
the other 214 are navigation-only entries.

Usage: python3 scripts/corpus_audit.py [--root .] [--out data/audit]
Standard library only, so nothing needs to be installed.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone

SCHEMA_VERSION = 1
AUDIT_TOOL_VERSION = "corpus_audit/1.0.0"

# Thresholds (saved in the manifest so the results can be interpreted later)
SHORT_TRANSCRIPT_WORDS = 100      # below this, transcript is unusable as content
HIGH_NO_SPEECH_PROB = 0.5         # per-segment threshold for "probably not speech"
HIGH_NO_SPEECH_FRAC = 0.30        # episode-level fraction that triggers a flag
LOW_CONFIDENCE_LOGPROB = -1.00    # mean avg_logprob below this is flagged
BOILERPLATE_MIN_EPISODES = 30     # a normalised segment seen in >= N episodes
BOILERPLATE_HEAD_SEGMENTS = 8     # how far into an episode we look for the intro

# Show group mapping. Episode numbers restart per group, so the group is part of
# the join key. Series not listed here belong to the main HoP numbering, which is
# continuous from Classical Greek through to Early Modern.
SERIES_TO_GROUP = {"Africana": "HAP", "India": "HPI", "China": "HPC"}
KNOWN_SHOWS = ("HoP", "HAP", "HPI", "HPC", "GPW")


# Helpers
def _mean(values):
    """Arithmetic mean, written out so the script also runs on Python 3.7."""
    values = list(values)
    return sum(values) / len(values) if values else None


def slugify(stem: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", stem).strip("_").lower()
    return re.sub(r"_+", "_", s)


def episode_id_for(stem: str) -> str:
    return f"ep_{slugify(stem)}"


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def group_of_series(series: str | None) -> str:
    return SERIES_TO_GROUP.get(series or "", "HOP")


def nav_number(nav_title: str | None):
    """Leading number and optional letter, for example '300b. ...' gives (300, 'b')."""
    if not nav_title:
        return None
    m = re.match(r"^\s*(\d+)\s*([a-z]?)", nav_title)
    if not m:
        return None
    return int(m.group(1)), (m.group(2) or "")


def transcript_key(stem: str):
    """(SHOW, number, suffix) parsed from a transcript filename stem.

    Filenames are inconsistent (`08HPI08-...`, `HoP249`, `OfhhekHoP_095_...`),
    so this searches rather than anchoring at the start.
    """
    m = re.search(
        r"(%s)[_\-\s]*0*(\d+)\s*([a-z]?)" % "|".join(KNOWN_SHOWS), stem, re.I
    )
    if not m:
        return None
    return m.group(1).upper(), int(m.group(2)), (m.group(3) or "").lower()


def normalise_segment(text: str) -> str:
    """Strong normalisation, only used to hash repeated boilerplate."""
    t = text.lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def word_count(text: str) -> int:
    return len(text.split())


# Loading
def load_metadata(root: str):
    path = os.path.join(root, "HoP", "00_infos", "hop_episodes_cache.jsonl")
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records, path


def build_indexes(meta):
    by_filename_stem, by_group_number = {}, defaultdict(list)
    for r in meta:
        fn = r.get("filename")
        if fn:
            by_filename_stem[os.path.splitext(fn)[0]] = r
        nn = nav_number(r.get("nav_title"))
        if nn:
            by_group_number[(group_of_series(r.get("series")), nn[0], nn[1])].append(r)
    return by_filename_stem, by_group_number


def join_metadata(stem, by_filename_stem, by_group_number):
    """Returns (record_or_None, tier, note)."""
    if stem in by_filename_stem:
        return by_filename_stem[stem], "tier1_filename", ""
    key = transcript_key(stem)
    if key is None:
        return None, "unresolved", "no show/number parsable from filename"
    cands = by_group_number.get(key, [])
    if len(cands) == 1:
        return cands[0], "tier2_group_number", f"matched {key}"
    if len(cands) > 1:
        return None, "unresolved", f"ambiguous {key} ({len(cands)} candidates)"
    return None, "unresolved", f"no metadata record for {key}"


# Pass 1: read transcripts, collect segment hashes for boilerplate detection
def read_transcript(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def collect_segment_hashes(transcript_paths):
    """Map each segment hash to the episodes that contain it."""
    hash_to_eps = defaultdict(set)
    for p in transcript_paths:
        stem = os.path.basename(p)[:-5]
        try:
            doc = read_transcript(p)
        except Exception:
            continue
        for seg in doc.get("segments", []) or []:
            norm = normalise_segment(seg.get("text", "") or "")
            if len(norm.split()) < 6:      # too short to be a reliable signal
                continue
            hash_to_eps[hashlib.sha1(norm.encode()).hexdigest()].add(stem)
    return hash_to_eps


# Pass 2: per-episode audit
def audit_episode(path, meta_rec, tier, note, boiler_hashes, run_id):
    stem = os.path.basename(path)[:-5]
    doc = read_transcript(path)
    segments = doc.get("segments", []) or []
    full_text = doc.get("text", "") or ""

    logprobs = [s["avg_logprob"] for s in segments if isinstance(s.get("avg_logprob"), (int, float))]
    nsp = [s["no_speech_prob"] for s in segments if isinstance(s.get("no_speech_prob"), (int, float))]
    duration = max((s.get("end") or 0) for s in segments) if segments else 0.0
    n_words = word_count(full_text)

    # boilerplate: which of the first N segments are corpus-wide repeats?
    head_hits = []
    for seg in segments[:BOILERPLATE_HEAD_SEGMENTS]:
        norm = normalise_segment(seg.get("text", "") or "")
        if len(norm.split()) < 6:
            continue
        h = hashlib.sha1(norm.encode()).hexdigest()
        if h in boiler_hashes:
            head_hits.append(seg.get("id", 0))
    # any repeated segment anywhere in the episode
    all_hits = 0
    for seg in segments:
        norm = normalise_segment(seg.get("text", "") or "")
        if len(norm.split()) < 6:
            continue
        if hashlib.sha1(norm.encode()).hexdigest() in boiler_hashes:
            all_hits += 1

    letters = [c for c in full_text if c.isalpha()]
    caps_ratio = (sum(1 for c in letters if c.isupper()) / len(letters)) if letters else 0.0
    punct_density = (sum(full_text.count(c) for c in ".,;:!?") / n_words) if n_words else 0.0

    flags = []
    if meta_rec is None:
        flags.append("no_metadata_match")
    if n_words < SHORT_TRANSCRIPT_WORDS:
        flags.append("very_short_transcript")
    if not segments:
        flags.append("no_segments")
    if logprobs and _mean(logprobs) < LOW_CONFIDENCE_LOGPROB:
        flags.append("low_asr_confidence")
    if nsp and (sum(1 for v in nsp if v > HIGH_NO_SPEECH_PROB) / len(nsp)) > HIGH_NO_SPEECH_FRAC:
        flags.append("high_no_speech")
    if transcript_key(stem) is None:
        flags.append("unparsable_filename")
    if not head_hits and segments:
        flags.append("no_boilerplate_head")

    show = (transcript_key(stem) or (None,))[0]
    return {
        "schema_version": SCHEMA_VERSION,
        "episode_id": episode_id_for(stem),
        "source_stem": stem,
        "source_path": os.path.relpath(path),
        "source_sha256": sha256_of(path),
        "metadata_matched": meta_rec is not None,
        "join_tier": tier,
        "join_note": note,
        "sub_show": show,
        "content_era": (meta_rec or {}).get("series"),
        "title": (meta_rec or {}).get("title"),
        "nav_title": (meta_rec or {}).get("nav_title"),
        "pub_date": (meta_rec or {}).get("pub_date"),
        "audio_url": (meta_rec or {}).get("audio_url"),
        "metadata_link": (meta_rec or {}).get("link"),
        "language": doc.get("language"),
        "duration_s": round(duration, 2),
        "n_segments": len(segments),
        "n_words": n_words,
        "mean_words_per_segment": round(n_words / len(segments), 2) if segments else 0.0,
        "mean_avg_logprob": round(_mean(logprobs), 4) if logprobs else None,
        "min_avg_logprob": round(min(logprobs), 4) if logprobs else None,
        "frac_high_no_speech": round(sum(1 for v in nsp if v > HIGH_NO_SPEECH_PROB) / len(nsp), 4) if nsp else None,
        "punct_density": round(punct_density, 4),
        "caps_ratio": round(caps_ratio, 4),
        "has_boilerplate_head": bool(head_hits),
        "boilerplate_head_segments": head_hits,
        "n_repeated_segments": all_hits,
        "anomaly_flags": flags,
        "audit_run_id": run_id,
    }


# 03_notes assessment
def assess_notes(root):
    paths = sorted(glob.glob(os.path.join(root, "HoP", "03_notes", "*.md")))
    ent_re = re.compile(r"^named_entities::\s*(.*)$", re.M)
    bow_re = re.compile(r"^bag_of_words::\s*(.*)$", re.M)
    rel_re = re.compile(r"^Related Notes::\s*(.*)$", re.M)
    link_re = re.compile(r"\[\[([^\]|#]+)")

    total = len(paths)
    with_ents = with_bow = with_rel = 0
    ent_counts, all_entities = [], Counter()
    for p in paths:
        try:
            txt = open(p, encoding="utf-8").read()
        except Exception:
            continue
        m = ent_re.search(txt)
        if m and m.group(1).strip():
            with_ents += 1
            ents = [e.strip() for e in link_re.findall(m.group(1))]
            ent_counts.append(len(ents))
            all_entities.update(ents)
        if bow_re.search(txt):
            with_bow += 1
        if rel_re.search(txt):
            with_rel += 1

    return {
        "n_notes": total,
        "with_named_entities": with_ents,
        "with_bag_of_words": with_bow,
        "with_related_notes": with_rel,
        "mean_entities_per_note": round(_mean(ent_counts), 2) if ent_counts else 0.0,
        "median_entities_per_note": statistics.median(ent_counts) if ent_counts else 0,
        "distinct_entity_strings": len(all_entities),
        "top_entities": all_entities.most_common(30),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".", help="repository root (default: .)")
    ap.add_argument("--out", default=os.path.join("data", "audit"), help="output directory")
    args = ap.parse_args()

    root, outdir = args.root, os.path.join(args.root, args.out)
    os.makedirs(outdir, exist_ok=True)

    started = datetime.now(timezone.utc)
    run_id = f"run_s01_{started.strftime('%Y%m%dT%H%M%SZ')}"

    meta, meta_path = load_metadata(root)
    by_fn, by_gn = build_indexes(meta)
    paths = sorted(glob.glob(os.path.join(root, "HoP", "01_whisper", "*.json")))
    print(f"[1/4] {len(paths)} transcripts, {len(meta)} metadata records", file=sys.stderr)

    print("[2/4] scanning segments for cross-episode repeats ...", file=sys.stderr)
    hash_to_eps = collect_segment_hashes(paths)
    boiler_hashes = {h for h, eps in hash_to_eps.items() if len(eps) >= BOILERPLATE_MIN_EPISODES}
    print(f"      {len(boiler_hashes)} boilerplate segment hashes "
          f"(seen in >= {BOILERPLATE_MIN_EPISODES} episodes)", file=sys.stderr)

    print("[3/4] auditing episodes ...", file=sys.stderr)
    audits, quarantine = [], []
    for p in paths:
        stem = os.path.basename(p)[:-5]
        rec, tier, note = join_metadata(stem, by_fn, by_gn)
        try:
            row = audit_episode(p, rec, tier, note, boiler_hashes, run_id)
        except Exception as exc:                                    # pragma: no cover
            quarantine.append({"schema_version": SCHEMA_VERSION, "episode_id": episode_id_for(stem),
                               "source_stem": stem, "source_path": os.path.relpath(p),
                               "reason_code": "unreadable_transcript", "detail": str(exc),
                               "audit_run_id": run_id})
            continue
        reasons = []
        if not row["metadata_matched"]:
            reasons.append("no_metadata_match")
        if "very_short_transcript" in row["anomaly_flags"]:
            reasons.append("very_short_transcript")
        if "no_segments" in row["anomaly_flags"]:
            reasons.append("no_segments")
        if reasons:
            quarantine.append({"schema_version": SCHEMA_VERSION, "episode_id": row["episode_id"],
                               "source_stem": stem, "source_path": row["source_path"],
                               "reason_code": ";".join(reasons), "detail": row["join_note"],
                               "sub_show": row["sub_show"], "n_words": row["n_words"],
                               "audit_run_id": run_id})
        else:
            audits.append(row)

    print("[4/4] assessing 03_notes ...", file=sys.stderr)
    notes = assess_notes(root)

    with open(os.path.join(outdir, "episode_audit.jsonl"), "w", encoding="utf-8") as fh:
        for r in audits:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(outdir, "quarantine.jsonl"), "w", encoding="utf-8") as fh:
        for r in quarantine:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(outdir, "notes_assessment.json"), "w", encoding="utf-8") as fh:
        json.dump(notes, fh, ensure_ascii=False, indent=2)

    tiers = Counter(r["join_tier"] for r in audits)
    manifest = {
        "audit_run_id": run_id,
        "tool_version": AUDIT_TOOL_VERSION,
        "schema_version": SCHEMA_VERSION,
        "started_utc": started.isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "inputs": {
            "transcripts_dir": os.path.relpath(os.path.join(root, "HoP", "01_whisper")),
            "n_transcripts": len(paths),
            "metadata_file": os.path.relpath(meta_path),
            "n_metadata_records": len(meta),
            "metadata_with_filename": sum(1 for r in meta if r.get("filename")),
        },
        "thresholds": {
            "SHORT_TRANSCRIPT_WORDS": SHORT_TRANSCRIPT_WORDS,
            "HIGH_NO_SPEECH_PROB": HIGH_NO_SPEECH_PROB,
            "HIGH_NO_SPEECH_FRAC": HIGH_NO_SPEECH_FRAC,
            "LOW_CONFIDENCE_LOGPROB": LOW_CONFIDENCE_LOGPROB,
            "BOILERPLATE_MIN_EPISODES": BOILERPLATE_MIN_EPISODES,
            "BOILERPLATE_HEAD_SEGMENTS": BOILERPLATE_HEAD_SEGMENTS,
        },
        "results": {
            "n_audited": len(audits),
            "n_quarantined": len(quarantine),
            "join_tiers": dict(tiers),
            "n_boilerplate_hashes": len(boiler_hashes),
            "total_words": sum(r["n_words"] for r in audits) + sum(q.get("n_words", 0) for q in quarantine),
            "total_hours": round((sum(r["duration_s"] for r in audits)) / 3600, 1),
        },
    }
    with open(os.path.join(outdir, "run_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)

    print(f"\naudited={len(audits)} quarantined={len(quarantine)} "
          f"total={len(audits)+len(quarantine)}", file=sys.stderr)
    print(f"join tiers: {dict(tiers)}", file=sys.stderr)
    print(f"wrote -> {outdir}", file=sys.stderr)


if __name__ == "__main__":
    main()
