// Following the audio in the transcript: which word is being spoken right now.
//
// Whisper gives times per segment (a phrase of a few seconds), not per word. Inside a
// segment I spread its time evenly over its characters, so a long word takes longer than a
// short one. Over a few seconds this stays close to the speech, and it can't drift far
// because every segment starts again at its own time.
import type { Passage } from "./types.ts";

export interface TimedWord {
  /** Character range in the passage text. */
  start: number;
  end: number;
  /** Estimated time the word starts and ends. */
  t0: number;
  t1: number;
}

export function timedWords(p: Pick<Passage, "text" | "timing">): TimedWord[] {
  const text = p.text ?? "";
  const out: TimedWord[] = [];
  for (const [from, to, t0, t1] of p.timing ?? []) {
    // A segment can start before the passage or end after it. Its time is still spread over
    // its whole length, so only the part inside this passage is used.
    const at = (char: number) => t0 + ((char - from) / Math.max(1, to - from)) * (t1 - t0);
    const lo = Math.max(0, from), hi = Math.min(text.length, to);
    for (const w of text.slice(lo, hi).matchAll(/\S+/g)) {
      const start = lo + w.index!, end = start + w[0].length;
      out.push({ start, end, t0: at(start), t1: at(end + 1) });
    }
  }
  return out;
}

/** Index of the word spoken at `time`, or -1 before the first word and after the last. */
export function wordAt(words: TimedWord[], time: number): number {
  if (!words.length || time < words[0].t0 || time >= words[words.length - 1].t1) return -1;
  let lo = 0, hi = words.length - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (words[mid].t0 <= time) lo = mid;
    else hi = mid - 1;
  }
  return lo;
}

/**
 * Which passage is being spoken at `time`: the last one whose first word has started, so a short
 * pause between two passages still counts as the earlier one. -1 before the first. `starts` holds
 * the time of each passage's first word, in order.
 */
export function passageAt(starts: number[], time: number): number {
  let lo = -1, hi = starts.length - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (starts[mid] <= time) lo = mid;
    else hi = mid - 1;
  }
  return lo;
}

export interface Piece {
  text: string;
  /** Index of the word this piece belongs to, -1 for the spaces between words. */
  w: number;
}

export interface Run {
  /** The entity mentioned here, if any. */
  id?: string;
  pieces: Piece[];
}

/**
 * The passage text cut at mentions and words. A mention becomes one run (one button in the
 * reader) and keeps its words inside, so a word that is part of a name still lights up.
 */
export function runs(text: string, mentions: { start: number; end: number; id: string }[], words: TimedWord[]): Run[] {
  const cuts = new Set([0, text.length]);
  const kept: typeof mentions = [];
  let pos = 0;
  for (const m of mentions) {
    if (m.start < pos) continue;
    kept.push(m);
    cuts.add(m.start).add(m.end);
    pos = m.end;
  }
  for (const w of words) cuts.add(w.start).add(w.end);
  const points = [...cuts].sort((a, b) => a - b);
  const out: Run[] = [];
  let m = 0, w = 0;
  for (let i = 0; i + 1 < points.length; i++) {
    const [a, b] = [points[i], points[i + 1]];
    while (m < kept.length && kept[m].end <= a) m++;
    while (w < words.length && words[w].end <= a) w++;
    const id = m < kept.length && kept[m].start <= a ? kept[m].id : undefined;
    const piece = { text: text.slice(a, b), w: w < words.length && words[w].start <= a ? w : -1 };
    const last = out[out.length - 1];
    // Pieces of the same mention, or plain text next to plain text, stay in one run.
    if (last && last.id === id) last.pieces.push(piece);
    else out.push({ id, pieces: [piece] });
  }
  return out;
}

/**
 * Keeps the spoken word in view while the reader is following along: if the last word was
 * on screen and the new one is not, it scrolls. Someone who scrolled away to read something
 * else is left alone. Each passage on screen keeps its own follower.
 */
export function follower() {
  let wasVisible = false;
  const visible = (el: Element) => {
    const r = el.getBoundingClientRect();
    // Leave room for the header at the top and the listening bar at the bottom.
    return r.top >= 60 && r.bottom <= window.innerHeight - 90;
  };
  return (el: Element | null) => {
    if (!el) return;
    if (!visible(el) && wasVisible) el.scrollIntoView({ block: "center", behavior: "smooth" });
    else wasVisible = visible(el);
  };
}
