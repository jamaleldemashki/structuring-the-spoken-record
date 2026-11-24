import type { Life } from "./types.ts";

export function formatTime(seconds: number | null): string {
  if (seconds === null) return "--:--";
  const s = Math.max(0, Math.round(seconds));
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), r = s % 60;
  const mm = String(m).padStart(2, "0"), ss = String(r).padStart(2, "0");
  return h ? `${h}:${mm}:${ss}` : `${mm}:${ss}`;
}

export const formatYear = (y: number) => (y < 0 ? `${-y} BCE` : String(y));

/** Inclusive year range for drawing and the overlap filter. A single year if one end is unknown. */
export function lifeRange(life: Life, now = new Date().getFullYear()): [number, number] {
  const start = life.start ?? life.end!;
  const end = life.living ? now : life.end ?? start;
  return [start, end];
}

export function lifeLabel(life: Life | undefined): string | null {
  if (!life) return null;
  const c = life.approximate ? "c. " : "";
  const prefix = life.basis === "floruit" ? "active " : "";
  if (life.start !== null && life.living) return `born ${c}${formatYear(life.start)}`;
  if (life.start !== null && life.end !== null) return `${prefix}${c}${formatYear(life.start)} – ${formatYear(life.end)}`;
  if (life.start !== null) return `${life.basis === "floruit" ? "active from" : "born"} ${c}${formatYear(life.start)}`;
  return `${life.basis === "floruit" ? "active until" : "died"} ${c}${formatYear(life.end!)}`;
}

export interface Segment {
  text: string;
  mark: boolean;
  /** The topic this text names, when names are linked. */
  id?: string;
}

/**
 * A readable excerpt around one or more mentions, cut at word boundaries, with every
 * range inside the window marked. Ranges are passage-relative character offsets. Text inside one
 * of the `names` (in order, not overlapping) carries that name's id, so it can become a link.
 */
export function highlight(text: string, ranges: { start: number; end: number }[], radius = 120,
  names: { start: number; end: number; id: string }[] = []): Segment[] {
  const sorted = [...ranges].sort((x, y) => x.start - y.start);
  const first = sorted[0]?.start ?? 0, last = sorted.length ? Math.max(...sorted.map((r) => r.end)) : 0;
  let a = Math.max(0, first - radius), b = Math.min(text.length, last + radius);
  if (a > 0) a = text.indexOf(" ", a) + 1 || a;
  if (b < text.length) { const cut = text.lastIndexOf(" ", b); if (cut > last) b = cut; }
  const squash = (s: string) => s.replace(/\s+/g, " ");
  const out: Segment[] = [];
  // Adds a stretch of text, cut where a name begins or ends.
  const add = (from: number, to: number, mark: boolean) => {
    const piece = (x: number, y: number, id?: string) => {
      const part = text.slice(x, y);
      out.push({ text: mark ? part : squash(part), mark, ...(id ? { id } : {}) });
    };
    let at = from;
    for (const n of names) {
      if (n.end <= at || n.start >= to) continue;
      if (n.start > at) piece(at, n.start);
      piece(Math.max(at, n.start), Math.min(to, n.end), n.id);
      at = Math.min(to, n.end);
    }
    piece(at, to);
  };
  let pos = a;
  for (const r of sorted) {
    if (r.start < pos) continue;
    add(pos, r.start, false);
    add(r.start, r.end, true);
    pos = r.end;
  }
  add(pos, b, false);
  // The dots of a cut stay outside a name, so they don't become part of its link.
  if (a > 0) {
    if (out[0].id) out.unshift({ text: "…", mark: false });
    else out[0].text = "…" + out[0].text.trimStart();
  }
  if (b < text.length) {
    if (out.at(-1)!.id) out.push({ text: "…", mark: false });
    else out[out.length - 1].text = out[out.length - 1].text.trimEnd() + "…";
  }
  return out.filter((s) => s.text);
}

export function plural(n: number, word: string, many = `${word}s`): string {
  return `${n} ${n === 1 ? word : many}`;
}
