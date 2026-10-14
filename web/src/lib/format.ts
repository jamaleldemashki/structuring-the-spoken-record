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
}

/**
 * A readable excerpt around one or more mentions, cut at word boundaries, with every
 * range inside the window marked. Ranges are passage-relative character offsets.
 */
export function highlight(text: string, ranges: { start: number; end: number }[], radius = 120): Segment[] {
  const sorted = [...ranges].sort((x, y) => x.start - y.start);
  const first = sorted[0]?.start ?? 0, last = sorted.length ? Math.max(...sorted.map((r) => r.end)) : 0;
  let a = Math.max(0, first - radius), b = Math.min(text.length, last + radius);
  if (a > 0) a = text.indexOf(" ", a) + 1 || a;
  if (b < text.length) { const cut = text.lastIndexOf(" ", b); if (cut > last) b = cut; }
  const squash = (s: string) => s.replace(/\s+/g, " ");
  const out: Segment[] = [];
  let pos = a;
  for (const r of sorted) {
    if (r.start < pos) continue;
    out.push({ text: squash(text.slice(pos, r.start)), mark: false }, { text: text.slice(r.start, r.end), mark: true });
    pos = r.end;
  }
  out.push({ text: squash(text.slice(pos, b)), mark: false });
  if (a > 0) out[0].text = "…" + out[0].text.trimStart();
  if (b < text.length) out[out.length - 1].text = out[out.length - 1].text.trimEnd() + "…";
  return out.filter((s) => s.text);
}

export function plural(n: number, word: string, many = `${word}s`): string {
  return `${n} ${n === 1 ? word : many}`;
}
