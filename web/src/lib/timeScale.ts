// The time axis of the 3D graph (a "space-time cube"): height stands for when a person
// lived, the network spreads out sideways.
//
// A podcast's people are not spread evenly over time. Most of ours lived in the 13th
// century, a few in antiquity. A linear axis would press the dense period into a thin
// band and leave long empty stretches. So the height mixes the linear position with the
// share of people born before that year (their cumulative distribution): dense periods
// get more room, empty ones less. The century planes are drawn at their true years, so
// the reader always sees where the axis is stretched.
import { formatYear, lifeRange } from "./format.ts";
import type { Edge, Entity } from "./types.ts";

export interface Tick {
  year: number;
  /** Height on the axis. */
  y: number;
  label: string;
}

export interface TimeScale {
  lo: number;
  hi: number;
  height: (year: number) => number;
  ticks: Tick[];
}

/** How much of the height follows the linear year, the rest follows the people's distribution. */
const LINEAR_SHARE = 0.4;
/** Roundest first: a year divisible by 1000 is labelled before one divisible by 500, and so on. */
const TICK_LEVELS = [1000, 500, 250, 100, 50, 25];

/** The middle of a person's lifetime. */
export function yearOf(e: Entity): number | null {
  if (!e.life) return null;
  const [a, b] = lifeRange(e.life);
  return (a + b) / 2;
}

/**
 * A scale from years to heights between 0 and `height`. Years outside `clamp` (for
 * example a biblical figure dated 4000 BCE) are drawn at the nearest end.
 */
export function timeScale(years: number[], height: number, clamp: [number, number] | null = null, minGap = 40): TimeScale {
  const limit = (y: number) => (clamp ? Math.min(clamp[1], Math.max(clamp[0], y)) : y);
  const inside = years.map(limit).sort((a, b) => a - b);
  if (!inside.length) return { lo: 0, hi: 1, height: () => height / 2, ticks: [] };
  const pad = Math.max(10, (inside.at(-1)! - inside[0]) * 0.03);
  const lo = inside[0] - pad, hi = inside.at(-1)! + pad;
  // Share of people at or before a year, interpolated so the scale has no steps.
  const share = (year: number) => {
    let below = 0;
    while (below < inside.length && inside[below] <= year) below++;
    if (below === 0) return 0;
    if (below === inside.length) return 1;
    const [a, b] = [inside[below - 1], inside[below]];
    return (below - 1 + (b > a ? (year - a) / (b - a) : 1)) / (inside.length - 1);
  };
  const at = (year: number) => {
    const y = Math.min(hi, Math.max(lo, limit(year)));
    return height * (LINEAR_SHARE * (y - lo) / (hi - lo) + (1 - LINEAR_SHARE) * share(y));
  };
  // Round years first, then finer ones wherever they still leave room to read each label.
  // So a stretched period gets labels every 50 years and an empty stretch only every 500.
  const ticks: Tick[] = [];
  for (const step of TICK_LEVELS) {
    for (let year = Math.ceil(lo / step) * step; year <= hi; year += step) {
      const y = at(year);
      if (ticks.every((t) => t.year !== year && Math.abs(t.y - y) >= minGap)) ticks.push({ year, y, label: formatYear(year) });
    }
  }
  ticks.sort((a, b) => a.year - b.year);
  return { lo, hi, height: at, ticks };
}

/**
 * Years for every item in the graph. People have their own; an item without dates
 * (a concept, a work) gets the average of its neighbours' years, worked out over a few
 * rounds so a work linked only to a concept still finds its time. Items with no dated
 * neighbour at all stay null. `inferred` lists the items placed this way.
 */
export function placeInTime(ids: string[], edges: Pick<Edge, "source" | "target">[], dated: Map<string, number>, rounds = 3) {
  const year = new Map<string, number>(dated);
  const inferred = new Set<string>();
  const neighbours = new Map<string, string[]>(ids.map((id) => [id, []]));
  for (const ed of edges) {
    neighbours.get(ed.source)?.push(ed.target);
    neighbours.get(ed.target)?.push(ed.source);
  }
  for (let r = 0; r < rounds; r++) {
    const found = new Map<string, number>();
    for (const id of ids) {
      if (year.has(id)) continue;
      const known = neighbours.get(id)!.filter((n) => year.has(n)).map((n) => year.get(n)!);
      if (known.length) found.set(id, known.reduce((a, b) => a + b, 0) / known.length);
    }
    if (!found.size) break;
    for (const [id, y] of found) { year.set(id, y); inferred.add(id); }
  }
  return { year: new Map(ids.map((id) => [id, year.get(id) ?? null])), inferred };
}
