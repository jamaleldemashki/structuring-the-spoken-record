// Episodes as nodes of the graph, and the publication axis they sit on.
//
// An episode node has one link to every item named in it, so an item named in five episodes
// has five such links. They come from the mentions the export already holds, and only from the
// passages inside the current filters, so nothing here needs its own data file.
//
// These links stay out of the edges used for exploring and for finding a chain between two
// items. "Both are named in HoP 250" would otherwise be the shortest answer to almost every
// question about how two items are connected.
import { entityVisible, type Filters } from "./filters.ts";
import { monthLabels } from "./timeAxis.ts";
import type { Dataset, Episode, Index } from "./types.ts";

export interface EpisodeLink {
  id: string;
  kind: "mentioned_in";
  /** The item. */
  source: string;
  /** The episode. */
  target: string;
  count: number;
  evidence: string[];
}

/** One link per visible item and episode that names it. General words get no link. */
export function episodeLinks(ds: Dataset, idx: Index, vp: Set<string>, f: Filters): EpisodeLink[] {
  const out: EpisodeLink[] = [];
  for (const e of ds.entities) {
    if (e.generic || !entityVisible(e, idx, vp, f)) continue;
    const inEpisode = new Map<string, string[]>();
    for (const pid of idx.entityPassages.get(e.id) ?? []) {
      if (!vp.has(pid)) continue;
      const ep = idx.passage.get(pid)!.episode_id;
      inEpisode.set(ep, [...(inEpisode.get(ep) ?? []), pid]);
    }
    for (const [ep, evidence] of inEpisode)
      out.push({ id: `${e.id}|${ep}`, kind: "mentioned_in", source: e.id, target: ep, count: evidence.length, evidence: evidence.sort() });
  }
  return out;
}

/** Per episode the items it names, the ones with most passages first. */
export function itemsOfEpisodes(links: EpisodeLink[], idx: Index): Map<string, string[]> {
  const out = new Map<string, EpisodeLink[]>();
  for (const l of links) out.set(l.target, [...(out.get(l.target) ?? []), l]);
  const mentions = (id: string) => idx.entity.get(id)?.mention_count ?? 0;
  return new Map([...out].map(([ep, list]) => [ep, list
    .sort((a, b) => b.count - a.count || mentions(b.source) - mentions(a.source) || a.source.localeCompare(b.source))
    .map((l) => l.source)]));
}

export interface PublicationScale {
  /** Graph position of a date ("YYYY-MM-DD"). */
  x(date: string): number;
  /** The date at a graph position, as "YYYY-MM-DD". */
  dateAt(x: number): string;
  left: number;
  right: number;
}

/**
 * How long the axis is in graph units. The whole map needs room for all its items, and an
 * area grows with the square of its side, so the axis follows the square root of their number.
 * The view zooms so that the axis fills the width of the screen, whatever its length. An
 * exploration holds only a few items, so there each episode gets a fixed room. A small archive still
 * gets a line about as long as a screen is wide. Fitting a few items to the width would otherwise
 * blow them up, and their picture would be higher than the screen.
 */
const SHORTEST_LINE = 1300;
export function axisLength(mode: "full" | "focused", episodes: number, items: number): number {
  return Math.max(SHORTEST_LINE, mode === "full" ? 215 * Math.sqrt(items) : Math.min(16000, 70 * Math.max(0, episodes - 1)));
}

/**
 * How episodes share the line when there are many: their size on screen and how many rows they take
 * turns on. A few episodes keep the full size on two rows. The more there are the smaller they get,
 * and when even the smallest size does not fit there are more rows, so that the markers stay apart.
 *
 * The spacing is worked out from the marker as it is drawn: its size times the size setting, plus its
 * border, plus a gap. The view is fitted after the layout and can end up a little further out than it
 * was laid out for (0.51 instead of 0.55 was seen), so there is some slack on top.
 */
export function episodeMarks(count: number, widthOnScreen: number, scale = 1): { size: number; rows: number; rowGap: number } {
  const FULL = 22, SMALLEST = 7, BORDER = 3, GAP = 3, SLACK = 1.15;
  const pitch = widthOnScreen / Math.max(1, count - 1);
  const apart = (size: number) => (size * scale + BORDER + GAP) * SLACK;
  // On two rows a marker has two places of the line to itself.
  const size = Math.max(SMALLEST, Math.min(FULL, (2 * pitch / SLACK - BORDER - GAP) / scale));
  const rows = Math.max(2, Math.ceil(apart(size) / pitch));
  return { size, rows, rowGap: apart(size) };
}

/** How long the line is for a chosen set of episodes with this many items around them. */
export const selectionLength = (items: number) => Math.max(SHORTEST_LINE, 300 * Math.sqrt(items));

/**
 * Old to new from left to right, linear in time, so a break in publishing shows as a gap. The
 * scale covers the whole archive and not what the filters leave, so an item does not move when
 * a filter changes. Null with fewer than two dates.
 */
export function publicationScale(episodes: Episode[], width: number): PublicationScale | null {
  const times = episodes.filter((e) => e.date).map((e) => Date.parse(e.date!)).sort((a, b) => a - b);
  const first = times[0], last = times.at(-1)!;
  if (times.length < 2 || first === last) return null;
  const perMs = width / (last - first);
  return {
    x: (date) => (Date.parse(date) - first) * perMs - width / 2,
    dateAt: (x) => new Date((x + width / 2) / perMs + first).toISOString().slice(0, 10),
    left: -width / 2,
    right: width / 2,
  };
}

/**
 * Positions for a chosen set of episodes along a line of this length: in order of publication
 * and evenly apart from one end to the other. One episode stands in the middle.
 */
export function selectionPositions(episodes: Episode[], length: number): { x(date: string): number } {
  const dates = [...new Set(episodes.map((e) => e.date).filter((d): d is string => !!d))].sort();
  const spacing = length / Math.max(1, dates.length - 1);
  const at = new Map(dates.map((d, i) => [d, (i - (dates.length - 1) / 2) * spacing]));
  return { x: (date) => at.get(date) ?? 0 };
}

/**
 * Where an item belongs on the axis: the average date of the episodes that name it, weighted by
 * passages. Someone discussed in June and again in May sits between the two.
 */
export function itemPositions(links: EpisodeLink[], idx: Index, scale: Pick<PublicationScale, "x">): Map<string, number> {
  const sums = new Map<string, [number, number]>();
  for (const l of links) {
    const date = idx.episode.get(l.target)?.date;
    if (!date) continue;
    const [sum, n] = sums.get(l.source) ?? [0, 0];
    sums.set(l.source, [sum + scale.x(date) * l.count, n + l.count]);
  }
  return new Map([...sums].map(([id, [sum, n]]) => [id, sum / n]));
}

/** The labels for the part of the axis on screen, between graph positions x0 and x1. */
export function axisTicks(scale: PublicationScale, x0: number, x1: number, pixels: number): { x: number; text: string }[] {
  const from = Math.max(x0, scale.left), to = Math.min(x1, scale.right);
  if (from >= to || pixels <= 0) return [];
  // Only the share of the screen the axis covers is available to its labels.
  const room = pixels * ((to - from) / (x1 - x0));
  return monthLabels(scale.dateAt(from).slice(0, 7), scale.dateAt(to).slice(0, 7), room)
    .map((t) => ({ x: scale.x(`${t.month}-01`), text: t.text }))
    .filter((t) => t.x >= from && t.x <= to);
}
