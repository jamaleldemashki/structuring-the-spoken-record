// Every count, node and edge in the app comes from the passages these filters leave
// visible, so each connection shown has evidence behind it that can be opened.
import type { CoEdge, Dataset, Edge, Entity, Episode, Index } from "./types.ts";
import { allTypes } from "./entityStyle.ts";
import { lifeRange } from "./format.ts";
import { drawnLinks, weighted } from "./strength.ts";

export interface Filters {
  /** Inclusive publication window in months, as "YYYY-MM". */
  pub: [string, string];
  /** Lifespan window. A null end is open, negative years are BCE. */
  lived: [number | null, number | null];
  /** "within" = born and died inside the window, "overlap" = alive at some point in it. */
  livedMatch: "within" | "overlap";
  /** Keep people without a known lifespan when a lifespan window is active. */
  undated: boolean;
  series: Set<string>;
  types: Set<string>;
  showRelations: boolean;
  showCoDiscussed: boolean;
  /** Share of each item's links that is drawn, 0 to 1 (strength.ts). */
  share: number;
  /** Draw the episodes and the publication axis. */
  episodes: boolean;
  /** Only these episodes, whatever the publication window says. Null for no such choice. */
  only: Set<string> | null;
}

const NO_SERIES = "";

/** First and last month with a published episode. */
export function monthRange(ds: Dataset): [string, string] {
  const months = ds.episodes.filter((e) => e.date).map((e) => e.date!.slice(0, 7)).sort();
  if (!months.length) {
    const now = new Date().toISOString().slice(0, 7);
    return [now, now];
  }
  return [months[0], months.at(-1)!];
}

export function allSeries(ds: Dataset): string[] {
  return [...new Set(ds.episodes.map((e) => e.series ?? NO_SERIES))].sort();
}

export function defaultFilters(ds: Dataset): Filters {
  return {
    pub: monthRange(ds),
    lived: [null, null],
    livedMatch: "within",
    undated: false,
    series: new Set(allSeries(ds)),
    types: new Set(allTypes().filter((t) => t.visibleByDefault).map((t) => t.type)),
    showRelations: true,
    showCoDiscussed: true,
    share: ds.site.co_discussed_share ?? 0.1,
    episodes: true,
    only: null,
  };
}

/** Episodes without a publication date stay visible, since they can't be placed in a window. */
export function episodeVisible(ep: Episode, f: Filters): boolean {
  if (f.only) return f.only.has(ep.id);
  if (!f.series.has(ep.series ?? NO_SERIES)) return false;
  if (!ep.date) return true;
  const month = ep.date.slice(0, 7);
  return month >= f.pub[0] && month <= f.pub[1];
}

export function visiblePassages(ds: Dataset, idx: Index, f: Filters): Set<string> {
  const out = new Set<string>();
  for (const p of ds.passages) if (episodeVisible(idx.episode.get(p.episode_id)!, f)) out.add(p.id);
  return out;
}

export const livedActive = (f: Filters) => f.lived[0] !== null || f.lived[1] !== null;

/**
 * Whether an entity of the archive's dated type (people, for most podcasts) matches the
 * lifespan window. Other types pass here, visibleEdges keeps them only next to a match.
 */
export function livedOk(e: Entity, f: Filters, datedType: string | null): boolean {
  const [from, to] = f.lived;
  if (e.type !== datedType || !livedActive(f)) return true;
  if (!e.life) return f.undated;
  const [start, end] = lifeRange(e.life);
  if (f.livedMatch === "within") return (from === null || start >= from) && (to === null || end <= to);
  return (from === null || end >= from) && (to === null || start <= to);
}

export function entityPassagesInView(e: Entity, idx: Index, vp: Set<string>): string[] {
  return [...(idx.entityPassages.get(e.id) ?? [])].filter((p) => vp.has(p)).sort();
}

export function entityVisible(e: Entity, idx: Index, vp: Set<string>, f: Filters): boolean {
  return f.types.has(e.type) && livedOk(e, f, idx.site.lifespan_type) && entityPassagesInView(e, idx, vp).length > 0;
}

export function evidenceInView(ed: Edge, vp: Set<string>): string[] {
  return ed.evidence.filter((p) => vp.has(p));
}

export function edgeVisible(ed: Edge, idx: Index, vp: Set<string>, f: Filters): boolean {
  if (ed.kind === "relation" ? !f.showRelations : !f.showCoDiscussed) return false;
  const a = idx.entity.get(ed.source), b = idx.entity.get(ed.target);
  if (!a || !b || !entityVisible(a, idx, vp, f) || !entityVisible(b, idx, vp, f)) return false;
  return evidenceInView(ed, vp).length > 0;
}

/** Entities the lifespan window selects. They are drawn even without a visible link. */
export function livedSelection(ds: Dataset, idx: Index, vp: Set<string>, f: Filters): string[] {
  if (!livedActive(f) || !idx.site.lifespan_type) return [];
  return ds.entities.filter((e) => e.type === idx.site.lifespan_type && entityVisible(e, idx, vp, f)).map((e) => e.id);
}

/**
 * Every link inside the filters, before the percentage chooses among them. A lifespan window
 * selects dated entities (people), so other items (concepts, works, places) stay only when they
 * connect directly to one of the selected people.
 */
export function linksInView(ds: Dataset, idx: Index, vp: Set<string>, f: Filters): Edge[] {
  const shown = ds.edges.filter((ed) => edgeVisible(ed, idx, vp, f));
  const edges: Edge[] = [...shown.filter((ed) => ed.kind === "relation"),
    ...weighted(shown.filter((ed): ed is CoEdge => ed.kind === "co_discussed"), vp, idx.passageWeight)];
  if (!livedActive(f) || !idx.site.lifespan_type) return edges;
  const isPerson = (id: string) => idx.entity.get(id)!.type === idx.site.lifespan_type;
  const anchored = new Set<string>();
  for (const ed of edges) {
    if (isPerson(ed.source)) anchored.add(ed.target);
    if (isPerson(ed.target)) anchored.add(ed.source);
  }
  return edges.filter((ed) => [ed.source, ed.target].every((id) => isPerson(id) || anchored.has(id)));
}

/** The links drawn for the current percentage. */
export const visibleEdges = (ds: Dataset, idx: Index, vp: Set<string>, f: Filters): Edge[] =>
  drawnLinks(linksInView(ds, idx, vp, f), f.share);

/** Filters differing from the defaults, for the header badge. */
export function activeFilterCount(ds: Dataset, f: Filters): number {
  const d = defaultFilters(ds);
  let n = 0;
  if (f.pub[0] !== d.pub[0] || f.pub[1] !== d.pub[1]) n++;
  if (f.lived[0] !== null || f.lived[1] !== null) n++;
  if (f.series.size !== d.series.size) n++;
  if (!f.episodes || f.only) n++;
  if (f.types.size !== d.types.size || [...f.types].some((t) => !d.types.has(t))) n++;
  return n;
}
