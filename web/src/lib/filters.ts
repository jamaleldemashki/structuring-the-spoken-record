// Every count, node and edge in the app comes from the passages these filters leave
// visible, so each connection shown has evidence behind it that can be opened.
import type { Dataset, Edge, Entity, Episode, Index } from "./types.ts";
import { allTypes } from "./entityStyle.ts";
import { lifeRange } from "./format.ts";

export interface Filters {
  /** Inclusive publication-year window. */
  pub: [number, number];
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
  /** Minimum shared passages, counted within the current filters, for a "discussed together" edge. */
  minShared: number;
}

const NO_SERIES = "";

export function yearRange(ds: Dataset): [number, number] {
  const years = ds.episodes.filter((e) => e.date).map((e) => Number(e.date!.slice(0, 4)));
  if (!years.length) {
    const now = new Date().getFullYear();
    return [now, now];
  }
  return [Math.min(...years), Math.max(...years)];
}

export function allSeries(ds: Dataset): string[] {
  return [...new Set(ds.episodes.map((e) => e.series ?? NO_SERIES))].sort();
}

export function defaultFilters(ds: Dataset): Filters {
  return {
    pub: yearRange(ds),
    lived: [null, null],
    livedMatch: "within",
    undated: false,
    series: new Set(allSeries(ds)),
    types: new Set(allTypes().filter((t) => t.visibleByDefault).map((t) => t.type)),
    showRelations: true,
    showCoDiscussed: true,
    minShared: 2,
  };
}

/** Episodes without a publication date stay visible, since they can't be placed in a window. */
export function episodeVisible(ep: Episode, f: Filters): boolean {
  if (!f.series.has(ep.series ?? NO_SERIES)) return false;
  if (!ep.date) return true;
  const year = Number(ep.date.slice(0, 4));
  return year >= f.pub[0] && year <= f.pub[1];
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
  return evidenceInView(ed, vp).length >= (ed.kind === "co_discussed" ? f.minShared : 1);
}

/** Entities the lifespan window selects. They are drawn even without a visible link. */
export function livedSelection(ds: Dataset, idx: Index, vp: Set<string>, f: Filters): string[] {
  if (!livedActive(f) || !idx.site.lifespan_type) return [];
  return ds.entities.filter((e) => e.type === idx.site.lifespan_type && entityVisible(e, idx, vp, f)).map((e) => e.id);
}

/**
 * A lifespan window selects dated entities (people), so other items (concepts, works,
 * places) stay only when they connect directly to one of the selected people.
 */
export function visibleEdges(ds: Dataset, idx: Index, vp: Set<string>, f: Filters): Edge[] {
  const edges = ds.edges.filter((ed) => edgeVisible(ed, idx, vp, f));
  if (!livedActive(f) || !idx.site.lifespan_type) return edges;
  const isPerson = (id: string) => idx.entity.get(id)!.type === idx.site.lifespan_type;
  const anchored = new Set<string>();
  for (const ed of edges) {
    if (isPerson(ed.source)) anchored.add(ed.target);
    if (isPerson(ed.target)) anchored.add(ed.source);
  }
  return edges.filter((ed) => [ed.source, ed.target].every((id) => isPerson(id) || anchored.has(id)));
}

/** Filters differing from the defaults, for the header badge. */
export function activeFilterCount(ds: Dataset, f: Filters): number {
  const d = defaultFilters(ds);
  let n = 0;
  if (f.pub[0] !== d.pub[0] || f.pub[1] !== d.pub[1]) n++;
  if (f.lived[0] !== null || f.lived[1] !== null) n++;
  if (f.series.size !== d.series.size) n++;
  if (f.types.size !== d.types.size || [...f.types].some((t) => !d.types.has(t))) n++;
  return n;
}
