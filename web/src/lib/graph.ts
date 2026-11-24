// Turns the visible edges into the node/edge sets the graph view draws.
import { EPISODE_TYPE } from "./entityStyle.ts";
import type { EpisodeLink, PublicationScale } from "./episodes.ts";
import type { Edge, EntityType, Index } from "./types.ts";

export type Mode = "full" | "focused";

export interface GraphNode {
  id: string;
  label: string;
  type: EntityType;
  passages: number;
  focus: boolean;
  /** People (the archive's dated type) are drawn a little bigger. */
  lead: boolean;
  /** Thumbnail URL, when the item has a Wikidata image. */
  image?: string;
  /** An episode, not an entity of the archive. */
  episode?: boolean;
  /** Place on the publication axis, when the axis is on and the date is known. */
  tx?: number;
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  kind: Edge["kind"] | EpisodeLink["kind"];
  label: string;
  verified: boolean;
  showLabel: boolean;
  /** Line width in pixels. */
  width: number;
}

export interface GraphSpec {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export interface PathResult {
  nodes: string[];
  edges: Edge[];
}

/** Named relations come before co-discussion, and among those specific (high PMI) pairs win. */
export function edgeRank(ed: Edge): number {
  if (ed.kind === "relation") return ed.status === "nli_supported" ? 1000 : 900;
  // Weight from the passages in view (strength.ts); the plain count when it isn't set.
  return ed.weight ?? ed.count;
}

export function edgeLabel(ed: Edge): string {
  if (ed.kind === "relation") return ed.status === "nli_supported" ? `${ed.label} ✓` : ed.label;
  return `discussed together (${ed.count})`;
}

/** Edges per entity, best first: named relations, then the most specific co-discussion. */
export function adjacency(edges: Edge[]): Map<string, Edge[]> {
  const adj = new Map<string, Edge[]>();
  for (const ed of edges) {
    for (const id of [ed.source, ed.target]) {
      if (!adj.has(id)) adj.set(id, []);
      adj.get(id)!.push(ed);
    }
  }
  for (const list of adj.values()) list.sort((a, b) => edgeRank(b) - edgeRank(a) || a.id.localeCompare(b.id));
  return adj;
}

export const other = (ed: Edge, id: string) => (ed.source === id ? ed.target : ed.source);

/**
 * Shortest chain between two entities over the visible edges. Breadth-first search finds
 * the shortest one, and sorting the neighbours prefers named relations when lengths tie.
 */
export function findPath(edges: Edge[], from: string, to: string): PathResult | null {
  if (from === to) return null;
  return shortestPath(adjacency(edges), from, (id) => id === to);
}

/** Breadth-first search from one entity to the nearest one that passes the test. */
export function shortestPath(adj: Map<string, Edge[]>, from: string, isTarget: (id: string) => boolean): PathResult | null {
  const prev = new Map<string, [string, Edge] | null>([[from, null]]);
  const queue = [from];
  let to: string | null = null;
  while (queue.length) {
    const u = queue.shift()!;
    if (u !== from && isTarget(u)) { to = u; break; }
    for (const ed of adj.get(u) ?? []) {
      const v = other(ed, u);
      if (!prev.has(v)) {
        prev.set(v, [u, ed]);
        queue.push(v);
      }
    }
  }
  if (to === null) return null;
  const nodes = [to];
  const path: Edge[] = [];
  let step = prev.get(to);
  while (step) {
    path.push(step[1]);
    nodes.push(step[0]);
    step = prev.get(step[0]);
  }
  return { nodes: nodes.reverse(), edges: path.reverse() };
}

export interface BuildOptions {
  mode: Mode;
  focus: string | null;
  path: PathResult | null;
  /** The exploration ("focused" mode): its nodes and the edges to draw between them. */
  scene?: { nodes: string[]; edges: Edge[] } | null;
  /** Entities drawn in the full graph even without a visible link (those a lifespan window selects). */
  always?: string[];
  /** Episode nodes. With `everyItem`, the whole map also draws the items that have no other link. */
  episodes?: { links: EpisodeLink[]; show: boolean; everyItem: boolean } | null;
  /** The publication axis: its scale and where each item sits on it. */
  time?: { scale: Pick<PublicationScale, "x">; items: Map<string, number> } | null;
}

/** How many relation labels are written at once around a topic or in an exploration. */
export const MOST_RELATION_LABELS = 12;

export function buildGraph(edges: Edge[], idx: Index, vp: Set<string>, opt: BuildOptions): GraphSpec {
  const exploring = opt.mode === "focused" && !opt.path && opt.scene;
  const chosen = opt.path ? opt.path.edges : exploring ? opt.scene!.edges : edges;
  const ids = new Set(chosen.flatMap((ed) => [ed.source, ed.target]));
  // An exploration keeps items without a drawn link too, for example someone just added.
  if (exploring) for (const id of opt.scene!.nodes) ids.add(id);
  if (opt.mode === "full" && !opt.path) for (const id of opt.always ?? []) ids.add(id);
  // A chain between two items is drawn on its own, without episodes and without the axis.
  const episodes = opt.path || !opt.episodes?.show ? null : opt.episodes;
  const time = opt.path ? null : opt.time;
  if (episodes?.everyItem && opt.mode === "full") for (const l of episodes.links) ids.add(l.source);
  const mentions = episodes ? episodes.links.filter((l) => ids.has(l.source)) : [];
  const nodes: GraphNode[] = [...ids].sort().map((id) => {
    const e = idx.entity.get(id)!;
    const passages = [...(idx.entityPassages.get(id) ?? [])].filter((p) => vp.has(p)).length;
    return { id, label: e.label, type: e.type, passages, focus: id === opt.focus, lead: e.type === idx.site.lifespan_type,
      image: e.image?.thumb, tx: time?.items.get(id) };
  });
  for (const id of [...new Set(mentions.map((l) => l.target))].sort()) {
    const ep = idx.episode.get(id)!;
    nodes.push({ id, label: ep.code, type: EPISODE_TYPE, passages: ep.passage_count, focus: id === opt.focus, lead: false, episode: true,
      tx: time && ep.date ? time.scale.x(ep.date) : undefined });
  }
  const graphEdges: GraphEdge[] = chosen.map((ed) => ({
    id: ed.id,
    source: ed.source,
    target: ed.target,
    kind: ed.kind,
    label: edgeLabel(ed),
    verified: ed.kind === "relation" && ed.status === "nli_supported",
    showLabel: Boolean(opt.path) || (ed.kind === "relation" && (Boolean(exploring) || ed.source === opt.focus || ed.target === opt.focus)),
    // Thin, as in Obsidian: the links are a faint web until something is pointed at.
    width: ed.kind === "relation" ? 1.6 : Math.max(0.5, Math.min(1.6, 0.3 + ed.pmi * 0.3)),
  }));
  // A topic with many named relations would be buried under their labels. Then only the
  // strongest are written, the others show theirs when pointed at.
  const labelled = graphEdges.filter((e) => e.showLabel && e.kind === "relation");
  if (!opt.path && labelled.length > MOST_RELATION_LABELS) {
    const confidence = new Map(chosen.map((ed) => [ed.id, ed.kind === "relation" ? ed.confidence : 0]));
    const keep = new Set(labelled.sort((a, b) => confidence.get(b.id)! - confidence.get(a.id)! || a.id.localeCompare(b.id))
      .slice(0, MOST_RELATION_LABELS).map((e) => e.id));
    for (const e of labelled) e.showLabel = keep.has(e.id);
  }
  for (const l of mentions) {
    graphEdges.push({ id: l.id, source: l.source, target: l.target, kind: l.kind, label: `named in ${l.count} ${l.count === 1 ? "passage" : "passages"}`,
      verified: false, showLabel: false, width: Math.min(1.4, 0.4 + Math.sqrt(l.count) * 0.25) });
  }
  return { nodes, edges: graphEdges };
}
