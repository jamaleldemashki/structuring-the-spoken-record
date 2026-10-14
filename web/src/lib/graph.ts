// Turns the visible edges into the node/edge sets the graph view draws.
import type { Edge, EntityType, Index } from "./types.ts";

export type Mode = "full" | "focused";

export interface GraphNode {
  id: string;
  label: string;
  type: EntityType;
  passages: number;
  focus: boolean;
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  kind: Edge["kind"];
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
  return ed.pmi + ed.count;
}

export function edgeLabel(ed: Edge): string {
  if (ed.kind === "relation") return ed.status === "nli_supported" ? `${ed.label} ✓` : ed.label;
  return `discussed together (${ed.count})`;
}

function adjacency(edges: Edge[]): Map<string, Edge[]> {
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

const other = (ed: Edge, id: string) => (ed.source === id ? ed.target : ed.source);

/**
 * Shortest chain between two entities over the visible edges. Breadth-first search finds
 * the shortest one, and sorting the neighbours prefers named relations when lengths tie.
 */
export function findPath(edges: Edge[], from: string, to: string): PathResult | null {
  if (from === to) return null;
  const adj = adjacency(edges);
  const prev = new Map<string, [string, Edge] | null>([[from, null]]);
  const queue = [from];
  while (queue.length) {
    const u = queue.shift()!;
    if (u === to) break;
    for (const ed of adj.get(u) ?? []) {
      const v = other(ed, u);
      if (!prev.has(v)) {
        prev.set(v, [u, ed]);
        queue.push(v);
      }
    }
  }
  if (!prev.has(to)) return null;
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
  depth: number;
  path: PathResult | null;
  /** Co-discussed edges kept per node when expanding a focused view. */
  coPerNode: number;
  /** Entities drawn in the full graph even without a visible link (those a lifespan window selects). */
  always?: string[];
}

export function buildGraph(edges: Edge[], idx: Index, vp: Set<string>, opt: BuildOptions): GraphSpec {
  let chosen: Edge[];
  if (opt.path) {
    chosen = opt.path.edges;
  } else if (opt.mode === "full" || !opt.focus) {
    chosen = edges;
  } else {
    const adj = adjacency(edges);
    const reached = new Set([opt.focus]);
    const picked = new Set<Edge>();
    let frontier = [opt.focus];
    for (let d = 0; d < opt.depth; d++) {
      const next: string[] = [];
      for (const u of frontier) {
        let co = 0;
        for (const ed of adj.get(u) ?? []) {
          if (ed.kind === "co_discussed" && ++co > (d === 0 ? opt.coPerNode : Math.ceil(opt.coPerNode / 2))) continue;
          picked.add(ed);
          const v = other(ed, u);
          if (!reached.has(v)) {
            reached.add(v);
            next.push(v);
          }
        }
      }
      frontier = next;
    }
    // Named relations among already shown nodes are always worth drawing.
    for (const ed of edges) if (ed.kind === "relation" && reached.has(ed.source) && reached.has(ed.target)) picked.add(ed);
    chosen = [...picked];
  }
  const ids = new Set(chosen.flatMap((ed) => [ed.source, ed.target]));
  // The caller only passes a focus that the filters let through.
  if (opt.focus && idx.entity.has(opt.focus) && !opt.path) ids.add(opt.focus);
  if (opt.mode === "full" && !opt.path) for (const id of opt.always ?? []) ids.add(id);
  const nodes: GraphNode[] = [...ids].sort().map((id) => {
    const e = idx.entity.get(id)!;
    const passages = [...(idx.entityPassages.get(id) ?? [])].filter((p) => vp.has(p)).length;
    return { id, label: e.label, type: e.type, passages, focus: id === opt.focus };
  });
  const graphEdges: GraphEdge[] = chosen.map((ed) => ({
    id: ed.id,
    source: ed.source,
    target: ed.target,
    kind: ed.kind,
    label: edgeLabel(ed),
    verified: ed.kind === "relation" && ed.status === "nli_supported",
    showLabel: Boolean(opt.path) || (ed.kind === "relation" && (ed.source === opt.focus || ed.target === opt.focus)),
    width: ed.kind === "relation" ? 2.5 : Math.max(1, Math.min(3.5, ed.pmi)),
  }));
  return { nodes, edges: graphEdges };
}
