// The exploration view: it starts with one item and its strongest links and only grows when
// the user asks ("search, show context, expand on demand").
//
// The view is not stored as a list of nodes but as the user's steps. Replaying the steps
// over the links in view gives the nodes, so Back is just dropping the last step, and a
// filter change applies to the whole exploration at once.
//
// Every item the user started from or chose a share for brings its strongest share of partners
// (strength.ts). Turning that share down takes them away again. Other items only show the
// links between what is already on screen, so nothing grows without being asked for.
//
// Two items can also be explored together. Their share is of what both are connected to, with both of
// its links drawn, so the picture answers what the two have in common. See sharedPartners.
//
// A step can also name an episode. Episodes are not nodes of the scene, they are drawn next to
// the items that are in it, so such a step brings in that share of the items the episode names.
import { edgeRank, shortestPath, adjacency } from "./graph.ts";
import { applyShares, portionOf, type RankedPartner } from "./strength.ts";
import type { Edge } from "./types.ts";

// `share` overrides the starting share of a start or add step. A question uses 0: only the asked
// items and the chains between them. In a share step, null goes back to the starting share.
export type Step =
  | { kind: "start"; id: string; share?: number }
  | { kind: "share"; id: string; share: number | null }
  | { kind: "add"; id: string; share?: number }
  | { kind: "hide"; id: string }
  // `id` is their pairKey, so a share step can name the pair as it names an item.
  | { kind: "together"; id: string; a: string; b: string; share?: number };

export interface Scene {
  /** Root first, then in the order they were added. */
  nodes: string[];
  /** The chains that link added items. They are always drawn. */
  tree: Edge[];
  /** The items and pairs with a share of their own. */
  shares: Map<string, number>;
}

export const pairKey = (a: string, b: string) => (a < b ? `${a}|${b}` : `${b}|${a}`);

export interface SharedPartner {
  id: string;
  /** Passages with the first item of the pair, and with the second. */
  a: number;
  b: number;
}

/** How many passages stand behind a link. A named relation has its evidence, shared passages their count. */
const passagesBehind = (p: RankedPartner) => Math.max(...p.edges.map((ed) => (ed.kind === "co_discussed" ? ed.count : ed.evidence.length)));

/**
 * What two items are both connected to. Ordered by the weaker of the two links, so what ranks high
 * matters to both. A topic discussed 50 times with one and once with the other says little about
 * what they have in common. Next comes the sum, and the id so that ties stay in the same order.
 */
export function sharedPartners(a: string, b: string, ranked: Map<string, RankedPartner[]>): SharedPartner[] {
  const other = new Map((ranked.get(b) ?? []).map((p) => [p.id, p]));
  const out: SharedPartner[] = [];
  for (const p of ranked.get(a) ?? []) {
    const q = other.get(p.id);
    if (q && p.id !== b) out.push({ id: p.id, a: passagesBehind(p), b: passagesBehind(q) });
  }
  return out.sort((x, y) => Math.min(y.a, y.b) - Math.min(x.a, x.b) || y.a + y.b - x.a - x.b || x.id.localeCompare(y.id));
}

export interface SceneInput {
  /** Every link inside the filters. */
  links: Edge[];
  ranked: Map<string, RankedPartner[]>;
  /** The share an item starts with. */
  share: number;
  episodeItems?: Map<string, string[]>;
}

export function replay(steps: Step[], { links, ranked, share, episodeItems = new Map() }: SceneInput): Scene {
  const adj = adjacency(links);
  let nodes: string[] = [];
  let shown = new Set<string>();
  let tree = new Map<string, Edge>();
  let hidden = new Set<string>();
  let shares = new Map<string, number>();
  let origins = new Map<string, number>();
  const pairs = new Map<string, [string, string]>();
  const add = (id: string) => { if (!shown.has(id)) { shown.add(id); nodes.push(id); } };
  for (const step of steps) {
    if (step.kind === "start") {
      nodes = []; shown = new Set(); tree = new Map(); hidden = new Set(); shares = new Map(); origins = new Map();
      add(step.id);
      origins.set(step.id, step.share ?? share);
      shares.set(step.id, step.share ?? share);
    } else if (step.kind === "share") {
      if (step.share !== null) shares.set(step.id, step.share);
      else if (origins.has(step.id)) shares.set(step.id, origins.get(step.id)!);
      else shares.delete(step.id);
    } else if (step.kind === "add") {
      hidden.delete(step.id);
      // A second item comes with the shortest chain to what is already shown, if there is one.
      const path = shown.has(step.id) ? null : shortestPath(adj, step.id, (id) => shown.has(id));
      add(step.id);
      for (const id of path?.nodes ?? []) add(id);
      for (const ed of path?.edges ?? []) tree.set(ed.id, ed);
      origins.set(step.id, step.share ?? share);
      shares.set(step.id, step.share ?? share);
    } else if (step.kind === "together") {
      // Both ends are shown, with what links them, and keep whatever share they already have.
      for (const id of [step.a, step.b]) {
        hidden.delete(id);
        const path = shown.has(id) ? null : shortestPath(adj, id, (x) => shown.has(x));
        add(id);
        for (const x of path?.nodes ?? []) add(x);
        for (const ed of path?.edges ?? []) tree.set(ed.id, ed);
      }
      for (const p of ranked.get(step.a) ?? []) if (p.id === step.b) for (const ed of p.edges) tree.set(ed.id, ed);
      pairs.set(step.id, [step.a, step.b]);
      origins.set(step.id, step.share ?? share);
      shares.set(step.id, step.share ?? share);
    } else {
      hidden.add(step.id);
      shown.delete(step.id);
    }
  }
  for (const [id, part] of shares) {
    const pair = pairs.get(id);
    if (pair) {
      const shared = sharedPartners(pair[0], pair[1], ranked).map((s) => s.id);
      for (const x of shared.slice(0, portionOf(part, shared.length))) {
        if (hidden.has(x)) continue;
        add(x);
        // Both links, so it can be seen that it belongs to both.
        for (const end of pair) for (const p of ranked.get(end) ?? []) if (p.id === x) for (const ed of p.edges) tree.set(ed.id, ed);
      }
    } else if (episodeItems.has(id)) {
      const items = episodeItems.get(id)!;
      for (const item of items.slice(0, portionOf(part, items.length))) if (!hidden.has(item)) add(item);
    } else if (shown.has(id)) {
      const partners = ranked.get(id) ?? [];
      for (const p of partners.slice(0, portionOf(part, partners.length))) if (!hidden.has(p.id)) add(p.id);
    }
  }
  nodes = nodes.filter((id) => shown.has(id));
  return { nodes, shares, tree: [...tree.values()].filter((ed) => shown.has(ed.source) && shown.has(ed.target)) };
}

/** Edges to draw: the chains, and the drawn links between items on screen. */
export function sceneEdges(scene: Scene, drawn: Edge[], ranked: Map<string, RankedPartner[]>): Edge[] {
  const shown = new Set(scene.nodes);
  const chosen = new Map(scene.tree.map((ed) => [ed.id, ed]));
  for (const ed of applyShares(drawn, ranked, scene.shares)) if (shown.has(ed.source) && shown.has(ed.target)) chosen.set(ed.id, ed);
  return [...chosen.values()].sort((a, b) => edgeRank(b) - edgeRank(a) || a.id.localeCompare(b.id));
}

/** Per episode, how many of the items it names are not shown yet. */
export function hiddenItems(scene: Scene, episodeItems: Map<string, string[]>): Map<string, number> {
  const shown = new Set(scene.nodes);
  return new Map([...episodeItems].map(([ep, items]) => [ep, items.filter((id) => !shown.has(id)).length]));
}
