// Graph settings (like Obsidian's) and the force simulation that lays out the graph.
// The simulation keeps running while nodes are dragged or a slider moves, and the
// collision force stops nodes and their labels from overlapping.
import { forceCollide, forceLink, forceManyBody, forceSimulation, forceX, forceY,
  type Simulation, type SimulationLinkDatum, type SimulationNodeDatum } from "d3-force";

export interface GraphSettings {
  arrows: boolean;
  /** Labels hide when they get smaller than this many pixels on screen. 0 always shows them. */
  textFade: number;
  nodeSize: number;
  linkThickness: number;
  /** Pull towards the centre, 0 to 1. */
  center: number;
  /** Push between all nodes, 0 to 20. */
  repel: number;
  /** How strongly links pull their ends together, 0 to 1. */
  linkForce: number;
  /** Preferred link length in pixels. */
  linkDistance: number;
  /** Extra free space kept around every node and its label, in pixels. */
  gap: number;
}

export const DEFAULT_SETTINGS: GraphSettings = {
  arrows: true, textFade: 7, nodeSize: 1, linkThickness: 1,
  center: 0.3, repel: 10, linkForce: 0.5, linkDistance: 160, gap: 12,
};

const KEY = "graph-settings";

export function loadSettings(): GraphSettings {
  try {
    const raw = localStorage.getItem(KEY);
    if (raw) return { ...DEFAULT_SETTINGS, ...JSON.parse(raw) };
  } catch { /* storage unavailable or corrupt: use defaults */ }
  return { ...DEFAULT_SETTINGS };
}

export function saveSettings(s: GraphSettings) {
  try { localStorage.setItem(KEY, JSON.stringify(s)); } catch { /* storage unavailable */ }
}

export interface SimNode extends SimulationNodeDatum {
  id: string;
  /** Collision radius covering the node and its label. */
  r: number;
}

export interface SimLink extends SimulationLinkDatum<SimNode> {
  id: string;
}

export const nodeDiameter = (passages: number, scale: number) => (20 + Math.sqrt(passages) * 7) * scale;

/** A circle around the node and the label under it. Labels are cut at 140px. */
export function collisionRadius(label: string, passages: number, s: GraphSettings): number {
  const d = nodeDiameter(passages, s.nodeSize);
  const labelWidth = Math.min(140, label.length * 6.5);
  return Math.max(d / 2 + 14, labelWidth / 2 + 4) + s.gap / 2;
}

/** Repel grows faster than linear, so the top of the slider spreads dense graphs far apart. */
export const repelStrength = (repel: number) => 30 * repel ** 1.4;

const endId = (end: SimLink["source"]) => (typeof end === "object" ? (end as SimNode).id : String(end));

export function applyForces(sim: Simulation<SimNode, SimLink>, links: SimLink[], s: GraphSettings) {
  const degree = new Map<string, number>();
  for (const l of links) for (const id of [endId(l.source), endId(l.target)]) degree.set(id, (degree.get(id) ?? 0) + 1);
  const deg = (end: SimLink["source"]) => degree.get(endId(end)) ?? 1;
  sim
    .force("link", forceLink<SimNode, SimLink>(links).id((n) => n.id).distance(s.linkDistance)
      .strength((l) => s.linkForce / Math.min(deg(l.source), deg(l.target))))
    .force("charge", forceManyBody<SimNode>().strength(-repelStrength(s.repel)))
    .force("collide", forceCollide<SimNode>((n) => n.r).strength(0.9).iterations(2))
    .force("x", forceX<SimNode>(0).strength(s.center * 0.1))
    .force("y", forceY<SimNode>(0).strength(s.center * 0.1));
}

export function makeSimulation(nodes: SimNode[], links: SimLink[], s: GraphSettings): Simulation<SimNode, SimLink> {
  const sim = forceSimulation<SimNode, SimLink>(nodes).stop();
  applyForces(sim, links, s);
  return sim;
}

/** Runs the simulation until it settles, without drawing. Used for the first layout and reduced motion. */
export function settle(sim: Simulation<SimNode, SimLink>, maxTicks = 400) {
  for (let i = 0; i < maxTicks && sim.alpha() > sim.alphaMin(); i++) sim.tick();
}

/**
 * Starting positions. Nodes keep their last position, so filtering doesn't scramble the
 * picture. New nodes start next to a placed neighbour, or on a spiral.
 * Returns the share of nodes that had no earlier position.
 */
export function seed(nodes: SimNode[], links: SimLink[], known: Map<string, { x: number; y: number }>) {
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const placed = new Set<string>();
  for (const n of nodes) {
    const p = known.get(n.id);
    if (p) { n.x = p.x; n.y = p.y; placed.add(n.id); }
  }
  let i = 0;
  for (const n of nodes) {
    if (placed.has(n.id)) continue;
    const link = links.find((l) => (endId(l.source) === n.id && placed.has(endId(l.target))) || (endId(l.target) === n.id && placed.has(endId(l.source))));
    const anchor = link ? byId.get(endId(link.source) === n.id ? endId(link.target) : endId(link.source)) : undefined;
    const angle = i * 2.4, radius = anchor ? 40 : 30 * Math.sqrt(i + 1);
    n.x = (anchor?.x ?? 0) + radius * Math.cos(angle);
    n.y = (anchor?.y ?? 0) + radius * Math.sin(angle);
    placed.add(n.id);
    i++;
  }
  return nodes.length ? i / nodes.length : 0;
}
