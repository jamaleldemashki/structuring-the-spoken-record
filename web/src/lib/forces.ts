// Graph settings (like Obsidian's) and the force simulation that lays out the graph.
// The simulation keeps running while nodes are dragged or a slider moves, and the
// collision force stops nodes and their labels from overlapping. It runs in a worker
// (layout.worker.ts), so a large graph never blocks scrolling, clicks or typing.
import { forceCollide, forceLink, forceManyBody, forceSimulation, forceX, forceY,
  type Simulation, type SimulationLinkDatum, type SimulationNodeDatum } from "d3-force";

export interface GraphSettings {
  arrows: boolean;
  /** Show Wikidata portraits inside the nodes. */
  portraits: boolean;
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
  /** Width of the canvas divided by its height, so a free layout can use the shape of the screen. */
  aspect?: number;
}

export const DEFAULT_SETTINGS: GraphSettings = {
  arrows: true, portraits: true, nodeSize: 1, linkThickness: 1,
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
  /** Place on the publication axis, when the axis is on. */
  tx?: number;
  /** Height the item is drawn to: above or below the line of episodes. */
  ty?: number;
  /** The area the item stays inside: left, right, top, bottom. */
  box?: [number, number, number, number];
  /** The room it must have for itself when the layout ends, if it differs from r. */
  keep?: number;
}

export interface SimLink extends SimulationLinkDatum<SimNode> {
  id: string;
  /** A link that says little about where things belong, it pulls only gently. */
  weak?: boolean;
}

/** People (the archive's dated type) are drawn this much bigger, so they stand out a little. */
export const LEAD_SCALE = 1.35;

/**
 * Size follows the number of passages, but must not follow the length of the archive: in a podcast
 * with 700 episodes the main topic has thousands of passages and would cover the screen. Sizes
 * grow as they are up to a largest topic of this many passages. Beyond it all passage counts are
 * scaled down together, so the largest topic is drawn as large as one with this many.
 */
const LARGEST_AS_IS = 150;
let sizeShare = 1;
/** Set once per archive from its most discussed topic. */
export function setLargestTopic(passages: number) {
  sizeShare = Math.min(1, LARGEST_AS_IS / Math.max(1, passages));
}
/** The passage count a size is worked out from. */
export const sizedPassages = (passages: number) => passages * sizeShare;

export const nodeDiameter = (passages: number, scale: number, lead = false) =>
  (9 + Math.sqrt(sizedPassages(passages)) * 4.2) * scale * (lead ? LEAD_SCALE : 1);

/** Nodes never get smaller on screen than this, or they could not be seen or clicked. */
export const SMALLEST_NODE = 8;

/**
 * How wide a node is on screen, in pixels, at this zoom. A photo would scale with the zoom. Here
 * zoom mostly changes the distance between nodes: zoomed out they shrink slowly down to a minimum,
 * zoomed in they grow only a little, so crowded parts open up. Important items stay larger.
 */
export function screenDiameter(passages: number, scale: number, lead: boolean, zoom: number, smallest = SMALLEST_NODE): number {
  const growth = zoom <= 1 ? Math.sqrt(zoom) : zoom ** 0.25;
  return Math.max(smallest * scale, nodeDiameter(passages, scale, lead) * growth);
}

/** How firmly an item is held at its place on the publication axis. Links can still bend it. */
export const TIME_PULL = 0.5;

/** On the line of episodes the date and side already place an item, so links to episodes pull less. */
const WEAK_LINK = 0.08;
/** Inside a box, only near neighbours push. Pressure from far away would pile items up at its edges. */
const REACH_IN_BOX = 650;

function charge(s: GraphSettings, boxed: boolean) {
  const force = forceManyBody<SimNode>().strength(-repelStrength(s.repel));
  return boxed ? force.distanceMax(REACH_IN_BOX) : force;
}

/** How many places a wedged node tries before it gives up. */
const SPIRAL_STEPS = 6000;

/** How firmly an item is drawn to its side of the line of episodes. */
export const SIDE_PULL = 0.2;

/** Keeps every node with a box inside it, by shortening the step that would take it out. */
function stayInBox() {
  let nodes: SimNode[] = [];
  const force = () => {
    for (const n of nodes) {
      if (!n.box || n.fx != null) continue;
      const [left, right, top, bottom] = n.box;
      n.vx = Math.max(left - n.x!, Math.min(right - n.x!, n.vx!));
      n.vy = Math.max(top - n.y!, Math.min(bottom - n.y!, n.vy!));
    }
  };
  force.initialize = (all: SimNode[]) => { nodes = all; };
  return force;
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
      .strength((l) => (l.weak ? WEAK_LINK : 1) * s.linkForce / Math.min(deg(l.source), deg(l.target))))
    .force("charge", charge(s, sim.nodes().some((n) => n.box)))
    .force("collide", forceCollide<SimNode>((n) => n.r).strength(0.9).iterations(2))
    // Without the line of episodes, a wide screen gets a wide picture: less pull sideways, more up and down.
    .force("x", forceX<SimNode>((n) => n.tx ?? 0).strength((n) => (n.tx === undefined ? (s.center * 0.1) / (s.aspect ?? 1) : TIME_PULL)))
    .force("y", forceY<SimNode>((n) => n.ty ?? 0).strength((n) => (n.ty === undefined ? s.center * 0.1 * (s.aspect ?? 1) : SIDE_PULL)))
    .force("box", stayInBox());
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
 * Moves nodes apart until none overlaps another. The collision force only pushes while other forces
 * pull, so it can leave overlaps behind; this runs after it and has the last word. Nodes give way
 * by how fixed they are: a free one moves first, an item the layout pinned moves only for another
 * pinned one or for a node that can't move at all, like an episode at its date. Items stay inside their box. A grid
 * means only near pairs are compared. Returns whether anything moved.
 */
export function separate(nodes: SimNode[], rounds = 2000): boolean {
  const reach = (n: SimNode) => n.keep ?? n.r;
  const largest = Math.max(1, ...nodes.map(reach));
  const cell = 2 * largest;
  const fixed = (n: SimNode) => n.fx != null && n.fy != null;
  const firmness = (n: SimNode) => (!fixed(n) ? 0 : n.box ? 1 : 2);
  const move = (n: SimNode, dx: number, dy: number) => {
    n.x! += dx;
    n.y! += dy;
    if (fixed(n)) { n.fx = n.x; n.fy = n.y; }
  };
  let movedAny = false;
  for (let round = 0; round < rounds; round++) {
    const grid = new Map<string, SimNode[]>();
    for (const n of nodes) {
      const key = `${Math.floor(n.x! / cell)},${Math.floor(n.y! / cell)}`;
      if (!grid.has(key)) grid.set(key, []);
      grid.get(key)!.push(n);
    }
    let moved = false;
    nodes.forEach((a, index) => {
      const cx = Math.floor(a.x! / cell), cy = Math.floor(a.y! / cell);
      for (let i = -1; i <= 1; i++) for (let j = -1; j <= 1; j++) for (const b of grid.get(`${cx + i},${cy + j}`) ?? []) {
        if (a === b || a.id > b.id || (firmness(a) === 2 && firmness(b) === 2)) continue;
        let dx = b.x! - a.x!, dy = b.y! - a.y!;
        const need = reach(a) + reach(b);
        let dist = Math.hypot(dx, dy);
        if (dist >= need) continue;
        // Two nodes on the same spot part in a direction that differs from pair to pair.
        if (dist < 1e-6) { dx = Math.cos(index * 2.4); dy = Math.sin(index * 2.4); dist = 1; }
        // A little more than needed, so a pair does not stay a hair too close.
        const push = (1.02 * need - dist) / dist;
        const share = firmness(a) === firmness(b) ? 0.5 : firmness(a) > firmness(b) ? 1 : 0;
        if (share > 0) move(b, dx * push * share, dy * push * share);
        if (share < 1) move(a, -dx * push * (1 - share), -dy * push * (1 - share));
        moved = true;
      }
    });
    // If the box turns out too small, its outer edge gives way: no overlap matters more than the
    // height. The side towards the line and the left and right ends always hold.
    const tight = round < 300;
    for (const n of nodes) {
      if (n.box) {
        const [left, right, top, bottom] = n.box;
        n.x = Math.max(left, Math.min(right, n.x!));
        const above = bottom <= 0;
        n.y = Math.max(above && !tight ? -Infinity : top, Math.min(!above && !tight ? Infinity : bottom, n.y!));
        if (fixed(n)) { n.fx = n.x; n.fy = n.y; }
      }
    }
    if (!moved) return movedAny;
    movedAny = true;
  }
  findRoom(nodes, reach, fixed);
  return true;
}

/**
 * The last resort of separate. Pushing pairs apart cannot free a node wedged between pinned ones, so
 * a free node that still lies on another looks for the nearest spot where it fits, on a spiral around
 * where it is. Its box still holds, apart from the outer edge, which gives way as in separate.
 */
function findRoom(nodes: SimNode[], reach: (n: SimNode) => number, fixed: (n: SimNode) => boolean) {
  const fits = (n: SimNode, x: number, y: number) => nodes.every((o) => o === n || Math.hypot(o.x! - x, o.y! - y) >= reach(n) + reach(o));
  const allowed = (n: SimNode, x: number, y: number) => {
    if (!n.box) return true;
    const [left, right, top, bottom] = n.box;
    return x >= left && x <= right && (bottom <= 0 ? y <= bottom : y >= top);
  };
  for (const n of nodes) {
    if (fixed(n) || fits(n, n.x!, n.y!)) continue;
    const step = reach(n) * 0.6;
    for (let i = 1; i < SPIRAL_STEPS; i++) {
      const angle = i * 2.399963, far = step * Math.sqrt(i);
      const x = n.x! + far * Math.cos(angle), y = n.y! + far * Math.sin(angle);
      if (allowed(n, x, y) && fits(n, x, y)) { n.x = x; n.y = y; break; }
    }
  }
}

/**
 * Starting positions. Nodes keep their last position, so filtering doesn't scramble the
 * picture. New nodes start next to a placed neighbour, or on a spiral.
 * With outward > 0 (growing an exploration), new neighbours of a placed node fan out on the
 * side away from the middle of the picture, at that distance, where there is free space.
 * Returns the share of nodes that had no earlier position.
 */
export function seed(nodes: SimNode[], links: SimLink[], known: Map<string, { x: number; y: number }>, outward = 0) {
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const placed = new Set<string>();
  for (const n of nodes) {
    const p = known.get(n.id);
    if (p) { n.x = p.x; n.y = p.y; placed.add(n.id); }
  }
  const old = [...placed].map((id) => byId.get(id)!);
  const middle = old.length
    ? { x: old.reduce((sum, n) => sum + n.x!, 0) / old.length, y: old.reduce((sum, n) => sum + n.y!, 0) / old.length }
    : { x: 0, y: 0 };
  const neighbours = new Map<string, string[]>();
  for (const l of links) {
    const [a, b] = [endId(l.source), endId(l.target)];
    neighbours.set(a, [...(neighbours.get(a) ?? []), b]);
    neighbours.set(b, [...(neighbours.get(b) ?? []), a]);
  }
  const fanned = new Map<string, number>();
  let i = 0;
  for (const n of nodes) {
    if (placed.has(n.id)) continue;
    const near = neighbours.get(n.id)?.find((id) => placed.has(id));
    const anchor = near ? byId.get(near) : undefined;
    if (outward && anchor && known.has(anchor.id)) {
      // Alternate left and right of the outward direction: 0, +1, -1, +2, -2 ... steps of 0.5 rad.
      const k = fanned.get(anchor.id) ?? 0;
      fanned.set(anchor.id, k + 1);
      const away = Math.atan2(anchor.y! - middle.y, anchor.x! - middle.x) || i * 2.4;
      const angle = away + (k % 2 ? 1 : -1) * Math.ceil(k / 2) * 0.5;
      n.x = anchor.x! + outward * Math.cos(angle);
      n.y = anchor.y! + outward * Math.sin(angle);
    } else {
      const angle = i * 2.4, radius = anchor ? 40 : 30 * Math.sqrt(i + 1);
      n.x = (anchor?.x ?? 0) + radius * Math.cos(angle);
      n.y = (anchor?.y ?? 0) + radius * Math.sin(angle);
    }
    placed.add(n.id);
    i++;
  }
  return nodes.length ? i / nodes.length : 0;
}
