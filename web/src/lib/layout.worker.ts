// Runs the force simulation off the main thread. The graph view sends the nodes, links
// and settings, the worker answers with node positions, in the order of the nodes it got.
//
//   start    new nodes and links; settle first (fresh layout) or keep moving from the seeds
//   forces   new settings or collision radii, the simulation heats up again
//   pin      a node is held at a point while it is dragged
//   release  the drag ended
//   links    the links changed but not the nodes: nothing moves, dragging follows the new links
//   room     nodes need a different amount of room, for example on a narrower screen
//   stop     drop the simulation
import { applyForces, makeSimulation, separate, settle, type GraphSettings, type SimLink, type SimNode } from "./forces.ts";
import type { Simulation } from "d3-force";

export type LayoutRequest =
  | { type: "start"; run: number; nodes: SimNode[]; links: SimLink[]; settings: GraphSettings; settle: boolean }
  | { type: "forces"; run: number; settings: GraphSettings; radii: Record<string, number>; settle: boolean }
  | { type: "pin"; run: number; id: string; x: number; y: number; heat: boolean }
  | { type: "release"; run: number; id: string; keep: boolean }
  | { type: "links"; run: number; links: SimLink[]; settings: GraphSettings }
  | { type: "room"; run: number; keep: Record<string, number> }
  | { type: "stop" };

/** Positions as x0, y0, x1, y1, ... and whether this was the end of a fresh layout. */
export type LayoutReply = { run: number; xy: Float32Array; settled: boolean };

let run = 0;
let sim: Simulation<SimNode, SimLink> | null = null;
let nodes: SimNode[] = [];
let byId = new Map<string, SimNode>();
let links: SimLink[] = [];

function send(settled = false) {
  const xy = new Float32Array(nodes.length * 2);
  nodes.forEach((n, i) => { xy[2 * i] = n.x!; xy[2 * i + 1] = n.y!; });
  postMessage({ run, xy, settled } satisfies LayoutReply, { transfer: [xy.buffer] });
}

onmessage = (ev: MessageEvent<LayoutRequest>) => {
  const m = ev.data;
  if (m.type === "stop") { sim?.stop(); sim = null; return; }
  // Messages about an older graph can still arrive after a new start.
  if (m.type !== "start" && m.run !== run) return;
  if (m.type === "start") {
    sim?.stop();
    run = m.run;
    nodes = m.nodes;
    links = m.links;
    byId = new Map(nodes.map((n) => [n.id, n]));
    sim = makeSimulation(nodes, links, m.settings);
    sim.on("tick", () => send());
    // Once the movement has ended, no node may lie on another.
    sim.on("end", () => { if (separate(nodes)) send(); });
    if (m.settle) { settle(sim.alpha(1)); separate(nodes); send(true); }
    else sim.alpha(0.5).restart();
    return;
  }
  if (!sim) return;
  if (m.type === "room") {
    for (const n of nodes) n.keep = m.keep[n.id] ?? n.keep;
    if (separate(nodes)) send();
  } else if (m.type === "links") {
    links = m.links;
    applyForces(sim, links, m.settings);
  } else if (m.type === "forces") {
    for (const n of nodes) { n.r = m.radii[n.id] ?? n.r; n.keep = n.r; }
    applyForces(sim, links, m.settings);
    if (m.settle) { settle(sim.alpha(1)); separate(nodes); send(); }
    else sim.alpha(0.6).restart();
  } else if (m.type === "pin") {
    const n = byId.get(m.id);
    if (!n) return;
    n.fx = n.x = m.x;
    n.fy = n.y = m.y;
    if (m.heat) sim.alphaTarget(0.3).restart();
  } else if (m.type === "release") {
    const n = byId.get(m.id);
    if (n && !m.keep) { n.fx = null; n.fy = null; }
    sim.alphaTarget(0);
  }
};
