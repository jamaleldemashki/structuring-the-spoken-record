<script lang="ts">
  import cytoscape from "cytoscape";
  import type { Simulation } from "d3-force";
  import { onDestroy, untrack } from "svelte";
  import { app } from "../lib/state.svelte.ts";
  import { buildGraph, findPath, type GraphSpec } from "../lib/graph.ts";
  import { applyForces, collisionRadius, makeSimulation, nodeDiameter, seed, settle, type SimLink, type SimNode } from "../lib/forces.ts";
  import { allTypes, cssToken, typeStyle, typeVar } from "../lib/entityStyle.ts";
  import GraphSettings from "./GraphSettings.svelte";

  let container: HTMLDivElement;
  let cy: cytoscape.Core | null = null;
  let structureKey = "";
  let sim: Simulation<SimNode, SimLink> | null = null;
  let simNodes = new Map<string, SimNode>();
  let simLinks: SimLink[] = [];
  let drawn = new Map<string, cytoscape.NodeSingular>();
  // Last drawn positions per mode, so filtering moves nodes instead of scrambling them.
  const positions = { full: new Map<string, { x: number; y: number }>(), focused: new Map<string, { x: number; y: number }>() };
  let from = $state(""), to = $state("");

  const spec = $derived<GraphSpec>(app.idx
    ? buildGraph(app.edges, app.idx, app.vp, { mode: app.mode, focus: app.visibleFocus, depth: app.depth, path: app.path,
        coPerNode: 10, always: app.livedIds })
    : { nodes: [], edges: [] });

  // Entities that can take part in a connection: anything with at least one visible edge.
  const connectable = $derived.by(() => {
    const ids = new Set(app.edges.flatMap((e) => [e.source, e.target]));
    return [...ids].map((id) => app.idx!.entity.get(id)!).sort((a, b) => a.label.localeCompare(b.label));
  });
  const reducedMotion = () => matchMedia("(prefers-reduced-motion: reduce)").matches;

  function stylesheet(): cytoscape.StylesheetJson {
    const t = cssToken, g = app.graph;
    const font = "IBM Plex Sans, system-ui, sans-serif";
    return [
      { selector: "node", style: {
        "background-color": (n: cytoscape.NodeSingular) => t(typeStyle(n.data("type")).token),
        shape: (n: cytoscape.NodeSingular) => typeStyle(n.data("type")).shape,
        width: (n: cytoscape.NodeSingular) => nodeDiameter(n.data("passages"), g.nodeSize),
        height: (n: cytoscape.NodeSingular) => nodeDiameter(n.data("passages"), g.nodeSize),
        label: "data(label)", color: t("ink"), "font-family": font,
        // Bigger nodes get bigger labels, so they stay readable longest when zooming out.
        "font-size": (n: cytoscape.NodeSingular) => 10 + Math.min(6, Math.sqrt(n.data("passages"))),
        "min-zoomed-font-size": g.textFade,
        "text-valign": "bottom", "text-margin-y": 5, "text-wrap": "ellipsis", "text-max-width": "140px",
        "text-background-color": t("surface"), "text-background-opacity": 0.85, "text-background-padding": "2px",
        "text-background-shape": "roundrectangle", "border-width": 2, "border-color": t("surface") } },
      { selector: "node.focus", style: { "border-width": 4, "border-color": t("ink"), "font-weight": 600, "min-zoomed-font-size": 0 } },
      { selector: "edge", style: {
        "curve-style": "bezier", width: (e: cytoscape.EdgeSingular) => e.data("width") * g.linkThickness,
        "line-color": t("edge-co"), "line-style": "dotted",
        label: "", "font-family": font, "font-size": 10.5, color: t("ink-2"), "text-rotation": "autorotate",
        "text-background-color": t("surface"), "text-background-opacity": 0.95, "text-background-padding": "2px",
        "text-background-shape": "roundrectangle" } },
      { selector: "edge.relation", style: { "line-style": "solid", "line-color": t("edge-relation"),
        "target-arrow-color": t("edge-relation"), "target-arrow-shape": g.arrows ? "triangle" : "none" } },
      { selector: "edge.relation.unverified", style: { "line-style": "dashed" } },
      { selector: "edge.lab, edge.hover, edge:selected", style: { label: "data(label)" } },
      { selector: "edge:selected", style: { width: 4, "line-color": t("ink"), "target-arrow-color": t("ink") } },
      { selector: "node.hover", style: { "min-zoomed-font-size": 0, "z-index": 10 } },
      { selector: ".dim", style: { opacity: 0.18 } },
    ];
  }

  function elements(s: GraphSpec): cytoscape.ElementDefinition[] {
    return [
      ...s.nodes.map((n) => ({ group: "nodes" as const, data: { id: n.id, label: n.label, type: n.type, passages: n.passages },
        classes: n.focus ? "focus" : "" })),
      ...s.edges.map((e) => ({ group: "edges" as const, data: { id: e.id, source: e.source, target: e.target, label: e.label, width: e.width },
        classes: [e.kind === "relation" ? "relation" : "co", e.kind === "relation" && !e.verified ? "unverified" : "", e.showLabel ? "lab" : ""].join(" ") })),
    ];
  }

  /** Copies simulation positions onto the drawing, leaving a node the user is dragging alone. */
  function draw() {
    if (!cy || !sim) return;
    const store = positions[app.mode];
    cy.batch(() => {
      for (const n of sim!.nodes()) {
        const el = drawn.get(n.id);
        if (el && !el.grabbed()) el.position({ x: n.x!, y: n.y! });
        store.set(n.id, { x: n.x!, y: n.y! });
      }
    });
  }

  function stopForces() {
    sim?.stop();
    sim = null;
    simNodes = new Map();
  }

  /** Lays out the current elements with the force simulation (full and focused modes). */
  function startForces(s: GraphSpec) {
    stopForces();
    const g = untrack(() => $state.snapshot(app.graph));
    const mode = untrack(() => app.mode), focus = untrack(() => app.focus);
    const nodes: SimNode[] = s.nodes.map((n) => ({ id: n.id, r: collisionRadius(n.label, n.passages, g) }));
    simLinks = s.edges.map((e) => ({ id: e.id, source: e.source, target: e.target }));
    // Focused mode always starts fresh around the pinned focus; full mode keeps earlier positions.
    const fresh = mode === "focused" ? (seed(nodes, simLinks, new Map()), true) : seed(nodes, simLinks, positions.full) > 0.5;
    for (const n of nodes) if (mode === "focused" && n.id === focus) { n.fx = 0; n.fy = 0; }
    simNodes = new Map(nodes.map((n) => [n.id, n]));
    sim = makeSimulation(nodes, simLinks, g);
    sim.on("tick", draw);
    if (fresh || reducedMotion()) {
      settle(sim.alpha(1));
      draw();
      cy!.fit(undefined, 40);
      return;
    }
    sim.alpha(0.5).restart();
  }

  function pathLayout() {
    stopForces();
    cy!.layout({ name: "breadthfirst", directed: false, roots: [app.path!.nodes[0]], spacingFactor: 1.4, fit: true, padding: 40 } as cytoscape.LayoutOptions).run();
  }

  function create(): cytoscape.Core {
    const core = cytoscape({ container, style: stylesheet(), minZoom: 0.05, maxZoom: 4, wheelSensitivity: 0.25 });
    core.on("tap", "node", (ev) => app.selectEntity(ev.target.id()));
    core.on("tap", "edge", (ev) => {
      const edge = app.ds!.edges.find((e) => e.id === ev.target.id());
      if (edge) app.selectEdge(edge);
    });
    core.on("mouseover", "edge", (ev) => ev.target.addClass("hover"));
    core.on("mouseout", "edge", (ev) => ev.target.removeClass("hover"));
    core.on("mouseover", "node", (ev) => {
      const near = ev.target.closedNeighborhood();
      core.elements().not(near).addClass("dim");
      near.nodes().addClass("hover");
      ev.target.connectedEdges().addClass("hover");
    });
    core.on("mouseout", "node", () => core.elements().removeClass("dim hover"));
    // Dragging pins a node and the rest of the graph follows, like in Obsidian.
    core.on("grab", "node", (ev) => {
      const n = simNodes.get(ev.target.id());
      if (!n || !sim) return;
      const p = ev.target.position();
      n.fx = p.x; n.fy = p.y;
      if (!reducedMotion()) sim.alphaTarget(0.3).restart();
    });
    core.on("drag", "node", (ev) => {
      const n = simNodes.get(ev.target.id());
      if (!n) return;
      const p = ev.target.position();
      n.fx = p.x; n.fy = p.y; n.x = p.x; n.y = p.y;
    });
    core.on("free", "node", (ev) => {
      const n = simNodes.get(ev.target.id());
      if (!n || !sim) return;
      if (!(app.mode === "focused" && n.id === app.focus)) { n.fx = null; n.fy = null; }
      sim.alphaTarget(0);
    });
    return core;
  }

  // Only rebuild when the set of drawn elements changes. Otherwise just restyle, so the
  // layout stays put while clicking around.
  $effect(() => {
    const s = spec;
    if (!container) return;
    cy ??= create();
    const key = `${app.mode}|${app.mode === "focused" ? app.focus : ""}|${app.path ? "path" : ""}|${s.nodes.map((n) => n.id).join()}|${s.edges.map((e) => e.id).join()}`;
    if (key !== structureKey) {
      structureKey = key;
      cy.elements().remove();
      cy.add(elements(s));
      drawn = new Map(cy.nodes().map((n) => [n.id(), n]));
      if (untrack(() => app.path)) pathLayout();
      else startForces(s);
      return;
    }
    cy.batch(() => {
      for (const n of s.nodes) cy!.getElementById(n.id).toggleClass("focus", n.focus);
      for (const e of s.edges) cy!.getElementById(e.id).toggleClass("lab", e.showLabel);
    });
    const focused = app.focus ? cy.getElementById(app.focus) : null;
    if (focused?.length) cy.animate({ center: { eles: focused }, zoom: Math.max(cy.zoom(), 1.1) }, { duration: 300 });
  });

  // Force sliders work live: the running simulation gets the new forces and heats up again.
  $effect(() => {
    void [app.graph.center, app.graph.repel, app.graph.linkForce, app.graph.linkDistance, app.graph.gap, app.graph.nodeSize];
    untrack(() => {
      const g = $state.snapshot(app.graph);
      if (!sim) return;
      for (const n of spec.nodes) { const sn = simNodes.get(n.id); if (sn) sn.r = collisionRadius(n.label, n.passages, g); }
      applyForces(sim, simLinks, g);
      if (reducedMotion()) { settle(sim.alpha(1)); draw(); }
      else sim.alpha(0.6).restart();
    });
  });

  // Display settings and theme changes only restyle.
  $effect(() => {
    void $state.snapshot(app.graph);
    cy?.style(stylesheet());
  });

  // Follow theme changes made by the OS or the in-app toggle, and size changes of the canvas.
  $effect(() => {
    const restyle = () => cy?.style(stylesheet());
    const media = matchMedia("(prefers-color-scheme: dark)");
    media.addEventListener("change", restyle);
    const observer = new MutationObserver(restyle);
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    const resize = new ResizeObserver(() => cy?.resize());
    resize.observe(container);
    return () => { media.removeEventListener("change", restyle); observer.disconnect(); resize.disconnect(); };
  });

  onDestroy(() => { stopForces(); cy?.destroy(); });

  function setMode(mode: "full" | "focused") {
    app.mode = mode;
    app.path = null;
  }

  function connect() {
    if (!from || !to) return;
    const path = findPath(app.edges, from, to);
    if (path) { app.path = path; app.selection = { kind: "path", path }; }
    else { app.path = null; app.selection = { kind: "nopath", from, to }; }
    app.ui.panelOpen = true;
  }

  $effect(() => { if (app.focus && !from) from = app.focus; });

  const zoomBy = (factor: number) => cy?.zoom({ level: cy.zoom() * factor, renderedPosition: { x: cy.width() / 2, y: cy.height() / 2 } });
  const legendTypes = $derived(allTypes().filter((t) => app.filters!.types.has(t.type)));
</script>

<div class="canvas card">
  {#if app.ui.toolbarOpen}
    <div class="toolbar">
      <span class="seg" role="group" aria-label="Graph mode">
        <button aria-pressed={app.mode === "full"} onclick={() => setMode("full")}>Full graph</button>
        <button aria-pressed={app.mode === "focused"} onclick={() => setMode("focused")} disabled={!app.focus}
          title={app.focus ? `Show ${app.idx!.entity.get(app.focus)?.label} and its neighbours` : "Search for an item or click a node first"}>Focused</button>
      </span>
      {#if app.mode === "focused"}
        <span class="seg" role="group" aria-label="Depth">
          {#each [1, 2] as d}<button aria-pressed={app.depth === d} onclick={() => (app.depth = d)}>{d} step{d > 1 ? "s" : ""}</button>{/each}
        </span>
      {/if}
      <span class="connect" title="Finds the shortest chain of connections between two items, for example Aquinas → influenced by → Aristotle → discussed together → soul.">
        <span class="note">How is</span>
        <select bind:value={from} aria-label="First item"><option value="" disabled>choose…</option>{#each connectable as e (e.id)}<option value={e.id}>{e.label}</option>{/each}</select>
        <span class="note">connected to</span>
        <select bind:value={to} aria-label="Second item"><option value="" disabled>choose…</option>{#each connectable as e (e.id)}<option value={e.id}>{e.label}</option>{/each}</select>
        <button class="btn" onclick={connect} disabled={!from || !to || from === to}>Show</button>
      </span>
    </div>
  {/if}
  <div class="stage">
    <div class="cy" bind:this={container} role="img" aria-label="Graph of {spec.nodes.length} entities and {spec.edges.length} connections"></div>
    {#if !spec.nodes.length}
      <p class="empty center">Nothing to draw with the current filters. Widen the time window or turn on more types.</p>
    {/if}

    <div class="overlay top-left">
      <span class="count note">{spec.nodes.length} items · {spec.edges.length} links</span>
    </div>

    {#if app.ui.settingsOpen}
      <div class="overlay top-right"><GraphSettings /></div>
    {/if}

    {#if app.ui.legendOpen}
      <div class="overlay legend card">
        {#each legendTypes as t (t.type)}<span><i class="swatch {t.shape}" style:background={typeVar(t.type)}></i>{t.label}</span>{/each}
        <span><svg width="30" height="10" aria-hidden="true"><line x1="1" y1="5" x2="23" y2="5" stroke="var(--edge-relation)" stroke-width="2" /><path d="M23 1 29 5 23 9z" fill="var(--edge-relation)" /></svg>Named relation (✓ checked against the transcript)</span>
        <span><svg width="30" height="10" aria-hidden="true"><line x1="1" y1="5" x2="29" y2="5" stroke="var(--edge-co)" stroke-width="2" stroke-dasharray="2 3" /></svg>Discussed together in the same passages</span>
        <span class="note">Drag a node to pull its neighbours along. Zoom in to read more labels.</span>
      </div>
    {/if}
    <div class="overlay zoom card" role="group" aria-label="Zoom">
      <button aria-label="Zoom in" onclick={() => zoomBy(1.25)}>+</button>
      <button aria-label="Zoom out" onclick={() => zoomBy(0.8)}>−</button>
      <button aria-label="Fit graph to screen" onclick={() => cy?.fit(undefined, 40)}>⤢</button>
    </div>
  </div>
</div>

<style>
  .seg button:disabled { opacity: 0.45; cursor: not-allowed; }
  .canvas { display: flex; flex-direction: column; min-width: 0; min-height: 0; height: 100%; }
  .toolbar { display: flex; flex-wrap: wrap; gap: 8px 14px; align-items: center; padding: 8px 12px; border-bottom: 1px solid var(--line); }
  .connect { display: inline-flex; gap: 6px; align-items: center; flex-wrap: wrap; }
  .connect select { max-width: 160px; }
  .stage { position: relative; flex: 1; min-height: 0; }
  .cy { position: absolute; inset: 0; }
  .center { position: absolute; inset: 0; display: grid; place-items: center; text-align: center; padding: 24px; pointer-events: none; }
  .overlay { position: absolute; z-index: 5; }
  .top-left { left: 10px; top: 10px; }
  .count { background: color-mix(in srgb, var(--surface) 85%, transparent); border-radius: 6px; padding: 2px 6px; font-family: var(--font-data); font-size: 11px; }
  /* Settings float over the graph's top right, like Obsidian's graph controls. */
  .top-right { right: 10px; top: 10px; bottom: 110px; display: flex; flex-direction: column; }
  .legend { left: 10px; bottom: 10px; display: flex; flex-wrap: wrap; gap: 6px 14px; padding: 8px 12px; font-size: 11.5px; color: var(--ink-2); max-width: calc(100% - 70px); }
  .legend span { display: inline-flex; align-items: center; gap: 6px; }
  .swatch { width: 10px; height: 10px; display: inline-block; border-radius: 2px; }
  .swatch.ellipse { border-radius: 50%; }
  .swatch.round-rectangle { border-radius: 3px; }
  .swatch.diamond { transform: rotate(45deg) scale(0.85); }
  .swatch.triangle { clip-path: polygon(50% 0, 100% 100%, 0 100%); }
  .swatch.round-hexagon, .swatch.hexagon { clip-path: polygon(25% 0, 75% 0, 100% 50%, 75% 100%, 25% 100%, 0 50%); }
  .swatch.vee { clip-path: polygon(0 0, 50% 40%, 100% 0, 50% 100%); }
  .zoom { right: 10px; bottom: 10px; display: flex; flex-direction: column; overflow: hidden; }
  .zoom button { border: 0; background: none; width: 32px; height: 30px; cursor: pointer; font-size: 15px; color: var(--ink-2); }
  .zoom button + button { border-top: 1px solid var(--line); }
  .zoom button:hover { background: var(--surface-2); }
</style>
