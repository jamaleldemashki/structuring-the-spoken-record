<script lang="ts">
  import cytoscape from "cytoscape";
  import { onDestroy, untrack } from "svelte";
  import { app, DETAIL_SIZE } from "../lib/state.svelte.ts";
  import { buildGraph, type GraphSpec } from "../lib/graph.ts";
  import { screenDiameter, seed, type GraphSettings as Forces, type SimLink, type SimNode } from "../lib/forces.ts";
  import { chooseLabels } from "../lib/labels.ts";
  import type { LayoutReply, LayoutRequest } from "../lib/layout.worker.ts";
  import { cssToken, typeStyle } from "../lib/entityStyle.ts";
  import { livedActive } from "../lib/filters.ts";
  import { episodeMarks, itemPositions, selectionLength, selectionPositions } from "../lib/episodes.ts";
  import { zoomGlide } from "../lib/zoom.ts";
  import BackToResults from "./BackToResults.svelte";
  import ConnectBox from "./ConnectBox.svelte";
  import LegendRow from "./LegendRow.svelte";
  import SelectionBar from "./SelectionBar.svelte";
  import PublicationAxis from "./PublicationAxis.svelte";
  import Search from "./Search.svelte";

  let container: HTMLDivElement;
  let cy: cytoscape.Core | null = null;
  let nodesKey = "";
  let edgesKey = "";
  // The force simulation runs in a worker. run numbers the layouts, so late positions of an
  // earlier graph are ignored; order is the node order the worker sends positions in.
  let worker: Worker | null = null;
  let run = 0;
  let order: string[] = [];
  let latest: Float32Array | null = null;
  let frame = 0;
  let fitWhenSettled = false;
  let arranging = $state(false);
  let drawn = new Map<string, cytoscape.NodeSingular>();
  // Episodes stand on a line through the middle, taking turns a little above and below it so
  // their labels stay clear of each other. Pixels on screen. How many there are decides the numbers.
  let marks = episodeMarks(0, 1);
  const UNNAMED_OPACITY = 0.55;
  /** A portrait is only drawn on a node at least this wide on screen. Smaller, it would be a blur. */
  const PORTRAIT_SIZE = 24;
  /** Free room between the line of episodes and the items. */
  const LINE_ROOM = 30;
  /** Free room around a node when the layout ends: names can be read in an exploration, so more there. */
  const NODE_ROOM = { full: 2.5, focused: 12 };
  /** The details panel usually opens, so the whole map is laid out for the width that leaves. */
  const PANEL_WIDTH = 390;
  /** How much of an item's place along the line comes from even spreading, the rest from its date. */
  const EVEN_SHARE = 0.92;
  const ROOM_MARGIN = { full: 1.3, focused: 2.4 };
  // A number from an item's id. It keeps an item on the same side whatever else is in view.
  function spot(id: string) {
    let h = 2166136261;
    for (let i = 0; i < id.length; i++) h = Math.imul(h ^ id.charCodeAt(i), 16777619);
    return h >>> 0;
  }
  const sideOf = (id: string) => (spot(id) % 2 ? -1 : 1);
  /** Room kept at the left and right end of the axis, and above and below, in pixels. */
  const AXIS_MARGIN = 36, HEIGHT_MARGIN = 24;
  /** How far out a graph without the line of episodes can be zoomed. */
  const MIN_ZOOM = 0.05;
  /** Height of the time axis under the graph, in pixels. */
  const AXIS_HEIGHT = 34;
  // Above this many links, panning and zooming move a snapshot of the graph instead of redrawing it.
  const LARGE = 800;
  // Last drawn positions per mode, so filtering moves nodes instead of scrambling them.
  const positions = { full: new Map<string, { x: number; y: number }>(), focused: new Map<string, { x: number; y: number }>() };
  // The zoom each mode's picture was first laid out for.
  const placedAt: { full: number | null; focused: number | null } = { full: null, focused: null };
  let stage: HTMLDivElement;
  let glide: ReturnType<typeof zoomGlide> | null = null;
  // The exploration whose positions are stored, so a new one starts from a clean layout.
  let laidOutRoot: string | null = null;
  let fitTimer = 0;
  // The right-click menu of a node.
  let menu = $state<{ id: string; x: number; y: number } | null>(null);

  // The current pan and zoom, for the time axis, and whether the axis is in use.
  let viewport = $state({ zoom: 1, pan: 0, width: 0 });
  let viewportFrame = 0;
  // The positions stored above were made with or without the axis.
  let laidOutWith: string | null = null;
  // One switch, the Episodes filter: the axis only means something with the episodes on it.
  const timeOn = $derived(app.filters!.episodes && !app.path && !!app.publication);
  const only = $derived(app.filters!.only);
  // With only some episodes, each stands with its items around it and the axis does not place them.
  const onlyLength = $derived(selectionLength(new Set(app.links.map((l) => l.source)).size));
  // The length of the line of episodes in graph units.
  const lineLength = $derived(only ? onlyLength : app.publication ? app.publication.right - app.publication.left : 0);
  // The view follows the size of the screen until the user moves it. The fit button brings that back.
  let followScreen = true;
  // Zooming out past full width: the axis stays across the whole screen and the graph spreads
  // with it. Cytoscape can only zoom evenly, so the drawn x positions are multiplied by this
  // number while the zoom goes down. Nodes get smaller and more of the height comes into view.
  // Everything stored and sent to the worker stays without it.
  let stretch = 1;
  let laidOutMode: "full" | "focused" | null = null;
  const layoutKey = $derived(`${timeOn}|${only ? [...only].sort().join() : ""}|${app.detail}`);
  const selectedEpisode = $derived(app.selection?.kind === "episode" ? app.selection.id : null);

  const onlyTime = $derived.by(() => {
    if (!only || !app.idx) return null;
    const scale = selectionPositions(app.ds!.episodes.filter((e) => only.has(e.id)), onlyLength);
    return { scale, items: itemPositions(app.links, app.idx, scale) };
  });

  // What was selected is drawn, even if the map's choice of topics would leave it out.
  const alwaysDrawn = $derived(app.visibleFocus ? [...app.livedIds, app.visibleFocus] : app.livedIds);
  const spec = $derived<GraphSpec>(app.idx
    ? buildGraph(app.mapEdges, app.idx, app.vp, { mode: app.mode, focus: selectedEpisode ?? app.visibleFocus, path: app.path,
        scene: app.scene, always: alwaysDrawn,
        // A lifespan window selects people, so the map keeps its own choice of items then.
        episodes: { links: app.mode === "focused" ? app.links : app.mapMentions, show: app.filters!.episodes, everyItem: !livedActive(app.filters!) },
        time: !timeOn ? null : only ? onlyTime : { scale: app.publication!, items: app.itemTimes } })
    : { nodes: [], edges: [] });
  // Height in the 3D view is when people lived, so publication dates and episodes stay out of it.
  const spec3D = $derived<GraphSpec>(app.ui.threeD && app.idx
    ? buildGraph(app.mapEdges, app.idx, app.vp, { mode: app.mode, focus: app.visibleFocus, path: app.path, scene: app.scene, always: alwaysDrawn })
    : { nodes: [], edges: [] });
  const exploring = $derived(app.mode === "focused" && !app.path && app.steps.length > 0);
  const root = $derived(app.steps[0]?.id ?? null);

  const reducedMotion = () => matchMedia("(prefers-reduced-motion: reduce)").matches;

  // The zoom the sizes are worked out for. It follows the real zoom in small steps.
  const ZOOM_STEP = Math.log(1.15);
  let styleZoom = 1;
  const z = () => styleZoom;
  const sizeOnScreen = (n: cytoscape.NodeSingular) => n.data("episode")
    ? marks.size * app.graph.nodeSize : screenDiameter(n.data("passages"), app.graph.nodeSize, n.data("lead"), z());
  const labelFont = (n: cytoscape.NodeSingular) => (n.data("episode") ? 10.5 : 10.5 + Math.min(2.5, Math.sqrt(n.data("passages")) / 3));

  function applyZoomToSizes() {
    if (!cy) return;
    const next = Math.exp(Math.round(Math.log(cy.zoom()) / ZOOM_STEP) * ZOOM_STEP);
    if (next === styleZoom) return;
    styleZoom = next;
    cy.style().update();
    writeLabels();
  }
  // Restyling thousands of elements takes a moment, so it waits until the zoom has come to rest.
  // Meanwhile everything scales with the view, which is the cheapest thing to draw.
  let sizeTimer = 0;
  function sizesSoon() {
    clearTimeout(sizeTimer);
    sizeTimer = window.setTimeout(applyZoomToSizes, 110);
  }

  /** Writes the names that fit, the most important first. Runs when the view has come to rest. */
  let labelTimer = 0;
  function labelsSoon() {
    clearTimeout(labelTimer);
    labelTimer = window.setTimeout(writeLabels, 90);
  }
  function writeLabels() {
    if (!cy) return;
    const list = cy.nodes().map((n) => {
      const p = n.renderedPosition();
      // Episodes come first as long as their names can all be written. A long run keeps them for zooming in.
      const important = n.hasClass("focus") ? 1e9 : n.hasClass("near") ? 1e6 : n.data("episode") && marks.rows <= 2 && marks.size >= 16 ? 1e5 : 0;
      return { id: n.id(), x: p.x, y: p.y, r: n.renderedWidth() / 2, text: n.data("label") as string, font: labelFont(n),
        priority: important + n.data("passages") * (n.data("lead") ? 1.35 : 1) };
    });
    const axis = untrack(() => timeOn) ? AXIS_HEIGHT : 0;
    const chosen = chooseLabels(list, cy.width(), cy.height() - axis);
    cy.batch(() => { for (const n of cy!.nodes()) { const on = chosen.has(n.id()); if (n.hasClass("named") !== on) n.toggleClass("named", on); } });
  }

  function stylesheet(): cytoscape.StylesheetJson {
    // Reading a CSS variable is slow, so each colour is read once per restyle, not once per node.
    const colours = new Map<string, string>();
    const t = (name: string) => colours.get(name) ?? (colours.set(name, cssToken(name)), colours.get(name)!);
    const g = app.graph;
    const font = "IBM Plex Sans, system-ui, sans-serif";
    return [
      { selector: "node", style: {
        "background-color": (n: cytoscape.NodeSingular) => t(typeStyle(n.data("type")).token),
        shape: (n: cytoscape.NodeSingular) => typeStyle(n.data("type")).shape,
        // Sizes are given in graph units, which the zoom scales. Dividing by the zoom makes them
        // sizes on screen. The zoom used here follows the view when it rests, see applyZoomToSizes.
        width: (n: cytoscape.NodeSingular) => sizeOnScreen(n) / z(),
        height: (n: cytoscape.NodeSingular) => sizeOnScreen(n) / z(),
        label: "", color: t("ink"), "font-family": font,
        "font-size": (n: cytoscape.NodeSingular) => labelFont(n) / z(),
        "min-zoomed-font-size": 0,
        "text-valign": "bottom", "text-margin-y": () => 2 / z(), "text-wrap": "ellipsis", "text-max-width": () => `${150 / z()}px`,
        "text-background-color": t("surface"), "text-background-opacity": 0.8, "text-background-padding": () => `${1.5 / z()}px`,
        "text-background-shape": "roundrectangle", "border-width": () => 1 / z(), "border-color": t("canvas"),
        // A name fades in when it is written, instead of appearing at once.
        "text-opacity": 0, "transition-property": "text-opacity", "transition-duration": reducedMotion() ? 0 : 0.2 } },
      // The fifth type on: a ring in place of a filled circle.
      { selector: "node[?ring]", style: { "background-color": t("canvas"), "border-width": () => 2 / z(), "border-color": t("neutral-2") } },
      // A portrait fills the node, if the node is large enough to show it. The ring keeps the type colour.
      ...(g.portraits ? [{ selector: "node[image]", style: {
        "background-image": "data(image)", "background-fit": "cover" as const, "background-clip": "node" as const,
        "background-image-crossorigin": "anonymous" as const, "background-image-opacity": (n: cytoscape.NodeSingular) => (sizeOnScreen(n) >= PORTRAIT_SIZE ? 1 : 0),
        "border-width": () => 2 / z(), "border-color": (n: cytoscape.NodeSingular) => t(typeStyle(n.data("type")).token) } }] : []),
      { selector: "node[?episode]", style: { "font-weight": 600 } },
      // Which names are written is decided by writeLabels. The selected and the pointed-at item always are.
      // Named items stand in front. The others recede, so a full map has a foreground to read.
      { selector: "node", style: { opacity: UNNAMED_OPACITY } },
      { selector: "node[?episode]", style: { opacity: 1, "z-index": 6 } },
      { selector: "node.named, node.focus, node.pointed", style: { label: "data(label)", "text-opacity": 1, opacity: 1, "z-index": 5 } },
      { selector: "node.focus", style: { "border-width": () => 3 / z(), "border-color": t("ink"), "font-weight": 600 } },
      { selector: "edge", style: {
        "curve-style": "bezier", width: (e: cytoscape.EdgeSingular) => Math.max(0.35, e.data("width") * g.linkThickness * Math.min(1, Math.sqrt(z()))) / z(),
        "line-color": t("edge-co"), "line-style": "solid",
        label: "", "font-family": font, "font-size": () => 10.5 / z(), color: t("ink-2"), "text-rotation": "autorotate",
        "text-background-color": t("surface"), "text-background-opacity": 0.95, "text-background-padding": () => `${2 / z()}px`,
        "text-background-shape": "roundrectangle" } },
      // Straight "haystack" lines are the cheapest edges to draw, and co-discussion has no direction.
      { selector: "edge.co", style: { "curve-style": "haystack", "haystack-radius": 0 } },
      // Links to episodes stay in the background, they are many and mostly say "named here".
      { selector: "edge.mention", style: { "curve-style": "haystack", "haystack-radius": 0, "line-style": "solid", "line-color": t("edge-episode") } },
      { selector: "edge.relation", style: { "line-style": "solid", "line-color": t("edge-relation"),
        "target-arrow-color": t("edge-relation"), "target-arrow-shape": g.arrows ? "triangle" : "none" } },
      { selector: "edge.relation.unverified", style: { "line-style": "dashed" } },
      { selector: "edge.lab, edge.hover, edge:selected", style: { label: "data(label)" } },
      // The links of what is pointed at or selected stand out from the faint web.
      { selector: "edge.near, edge.hover", style: { "line-color": t("ink-2"), "z-index": 9,
        width: (e: cytoscape.EdgeSingular) => Math.max(1.2, 1.8 * e.data("width") * g.linkThickness) / z() } },
      { selector: "edge:selected", style: { width: () => 3 / z(), "line-color": t("ink"), "target-arrow-color": t("ink") } },
      { selector: "node.hover, node.near, node.pointed", style: { "z-index": 10, opacity: 1 } },
      { selector: ".dim", style: { opacity: 0.15 } },
    ];
  }

  function elements(s: GraphSpec): cytoscape.ElementDefinition[] {
    return [
      ...s.nodes.map((n) => ({ group: "nodes" as const, data: { id: n.id, label: n.label, type: n.type, passages: n.passages, lead: n.lead, ...(n.episode ? { episode: true } : {}), ...(n.image ? { image: n.image } : {}), ...(typeStyle(n.type).ring ? { ring: true } : {}) },
        classes: n.focus ? "focus" : "" })),
      ...s.edges.map((e) => ({ group: "edges" as const, data: { id: e.id, source: e.source, target: e.target, label: e.label, width: e.width },
        classes: [e.kind === "relation" ? "relation" : e.kind === "mentioned_in" ? "mention" : "co", e.kind === "relation" && !e.verified ? "unverified" : "", e.showLabel ? "lab" : ""].join(" ") })),
    ];
  }

  function post(message: LayoutRequest) {
    worker?.postMessage(message);
  }

  /** Copies the newest positions onto the drawing once per frame, leaving a dragged node alone. */
  function draw() {
    frame = 0;
    if (!cy || !latest) return;
    labelsSoon();
    const xy = latest, store = positions[app.mode];
    latest = null;
    cy.batch(() => {
      order.forEach((id, i) => {
        const p = { x: xy[2 * i], y: xy[2 * i + 1] };
        const el = drawn.get(id);
        if (el && !el.grabbed()) el.position({ x: p.x * stretch, y: p.y });
        store.set(id, p);
      });
    });
    if (fitWhenSettled) { fitWhenSettled = false; arranging = false; if (followScreen) fitView(); sizesSoon(); }
  }

  function receive(ev: MessageEvent<LayoutReply>) {
    if (ev.data.run !== run) return;
    latest = ev.data.xy;
    if (ev.data.settled) fitWhenSettled = true;
    frame ||= requestAnimationFrame(draw);
  }

  function stopForces() {
    run++;
    post({ type: "stop" });
  }

  const forceSettings = (): Forces => ({ ...$state.snapshot(app.graph), aspect: cy && cy.height() ? cy.width() / cy.height() : 1 });

  /**
   * The zoom the picture will be seen at, as the room nodes need depends on it: their size is
   * fixed on screen, so at a low zoom they take up more of the graph. With the line of episodes it
   * is known. Without it, it is estimated from how long a line these items would need.
   */
  function plannedZoom(s: GraphSpec, mode: "full" | "focused"): number {
    const width = cy!.width();
    // Laid out for the details panel being open, so that opening it moves nothing.
    const usable = untrack(() => app.ui.panelOpen) || width < 2 * PANEL_WIDTH ? width : width - PANEL_WIDTH;
    const length = untrack(() => timeOn) ? untrack(() => lineLength) : 215 * Math.sqrt(Math.max(1, s.nodes.length));
    laidOutForZoom = (usable - 2 * AXIS_MARGIN) / length;
    return mode === "focused" && !untrack(() => timeOn) ? Math.min(1, laidOutForZoom) : laidOutForZoom;
  }

  /** The room a node must have for itself, in graph units, when seen at this zoom. */
  function roomFor(n: GraphSpec["nodes"][number], g: Forces, mode: "full" | "focused", zoom: number) {
    const size = n.episode ? marks.size * g.nodeSize : screenDiameter(n.passages, g.nodeSize, n.lead, zoom);
    return (size / 2 + NODE_ROOM[mode] * (g.gap / 12)) / zoom;
  }

  const simLinks = (s: GraphSpec, time: boolean): SimLink[] =>
    s.edges.map((e) => ({ id: e.id, source: e.source, target: e.target, weak: time && e.kind === "mentioned_in" }));

  /** Lays out the current elements with the force simulation (full and focused modes). */
  function startForces(s: GraphSpec) {
    worker ??= Object.assign(new Worker(new URL("../lib/layout.worker.ts", import.meta.url), { type: "module" }), { onmessage: receive });
    run++;
    const g = untrack(forceSettings);
    const mode = untrack(() => app.mode), first = untrack(() => root);
    // Each mode has its own length of the line, so a change of mode needs a new fit.
    if (mode !== laidOutMode) { laidOutMode = mode; followScreen = true; stretch = 1; }
    const time = untrack(() => timeOn), hub = untrack(() => !!only);
    const layout = untrack(() => layoutKey);
    if (laidOutWith !== layout) { positions.full.clear(); positions.focused.clear(); laidOutWith = layout; followScreen = true; stretch = 1; }
    const planned = plannedZoom(s, mode);
    const nodes: SimNode[] = s.nodes.map((n) => { const r = roomFor(n, g, mode, planned); return { id: n.id, r, keep: r, tx: n.tx }; });
    const links = simLinks(s, time);
    let fresh: boolean;
    if (mode === "focused") {
      if (first !== laidOutRoot) { positions.focused.clear(); laidOutRoot = first; followScreen = true; stretch = 1; }
      const known = new Set(nodes.filter((n) => positions.focused.has(n.id)).map((n) => n.id));
      fresh = seed(nodes, links, positions.focused, g.linkDistance) === 1;
      // In an exploration, items already placed stay where they are and only new ones move.
      for (const n of nodes) {
        if (known.has(n.id)) { n.fx = n.x; n.fy = n.y; }
        else if (fresh && !time && n.id === first) { n.x = n.fx = 0; n.y = n.fy = 0; }
      }
      clearTimeout(fitTimer);
      // With the axis the view stays where it is: it already shows the whole line.
      if (!fresh && !time && followScreen) fitTimer = window.setTimeout(() => cy?.animate({ fit: { eles: cy.elements(), padding: 50 } }, { duration: 400, complete: capZoom }), 900);
    } else {
      fresh = seed(nodes, links, positions.full) > 0.5;
      // Selecting something can add a few topics to the map. Those already there stay put.
      if (!fresh && time) for (const n of nodes) if (positions.full.has(n.id)) { n.fx = n.x; n.fy = n.y; }
    }
    if (time) placeAlongTheLine(s, nodes, mode, g);
    if (time && followScreen) fitView();
    // New elements are added at the origin. They wait at their starting place, not in one pile,
    // until the worker's first answer.
    cy!.batch(() => { for (const n of nodes) drawn.get(n.id)?.position({ x: n.x! * stretch, y: n.y! }); });
    order = nodes.map((n) => n.id);
    // A fresh layout is worked out before anything is shown, so nodes don't fly in from a spiral.
    // A large graph is also settled first: redrawing thousands of links on every step is slow.
    // On the line of episodes it is always settled first: new items appear at their place at once,
    // and there is no moment in which they lie on top of others.
    const settleFirst = fresh || time || reducedMotion() || links.length > LARGE;
    arranging = (fresh || links.length > LARGE) && settleFirst && nodes.length > 150;
    post({ type: "start", run, nodes, links, settings: g, settle: settleFirst });
  }

  /**
   * The picture when the episodes are on: they stand on a line through the middle, from one end of
   * the screen to the other, with items of all types above and below. The view zooms so the
   * line fills the width, so the height available in graph units follows from that.
   */
  function placeAlongTheLine(s: GraphSpec, nodes: SimNode[], mode: "full" | "focused", g: Forces) {
    const length = untrack(() => lineLength), alone = untrack(() => !!only);
    // The room around the line is worked out once per layout. Opening the details makes the graph
    // narrower, and items added after that must not push the others to new places.
    const store = positions[mode];
    const zoom = store.size && placedAt[mode] ? placedAt[mode]! : (placedAt[mode] = (cy!.width() - 2 * AXIS_MARGIN) / length);
    const half = Math.max(240, (cy!.height() - AXIS_HEIGHT - 2 * HEIGHT_MARGIN) / zoom / 2);
    const byId = new Map(s.nodes.map((n) => [n.id, n]));
    const episodes = nodes.filter((n) => byId.get(n.id)!.episode).sort((a, b) => (a.tx ?? 0) - (b.tx ?? 0));
    const fewer = Math.min(zoom, untrack(() => laidOutForZoom));
    const before = marks.size;
    marks = episodeMarks(episodes.length, length * fewer, g.nodeSize);
    if (marks.size !== before) cy!.style().update();
    const band = (marks.rows - 1) * marks.rowGap;
    const room = (LINE_ROOM + band / 2 + marks.size) / zoom;
    episodes.forEach((n, i) => {
      if (n.tx !== undefined) n.x = n.fx = n.tx;
      // First row to last row in turn, around the middle. The first of two is the lower one.
      const row = marks.rows === 2 ? (i % 2 ? -0.5 : 0.5) : i % marks.rows - (marks.rows - 1) / 2;
      n.y = n.fy = alone && episodes.length === 1 ? 0 : (row * marks.rowGap) / zoom;
      n.r = n.keep = roomFor(byId.get(n.id)!, g, mode, fewer);
    });
    const items = nodes.filter((n) => !byId.get(n.id)!.episode);
    // If the items need more room than the screen has, the picture gets higher and can be panned.
    // The width stays: that is what the time axis is fitted to.
    // Circles fill an area only so far, hence the margin.
    const high = Math.max(half, (items.reduce((sum, n) => sum + (2 * n.keep!) ** 2, 0) * ROOM_MARGIN[mode]) / (2 * length) + room);
    for (const side of [-1, 1]) {
      const here = items.filter((n) => sideOf(n.id) === side)
        .sort((a, b) => (a.tx ?? 0) - (b.tx ?? 0) || spot(a.id) - spot(b.id));
      here.forEach((n, i) => {
        // Items are spread evenly over the width, in order of when they are discussed, with their
        // date still counting for part of it. Most averages fall in the middle of the run, so by
        // date alone they would all gather there. Heights follow the golden ratio, which fills
        // the height evenly without any pattern of rows.
        const even = -length / 2 + ((i + 0.5) / here.length) * length;
        n.tx = n.tx === undefined || alone ? even : EVEN_SHARE * even + (1 - EVEN_SHARE) * n.tx;
        n.ty = side * (room + ((i * 0.618034) % 1) * (high - room));
        n.box = [-length / 2, length / 2, side < 0 ? -high : room, side < 0 ? -room : high];
        if (!store.has(n.id)) { n.x = n.tx; n.y = n.ty; }
      });
    }
  }

  /** The zoom at which the line of episodes fills the width, if there is a line. */
  function lineZoom(): number | null {
    return cy && timeOn && lineLength > 0 ? (cy.width() - 2 * AXIS_MARGIN) / lineLength : null;
  }
  const atFullWidth = () => { const z = untrack(lineZoom); return z !== null && cy!.zoom() <= z * 1.001; };

  /** How far the view can zoom out: to full width, and further only if the graph is taller than
   * the screen, until all of it fits. Going further would show nothing new. */
  function zoomFloor(): number {
    const fit = lineZoom();
    if (fit === null || !cy) return MIN_ZOOM;
    const box = cy.nodes().boundingBox({ includeLabels: false });
    const room = cy.height() - AXIS_HEIGHT;
    return box.h * fit <= room ? fit : Math.min(fit, (room / box.h) * 0.95);
  }

  /** Where the view can pan to sideways at this zoom, so that the line always spans the screen. */
  function keepLineOnScreen(zoom: number, x: number): number {
    const w = cy!.width(), half = (untrack(() => lineLength) * zoom * stretch) / 2;
    return Math.min(AXIS_MARGIN + half, Math.max(w - AXIS_MARGIN - half, x));
  }

  /** Below full width the x positions spread by exactly as much as the zoom went down. */
  function setStretch(next: number) {
    if (next === stretch) return;
    const ratio = next / stretch;
    stretch = next;
    cy!.batch(() => { for (const el of drawn.values()) el.position("x", el.position("x") * ratio); });
  }

  /** One step of zooming, around a point on the screen. */
  function zoomTo(level: number, at: { x: number; y: number }) {
    if (!cy) return;
    const fit = untrack(lineZoom);
    if (fit === null) { cy.zoom({ level, renderedPosition: at }); return; }
    const old = cy.zoom(), pan = cy.pan();
    let y = at.y - ((at.y - pan.y) / old) * level;
    const x = at.x - ((at.x - pan.x) / old) * level;
    setStretch(level < fit ? fit / level : 1);
    if (level <= fit) {
      // At or below full width the graph is kept in view: centred on the line while its height
      // fits on the screen, and inside its height when it does not.
      const box = cy.nodes().boundingBox({ includeLabels: false });
      const room = cy.height() - AXIS_HEIGHT, high = box.h * level;
      y = high <= room ? room / 2 - ((box.y1 + box.y2) / 2) * level : Math.min(HEIGHT_MARGIN - box.y1 * level, Math.max(room - HEIGHT_MARGIN - box.y2 * level, y));
      followScreen = true;
    }
    cy.viewport({ zoom: level, pan: { x: level < fit ? cy.width() / 2 : keepLineOnScreen(level, x), y } });
  }

  /** The whole line of episodes across the width with the line itself in the middle, or everything. */
  function fitView(animate = false) {
    if (!cy) return;
    const length = untrack(() => lineLength), line = untrack(() => timeOn) && length > 0;
    if (!line) { setStretch(1); cy.fit(undefined, 40); capZoom(); return; }
    // A view zoomed out past full width stays as far out when the screen changes.
    const zoom = (cy.width() - 2 * AXIS_MARGIN) / length / stretch;
    const pan = { x: cy.width() / 2, y: (cy.height() - AXIS_HEIGHT) / 2 };
    if (animate && !reducedMotion()) cy.animate({ zoom, pan }, { duration: 250 });
    else cy.viewport({ zoom, pan });
    // Sizes depend on the zoom. A view set before the graph is first drawn must not keep the first ones.
    sizesSoon();
  }

  /** On a narrower screen than the layout was made for, nodes that now touch move apart. */
  let roomTimer = 0;
  let laidOutForZoom = Infinity;
  function makeRoom() {
    const zoom = untrack(lineZoom);
    if (zoom === null || zoom >= laidOutForZoom * 0.99) return;
    laidOutForZoom = zoom;
    const g = untrack(forceSettings), mode = untrack(() => app.mode);
    post({ type: "room", run, keep: Object.fromEntries(untrack(() => spec).nodes.map((n) => [n.id, roomFor(n, g, mode, zoom)])) });
  }

  /** Brings the selected item's neighbourhood forward, so one item can be looked at in a full map. */
  function showSelection() {
    if (!cy) return;
    const sel = untrack(() => app.selection);
    const id = sel?.kind === "entity" ? sel.id : sel?.kind === "episode" ? sel.id : null;
    const el = id ? cy.getElementById(id) : null;
    cy.batch(() => {
      cy!.elements().removeClass("dim near hover pointed");
      // Two topics together: the two, what is linked to both, and those links stand out.
      if (sel?.kind === "pair") {
        const [a, b] = [cy!.getElementById(sel.a), cy!.getElementById(sel.b)];
        if (!a.length || !b.length) return;
        const common = a.neighborhood("node").intersection(b.neighborhood("node")).filter((n) => !n.data("episode"));
        const nodes = common.union(a).union(b);
        const edges = nodes.edgesWith(nodes).intersection(a.connectedEdges().union(b.connectedEdges()));
        cy!.elements().not(nodes.union(edges)).addClass("dim");
        nodes.addClass("near");
        edges.addClass("near");
        return;
      }
      if (!el?.length) return;
      const around = el.closedNeighborhood();
      cy!.elements().not(around).addClass("dim");
      around.nodes().addClass("near");
      el.connectedEdges().addClass("near");
    });
    labelsSoon();
  }

  /** A connection reads left to right in one row, so the link labels stay level and clear of the nodes. */
  function pathLayout() {
    stopForces();
    const step = new Map(app.path!.nodes.map((id, i) => [id, { x: i * 280, y: 0 }]));
    cy!.layout({ name: "preset", positions: (n: cytoscape.NodeSingular) => step.get(n.id()) ?? { x: 0, y: 0 }, fit: true, padding: 80 } as cytoscape.LayoutOptions).run();
    capZoom();
    if (cy!.zoom() > 1.3) cy!.zoom({ level: 1.3, renderedPosition: { x: cy!.width() / 2, y: cy!.height() / 2 } });
  }

  // Fitting a handful of items would blow them up to fill the screen. Past this zoom the
  // view stays centred instead.
  const FIT_MAX_ZOOM = 1.3;
  function capZoom() {
    if (cy && cy.zoom() > FIT_MAX_ZOOM) cy.zoom({ level: FIT_MAX_ZOOM, renderedPosition: { x: cy.width() / 2, y: cy.height() / 2 } });
  }

  function create(): cytoscape.Core {
    const core = cytoscape({ container, style: stylesheet(), minZoom: MIN_ZOOM / 20, maxZoom: 4 });
    glide = zoomGlide(() => core.zoom(), zoomTo, () => [untrack(zoomFloor), core.maxZoom()], reducedMotion);
    viewport = { zoom: core.zoom(), pan: core.pan().x, width: core.width() };
    // Hover dims everything else, done in one batch so the graph is restyled once.
    core.on("tap", "node", (ev) => {
      menu = null;
      if (ev.target.data("episode")) app.selectEpisode(ev.target.id());
      else app.selectEntity(ev.target.id());
    });
    core.on("viewport resize", () => {
      viewportFrame ||= requestAnimationFrame(() => {
        viewportFrame = 0;
        viewport = { zoom: core.zoom() * stretch, pan: core.pan().x, width: core.width() };
      });
      sizesSoon();
      labelsSoon();
    });
    core.on("tap", (ev) => {
      if (ev.target !== core) return;
      menu = null;
      // A click on empty space lets go of what is selected. A connection stays until it is cleared.
      if (!app.path || app.selection?.kind !== "path") app.closePanel();
    });
    core.on("dragpan pinchzoom", () => {
      // The line never leaves the screen's width, so at full width only up and down can move.
      if (untrack(lineZoom) === null) { followScreen = false; return; }
      core.pan({ x: keepLineOnScreen(core.zoom(), core.pan().x), y: core.pan().y });
      if (!atFullWidth()) followScreen = false;
    });
    core.on("resize", () => {
      if (followScreen) fitView();
      clearTimeout(roomTimer);
      roomTimer = window.setTimeout(makeRoom, 200);
    });
    core.on("dbltap", "node", (ev) => app.moreOf(ev.target.id()));
    core.on("cxttap", "node", (ev) => {
      const p = ev.target.renderedPosition();
      menu = { id: ev.target.id(), x: p.x, y: p.y };
    });
    core.on("tap", "edge", (ev) => {
      const edge = app.allLinks.find((e) => e.id === ev.target.id());
      if (edge) return app.selectEdge(edge);
      const link = app.links.find((l) => l.id === ev.target.id());
      if (link) app.selectMention(link);
    });
    core.on("mouseover", "edge", (ev) => ev.target.addClass("hover"));
    core.on("mouseout", "edge", (ev) => ev.target.removeClass("hover"));
    core.on("mouseover", "node", (ev) => glide?.active() ? undefined : core.batch(() => {
      const near = ev.target.closedNeighborhood();
      core.elements().not(near).addClass("dim");
      core.elements().removeClass("dim");
      core.elements().not(near).addClass("dim");
      near.nodes().addClass("hover");
      ev.target.addClass("pointed");
      ev.target.connectedEdges().addClass("hover");
    }));
    core.on("mouseout", "node", () => glide?.active() ? undefined : showSelection());
    // Dragging pins a node and the rest of the graph follows, like in Obsidian.
    // Only a real drag sets things moving. A plain click to select an item must leave the graph still.
    let dragging = false;
    core.on("grab", "node", () => { dragging = false; });
    core.on("drag", "node", (ev) => {
      if (app.path) return;
      const p = ev.target.position();
      post({ type: "pin", run, id: ev.target.id(), x: p.x / stretch, y: p.y, heat: !dragging && !reducedMotion() });
      dragging = true;
    });
    core.on("free", "node", (ev) => {
      if (app.path || !dragging) return;
      // In an exploration a dragged item stays where it was put.
      post({ type: "release", run, id: ev.target.id(), keep: app.mode === "focused" });
    });
    return core;
  }

  // Only rebuild when the set of drawn elements changes. Otherwise just restyle, so the
  // layout stays put while clicking around.
  $effect(() => {
    const s = spec;
    if (!container) return;
    cy ??= create();
    const nodes = `${app.mode}|${app.path ? "path" : ""}|${layoutKey}|${s.nodes.map((n) => n.id).join()}`;
    const edges = s.edges.map((e) => e.id).join();
    // Only the links of one item changed: nothing else moves or is rebuilt.
    if (nodes === nodesKey && edges !== edgesKey && !untrack(() => app.path)) {
      edgesKey = edges;
      const wanted = new Set(s.edges.map((e) => e.id));
      cy.batch(() => {
        cy!.edges().filter((e) => !wanted.has(e.id())).remove();
        const have = new Set(cy!.edges().map((e) => e.id()));
        cy!.add(elements({ nodes: [], edges: s.edges.filter((e) => !have.has(e.id)) }));
      });
      post({ type: "links", run, links: simLinks(s, untrack(() => timeOn)), settings: untrack(forceSettings) });
      untrack(showSelection);
      return;
    }
    if (nodes !== nodesKey || edges !== edgesKey) {
      nodesKey = nodes;
      edgesKey = edges;
      cy.elements().remove();
      cy.add(elements(s));
      drawn = new Map(cy.nodes().map((n) => [n.id(), n]));
      // Internal renderer flag: cytoscape reads it on every frame, so it can follow the graph size.
      (cy as unknown as { renderer(): { textureOnViewport: boolean } }).renderer().textureOnViewport = s.edges.length > LARGE;
      if (untrack(() => app.path)) pathLayout();
      else startForces(s);
      // Episodes mark the time axis, so they stay at their date.
      if (untrack(() => timeOn)) cy.nodes("[?episode]").ungrabify();
      untrack(showSelection);
      return;
    }
    cy.batch(() => {
      for (const n of s.nodes) cy!.getElementById(n.id).toggleClass("focus", n.focus);
      for (const e of s.edges) cy!.getElementById(e.id).toggleClass("lab", e.showLabel);
    });
    void app.selection;
    untrack(showSelection);
    const focused = app.focus ? cy.getElementById(app.focus) : null;
    // Bring a selected item into view only when it is off screen.
    const box = focused?.renderedBoundingBox();
    if (focused?.length && box && (box.x2 < 0 || box.y2 < 0 || box.x1 > cy.width() || box.y1 > cy.height()))
      cy.animate({ center: { eles: focused } }, { duration: 300 });
  });

  // Force sliders work live: the running simulation gets the new forces and heats up again. Not
  // on the first run, when nothing has changed: that would set a finished layout moving again.
  let forcesSeen = false;
  $effect(() => {
    void [app.graph.center, app.graph.repel, app.graph.linkForce, app.graph.linkDistance, app.graph.gap, app.graph.nodeSize];
    if (!forcesSeen) { forcesSeen = true; return; }
    untrack(() => {
      const g = forceSettings();
      if (app.path) return;
      const radii = Object.fromEntries(spec.nodes.map((n) => [n.id, roomFor(n, g, app.mode, plannedZoom(spec, app.mode))]));
      post({ type: "forces", run, settings: g, radii, settle: reducedMotion() });
    });
  });

  // Display settings and theme changes only restyle.
  $effect(() => {
    void $state.snapshot(app.graph);
    cy?.style(stylesheet());
  });

  // The wheel is handled here and not by Cytoscape, so a mouse and a trackpad zoom alike (zoom.ts).
  // Listening one level above the canvas and before it means Cytoscape never gets these events.
  $effect(() => {
    const onWheel = (ev: WheelEvent) => {
      if (!cy || !container.contains(ev.target as Node)) return;
      ev.preventDefault();
      ev.stopPropagation();
      followScreen = false;
      const box = container.getBoundingClientRect();
      glide?.wheel(ev, { x: ev.clientX - box.left, y: ev.clientY - box.top });
    };
    stage.addEventListener("wheel", onWheel, { capture: true, passive: false });
    return () => { stage.removeEventListener("wheel", onWheel, { capture: true }); glide?.stop(); };
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

  onDestroy(() => { clearTimeout(sizeTimer); clearTimeout(labelTimer); clearTimeout(roomTimer); worker?.terminate(); cancelAnimationFrame(frame); cancelAnimationFrame(viewportFrame); clearTimeout(fitTimer); cy?.destroy(); });

  /** Closes the node menu, then runs its action on the item it was opened for. */
  function fromMenu(action: (id: string) => void) {
    const id = menu!.id;
    menu = null;
    action(id);
  }

  const label = (id: string) => app.idx!.entity.get(id)?.label ?? app.idx!.episode.get(id)?.code ?? id;
  const pairLabel = (a: string, b: string) => `${label(a)} and ${label(b)}`;
  const isEpisode = (id: string) => app.idx!.episode.has(id);
  // The steps that change what is in the picture. A slider is undone with Undo but needs no place in the line.
  const trail = $derived(app.steps.filter((s) => s.kind !== "share"));
  const STEP_WORDS = { start: "", share: "", add: "added", hide: "hid", together: "shared by" } as const;
  const onlyCodes = $derived(only ? [...only].map((id) => app.idx!.episode.get(id)?.code ?? id).sort() : []);
  const DETAILS = [["main", `Most discussed`, DETAIL_SIZE.main], ["more", "More", DETAIL_SIZE.more], ["all", "All", null]] as const;
  // The map only has a choice to make when there are more topics than the smallest picture holds.
  const canChoose = $derived(!only && !livedActive(app.filters!) && app.topicCount > DETAIL_SIZE.main);

  const zoomBy = (factor: number) => {
    if (!cy) return;
    const level = Math.max(zoomFloor(), Math.min(cy.maxZoom(), cy.zoom() * factor));
    if (level > (lineZoom() ?? Infinity) * 1.001 || lineZoom() === null) followScreen = false;
    zoomTo(level, { x: cy.width() / 2, y: cy.height() / 2 });
  };
</script>

<div class="canvas card">
  <!-- What is being looked at on the left, and things that open something on the right. -->
  {#if !app.ui.full}
    <div class="toolbar">
      <nav class="scope" aria-label="What the map shows">
        <BackToResults />
        {#if app.path || app.selection?.kind === "nopath"}
          <b>Connection</b>
          {#if app.path}<span class="crumbs"><span>{label(app.path.nodes[0])}</span><span>{label(app.path.nodes.at(-1)!)}</span></span>{/if}
          <button class="btn ghost" onclick={() => app.clearConnection()}>Clear</button>
        {:else if exploring}
          <button class="linkish crumb" onclick={() => app.wholeMap()} title="See the whole map. This exploration is kept.">Whole map</button>
          <ol class="crumbs">
            {#each trail as step, i (i)}
              {#if step.kind === "together"}
                <!-- The way back to the pair's list and slider after looking at something else. -->
                <li class:last={i === trail.length - 1}>{STEP_WORDS.together} <button class="linkish pairlink" onclick={() => app.selectPair(step.a, step.b)}>{pairLabel(step.a, step.b)}</button></li>
              {:else}
                <li class:last={i === trail.length - 1}>{STEP_WORDS[step.kind]}{STEP_WORDS[step.kind] ? " " : ""}<b>{label(step.id)}</b></li>
              {/if}
            {/each}
          </ol>
          <button class="btn ghost" disabled={app.steps.length < 2} onclick={() => app.back()} title="Undo the last step">Undo</button>
          <span class="add"><Search compact placeholder="Add a topic to this" onpick={(e) => app.addToScene(e.id)} /></span>
        {:else if only}
          <b>Only {onlyCodes.slice(0, 3).join(", ")}{onlyCodes.length > 3 ? ` and ${onlyCodes.length - 3} more` : ""}</b>
          <button class="btn ghost" onclick={() => app.showAllEpisodes()}>Show all episodes</button>
        {:else}
          <b>Whole map</b>
          {#if canChoose}
            <span class="seg" role="group" aria-label="How many topics to draw">
              {#each DETAILS as [id, name, size] (id)}
                <button aria-pressed={app.detail === id} title={size ? `The ${size} topics with most passages` : `All ${app.topicCount} topics`} onclick={() => (app.detail = id)}>{name}</button>
              {/each}
            </span>
          {/if}
          {#if app.steps.length}<button class="linkish" onclick={() => app.resumeExploration()}>Back to {label(app.steps[0].id)}</button>{/if}
        {/if}
      </nav>
      <span class="actions">
        <button class="btn ghost" aria-expanded={app.ui.connectOpen} onclick={() => (app.ui.connectOpen = !app.ui.connectOpen)}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M9 15l6-6M7 11l-2 2a4 4 0 0 0 6 6l2-2M17 13l2-2a4 4 0 0 0-6-6l-2 2" /></svg>Find a connection
        </button>
        <span class="seg" role="group" aria-label="2D or 3D">
          <button aria-pressed={!app.ui.threeD} onclick={() => (app.ui.threeD = false)} title="The flat graph">2D</button>
          <button aria-pressed={app.ui.threeD} onclick={() => (app.ui.threeD = true)} disabled={!app.hasLifespans}
            title={app.hasLifespans ? `A space-time cube: height is when the ${typeStyle(app.idx!.site.lifespan_type!).plural} lived, the network spreads out sideways` : "This archive has no lifespans to show in 3D"}>3D</button>
        </span>
        <button class="btn quiet icon" aria-label="Give the graph the whole window" onclick={() => (app.ui.full = true)} title="Give the graph the whole window">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 9V3h6M21 9V3h-6M3 15v6h6M21 15v6h-6" /></svg>
        </button>
      </span>
    </div>
    {#if !app.path}<LegendRow spec={app.ui.threeD ? spec3D : spec} />{/if}
  {/if}
  <div class="stage" bind:this={stage}>
    <div class="cy" class:arranging class:away={app.ui.threeD} bind:this={container} role="img" aria-label="Graph of {spec.nodes.length} topics and {spec.edges.length} connections"></div>
    {#if app.ui.threeD}
      {#await import("./Graph3D.svelte")}
        <p class="empty center">Loading the 3D view…</p>
      {:then three}
        <three.default spec={spec3D} />
      {:catch}
        <p class="empty center">The 3D view could not be started. It needs a browser with WebGL.</p>
      {/await}
    {/if}
    {#if arranging}
      <p class="empty center">Arranging {spec.nodes.length} topics…</p>
    {/if}
    {#if !spec.nodes.length}
      <div class="center nothing">
        <p class="empty">Nothing to draw with the current filters.</p>
        <button class="btn ghost" onclick={() => app.resetFilters()}>Reset the filters</button>
      </div>
    {/if}

    {#if timeOn && !app.ui.threeD}
      <PublicationAxis scale={app.publication!} {...viewport} />
    {/if}

    <!-- These stay in full view: it gives the graph more room, not fewer ways to work with it. -->
    <SelectionBar />
    {#if app.ui.connectOpen}<ConnectBox />{/if}

    {#if menu && !app.ui.threeD}
      <div class="menu card" role="menu" style:left="{menu.x}px" style:top="{menu.y}px">
        <strong>{label(menu.id)}</strong>
        {#if isEpisode(menu.id)}
          <button role="menuitem" onclick={() => fromMenu((id) => app.showOnly([id]))}>Show only this episode</button>
          <button role="menuitem" onclick={() => fromMenu((id) => app.pickEpisode(id))}>{app.picked.has(menu.id) ? "Remove from your choice" : "Pick to compare"}</button>
          <button role="menuitem" onclick={() => fromMenu((id) => app.readEpisode(id))}>Read the transcript</button>
          {#if only}<button role="menuitem" onclick={() => fromMenu(() => app.showAllEpisodes())}>Show all episodes</button>{/if}
        {:else}
          <button role="menuitem" onclick={() => fromMenu((id) => app.moreOf(id))}>Show more connections</button>
          {#if exploring && menu.id !== root}<button role="menuitem" onclick={() => fromMenu((id) => app.hide(id))}>Hide</button>{/if}
          <button role="menuitem" onclick={() => fromMenu((id) => app.openEntity(id))}>{exploring || app.path ? "Start a new exploration here" : "Explore from here"}</button>
          <button role="menuitem" onclick={() => fromMenu((id) => { app.connectFrom = id; app.ui.connectOpen = true; })}>Find a connection</button>
        {/if}
      </div>
    {/if}

    {#if app.ui.full && !app.ui.connectOpen}
      <span class="leave">
        {#if exploring && app.steps.length > 1}<button class="btn ghost" onclick={() => app.back()} title="Undo the last step">Undo</button>{/if}
        <button class="btn ghost" onclick={() => (app.ui.full = false)} title="Back to the normal view (Esc)">Leave full view</button>
      </span>
    {/if}
    {#if !app.ui.threeD}
      <div class="overlay zoom card" class:raised={timeOn} role="group" aria-label="Zoom">
        <button aria-label="Zoom in" title="Zoom in" onclick={() => zoomBy(1.25)}>+</button>
        <button aria-label="Zoom out" title="Zoom out" onclick={() => zoomBy(0.8)}>−</button>
        <button aria-label="Fit the graph to the screen" title="Fit the graph to the screen" onclick={() => { followScreen = true; setStretch(1); fitView(true); }}>⤢</button>
      </div>
    {/if}
  </div>
</div>

<style>
  .canvas { display: flex; flex-direction: column; min-width: 0; min-height: 0; height: 100%; container-type: inline-size; }
  .toolbar { display: flex; flex-wrap: wrap; gap: 6px 14px; align-items: center; padding: 6px 8px 6px 12px; border-bottom: 1px solid var(--line); }
  .scope { flex: 1 1 320px; min-width: 0; display: flex; flex-wrap: wrap; align-items: center; gap: 4px 10px; font-size: 13px; }
  .scope > b { font-weight: 600; font-size: 14px; }
  .crumb { flex: none; font-size: 13px; }
  .crumbs { display: flex; flex-wrap: wrap; gap: 4px; margin: 0; padding: 0; list-style: none; min-width: 0; color: var(--ink-3); }
  .crumbs > * + *::before { content: "›"; margin-right: 4px; color: var(--ink-3); }
  .crumbs li b { font-weight: 500; color: var(--ink-2); }
  .crumbs li.last b { color: var(--ink); font-weight: 600; }
  .pairlink { font-size: 13px; font-weight: 600; color: var(--ink); }
  .add { flex: 0 1 230px; min-width: 140px; display: flex; }
  .actions { margin-left: auto; display: inline-flex; align-items: center; flex-wrap: wrap; gap: 6px; }
  @media (max-width: 760px) { .toolbar { padding: 4px 6px 4px 10px; } .scope { flex-basis: 100%; } .actions { margin-left: 0; } }
  .leave { position: absolute; z-index: 7; top: 8px; right: 10px; display: flex; gap: 6px; }
  .leave .btn { box-shadow: var(--shadow); }
  .menu { position: absolute; z-index: 8; display: flex; flex-direction: column; padding: 6px; min-width: 210px; box-shadow: var(--shadow);
    transform: translate(12px, 12px); }
  .menu strong { font-size: 12.5px; padding: 4px 8px 6px; color: var(--ink-3); font-weight: 600; }
  .menu button { border: 0; background: none; text-align: left; padding: 7px 8px; border-radius: 6px; cursor: pointer; color: inherit; font: inherit; font-size: 13px; }
  .menu button:hover { background: var(--surface-2); }
  .stage { position: relative; flex: 1; min-height: 0; background: var(--canvas); border-radius: 0 0 var(--radius) var(--radius); }
  .cy { position: absolute; inset: 0; z-index: 1; transition: opacity 0.2s; }
  .cy.arranging { opacity: 0; }
  /* Kept alive under the 3D view, so switching back is instant and keeps the 2D layout. */
  .cy.away { visibility: hidden; }
  .center { position: absolute; inset: 0; z-index: 2; display: grid; place-items: center; text-align: center; padding: 24px; pointer-events: none; }
  .nothing { place-content: center; gap: 10px; }
  .nothing button { pointer-events: auto; }
  .nothing p { margin: 0; }
  .overlay { position: absolute; z-index: 5; }
  .raised { margin-bottom: 34px; }
  .zoom { right: 10px; bottom: 10px; display: flex; flex-direction: column; overflow: hidden; }
  .zoom button { border: 0; background: none; width: 32px; height: 30px; cursor: pointer; font-size: 15px; color: var(--ink-2); }
  .zoom button + button { border-top: 1px solid var(--line); }
  .zoom button:hover { background: var(--surface-2); }
  @media (pointer: coarse) { .zoom button { width: 40px; height: 40px; } }
</style>
