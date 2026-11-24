<script lang="ts">
  // The 3D graph as a space-time cube: height is when a person lived, the network spreads
  // out sideways. Other topics sit at the time of the dated ones they are linked to.
  // It draws the same nodes and links as the 2D view, so filters and the exploration apply.
  // This file and its libraries (three.js) only load when the 3D view is switched on.
  import ForceGraph3D, { type ForceGraph3DInstance } from "3d-force-graph";
  import * as THREE from "three";
  import SpriteText from "three-spritetext";
  import { onDestroy, untrack } from "svelte";
  import { app } from "../lib/state.svelte.ts";
  import { cssToken, typeStyle, type Shape } from "../lib/entityStyle.ts";
  import { formatYear, lifeLabel, lifeRange } from "../lib/format.ts";
  import { placeInTime, timeScale, yearOf } from "../lib/timeScale.ts";
  import { sizedPassages } from "../lib/forces.ts";
  import type { GraphSpec } from "../lib/graph.ts";
  // What the archive calls the type that has dates, "people" in most podcasts.
  const datedPlural = $derived(typeStyle(app.idx!.site.lifespan_type ?? "").plural);

  let { spec }: { spec: GraphSpec } = $props();

  interface Node3D {
    id: string; label: string; type: string; passages: number; lead: boolean; image?: string;
    year: number | null; inferred: boolean;
    x?: number; y?: number; z?: number; fy?: number;
  }
  interface Link3D { id: string; source: string | Node3D; target: string | Node3D; kind: string; label: string }

  /** Height of the time axis in scene units. */
  const AXIS = 900;
  /** Items with no date and no dated neighbour sit on their own plane under the axis. */
  const UNDATED_Y = -AXIS / 2 - 140;
  /** Above this many items only the larger ones get a written label; all show one on hover. */
  const LABELLED = 70;

  let container: HTMLDivElement;
  let graph: ForceGraph3DInstance | null = null;
  let previous = new Map<string, Node3D>();
  let materials = new Map<string, THREE.Material[]>();
  let rings: THREE.Group | null = null;
  let slab: THREE.Mesh | null = null;
  let fitted = false;
  let theme = $state(0);
  /** Index of the highlighted period between two century planes, -1 for none. */
  let period = $state(-1);

  const idx = $derived(app.idx!);
  const datedType = $derived(idx.site.lifespan_type);
  // The archive's own presets give the sensible range, so a stray date can't stretch the axis.
  const clamp = $derived.by((): [number, number] | null => {
    const p = idx.site.lived_presets;
    return p.length ? [Math.min(...p.map((x) => x.start)), Math.max(...p.map((x) => x.end))] : null;
  });

  const layout = $derived.by(() => {
    const dated = new Map<string, number>();
    for (const n of spec.nodes) {
      const e = idx.entity.get(n.id)!;
      const y = e.type === datedType ? yearOf(e) : null;
      if (y !== null) dated.set(n.id, y);
    }
    const placed = placeInTime(spec.nodes.map((n) => n.id), spec.edges, dated);
    // The axis follows the dated people of the whole archive, so it doesn't jump while exploring.
    const all = app.ds!.entities.filter((e) => e.type === datedType).map(yearOf).filter((y): y is number => y !== null);
    return { ...placed, scale: timeScale(all, AXIS, clamp) };
  });
  const height = (year: number | null) => (year === null ? UNDATED_Y : layout.scale.height(year) - AXIS / 2);
  const periods = $derived(layout.scale.ticks.slice(0, -1).map((t, i) => [t, layout.scale.ticks[i + 1]] as const));

  // Colours come from the page theme, read once per redraw.
  function colours() {
    const read = new Map<string, string>();
    return (name: string) => read.get(name) ?? (read.set(name, cssToken(name)), read.get(name)!);
  }
  let colour = colours();

  let circleTexture: THREE.Texture | null = null;
  /** A soft-edged white disc, used to cut portraits round and to draw their ring. */
  function circle(): THREE.Texture {
    if (circleTexture) return circleTexture;
    const canvas = document.createElement("canvas");
    canvas.width = canvas.height = 128;
    const g = canvas.getContext("2d")!;
    g.fillStyle = "#fff";
    g.beginPath(); g.arc(64, 64, 62, 0, Math.PI * 2); g.fill();
    circleTexture = new THREE.CanvasTexture(canvas);
    return circleTexture;
  }
  const loader = new THREE.TextureLoader().setCrossOrigin("anonymous");

  function geometry(shape: Shape, r: number): THREE.BufferGeometry {
    return shape === "tag" ? new THREE.ConeGeometry(r, r * 1.8, 12) : new THREE.SphereGeometry(r, 20, 14);
  }

  const radius = (n: Node3D) => (2.5 + Math.sqrt(sizedPassages(n.passages)) * 1.1) * (n.lead ? 1.35 : 1) * (n.inferred ? 0.8 : 1);

  function nodeObject(n: Node3D, labelled: boolean): THREE.Object3D {
    const group = new THREE.Group();
    const r = radius(n), tint = colour(typeStyle(n.type).token);
    const used: THREE.Material[] = [];
    if (n.image && app.graph.portraits) {
      // The ring keeps the type colour, as in 2D; the portrait sits in front of it.
      const ring = new THREE.SpriteMaterial({ color: tint, alphaMap: circle(), transparent: true, depthWrite: false });
      const photo = new THREE.SpriteMaterial({ map: loader.load(n.image), alphaMap: circle(), transparent: true });
      const back = new THREE.Sprite(ring), front = new THREE.Sprite(photo);
      back.scale.set(r * 2.5, r * 2.5, 1);
      front.scale.set(r * 2.2, r * 2.2, 1);
      front.renderOrder = 1;
      group.add(back, front);
      used.push(ring, photo);
    } else {
      // A ring type is a sphere of lines, the closest to a ring in 3D.
      const material = new THREE.MeshLambertMaterial({ color: tint, transparent: true, opacity: n.inferred ? 0.75 : 1, wireframe: typeStyle(n.type).ring });
      group.add(new THREE.Mesh(geometry(typeStyle(n.type).shape, r), material));
      used.push(material);
    }
    if (labelled) {
      const text = new SpriteText(n.label, Math.max(6, Math.min(11, r * 1.1)), colour("ink"));
      text.fontFace = "IBM Plex Sans, system-ui, sans-serif";
      text.position.y = -(r * 1.3 + 4);
      group.add(text);
      used.push(text.material);
    }
    materials.set(n.id, used);
    return group;
  }

  /** Century planes: a faint ring at each labelled year, wide enough to hold the network. */
  function drawRings() {
    if (!graph) return;
    if (rings) graph.scene().remove(rings);
    rings = new THREE.Group();
    const nodes = graph.graphData().nodes as Node3D[];
    const reach = Math.max(120, ...nodes.map((n) => Math.hypot(n.x ?? 0, n.z ?? 0))) + 30;
    const line = new THREE.LineBasicMaterial({ color: colour("line"), transparent: true, opacity: 0.7 });
    const ring = (y: number, label: string, ink: string) => {
      const points = new THREE.EllipseCurve(0, 0, reach, reach).getPoints(96).map((p) => new THREE.Vector3(p.x, y, p.y));
      rings!.add(new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(points), line));
      const text = new SpriteText(label, 15, colour(ink));
      text.fontFace = "IBM Plex Mono, ui-monospace, monospace";
      text.position.set(reach + 40, y, 0);
      rings!.add(text);
    };
    for (const t of layout.scale.ticks) ring(t.y - AXIS / 2, t.label, "ink-3");
    if (nodes.some((n) => n.year === null)) ring(UNDATED_Y, "no date", "ink-3");
    graph.scene().add(rings);
    drawSlab(reach);
  }

  /** A translucent band over the highlighted period. */
  function drawSlab(reach: number) {
    if (!graph) return;
    if (slab) { graph.scene().remove(slab); slab = null; }
    const chosen = periods[period];
    if (!chosen) return;
    const [a, b] = [chosen[0].y - AXIS / 2, chosen[1].y - AXIS / 2];
    slab = new THREE.Mesh(new THREE.CylinderGeometry(reach, reach, b - a, 64, 1, true),
      new THREE.MeshBasicMaterial({ color: colour("accent"), transparent: true, opacity: 0.08, side: THREE.DoubleSide, depthWrite: false }));
    slab.position.y = (a + b) / 2;
    graph.scene().add(slab);
  }

  /** Dims everything outside the highlighted period. People count if they were alive in it. */
  function applyPeriod() {
    const chosen = periods[period];
    for (const n of (graph?.graphData().nodes ?? []) as Node3D[]) {
      let inside = true;
      if (chosen) {
        const e = idx.entity.get(n.id)!;
        const [from, to] = [chosen[0].year, chosen[1].year];
        if (e.life && e.type === datedType) { const [s, d] = lifeRange(e.life); inside = d >= from && s <= to; }
        else inside = n.year !== null && n.year >= from && n.year <= to;
      }
      for (const m of materials.get(n.id) ?? []) { m.transparent = true; m.opacity = inside ? 1 : 0.12; }
    }
  }

  function tooltip(n: Node3D): string {
    const e = idx.entity.get(n.id)!;
    const when = e.life ? lifeLabel(e.life) : n.year !== null ? `placed near ${formatYear(Math.round(n.year))} by its links` : "no date known";
    return `<b>${n.label}</b><br><span>${typeStyle(n.type).label} · ${when}</span>`;
  }

  function create() {
    const g = new ForceGraph3D(container, { controlType: "orbit" })
      .backgroundColor(colour("bg"))
      .showNavInfo(false)
      // The layout settles within a few seconds; the camera and the century rings follow then.
      .cooldownTime(5000)
      .nodeId("id")
      .nodeLabel((n) => tooltip(n as Node3D))
      .nodeThreeObjectExtend(false)
      .linkColor((l) => colour((l as Link3D).kind === "relation" ? "edge-relation" : "edge-co"))
      .linkOpacity(0.55)
      .linkWidth((l) => ((l as Link3D).kind === "relation" ? 1.2 : 0))
      .linkDirectionalArrowLength((l) => ((l as Link3D).kind === "relation" && app.graph.arrows ? 5 : 0))
      .linkDirectionalArrowRelPos(1)
      .linkLabel((l) => (l as Link3D).label)
      .onNodeClick((n) => app.selectEntity((n as Node3D).id))
      .onNodeRightClick((n) => app.moreOf((n as Node3D).id))
      .onLinkClick((l) => { const edge = app.ds!.edges.find((e) => e.id === (l as Link3D).id); if (edge) app.selectEdge(edge); })
      .onEngineStop(() => {
        drawRings();
        if (!fitted) { fitted = true; view("free", 600); }
      });
    // Strong repulsion spreads the network sideways, so the cube is about as wide as it is tall.
    // The default wheel speed of the orbit controls takes a lot of scrolling for a scene this size.
    (g.controls() as unknown as { zoomSpeed: number }).zoomSpeed = 3;
    g.d3Force("charge")?.strength(-260);
    (g.d3Force("link") as unknown as { distance(d: number): void } | undefined)?.distance(45);
    return g;
  }

  // Every change of the drawn items or the theme rebuilds the data; items keep their place.
  $effect(() => {
    void theme;
    const s = spec, placed = layout;
    if (!container) return;
    untrack(() => {
      colour = colours();
      graph ??= create();
      graph.backgroundColor(colour("bg"));
      materials = new Map();
      const ranked = [...s.nodes].sort((a, b) => b.passages - a.passages).slice(0, LABELLED).map((n) => n.id);
      const labelled = new Set(s.nodes.length <= LABELLED ? s.nodes.map((n) => n.id) : ranked);
      const nodes: Node3D[] = s.nodes.map((n) => {
        const year = placed.year.get(n.id) ?? null, old = previous.get(n.id);
        return { id: n.id, label: n.label, type: n.type, passages: n.passages, lead: n.lead, image: n.image,
          year, inferred: placed.inferred.has(n.id), x: old?.x, z: old?.z, y: height(year), fy: height(year) };
      });
      previous = new Map(nodes.map((n) => [n.id, n]));
      const links: Link3D[] = s.edges.map((e) => ({ id: e.id, source: e.source, target: e.target, kind: e.kind, label: e.label }));
      graph.nodeThreeObject((n) => nodeObject(n as Node3D, labelled.has((n as Node3D).id)));
      graph.graphData({ nodes, links });
      if (!fitted) view("free", 0);
      applyPeriod();
    });
  });

  $effect(() => {
    void period;
    untrack(() => { applyPeriod(); drawRings(); });
  });

  // A selected item is brought to the middle of the view.
  $effect(() => {
    const id = app.focus;
    untrack(() => {
      const n = id ? (graph?.graphData().nodes as Node3D[] | undefined)?.find((x) => x.id === id) : undefined;
      if (!n || n.x === undefined) return;
      const pos = graph!.cameraPosition();
      const dist = Math.hypot(pos.x - n.x, pos.y - n.y!, pos.z - n.z!);
      const k = Math.min(1, 260 / dist);
      graph!.cameraPosition({ x: n.x + (pos.x - n.x) * k, y: n.y! + (pos.y - n.y!) * k, z: n.z! + (pos.z - n.z!) * k }, { x: n.x, y: n.y!, z: n.z! }, 700);
    });
  });

  /** Side: the time axis upright, like a timeline. Top: the network as a map. Free: both. */
  function view(kind: "side" | "top" | "free", ms = 800) {
    if (!graph) return;
    const nodes = graph.graphData().nodes as Node3D[];
    const reach = Math.max(150, ...nodes.map((n) => Math.hypot(n.x ?? 0, n.z ?? 0))) + 80;
    // Far enough that the whole cube fits the camera's field of view, with a margin.
    const camera = graph.camera() as THREE.PerspectiveCamera;
    const half = Math.tan(((camera.fov ?? 40) * Math.PI) / 360);
    const tall = (AXIS + (nodes.some((n) => n.year === null) ? 300 : 120)) / 2, wide = reach / (camera.aspect || 1);
    const d = (kind === "top" ? reach : Math.max(tall, wide)) / half * 1.1 + reach;
    const centre = { x: 0, y: 0, z: 0 };
    if (kind === "side") graph.cameraPosition({ x: 0, y: 0, z: d }, centre, ms);
    else if (kind === "top") graph.cameraPosition({ x: 0, y: d, z: 1 }, centre, ms);
    else graph.cameraPosition({ x: d * 0.62, y: d * 0.3, z: d * 0.62 }, centre, ms);
  }

  $effect(() => {
    const restyle = () => theme++;
    const media = matchMedia("(prefers-color-scheme: dark)");
    media.addEventListener("change", restyle);
    const observer = new MutationObserver(restyle);
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    const resize = new ResizeObserver(() => graph?.width(container.clientWidth).height(container.clientHeight));
    resize.observe(container);
    return () => { media.removeEventListener("change", restyle); observer.disconnect(); resize.disconnect(); };
  });

  onDestroy(() => { graph?._destructor(); circleTexture?.dispose(); });
</script>

<div class="three" bind:this={container}></div>

<div class="controls card" role="group" aria-label="3D view">
  <span class="seg" role="group" aria-label="Camera">
    <button onclick={() => view("side")} title="Look from the side: the time axis stands upright, like a timeline">Side</button>
    <button onclick={() => view("top")} title="Look from above: the network as a map">Top</button>
    <button onclick={() => view("free")} title="Look at an angle: time and network together">Free</button>
  </span>
  <label class="period">
    <span>Highlight</span>
    <select bind:value={period} aria-label="Highlight a period">
      <option value={-1}>all periods</option>
      {#each periods as [a, b], i}<option value={i}>{a.label} – {b.label}</option>{/each}
    </select>
  </label>
  <span class="note">Up is later in time. Other topics sit at the time of the {datedPlural} they are linked to.</span>
</div>

<style>
  .three { position: absolute; inset: 0; }
  .three :global(.scene-tooltip) { font-family: var(--font-ui); font-size: 12px; background: var(--surface); color: var(--ink);
    border: 1px solid var(--line); border-radius: 6px; padding: 5px 8px; }
  .three :global(.scene-tooltip span) { color: var(--ink-3); }
  .controls { position: absolute; z-index: 5; right: 10px; bottom: 10px; justify-content: flex-end; display: flex; flex-wrap: wrap; align-items: center; gap: 8px 12px;
    padding: 8px 10px; max-width: calc(100% - 80px); }
  .period { display: inline-flex; align-items: center; gap: 6px; font-size: 12px; color: var(--ink-2); }
  .controls .note { font-size: 11.5px; }
</style>
