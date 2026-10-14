<script lang="ts">
  import { app } from "../lib/state.svelte.ts";
  import { entityPassagesInView, livedOk, yearRange } from "../lib/filters.ts";
  import { formatYear, lifeLabel, lifeRange } from "../lib/format.ts";
  import { typeStyle, typeVar } from "../lib/entityStyle.ts";
  import { lifeAxis, yearLabels } from "../lib/timeAxis.ts";

  const PAD_L = 160, PAD_R = 24, ROW = 22, TOP = 30;
  let width = $state(900);

  const ds = $derived(app.ds!);
  const idx = $derived(app.idx!);
  const datedType = $derived(ds.site.lifespan_type);
  const dated = $derived(datedType ? typeStyle(datedType) : null);
  const [y0, y1] = $derived(yearRange(ds));
  const people = $derived(ds.entities
    .filter((e) => e.type === datedType && e.life && entityPassagesInView(e, idx, app.vp).length)
    .sort((a, b) => lifeRange(a.life!)[0] - lifeRange(b.life!)[0]));
  const undated = $derived(ds.entities.filter((e) => e.type === datedType && !e.life && entityPassagesInView(e, idx, app.vp).length)
    .sort((a, b) => b.mention_count - a.mention_count));

  const W = $derived(Math.max(640, width));
  const span = $derived(W - PAD_L - PAD_R);
  // The lifespan axis fits the people in view (for any podcast), with round tick years.
  const axis = $derived(lifeAxis(people.map((e) => lifeRange(e.life!))));
  const LO = $derived(axis.lo), HI = $derived(axis.hi);
  const x = (y: number) => PAD_L + ((y - LO) / (HI - LO)) * span;
  // Each publication year is a band from Jan 1 to Dec 31, labelled in its middle.
  const px = (y: number) => PAD_L + ((y - y0) / (y1 + 1 - y0)) * span;
  const lifeH = $derived(people.length ? TOP + people.length * ROW + 10 : 0);
  const pubTop = $derived(lifeH + 34);
  const H = $derived(pubTop + 46 + 28);
  const pubLabels = $derived(yearLabels(y0, y1, span));
  const epYear = (date: string) => { const d = new Date(date); return d.getFullYear() + d.getMonth() / 12 + (d.getDate() - 1) / 365; };

  let svg: SVGSVGElement;
  let drag = $state<{ zone: "lived" | "pub"; a: number; b: number } | null>(null);
  const localX = (ev: PointerEvent) => { const box = svg.getBoundingClientRect(); return ((ev.clientX - box.left) / box.width) * W; };
  function down(zone: "lived" | "pub", ev: PointerEvent) {
    (ev.currentTarget as Element).setPointerCapture(ev.pointerId);
    drag = { zone, a: localX(ev), b: localX(ev) };
  }
  function move(ev: PointerEvent) { if (drag) drag = { ...drag, b: localX(ev) }; }
  function up(ev: PointerEvent) {
    if (!drag) return;
    const [a, b] = [Math.min(drag.a, drag.b), Math.max(drag.a, drag.b)];
    const zone = drag.zone;
    drag = null;
    if (b - a < 6) {
      if (zone === "lived") {
        const row = Math.floor((ev.clientY - svg.getBoundingClientRect().top - TOP) / ROW);
        if (people[row]) app.openEntity(people[row].id);
      }
      return;
    }
    if (zone === "lived") app.setFilters({ lived: [Math.round(LO + ((a - PAD_L) / span) * (HI - LO)), Math.round(LO + ((b - PAD_L) / span) * (HI - LO))] });
    else app.setFilters({ pub: [Math.max(y0, Math.floor(y0 + ((a - PAD_L) / span) * (y1 + 1 - y0))), Math.min(y1, Math.floor(y0 + ((b - PAD_L) / span) * (y1 + 1 - y0)))] });
  }
  const f = $derived(app.filters!);
</script>

<section class="card timeline">
  <div class="head">
    <h2 class="serif">When they lived, and when the podcast discussed them</h2>
    <span class="note">Drag across an axis to set a time window. Click a name to open it.</span>
    <button class="btn ghost" onclick={() => app.setFilters({ pub: [y0, y1], lived: [null, null] })}>Clear time windows</button>
  </div>
  {#if dated && ds.summary.lifespans !== "enriched"}
    <p class="callout">Lifespans of {dated.plural} will appear here once the Wikidata enrichment stage has run. For now the timeline shows when episodes were published.</p>
  {/if}
  <div class="scroll" bind:clientWidth={width}>
    <svg bind:this={svg} width={W} height={H} viewBox="0 0 {W} {H}" role="img" aria-label="Lifespans and publication dates">
      {#if people.length}
        <g class="axis">
          {#each axis.ticks as t}
            <line x1={x(t)} x2={x(t)} y1={TOP - 6} y2={lifeH} stroke="var(--line)" />
            <text x={x(t)} y={TOP - 12} text-anchor="middle">{formatYear(t)}</text>
          {/each}
        </g>
        {#if f.lived[0] !== null || f.lived[1] !== null}
          <rect x={x(f.lived[0] ?? LO)} y={TOP - 6} width={Math.max(2, x(f.lived[1] ?? HI) - x(f.lived[0] ?? LO))} height={lifeH - TOP + 6} fill={typeVar(datedType!)} opacity="0.08" />
        {/if}
        {#each people as e, i (e.id)}
          {@const range = lifeRange(e.life!)}
          {@const a = x(range[0])}
          {@const b = x(range[1])}
          <g opacity={livedOk(e, f, datedType) ? 1 : 0.28}>
            <text x={PAD_L - 10} y={TOP + i * ROW + 13} text-anchor="end" class="name">{e.label}</text>
            <rect x={a} y={TOP + i * ROW + 5} width={Math.max(4, b - a)} height="10" rx="4" fill={typeVar(datedType!)}
              fill-opacity={e.life!.approximate ? 0.5 : 1} stroke={e.life!.approximate ? typeVar(datedType!) : "none"} stroke-dasharray="3 2">
              <title>{e.label} · {lifeLabel(e.life)}</title>
            </rect>
          </g>
        {/each}
        <rect role="presentation" class="brush" x={PAD_L} y={TOP - 6} width={W - PAD_L - PAD_R} height={lifeH - TOP + 6}
          onpointerdown={(ev) => down("lived", ev)} onpointermove={move} onpointerup={up} />
      {/if}

      <text x={PAD_L - 10} y={pubTop + 27} text-anchor="end" class="name strong">Published</text>
      <rect x={PAD_L} y={pubTop} width={W - PAD_L - PAD_R} height="46" rx="6" fill="var(--surface-2)" />
      {#if f.pub[0] !== y0 || f.pub[1] !== y1}
        <rect x={px(f.pub[0])} y={pubTop} width={px(f.pub[1] + 1) - px(f.pub[0])} height="46" fill="var(--accent)" opacity="0.12" />
      {/if}
      <g class="axis">
        {#each pubLabels as y}<text x={(px(y) + px(y + 1)) / 2} y={pubTop + 62} text-anchor="middle">{y}</text>{/each}
        {#each Array.from({ length: y1 - y0 }, (_, i) => y0 + i + 1) as y}<line x1={px(y)} x2={px(y)} y1={pubTop} y2={pubTop + 46} stroke="var(--line)" />{/each}
      </g>
      {#each ds.episodes.filter((e) => e.date) as ep, i (ep.id)}
        {@const cx = px(epYear(ep.date!))}
        {@const flip = cx > PAD_L + span * 0.85}
        <g opacity={app.vp.has(ds.passages.find((p) => p.episode_id === ep.id)?.id ?? "") ? 1 : 0.3}>
          <circle {cx} cy={pubTop + 16 + (i % 2) * 14} r="6" fill="var(--accent)" stroke="var(--surface)" stroke-width="2"><title>{ep.code} · {ep.title} · {ep.date}</title></circle>
          <text x={flip ? cx - 10 : cx + 10} y={pubTop + 20 + (i % 2) * 14} text-anchor={flip ? "end" : "start"} class="ep">{ep.code}</text>
        </g>
      {/each}
      <rect role="presentation" class="brush" x={PAD_L} y={pubTop} width={W - PAD_L - PAD_R} height="46"
        onpointerdown={(ev) => down("pub", ev)} onpointermove={move} onpointerup={up} />
      {#if drag}
        <rect x={Math.min(drag.a, drag.b)} y={drag.zone === "lived" ? TOP - 6 : pubTop} width={Math.abs(drag.b - drag.a)}
          height={drag.zone === "lived" ? lifeH - TOP + 6 : 46} fill="var(--accent)" opacity="0.15" pointer-events="none" />
      {/if}
    </svg>
  </div>
  {#if undated.length}
    <div class="undated">
      <strong>{undated.length} people without a known lifespan</strong> in view.
      {#each undated.slice(0, 40) as e (e.id)}<button onclick={() => app.openEntity(e.id)}>{e.label}</button>{/each}
    </div>
  {/if}
</section>

<style>
  .timeline { padding: 14px 16px; }
  .head { display: flex; flex-wrap: wrap; gap: 8px 16px; align-items: baseline; margin-bottom: 8px; }
  h2 { font-size: 20px; margin: 0; font-weight: 600; }
  .callout { background: var(--surface-2); border-radius: 8px; padding: 8px 12px; font-size: 13px; color: var(--ink-2); margin: 0 0 10px; }
  .scroll { overflow-x: auto; }
  .axis text { font-family: var(--font-data); font-size: 10.5px; fill: var(--ink-3); }
  .name { font-size: 12px; fill: var(--ink); }
  .strong { font-weight: 600; }
  .ep { font-size: 11px; fill: var(--ink); }
  .brush { fill: transparent; cursor: crosshair; }
  .undated { margin-top: 10px; font-size: 12.5px; color: var(--ink-2); }
  .undated button { border: 0; background: var(--surface-2); border-radius: 999px; padding: 2px 9px; margin: 3px 2px; cursor: pointer; font-size: 12px; }
</style>
