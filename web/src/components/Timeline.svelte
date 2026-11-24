<script lang="ts">
  // Who lived when. The years stay on top while scrolling. Someone far outside the rest, like a
  // legendary figure thousands of years earlier, does not set the scale: their bar is cut at the
  // edge and carries their dates. Dragging across the years narrows the view to that window.
  import { app } from "../lib/state.svelte.ts";
  import { entityPassagesInView, livedActive, livedOk } from "../lib/filters.ts";
  import { formatYear, lifeLabel, lifeRange, plural } from "../lib/format.ts";
  import { lifeAxis, withoutOutliers } from "../lib/timeAxis.ts";
  import { typeStyle, typeVar } from "../lib/entityStyle.ts";
  import BackToResults from "./BackToResults.svelte";

  const FIRST_ROWS = 40;
  const ds = $derived(app.ds!);
  const idx = $derived(app.idx!);
  const f = $derived(app.filters!);
  const datedType = $derived(ds.site.lifespan_type);
  const dated = $derived(datedType ? typeStyle(datedType) : null);
  const inView = $derived(ds.entities.filter((e) => e.type === datedType && entityPassagesInView(e, idx, app.vp).length));
  const everyone = $derived(inView.filter((e) => e.life && livedOk(e, { ...f, undated: false }, datedType))
    .sort((a, b) => lifeRange(a.life!)[0] - lifeRange(b.life!)[0]));
  const undated = $derived(inView.filter((e) => !e.life).sort((a, b) => b.mention_count - a.mention_count));
  let showAll = $state(false);
  // The most discussed, still in order of birth.
  const people = $derived.by(() => {
    if (showAll || everyone.length <= FIRST_ROWS) return everyone;
    const top = new Set([...everyone].sort((a, b) => b.mention_count - a.mention_count).slice(0, FIRST_ROWS).map((e) => e.id));
    return everyone.filter((e) => top.has(e.id));
  });
  // The ticks are round years. The scale itself ends just past the people, not at the next round year.
  const axis = $derived.by(() => {
    const ranges = withoutOutliers(people.map((e) => lifeRange(e.life!)));
    const round = lifeAxis(ranges);
    const low = Math.min(...ranges.map((r) => r[0])), high = Math.max(...ranges.map((r) => r[1]));
    const pad = Math.max(2, (high - low) * 0.02);
    const lo = Math.max(round.lo, low - pad), hi = Math.min(round.hi, high + pad);
    return { lo, hi, ticks: round.ticks.filter((t) => t >= lo && t <= hi) };
  });
  const pct = (year: number) => ((year - axis.lo) / (axis.hi - axis.lo)) * 100;
  const clamp = (v: number) => Math.max(0, Math.min(100, v));
  const passagesOf = (id: string) => entityPassagesInView(idx.entity.get(id)!, idx, app.vp).length;

  // Dragging across the years. Everyone alive at any point in the window stays.
  let track = $state<HTMLDivElement>();
  let drag = $state<{ a: number; b: number } | null>(null);
  const toPct = (ev: PointerEvent) => { const box = track!.getBoundingClientRect(); return clamp(((ev.clientX - box.left) / box.width) * 100); };
  function down(ev: PointerEvent) {
    (ev.currentTarget as Element).setPointerCapture(ev.pointerId);
    drag = { a: toPct(ev), b: toPct(ev) };
  }
  function up() {
    if (!drag) return;
    const [a, b] = [Math.min(drag.a, drag.b), Math.max(drag.a, drag.b)];
    drag = null;
    if (b - a < 1) return;
    const year = (p: number) => Math.round(axis.lo + (p / 100) * (axis.hi - axis.lo));
    app.setFilters({ lived: [year(a), year(b)], livedMatch: "overlap" });
  }
</script>

<section class="timeline">
  <div class="head">
    <BackToResults />
    <h1 class="serif">Who lived when</h1>
    <span class="note">{people.length === everyone.length ? plural(everyone.length, dated?.label.toLowerCase() ?? "person", dated?.plural) : `The ${people.length} most discussed of ${everyone.length} ${dated?.plural ?? ""}`}, in order of birth. Drag across the years to look closer.</span>
    <span class="acts">
      {#if everyone.length > FIRST_ROWS}<button class="btn ghost" aria-pressed={showAll} onclick={() => (showAll = !showAll)}>{showAll ? `Most discussed ${FIRST_ROWS}` : `All ${everyone.length}`}</button>{/if}
      {#if livedActive(f)}<button class="btn ghost" onclick={() => app.setFilters({ lived: [null, null] })}>See all years</button>{/if}
    </span>
  </div>

  {#if !people.length}
    <p class="empty">{ds.summary.lifespans !== "enriched" ? "Lifespans are not available in this archive yet." : "Nobody with known dates inside the current filters."}</p>
  {:else}
    <div class="chart card">
      <div class="years">
        <span class="corner"></span>
        <div class="scale">{#each axis.ticks as t}<span style:left="{pct(t)}%">{formatYear(t)}</span>{/each}</div>
        <span class="count" title="Passages">Passages</span>
      </div>
      <div class="body">
        <div class="brush" bind:this={track} role="presentation" onpointerdown={down} onpointermove={(ev) => { if (drag) drag = { ...drag, b: toPct(ev) }; }} onpointerup={up}>
          {#each axis.ticks as t}<i class="grid" style:left="{pct(t)}%"></i>{/each}
          {#if drag}<i class="sweep" style:left="{Math.min(drag.a, drag.b)}%" style:width="{Math.abs(drag.b - drag.a)}%"></i>{/if}
        </div>
        {#each people as e (e.id)}
          {@const [a, b] = lifeRange(e.life!)}
          {@const left = clamp(pct(a))}
          {@const right = clamp(pct(b))}
          {@const cut = pct(a) < 0 || pct(b) > 100}
          <div class="row" class:selected={app.selection?.kind === "entity" && app.selection.id === e.id}>
            <button class="name" title={e.label} onclick={() => app.selectEntity(e.id)}>{e.label}</button>
            <span class="lane">
              <button class="life" class:approx={e.life?.approximate} class:cut style:left="{left}%" style:width="max(5px, {right - left}%)" style:background={typeVar(datedType!)}
                aria-label="{e.label}: {lifeLabel(e.life)}" title="{e.label}, {lifeLabel(e.life)}" onclick={() => app.selectEntity(e.id)}></button>
              {#if cut}<small style:left="{right + 1}%">{lifeLabel(e.life)}</small>{/if}
            </span>
            <span class="count">{passagesOf(e.id)}</span>
          </div>
        {/each}
      </div>
    </div>
  {/if}

  {#if undated.length}
    <details class="undated">
      <summary>{plural(undated.length, dated?.label.toLowerCase() ?? "person", dated?.plural)} without known dates</summary>
      <div class="chips">{#each undated as e (e.id)}<button class="echip" onclick={() => app.selectEntity(e.id)}>{e.label}</button>{/each}</div>
    </details>
  {/if}
</section>

<style>
  .timeline { --name: 170px; --count: 62px; padding-bottom: 24px; display: flex; flex-direction: column; gap: 8px; }
  .head { display: flex; flex-wrap: wrap; gap: 4px 14px; align-items: baseline; padding-block: 4px; }
  h1 { font-size: 22px; margin: 0; font-weight: 600; }
  .acts { margin-left: auto; display: flex; gap: 6px; }
  .chart { overflow: clip; }
  /* The years stay on top while the rows scroll under them. */
  .years { position: sticky; top: -8px; z-index: 3; display: grid; grid-template-columns: var(--name) minmax(0, 1fr) var(--count); align-items: center; gap: 0 10px; height: 30px; background: var(--surface); border-bottom: 1px solid var(--line); }
  .scale { position: relative; height: 100%; }
  .scale span { position: absolute; top: 7px; transform: translateX(-50%); font-family: var(--font-data); font-size: 11.5px; color: var(--ink-3); white-space: nowrap; }
  .body { position: relative; padding-block: 4px; }
  .brush { position: absolute; top: 0; bottom: 0; left: calc(var(--name) + 10px); right: calc(var(--count) + 10px); cursor: crosshair; touch-action: pan-y; }
  .grid { position: absolute; top: 0; bottom: 0; width: 1px; background: var(--line); opacity: 0.6; }
  .sweep { position: absolute; top: 0; bottom: 0; background: var(--accent); opacity: 0.14; }
  .row { position: relative; display: grid; grid-template-columns: var(--name) minmax(0, 1fr) var(--count); align-items: center; gap: 0 10px; height: 24px; pointer-events: none; }
  .row:hover, .row.selected { background: color-mix(in srgb, var(--surface-2) 70%, transparent); }
  .name { pointer-events: auto; border: 0; background: none; padding: 0 0 0 10px; text-align: right; font-size: 13px; color: var(--ink); cursor: pointer; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .name:hover { text-decoration: underline; text-underline-offset: 3px; }
  .selected .name { font-weight: 600; }
  .lane { position: relative; height: 100%; }
  .life { pointer-events: auto; position: absolute; top: 7px; height: 10px; padding: 0; border: 0; border-radius: 3px; cursor: pointer; }
  .life:hover { outline: 2px solid var(--ink); outline-offset: 1px; }
  /* Dates only known roughly are lighter. */
  .approx { opacity: 0.55; }
  .cut { border-top-left-radius: 0; border-bottom-left-radius: 0; }
  .lane small { position: absolute; top: 4px; font-size: 11.5px; color: var(--ink-3); white-space: nowrap; }
  .count { font-family: var(--font-data); font-size: 11.5px; color: var(--ink-3); text-align: right; padding-right: 12px; font-variant-numeric: tabular-nums; }
  .undated { font-size: 13px; color: var(--ink-2); }
  .undated summary { cursor: pointer; padding: 6px 0; }
  @media (max-width: 760px) { .timeline { --name: 108px; --count: 0px; } .count { display: none; } .years, .row { grid-template-columns: var(--name) minmax(0, 1fr); } .name { font-size: 12px; } .brush { right: 10px; } }
</style>
