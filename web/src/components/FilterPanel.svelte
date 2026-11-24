<script lang="ts">
  // What to see and how it looks, in one place. It lies over the left edge of the page, so nothing
  // moves when it opens and the effect of each change can be watched next to it.
  import { app } from "../lib/state.svelte.ts";
  import { allSeries, monthRange } from "../lib/filters.ts";
  import { DEFAULT_SETTINGS } from "../lib/forces.ts";
  import { allTypes, EPISODE_TYPE, typeStyle } from "../lib/entityStyle.ts";
  import { formatMonth, monthsBetween } from "../lib/timeAxis.ts";
  import { formatYear } from "../lib/format.ts";
  import TypeMark from "./TypeMark.svelte";

  const ds = $derived(app.ds!);
  const f = $derived(app.filters!);
  const g = app.graph;
  const site = $derived(ds.site);
  const months = $derived(monthsBetween(...monthRange(ds)));
  const perMonth = $derived.by(() => {
    const m = new Map<string, number>();
    for (const e of ds.episodes) if (e.date) m.set(e.date.slice(0, 7), (m.get(e.date.slice(0, 7)) ?? 0) + 1);
    return m;
  });
  const maxMonth = $derived(Math.max(1, ...perMonth.values()));
  const hasLifespans = $derived(ds.summary.lifespans === "enriched");
  const dated = $derived(site.lifespan_type ? typeStyle(site.lifespan_type) : null);
  const series = $derived(allSeries(ds));

  // The two handles of the publication window. They show where they are while dragging, and the
  // filter applies when they are let go, since every change redraws the whole graph.
  let held = $state<[number, number] | null>(null);
  const window_ = $derived(held ?? [months.indexOf(f.pub[0]), months.indexOf(f.pub[1])] as [number, number]);
  function drag(end: 0 | 1, value: number) {
    const [a, b] = window_;
    // A handle pushes the other one along. Stopping there would leave two handles on one month stuck.
    held = end === 0 ? [value, Math.max(value, b)] : [Math.min(a, value), value];
  }
  function drop() {
    if (!held) return;
    app.setFilters({ pub: [months[held[0]], months[held[1]]] });
    held = null;
  }

  const year = (v: string) => (v.trim() === "" ? null : Number(v));
  function toggle(set: Set<string>, value: string): Set<string> {
    const next = new Set(set);
    if (!next.delete(value)) next.add(value);
    return next;
  }
  let forcesOpen = $state(false);
</script>

<section id="filters" class="card filters" aria-label="Filters">
  <header>
    <strong>Filters</strong>
    <button class="btn quiet" onclick={() => app.resetFilters()}>Reset</button>
    <button class="close" aria-label="Close the filters" title="Close (Esc)" onclick={() => (app.ui.filtersOpen = false)}>×</button>
  </header>

  <div class="group">
    <h3>Types</h3>
    <div class="chips">
      <button class="chip" aria-pressed={f.episodes} onclick={() => app.setFilters({ episodes: !f.episodes })}
        title="The episodes as nodes, each linked to what it names, on a time axis in the order they were published">
        <TypeMark type={EPISODE_TYPE} />Episodes
      </button>
      {#each allTypes() as t (t.type)}
        <button class="chip" aria-pressed={f.types.has(t.type)} onclick={() => app.setFilters({ types: toggle(f.types, t.type) })}><TypeMark type={t.type} />{t.label}</button>
      {/each}
    </div>
  </div>

  {#if app.hasPublicationRange}
  <div class="group">
    <h3>Published</h3>
    <div class="bars" aria-hidden="true" style:--months={months.length}>
      {#each months as m, i}
        {@const n = perMonth.get(m) ?? 0}
        <i class:in={i >= window_[0] && i <= window_[1]} style:height="{Math.max(6, (n / maxMonth) * 100)}%" title="{formatMonth(m)}: {n} episodes"></i>
      {/each}
    </div>
    <div class="range" style:--from={window_[0] / Math.max(1, months.length - 1)} style:--to={window_[1] / Math.max(1, months.length - 1)}>
      <i class="track"></i><i class="chosen"></i>
      <input type="range" min="0" max={months.length - 1} value={window_[0]} aria-label="Published from" aria-valuetext={formatMonth(months[window_[0]])}
        oninput={(e) => drag(0, Number(e.currentTarget.value))} onchange={drop} />
      <input type="range" min="0" max={months.length - 1} value={window_[1]} aria-label="Published to" aria-valuetext={formatMonth(months[window_[1]])}
        oninput={(e) => drag(1, Number(e.currentTarget.value))} onchange={drop} />
    </div>
    <p class="ends"><span>{formatMonth(months[window_[0]])}</span><span>{formatMonth(months[window_[1]])}</span></p>
  </div>
  {/if}

  {#if dated && hasLifespans}
    <div class="group">
      <h3>{dated.plural} who lived</h3>
      <div class="chips">
        {#each site.lived_presets as p (p.label)}
          <button class="chip" title="{formatYear(p.start)} to {formatYear(p.end)}" aria-pressed={f.lived[0] === p.start && f.lived[1] === p.end}
            onclick={() => app.setFilters(f.lived[0] === p.start && f.lived[1] === p.end ? { lived: [null, null] } : { lived: [p.start, p.end] })}>{p.label}</button>
        {/each}
      </div>
      <div class="years">
        <label>From year<input type="number" step="50" placeholder="any" value={f.lived[0] ?? ""} onchange={(e) => app.setFilters({ lived: [year(e.currentTarget.value), f.lived[1]] })} /></label>
        <label>To year<input type="number" step="50" placeholder="any" value={f.lived[1] ?? ""} onchange={(e) => app.setFilters({ lived: [f.lived[0], year(e.currentTarget.value)] })} /></label>
      </div>
      <p class="note">Years before the common era are negative: -500 is 500 BCE.</p>
      <label class="line">Count someone who
        <select value={f.livedMatch} onchange={(e) => app.setFilters({ livedMatch: e.currentTarget.value as "within" | "overlap" })}>
          <option value="within">was born and died in these years</option>
          <option value="overlap">was alive at any point in them</option>
        </select>
      </label>
      <label class="line check"><input type="checkbox" checked={f.undated} onchange={(e) => app.setFilters({ undated: e.currentTarget.checked })} />Also show {dated.plural} without known dates</label>
    </div>
  {/if}

  {#if site.series_label && series.length > 1}
    <div class="group">
      <h3>{site.series_label}</h3>
      <div class="chips">
        {#each series as s}<button class="chip" aria-pressed={f.series.has(s)} onclick={() => app.setFilters({ series: toggle(f.series, s) })}>{s || "Other"}</button>{/each}
      </div>
    </div>
  {/if}

  {#if app.view === "explore"}
    <div class="group">
      <h3>Display</h3>
      <label class="line check" title="Pictures from Wikimedia Commons, for topics linked to Wikidata that have one."><input type="checkbox" bind:checked={g.portraits} />Portraits</label>
      <label class="line check"><input type="checkbox" bind:checked={g.arrows} />Arrows on named relations</label>
      <label class="slider"><span>Node size</span><input type="range" min="0.5" max="2" step="0.05" bind:value={g.nodeSize} /></label>
      <label class="slider"><span>Line thickness</span><input type="range" min="0.3" max="3" step="0.1" bind:value={g.linkThickness} /></label>
    </div>
    <div class="group">
      <button class="fold" aria-expanded={forcesOpen} onclick={() => (forcesOpen = !forcesOpen)}>Layout forces</button>
      {#if forcesOpen}
        <label class="slider" title="Pulls everything towards the middle. Lower it to let clusters drift apart."><span>Center</span><input type="range" min="0" max="1" step="0.02" bind:value={g.center} /></label>
        <label class="slider" title="Pushes every node away from every other node. Raise it to spread a crowded graph."><span>Repel</span><input type="range" min="0" max="20" step="0.5" bind:value={g.repel} /></label>
        <label class="slider" title="How firmly a connection pulls its two ends together."><span>Connection strength</span><input type="range" min="0" max="1" step="0.02" bind:value={g.linkForce} /></label>
        <label class="slider" title="The length connections try to have."><span>Connection length</span><input type="range" min="30" max="500" step="10" bind:value={g.linkDistance} /></label>
        <label class="slider" title="Free space kept around every node and its name."><span>Space around nodes</span><input type="range" min="0" max="120" step="4" bind:value={g.gap} /></label>
        <button class="linkish" onclick={() => Object.assign(app.graph, DEFAULT_SETTINGS)}>Restore the defaults</button>
      {/if}
    </div>
  {/if}
</section>

<style>
  .filters { position: fixed; z-index: 24; left: 12px; top: 58px; bottom: calc(12px + var(--bottom-room, 0px)); width: min(340px, calc(100vw - 24px)); overflow: auto; box-shadow: var(--shadow); font-size: 13px; }
  header { position: sticky; top: 0; z-index: 1; display: flex; align-items: center; gap: 4px; padding: 8px 8px 8px 14px; background: var(--surface); border-bottom: 1px solid var(--line); }
  header strong { flex: 1; font-size: 14px; }
  .group { padding: 12px 14px; display: flex; flex-direction: column; gap: 8px; }
  .group + .group { border-top: 1px solid var(--line); }
  h3 { margin: 0; font-size: 11.5px; text-transform: uppercase; letter-spacing: 0.07em; color: var(--ink-3); font-weight: 600; }
  /* The handles move over the width less a handle. A bar stands exactly above the place of its month. */
  .bars { --handle: 14px; --pitch: calc((100% - var(--handle)) / max(1, var(--months) - 1)); display: flex; align-items: flex-end; gap: 1px; height: 34px; margin-left: calc(var(--handle) / 2 - var(--pitch) / 2); width: calc(var(--pitch) * var(--months)); }
  .bars i { flex: 1; min-width: 0; border-radius: 1px 1px 0 0; background: var(--line); }
  .bars i.in { background: var(--hue-1); }
  /* Two sliders on one track. Each would colour from the far left up to its own handle, so the
     browser's track is switched off and the stretch between the two handles is drawn here. Only the
     handles take the pointer. */
  .range { --handle: 14px; position: relative; height: 20px; margin-top: -2px; }
  .track, .chosen { position: absolute; top: 8px; height: 4px; border-radius: 2px; }
  .track { left: 0; right: 0; background: var(--line); }
  .chosen { left: calc(var(--handle) / 2 + (100% - var(--handle)) * var(--from)); right: calc(var(--handle) / 2 + (100% - var(--handle)) * (1 - var(--to))); background: var(--accent); }
  .range input { position: absolute; inset: 0; width: 100%; height: 20px; margin: 0; padding: 0; background: none; appearance: none; -webkit-appearance: none; pointer-events: none; }
  .range input::-webkit-slider-runnable-track { background: none; }
  .range input::-moz-range-track { background: none; }
  .range input::-webkit-slider-thumb { -webkit-appearance: none; pointer-events: auto; width: var(--handle); height: var(--handle); border-radius: 50%; background: var(--accent); border: 2px solid var(--surface); box-shadow: 0 0 0 1px var(--accent); cursor: grab; }
  .range input::-moz-range-thumb { pointer-events: auto; width: var(--handle); height: var(--handle); box-sizing: border-box; border-radius: 50%; background: var(--accent); border: 2px solid var(--surface); box-shadow: 0 0 0 1px var(--accent); cursor: grab; }
  .ends { display: flex; justify-content: space-between; margin: 0; font-family: var(--font-data); font-size: 12px; color: var(--ink-2); }
  .years { display: flex; gap: 10px; }
  .years label { flex: 1; display: flex; flex-direction: column; gap: 2px; color: var(--ink-2); font-size: 12.5px; }
  .years input { width: 100%; border: 1px solid var(--line); background: var(--surface); border-radius: 6px; padding: 4px 8px; font-family: var(--font-data); font-size: 13px; }
  .note { margin: 0; }
  .line { display: flex; flex-direction: column; gap: 3px; color: var(--ink-2); }
  .check { flex-direction: row; align-items: center; gap: 8px; cursor: pointer; }
  .slider { display: flex; flex-direction: column; gap: 2px; color: var(--ink-2); }
  .slider input { width: 100%; accent-color: var(--accent); }
  .fold { border: 0; background: none; padding: 0; text-align: left; cursor: pointer; font: inherit; font-size: 11.5px; text-transform: uppercase; letter-spacing: 0.07em; color: var(--ink-3); font-weight: 600; display: flex; gap: 6px; }
  .fold::before { content: "›"; display: inline-block; width: 10px; transition: transform 0.15s; }
  .fold[aria-expanded="true"]::before { transform: rotate(90deg); }
  .linkish { align-self: flex-start; }
  @media (max-width: 760px) { .filters { top: auto; left: 0; right: 0; width: auto; height: 72vh; border-radius: var(--radius) var(--radius) 0 0; bottom: var(--bottom-room, 0px); } }
</style>
