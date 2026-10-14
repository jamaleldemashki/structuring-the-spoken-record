<script lang="ts">
  import { app } from "../lib/state.svelte.ts";
  import { allSeries, yearRange } from "../lib/filters.ts";
  import { allTypes, typeStyle, typeVar } from "../lib/entityStyle.ts";

  const ds = $derived(app.ds!);
  const f = $derived(app.filters!);
  const [y0, y1] = $derived(yearRange(ds));
  const years = $derived(Array.from({ length: y1 - y0 + 1 }, (_, i) => y0 + i));
  const perYear = $derived.by(() => {
    const m = new Map<number, number>();
    for (const e of ds.episodes) if (e.date) m.set(+e.date.slice(0, 4), (m.get(+e.date.slice(0, 4)) ?? 0) + 1);
    return m;
  });
  const maxYear = $derived(Math.max(1, ...perYear.values()));
  const site = $derived(ds.site);
  const hasLifespans = $derived(ds.summary.lifespans === "enriched");
  const dated = $derived(site.lifespan_type ? typeStyle(site.lifespan_type) : null);

  const num = (v: string) => (v.trim() === "" ? null : Number(v));
  function toggle(set: Set<string>, value: string): Set<string> {
    const next = new Set(set);
    next.has(value) ? next.delete(value) : next.add(value);
    return next;
  }
</script>

<section id="filters" class="card filters" aria-label="Filters">
  <div class="group">
    <span class="label">Published</span>
    <svg width={years.length * 7} height="22" aria-hidden="true">
      {#each years as y, i}
        {@const n = perYear.get(y) ?? 0}
        <rect x={i * 7} y={20 - (n / maxYear) * 18} width="5" height={Math.max(1, (n / maxYear) * 18)} rx="1"
          fill={n && y >= f.pub[0] && y <= f.pub[1] ? "var(--hue-1)" : "var(--line)"}><title>{y}: {n} episodes</title></rect>
      {/each}
    </svg>
    <select aria-label="Published from" value={f.pub[0]} onchange={(e) => app.setFilters({ pub: [Math.min(+e.currentTarget.value, f.pub[1]), f.pub[1]] })}>
      {#each years as y}<option value={y}>{y}</option>{/each}
    </select>
    <span class="note">to</span>
    <select aria-label="Published to" value={f.pub[1]} onchange={(e) => app.setFilters({ pub: [f.pub[0], Math.max(+e.currentTarget.value, f.pub[0])] })}>
      {#each years as y}<option value={y}>{y}</option>{/each}
    </select>
  </div>

  {#if dated}
  <div class="group" title={hasLifespans ? "" : "Lifespans become available once the Wikidata enrichment stage runs."}>
    <span class="label">{dated.plural} who lived</span>
    <select aria-label="How a lifespan must match the window" disabled={!hasLifespans} value={f.livedMatch}
      onchange={(e) => app.setFilters({ livedMatch: e.currentTarget.value as "within" | "overlap" })}>
      <option value="within">born and died within</option>
      <option value="overlap">alive at any point in</option>
    </select>
    <input type="number" step="50" placeholder="any" aria-label="Lived from (negative years are BCE)" disabled={!hasLifespans}
      value={f.lived[0] ?? ""} onchange={(e) => app.setFilters({ lived: [num(e.currentTarget.value), f.lived[1]] })} />
    <span class="note">to</span>
    <input type="number" step="50" placeholder="any" aria-label="Lived to" disabled={!hasLifespans}
      value={f.lived[1] ?? ""} onchange={(e) => app.setFilters({ lived: [f.lived[0], num(e.currentTarget.value)] })} />
    {#each site.lived_presets as p (p.label)}
      <button class="chip" disabled={!hasLifespans} aria-pressed={f.lived[0] === p.start && f.lived[1] === p.end}
        onclick={() => app.setFilters({ lived: [p.start, p.end] })}>{p.label}</button>
    {/each}
    <label class="note"><input type="checkbox" disabled={!hasLifespans} checked={f.undated}
      onchange={(e) => app.setFilters({ undated: e.currentTarget.checked })} /> also show {dated.plural} without known dates</label>
    {#if !hasLifespans}<span class="note">(needs lifespan enrichment)</span>{/if}
  </div>
  {/if}

  {#if site.series_label}
  <div class="group">
    <span class="label">{site.series_label}</span>
    <span class="chips">
      {#each allSeries(ds) as s}
        <button class="chip" aria-pressed={f.series.has(s)} onclick={() => app.setFilters({ series: toggle(f.series, s) })}>{s || "Other"}</button>
      {/each}
    </span>
  </div>
  {/if}

  <div class="group">
    <span class="label">Types</span>
    <span class="chips">
      {#each allTypes() as t (t.type)}
        <button class="chip" aria-pressed={f.types.has(t.type)} onclick={() => app.setFilters({ types: toggle(f.types, t.type) })}>
          <span class="dot" style:background={typeVar(t.type)}></span>{t.label}
        </button>
      {/each}
    </span>
  </div>

  <span class="end">
    <button class="reset" onclick={() => app.resetFilters()}>Reset filters</button>
  </span>
</section>

<style>
  .filters { display: flex; flex-wrap: wrap; gap: 10px 22px; align-items: center; padding: 10px 14px; }
  .group { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
  .label { font-size: 11px; text-transform: uppercase; letter-spacing: 0.07em; color: var(--ink-3); font-weight: 600; }
  input[type="number"] { width: 78px; border: 1px solid var(--line); background: var(--bg); border-radius: 6px; padding: 3px 6px; font-family: var(--font-data); font-size: 12px; }
  select { font-family: var(--font-data); }
  :disabled { opacity: 0.5; cursor: not-allowed; }
  .end { margin-left: auto; display: flex; gap: 10px; align-items: center; }
  .reset { border: 0; background: none; color: var(--ink-2); text-decoration: underline; cursor: pointer; font-size: 12px; }
</style>
