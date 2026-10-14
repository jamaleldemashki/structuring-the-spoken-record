<script lang="ts">
  import { app } from "../lib/state.svelte.ts";
  import { entityPassagesInView, livedOk } from "../lib/filters.ts";
  import { allTypes, typeStyle, typeVar } from "../lib/entityStyle.ts";
  import type { EntityType } from "../lib/types.ts";

  const ds = $derived(app.ds!);
  const idx = $derived(app.idx!);
  const inView = $derived(ds.entities.map((e) => [e, entityPassagesInView(e, idx, app.vp).length] as const)
    .filter(([e, n]) => n > 0 && livedOk(e, app.filters!, ds.site.lifespan_type)));
  const episodesInView = $derived(new Set([...app.vp].map((p) => idx.passage.get(p)!.episode_id)).size);
  const count = (t: EntityType) => inView.filter(([e]) => e.type === t).length;
  // The archive's default-visible types, in its configured order.
  const groups = $derived(allTypes().filter((t) => t.visibleByDefault));
  const tiles = $derived([[episodesInView, "episodes in view"], [app.vp.size, "passages"],
    ...groups.map((t) => [count(t.type), t.plural] as const)]);
  const dated = $derived(ds.site.lifespan_type ? typeStyle(ds.site.lifespan_type) : null);
  const top = (t: EntityType) => inView.filter(([e]) => e.type === t).sort((a, b) => b[1] - a[1]).slice(0, 8);
  const eras = $derived([...new Set(ds.episodes.map((e) => e.era ?? "Unknown"))].sort());
  const series = $derived([...new Set(ds.episodes.map((e) => e.series ?? "Other"))].sort());
</script>

<div class="tiles">
  {#each tiles as [n, label]}<div class="card tile"><b>{n}</b><span>{label}</span></div>{/each}
</div>

<div class="cols">
  {#each groups as g (g.type)}
    {@const t = g.type}
    {@const rows = top(t)}
    {@const max = Math.max(1, ...rows.map((r) => r[1]))}
    <div class="card list">
      <h3><span class="dot" style:background={typeVar(t)}></span>Most discussed {g.plural}</h3>
      <div class="toplist">
        {#each rows as [e, n] (e.id)}
          <button onclick={() => app.openEntity(e.id)} title={e.label}>{e.label}</button>
          <span class="track"><i style:width="{(n / max) * 100}%" style:background={typeVar(t)}></i></span>
          <span class="n">{n}</span>
        {:else}
          <span class="empty">None in view</span>
        {/each}
      </div>
    </div>
  {/each}
</div>

{#if ds.site.era_label}
<div class="card list">
  <h3>Coverage by {ds.site.era_label.toLowerCase()}{ds.site.series_label ? ` and ${ds.site.series_label.toLowerCase()}` : ""}</h3>
  <div class="table">
    <table>
      <thead><tr><th>{ds.site.era_label}</th>{#each series as s}<th class="n">{s} episodes</th>{/each}{#if dated}<th class="n">{dated.plural} with a known lifespan</th>{/if}</tr></thead>
      <tbody>
        {#each eras as era}
          {@const people = ds.entities.filter((e) => e.type === ds.site.lifespan_type && e.episode_ids.some((id) => (idx.episode.get(id)!.era ?? "Unknown") === era))}
          <tr>
            <td>{era}</td>
            {#each series as s}<td class="n">{ds.episodes.filter((e) => (e.era ?? "Unknown") === era && (e.series ?? "Other") === s).length}</td>{/each}
            {#if dated}<td class="n">{people.filter((e) => e.life).length} / {people.length}</td>{/if}
          </tr>
        {/each}
      </tbody>
    </table>
  </div>
</div>
{/if}

<style>
  .tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; }
  .tile { padding: 14px 16px; }
  .tile b { display: block; font-family: var(--font-read); font-size: 32px; font-weight: 600; font-variant-numeric: tabular-nums; }
  .tile span { font-size: 12px; color: var(--ink-2); }
  .cols { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px; margin-block: 12px; }
  .list { padding: 14px 16px; }
  h3 { margin: 0 0 10px; font-size: 13px; font-weight: 600; display: flex; gap: 8px; align-items: center; }
  .toplist { display: grid; grid-template-columns: minmax(0, 1fr) 90px 28px; gap: 5px 8px; align-items: center; font-size: 13px; }
  .toplist button { border: 0; background: none; text-align: left; padding: 0; cursor: pointer; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .toplist button:hover { text-decoration: underline; }
  .track { height: 8px; background: var(--surface-2); border-radius: 4px; overflow: hidden; }
  .track i { display: block; height: 100%; border-radius: 0 4px 4px 0; }
  .n { font-family: var(--font-data); font-size: 11.5px; color: var(--ink-2); text-align: right; font-variant-numeric: tabular-nums; }
  .table { overflow-x: auto; }
  table { border-collapse: collapse; font-size: 12.5px; width: 100%; }
  th, td { border-bottom: 1px solid var(--line); padding: 6px 8px; text-align: left; }
  th.n, td.n { text-align: right; }
</style>
