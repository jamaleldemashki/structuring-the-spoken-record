<script lang="ts">
  // What the filters are leaving out at the moment, each with its own way to undo it. Connections
  // and chosen episodes are written at the top of the map itself.
  import { app } from "../lib/state.svelte.ts";
  import { defaultFilters } from "../lib/filters.ts";
  import { formatYear } from "../lib/format.ts";
  import { formatMonth } from "../lib/timeAxis.ts";
  import { typeStyle } from "../lib/entityStyle.ts";

  type Chip = { label: string; clear: () => void };
  const chips = $derived.by((): Chip[] => {
    if (!app.ds || !app.filters) return [];
    const d = defaultFilters(app.ds), f = app.filters, out: Chip[] = [];
    if (f.only && app.view !== "explore") {
      const codes = [...f.only].map((id) => app.idx!.episode.get(id)?.code ?? id).sort();
      out.push({ label: `Only ${codes.slice(0, 3).join(", ")}${codes.length > 3 ? ` and ${codes.length - 3} more` : ""}`, clear: () => app.showAllEpisodes() });
    }
    if (f.pub[0] !== d.pub[0] || f.pub[1] !== d.pub[1])
      out.push({ label: `Published ${formatMonth(f.pub[0])} to ${formatMonth(f.pub[1])}`, clear: () => app.setFilters({ pub: d.pub }) });
    if (f.lived[0] !== null || f.lived[1] !== null)
      out.push({ label: `${f.livedMatch === "within" ? "Born and died" : "Alive"} ${f.lived[0] === null ? "any time" : formatYear(f.lived[0])} to ${f.lived[1] === null ? "any time" : formatYear(f.lived[1])}${f.undated ? ", and undated" : ""}`,
        clear: () => app.setFilters({ lived: [null, null] }) });
    for (const s of d.series) if (!f.series.has(s))
      out.push({ label: `Hiding ${s || "other"} series`, clear: () => app.setFilters({ series: new Set([...f.series, s]) }) });
    // Types can be seen and switched on the map's own legend, so only the other pages list them.
    if (app.view !== "explore") for (const t of d.types) if (!f.types.has(t))
      out.push({ label: `Hiding ${typeStyle(t).plural}`, clear: () => app.setFilters({ types: new Set([...f.types, t]) }) });
    return out;
  });
</script>

{#if chips.length}
  <div class="active" aria-label="Active filters">
    {#each chips as c}
      <span>{c.label}<button aria-label="Remove: {c.label}" onclick={c.clear}>×</button></span>
    {/each}
    {#if chips.length > 1}<button class="linkish" onclick={() => app.resetFilters()}>Clear all</button>{/if}
  </div>
{/if}

<style>
  .active { display: flex; gap: 6px; flex-wrap: wrap; align-items: center; padding-block: 2px 4px; }
  span { font-size: 12.5px; background: var(--surface-2); border-radius: 999px; padding: 2px 4px 2px 11px; display: inline-flex; gap: 4px; align-items: center; }
  span button { border: 0; background: none; cursor: pointer; color: var(--ink-3); padding: 0 5px; font-size: 14px; }
  span button:hover { color: var(--ink); }
</style>
