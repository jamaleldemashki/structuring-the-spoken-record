<script lang="ts">
  import { app } from "../lib/state.svelte.ts";
  import { defaultFilters } from "../lib/filters.ts";
  import { formatYear } from "../lib/format.ts";

  type Chip = { label: string; clear: () => void };
  const chips = $derived.by((): Chip[] => {
    if (!app.ds || !app.filters) return [];
    const d = defaultFilters(app.ds), f = app.filters, out: Chip[] = [];
    if (f.pub[0] !== d.pub[0] || f.pub[1] !== d.pub[1])
      out.push({ label: `Published ${f.pub[0]}–${f.pub[1]}`, clear: () => app.setFilters({ pub: d.pub }) });
    if (f.lived[0] !== null || f.lived[1] !== null)
      out.push({ label: `${f.livedMatch === "within" ? "Born and died" : "Alive"} ${f.lived[0] === null ? "…" : formatYear(f.lived[0])} – ${f.lived[1] === null ? "…" : formatYear(f.lived[1])}${f.undated ? " (+ undated)" : ""}`,
        clear: () => app.setFilters({ lived: [null, null] }) });
    for (const s of d.series) if (!f.series.has(s))
      out.push({ label: `Hiding ${s || "other"} series`, clear: () => app.setFilters({ series: new Set([...f.series, s]) }) });
    if (app.path)
      out.push({ label: `Connection: ${app.idx!.entity.get(app.path.nodes[0])!.label} → ${app.idx!.entity.get(app.path.nodes.at(-1)!)!.label}`,
        clear: () => { app.path = null; app.selection = app.focus ? { kind: "entity", id: app.focus } : null; } });
    return out;
  });
</script>

{#if chips.length}
  <div class="active" aria-label="Active filters">
    {#each chips as c}
      <span>{c.label}<button aria-label="Remove {c.label}" onclick={c.clear}>×</button></span>
    {/each}
  </div>
{/if}

<style>
  .active { display: flex; gap: 6px; flex-wrap: wrap; padding-top: 8px; }
  span { font-size: 12px; background: var(--surface-2); border-radius: 999px; padding: 2px 4px 2px 10px; display: inline-flex; gap: 4px; align-items: center; }
  button { border: 0; background: none; cursor: pointer; color: var(--ink-3); padding: 0 4px; }
</style>
