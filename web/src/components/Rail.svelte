<script lang="ts">
  // One fixed column of show/hide buttons, like Obsidian's ribbon. Each bar has one button
  // that always stays in the same place, whatever else is open.
  import { app } from "../lib/state.svelte.ts";
  import { activeFilterCount } from "../lib/filters.ts";
  import { player } from "../lib/player.svelte.ts";

  type Bar = "headerOpen" | "filtersOpen" | "toolbarOpen" | "panelOpen" | "settingsOpen" | "legendOpen";
  interface Item { bar: Bar; label: string; name: string; icon: string; explore?: boolean }

  // Icon paths are strokes on a 24x24 grid.
  const items: Item[] = [
    { bar: "headerOpen", label: "Menu", name: "header with views and search", icon: "M3 5h18v4H3zM3 13h18M3 18h12" },
    { bar: "filtersOpen", label: "Filters", name: "filters", icon: "M3 5h18l-7 8v6l-4 2v-8z" },
    { bar: "toolbarOpen", label: "Tools", name: "graph toolbar", icon: "M4 7h16M4 12h10M4 17h6", explore: true },
    { bar: "settingsOpen", label: "Settings", name: "graph settings", icon: "M4 6h16M4 12h16M4 18h16M9 4v4M15 10v4M7 16v4", explore: true },
    { bar: "legendOpen", label: "Legend", name: "legend", icon: "M5 6h.01M9 6h10M5 12h.01M9 12h10M5 18h.01M9 18h10", explore: true },
    { bar: "panelOpen", label: "Details", name: "details panel", icon: "M3 4h18v16H3zM15 4v16", explore: true },
  ];
  const shown = $derived(items.filter((i) => !i.explore || app.view === "explore"));
  const filterCount = $derived(app.ds && app.filters ? activeFilterCount(app.ds, app.filters) : 0);
  const listening = $derived(!app.ui.audioMin);
</script>

<nav class="rail" aria-label="Show or hide bars">
  {#each shown as item (item.bar)}
    {@const open = app.ui[item.bar]}
    <button aria-pressed={open} title="{open ? 'Hide' : 'Show'} {item.name}" onclick={() => (app.ui[item.bar] = !open)}>
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d={item.icon} /></svg>
      <span>{item.label}</span>
      {#if item.bar === "filtersOpen" && filterCount}<b class="badge">{filterCount}</b>{/if}
    </button>
  {/each}
  <button aria-pressed={listening} title="{listening ? 'Hide' : 'Show'} listening bar" onclick={() => (app.ui.audioMin = listening)}>
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M4 14v-2a8 8 0 0 1 16 0v2M4 14h3v6H4zM17 14h3v6h-3z" /></svg>
    <span>Listen</span>
    {#if !player.paused && !listening}<b class="badge dot" aria-label="playing"></b>{/if}
  </button>
  {#if app.view === "explore"}
    <button class="end" aria-pressed={Boolean(app.restoreUi)} onclick={() => app.toggleMaximized()}
      title={app.restoreUi ? "Bring back the bars that were open (Esc)" : "Hide every bar and give the graph the whole window"}>
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <path d={app.restoreUi ? "M9 3v6H3M15 3v6h6M9 21v-6H3M15 21v-6h6" : "M3 9V3h6M21 9V3h-6M3 15v6h6M21 15v6h-6"} />
      </svg>
      <span>{app.restoreUi ? "Restore" : "Maximize"}</span>
    </button>
  {/if}
</nav>

<style>
  .rail { position: fixed; left: 0; top: 0; bottom: 0; width: var(--rail-w); z-index: 45; display: flex; flex-direction: column;
    align-items: stretch; gap: 2px; padding: calc(8px + env(safe-area-inset-top, 0px)) 4px calc(8px + env(safe-area-inset-bottom, 0px));
    background: var(--surface); border-right: 1px solid var(--line); overflow-y: auto; }
  button { position: relative; border: 0; background: none; border-radius: 8px; padding: 7px 0 5px; cursor: pointer; color: var(--ink-3);
    display: flex; flex-direction: column; align-items: center; gap: 2px; font-size: 10px; line-height: 1.1; }
  button:hover { background: var(--surface-2); color: var(--ink); }
  button[aria-pressed="true"] { color: var(--ink); background: var(--surface-2); }
  button[aria-pressed="true"]::before { content: ""; position: absolute; left: -4px; top: 8px; bottom: 8px; width: 3px; border-radius: 0 3px 3px 0; background: var(--accent); }
  .end { margin-top: auto; }
  .badge { position: absolute; top: 3px; right: 6px; background: var(--accent); color: var(--accent-ink); border-radius: 999px; font-size: 9.5px; padding: 0 5px; }
  .badge.dot { width: 8px; height: 8px; padding: 0; }
</style>
