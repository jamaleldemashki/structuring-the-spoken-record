<script lang="ts">
  import { app } from "../lib/state.svelte.ts";
  import { activeFilterCount } from "../lib/filters.ts";
  import Search from "./Search.svelte";

  let { filterable }: { filterable: boolean } = $props();

  type Theme = "system" | "light" | "dark";
  let theme = $state<Theme>(readTheme());
  function readTheme(): Theme {
    try { return (localStorage.getItem("theme") as Theme) || "system"; } catch { return "system"; }
  }
  $effect(() => {
    if (theme === "system") delete document.documentElement.dataset.theme;
    else document.documentElement.dataset.theme = theme;
    try { localStorage.setItem("theme", theme); } catch { /* storage unavailable */ }
  });
  const filterCount = $derived(activeFilterCount(app.ds!, app.filters!));
  const nextTheme: Record<Theme, Theme> = { system: "light", light: "dark", dark: "system" };
  const themeName: Record<Theme, string> = { system: "Theme follows the system", light: "Light theme", dark: "Dark theme" };
  const icons = {
    start: "M3 11l9-8 9 8M5 9.5V21h14V9.5",
    explore: "M6 6a2 2 0 1 0 0 .01M18 8a2 2 0 1 0 0 .01M9 18a2 2 0 1 0 0 .01M7.5 7.5 16 8M6.5 8 8.5 16M16.5 9.5 10.5 16.5",
    episodes: "M12 3a3 3 0 0 0-3 3v6a3 3 0 0 0 6 0V6a3 3 0 0 0-3-3M5 11a7 7 0 0 0 14 0M12 18v3",
    timeline: "M3 12h18M7 8v8M12 6v12M17 9v6",
  };
  // Results have no tab: they belong to a search. Back and the search field lead there again.
  const pages = $derived(([["start", "Home"], ["explore", "Map"], ["episodes", "Episodes"], ["timeline", "Timeline"]] as const)
    .filter(([id]) => id !== "timeline" || app.hasLifespans));
</script>

<header>
  <button class="brand" title="Home" onclick={() => (app.view = "start")}>
    <b>{app.ds!.site.title}</b>{#if app.ds!.site.subtitle}<span>{app.ds!.site.subtitle}</span>{/if}
  </button>
  <!-- The home page has its own large search field. -->
  {#if app.view !== "start"}<Search />{/if}
  <nav class="tabs" aria-label="Pages">
    {#each pages as [id, label]}
      <button class:home={id === "start"} aria-current={app.view === id ? "page" : undefined} class:on={app.view === id} onclick={() => (id === "explore" ? app.wholeMapOrExploration() : (app.view = id))}>
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d={icons[id]} /></svg>{label}
      </button>
    {/each}
  </nav>
  <div class="right">
    {#if filterable}
      <button class="btn ghost" aria-expanded={app.ui.filtersOpen} aria-controls="filters" onclick={() => (app.ui.filtersOpen = !app.ui.filtersOpen)}>
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 5h18l-7 8v6l-4 2v-8z" /></svg>
        Filters{#if filterCount}<b class="badge">{filterCount}</b>{/if}
      </button>
    {/if}
    <button class="btn quiet icon" title="{themeName[theme]}. Click to change." aria-label={themeName[theme]} onclick={() => (theme = nextTheme[theme])}>
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        {#if theme === "dark"}<path d="M20 14.5A8 8 0 1 1 9.5 4a6.5 6.5 0 0 0 10.5 10.5z" />
        {:else if theme === "light"}<circle cx="12" cy="12" r="4" /><path d="M12 2v2M12 20v2M4 12H2M22 12h-2M5.6 5.6 4.2 4.2M19.8 19.8l-1.4-1.4M18.4 5.6l1.4-1.4M4.2 19.8l1.4-1.4" />
        {:else}<circle cx="12" cy="12" r="9" /><path d="M12 3v18a9 9 0 0 0 0-18z" fill="currentColor" stroke="none" />{/if}
      </svg>
    </button>
  </div>
</header>

<style>
  header { display: flex; align-items: center; gap: 8px 16px; padding-block: 10px 6px; min-height: 52px; }
  .brand { display: flex; align-items: baseline; gap: 8px; border: 0; background: none; padding: 0; cursor: pointer; color: inherit; font: inherit; text-align: left; flex: none; }
  .brand:hover b { text-decoration: underline; text-underline-offset: 4px; text-decoration-thickness: 1px; }
  .brand b { font-family: var(--font-read); font-size: 20px; font-weight: 600; letter-spacing: -0.01em; white-space: nowrap; }
  .brand span { font-size: 12px; color: var(--ink-3); white-space: nowrap; }
  .tabs { display: flex; gap: 2px; flex: none; }
  .tabs button { display: inline-flex; align-items: center; gap: 6px; border: 0; background: none; padding: 6px 10px; border-radius: 7px; cursor: pointer; color: var(--ink-2); font-weight: 500; }
  .tabs button:hover { background: var(--surface-2); color: var(--ink); }
  .tabs button.on { background: var(--surface-2); color: var(--ink); }
  .tabs .home { display: none; }
  .tabs svg { display: none; }
  .right { margin-left: auto; display: flex; gap: 6px; align-items: center; flex: none; }
  .badge { background: var(--accent); color: var(--accent-ink); border-radius: 999px; font-size: 11px; padding: 0 6px; font-weight: 600; }
  @media (max-width: 1240px) { .brand span { display: none; } }
  @media (max-width: 900px) { header { flex-wrap: wrap; } header :global(.search) { order: 5; flex-basis: 100%; max-width: none; } }
  /* On a phone the pages go to a bar at the bottom, where a thumb reaches them. */
  @media (max-width: 760px) {
    .brand b { font-size: 17px; }
    .tabs { position: fixed; z-index: 30; left: 0; right: 0; bottom: 0; height: 56px; background: var(--surface); border-top: 1px solid var(--line); padding-bottom: env(safe-area-inset-bottom, 0px); }
    .tabs button { flex: 1; flex-direction: column; justify-content: center; gap: 1px; font-size: 11px; border-radius: 0; }
    .tabs button.on { background: none; color: var(--ink); font-weight: 600; }
    .tabs .home { display: inline-flex; }
    .tabs svg { display: block; }
  }
</style>
