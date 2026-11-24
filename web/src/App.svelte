<script lang="ts">
  import { onMount } from "svelte";
  import { app } from "./lib/state.svelte.ts";
  import Header from "./components/Header.svelte";
  import FilterPanel from "./components/FilterPanel.svelte";
  import ActiveFilters from "./components/ActiveFilters.svelte";
  import Explore from "./components/Explore.svelte";
  import Timeline from "./components/Timeline.svelte";
  import Episodes from "./components/Episodes.svelte";
  import Start from "./components/Start.svelte";
  import Results from "./components/Results.svelte";
  import AudioBar from "./components/AudioBar.svelte";
  import Panel from "./components/Panel.svelte";

  let error = $state<string | null>(null);
  // Filters only change something on these pages.
  const filterable = $derived(app.view === "explore" || app.view === "results" || app.view === "timeline");
  // The map has its own place for the details.
  // Full view belongs to the map. On other pages the header is needed.
  const bare = $derived(app.ui.full && app.view === "explore");
  const floatingPanel = $derived(app.ui.panelOpen && app.view !== "explore" && app.view !== "start");

  onMount(() => {
    app.load().then(() => (document.title = app.ds!.site.title)).catch((e: Error) => (error = e.message));
  });

  // Esc closes the nearest thing first. Not while typing.
  function onKey(ev: KeyboardEvent) {
    if (ev.key !== "Escape" || (ev.target as HTMLElement).closest?.("input:not([type=range]), select, textarea")) return;
    if (app.ui.connectOpen) app.ui.connectOpen = false;
    else if (app.ui.filtersOpen) app.ui.filtersOpen = false;
    else if (app.ui.panelOpen) app.closePanel();
    else if (app.ui.full) app.ui.full = false;
  }
</script>

<svelte:window onkeydown={onKey} />

{#if error}
  <div class="state"><h1>The archive could not be loaded</h1><p>{error}</p></div>
{:else if !app.ds}
  <div class="state"><p>Loading the archive…</p></div>
{:else}
  <div class="shell" class:audio={app.ui.audioOpen} class:wide={app.view === "explore"} class:bare>
    {#if !bare}
      <Header {filterable} />
      {#if app.view !== "start"}<ActiveFilters />{/if}
    {/if}
    <main>
      {#if app.view === "start"}<Start />
      {:else if app.view === "explore"}<Explore />
      {:else if app.view === "results"}<Results />
      {:else if app.view === "timeline"}<Timeline />
      {:else}<Episodes />{/if}
    </main>
  </div>
  {#if app.ui.filtersOpen && filterable && !bare}<FilterPanel />{/if}
  {#if floatingPanel}<Panel floating />{/if}
  <AudioBar />
{/if}

<style>
  .shell { display: flex; flex-direction: column; height: 100%; max-width: 1680px; margin-inline: auto;
    padding-inline: 12px; padding-block: 0 8px; }
  .shell.wide { max-width: none; }
  main { flex: 1; min-height: 0; overflow: auto; padding-block: 8px 0; }
  .bare main { padding-block-start: 8px; }
  .state { max-width: 560px; margin: 15vh auto; padding-inline: 16px; color: var(--ink-2); }
  .state h1 { font-family: var(--font-read); font-size: 24px; color: var(--ink); }
  /* What is fixed at the bottom of the window leaves its room free. */
  :global(:root) { --bottom-room: var(--tabs-h); }
  :global(:root:has(.shell.audio)) { --bottom-room: calc(var(--tabs-h) + var(--audio-h)); }
  .shell { padding-block-end: calc(8px + var(--bottom-room)); }
  @media (max-width: 760px) { :global(:root) { --tabs-h: 56px; } .shell { padding-inline: 8px; } }
</style>
