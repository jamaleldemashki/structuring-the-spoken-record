<script lang="ts">
  import { onMount } from "svelte";
  import { app } from "./lib/state.svelte.ts";
  import Header from "./components/Header.svelte";
  import FilterBar from "./components/FilterBar.svelte";
  import ActiveFilters from "./components/ActiveFilters.svelte";
  import Explore from "./components/Explore.svelte";
  import Timeline from "./components/Timeline.svelte";
  import Episodes from "./components/Episodes.svelte";
  import Overview from "./components/Overview.svelte";
  import AudioBar from "./components/AudioBar.svelte";
  import Rail from "./components/Rail.svelte";

  let error = $state<string | null>(null);

  onMount(() => {
    app.load().then(() => (document.title = app.ds!.site.title)).catch((e: Error) => (error = e.message));
  });

  // Esc leaves the maximised graph, unless the user is typing somewhere.
  function onKey(ev: KeyboardEvent) {
    const typing = (ev.target as HTMLElement).closest?.("input, select, textarea");
    if (ev.key === "Escape" && app.restoreUi && !typing) app.toggleMaximized();
  }
</script>

<svelte:window onkeydown={onKey} />

{#if error}
  <div class="state"><h1>The archive data could not be loaded</h1><p>{error}</p></div>
{:else if !app.ds}
  <div class="state"><p>Loading the archive…</p></div>
{:else}
  <div class="shell" class:audio-min={app.ui.audioMin} class:wide={app.view === "explore"} class:bare={!app.ui.headerOpen}>
    {#if app.ui.headerOpen}<Header />{/if}
    {#if app.ui.filtersOpen}<FilterBar />{/if}
    <ActiveFilters />
    <main>
      {#if app.view === "explore"}<Explore />
      {:else if app.view === "timeline"}<Timeline />
      {:else if app.view === "episodes"}<Episodes />
      {:else}<Overview />{/if}
    </main>
  </div>
  <Rail />
  <AudioBar />
{/if}

<style>
  .shell { display: flex; flex-direction: column; height: 100%; max-width: calc(1680px + var(--rail-w)); margin-inline: auto;
    padding-inline: calc(var(--rail-w) + 12px) 12px; padding-block: 0 var(--audio-h); }
  .shell.audio-min { padding-block-end: 8px; }
  .shell.wide { max-width: none; }
  main { flex: 1; min-height: 0; overflow: auto; padding-block: 10px; }
  .bare main { padding-block-start: 8px; }
  .state { max-width: 560px; margin: 15vh auto; padding-inline: 16px; color: var(--ink-2); }
  .state h1 { font-family: var(--font-read); font-size: 24px; color: var(--ink); }
</style>
