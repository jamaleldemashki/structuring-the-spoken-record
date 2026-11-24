<script lang="ts">
  import { app } from "../lib/state.svelte.ts";
  import GraphView from "./GraphView.svelte";
  import Panel from "./Panel.svelte";
</script>

<div class="explore" class:nopanel={!app.ui.panelOpen || app.ui.full}>
  <GraphView />
  {#if app.ui.panelOpen && !app.ui.full}<Panel />{/if}
</div>

<style>
  .explore { display: grid; grid-template-columns: minmax(0, 1fr) 380px; gap: 10px; height: 100%; }
  .explore.nopanel { grid-template-columns: minmax(0, 1fr); }
  /* On a narrow window the details lie over the graph, which keeps the whole width. */
  @media (max-width: 980px) { .explore { grid-template-columns: minmax(0, 1fr); } }
</style>
