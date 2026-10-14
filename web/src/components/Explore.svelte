<script lang="ts">
  import { app } from "../lib/state.svelte.ts";
  import GraphView from "./GraphView.svelte";
  import SidePanel from "./SidePanel.svelte";
</script>

<div class="explore" class:nopanel={!app.ui.panelOpen}>
  <GraphView />
  {#if app.ui.panelOpen}
    <aside class="card panel" aria-live="polite"><SidePanel /></aside>
  {/if}
</div>

<style>
  .explore { display: grid; grid-template-columns: minmax(0, 1fr) 380px; gap: 10px; height: 100%; }
  .explore.nopanel { grid-template-columns: minmax(0, 1fr); }
  .panel { padding: 16px; min-width: 0; min-height: 0; overflow: auto; }
  @media (max-width: 980px) {
    .explore { grid-template-columns: 1fr; height: auto; }
    .explore > :global(.canvas) { height: 72vh; }
  }
</style>
