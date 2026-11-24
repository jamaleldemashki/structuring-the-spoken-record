<script lang="ts">
  // The details, the same on every page. On the map they stand beside the graph, on the pages for
  // reading they lie over the right edge, so the place in the text stays where it was. Under 760 px
  // they come up from the bottom.
  import { app } from "../lib/state.svelte.ts";
  import SidePanel from "./SidePanel.svelte";

  let { floating = false }: { floating?: boolean } = $props();
</script>

<aside class="card panel" class:floating aria-live="polite" aria-label="Details">
  <button class="close" aria-label="Close the details" title="Close (Esc)" onclick={() => app.closePanel()}>×</button>
  <SidePanel />
</aside>

<style>
  .panel { padding: 16px; min-width: 0; min-height: 0; overflow: auto; position: relative; }
  .close { position: sticky; top: -6px; float: right; z-index: 3; margin: -8px -8px 0 6px; background: var(--surface); }
  .floating { position: fixed; z-index: 25; top: 58px; right: 12px; bottom: calc(12px + var(--bottom-room, 0px)); width: min(420px, calc(100vw - 24px)); box-shadow: var(--shadow); }
  @media (max-width: 980px) {
    .panel { position: fixed; z-index: 25; top: 58px; right: 12px; bottom: calc(12px + var(--bottom-room, 0px)); width: min(420px, calc(100vw - 24px)); box-shadow: var(--shadow); }
  }
  @media (max-width: 760px) {
    .panel, .floating { top: auto; left: 0; right: 0; bottom: var(--bottom-room, 0px); width: auto; height: 62vh; border-radius: var(--radius) var(--radius) 0 0; border-width: 1px 0 0; }
  }
</style>
