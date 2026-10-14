<script lang="ts">
  import { app } from "../lib/state.svelte.ts";
  import EntityPanel from "./EntityPanel.svelte";
  import EdgePanel from "./EdgePanel.svelte";
  import PathPanel from "./PathPanel.svelte";
</script>

{#if app.selection?.kind === "entity"}
  {#key app.selection.id}<EntityPanel id={app.selection.id} />{/key}
{:else if app.selection?.kind === "edge"}
  <EdgePanel edge={app.selection.edge} />
{:else if app.selection?.kind === "path"}
  <PathPanel path={app.selection.path} />
{:else if app.selection?.kind === "nopath"}
  <p class="eyebrow">Connection</p>
  <h2 class="serif">No connection found</h2>
  <p class="empty">{app.idx!.entity.get(app.selection.from)?.label} and {app.idx!.entity.get(app.selection.to)?.label} are not linked by any chain
    of relations or shared passages within the current filters. Widening the time window or turning on more entity types may connect them.</p>
{:else}
  <p class="empty">Search for a person, concept or work (⌘K) to focus the graph on it, or click a node to see where it is
    discussed and an edge to read the passages behind it.</p>
{/if}

<style>
  h2 { font-size: 24px; font-weight: 600; margin: 4px 0 8px; }
</style>
