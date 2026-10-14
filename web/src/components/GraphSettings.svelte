<script lang="ts">
  import { app } from "../lib/state.svelte.ts";
  import { DEFAULT_SETTINGS } from "../lib/forces.ts";

  // Sections open and close like Obsidian's graph settings.
  let open = $state({ links: true, display: false, forces: true });
  let minShared = $state(app.filters!.minShared);
  $effect(() => { minShared = app.filters!.minShared; });

  const g = app.graph;
  const restore = () => Object.assign(app.graph, DEFAULT_SETTINGS);
</script>

<div class="settings card" role="dialog" aria-label="Graph settings">
  <strong class="top">Graph settings</strong>

  <section>
    <button class="sec" aria-expanded={open.links} onclick={() => (open.links = !open.links)}>Links</button>
    {#if open.links}
      <label class="row"><span>Named relations</span>
        <input type="checkbox" checked={app.filters!.showRelations} onchange={(e) => app.setFilters({ showRelations: e.currentTarget.checked })} /></label>
      <label class="row"><span>Discussed together</span>
        <input type="checkbox" checked={app.filters!.showCoDiscussed} onchange={(e) => app.setFilters({ showCoDiscussed: e.currentTarget.checked })} /></label>
      <label class="row slider" title="Only draw a 'discussed together' link when the two share at least this many passages in view. Higher values keep the stronger links and thin out the graph.">
        <span>…in at least <output>{minShared}</output> {minShared === 1 ? "passage" : "passages"}</span>
        <input type="range" min="1" max="5" step="1" bind:value={minShared} disabled={!app.filters!.showCoDiscussed}
          onchange={() => app.setFilters({ minShared })} />
      </label>
      <label class="row"><span>Arrows</span><input type="checkbox" bind:checked={g.arrows} /></label>
    {/if}
  </section>

  <section>
    <button class="sec" aria-expanded={open.display} onclick={() => (open.display = !open.display)}>Display</button>
    {#if open.display}
      <label class="row slider" title="Labels fade out when you zoom out, so they do not pile on top of each other. Move left to always show them.">
        <span>Text fade threshold</span><input type="range" min="0" max="14" step="0.5" bind:value={g.textFade} /></label>
      <label class="row slider"><span>Node size</span><input type="range" min="0.5" max="2" step="0.05" bind:value={g.nodeSize} /></label>
      <label class="row slider"><span>Link thickness</span><input type="range" min="0.3" max="3" step="0.1" bind:value={g.linkThickness} /></label>
    {/if}
  </section>

  <section>
    <button class="sec" aria-expanded={open.forces} onclick={() => (open.forces = !open.forces)}>Forces</button>
    {#if open.forces}
      <label class="row slider" title="Pulls everything towards the middle. Lower it to let clusters drift apart.">
        <span>Center force</span><input type="range" min="0" max="1" step="0.02" bind:value={g.center} /></label>
      <label class="row slider" title="Pushes every node away from every other node. Raise it to spread a crowded graph.">
        <span>Repel force</span><input type="range" min="0" max="20" step="0.5" bind:value={g.repel} /></label>
      <label class="row slider" title="How firmly a link pulls its two ends together.">
        <span>Link force</span><input type="range" min="0" max="1" step="0.02" bind:value={g.linkForce} /></label>
      <label class="row slider" title="The length links try to have.">
        <span>Link distance</span><input type="range" min="30" max="500" step="10" bind:value={g.linkDistance} /></label>
      <label class="row slider" title="Free space kept around every node and its label. Nodes never overlap; raise it for more air between them.">
        <span>Node gap</span><input type="range" min="0" max="120" step="4" bind:value={g.gap} /></label>
    {/if}
  </section>

  <button class="restore" onclick={restore}>Restore default settings</button>
</div>

<style>
  .settings { width: 260px; min-height: 0; overflow: auto; padding: 8px 12px 10px; box-shadow: var(--shadow); font-size: 12.5px; }
  .top { display: block; padding-block: 2px 6px; }
  section { border-top: 1px solid var(--line); padding-block: 6px; }
  .sec { border: 0; background: none; padding: 2px 0; width: 100%; text-align: left; cursor: pointer; font-weight: 600; color: var(--ink); display: flex; gap: 6px; align-items: center; }
  .sec::before { content: "›"; display: inline-block; width: 10px; transition: transform 0.15s; color: var(--ink-3); }
  .sec[aria-expanded="true"]::before { transform: rotate(90deg); }
  .row { display: flex; align-items: center; justify-content: space-between; gap: 10px; padding: 4px 0 4px 16px; color: var(--ink-2); cursor: pointer; }
  .row.slider { flex-direction: column; align-items: stretch; gap: 2px; }
  .row input[type="range"] { width: 100%; accent-color: var(--accent); }
  .row:has(:disabled) { opacity: 0.5; }
  output { font-family: var(--font-data); }
  .restore { border: 0; background: none; color: var(--ink-2); text-decoration: underline; cursor: pointer; font-size: 12px; padding: 6px 0 0; }
</style>
