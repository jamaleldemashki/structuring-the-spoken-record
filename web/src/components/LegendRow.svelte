<script lang="ts">
  // The legend and the switches are one thing: each type and kind of connection, with what it looks
  // like and how many are drawn, and a click to switch it on or off. Types that are off by default
  // wait behind one button, so the row stays short.
  import { app } from "../lib/state.svelte.ts";
  import { allTypes, EPISODE_TYPE } from "../lib/entityStyle.ts";
  import type { GraphSpec } from "../lib/graph.ts";
  import TypeMark from "./TypeMark.svelte";

  let { spec }: { spec: GraphSpec } = $props();
  const f = $derived(app.filters!);
  const counts = $derived.by(() => {
    const n = new Map<string, number>();
    for (const node of spec.nodes) n.set(node.type, (n.get(node.type) ?? 0) + 1);
    return n;
  });
  let allOpen = $state(false);
  const rare = $derived(allTypes().filter((t) => !t.visibleByDefault && !f.types.has(t.type)));
  const shownTypes = $derived(allTypes().filter((t) => allOpen || t.visibleByDefault || f.types.has(t.type)));
  function toggle(type: string) {
    const next = new Set(f.types);
    if (!next.delete(type)) next.add(type);
    app.setFilters({ types: next });
  }
</script>

<div class="legend" role="group" aria-label="What is drawn. Click to switch on or off.">
  <button class="key" aria-pressed={f.episodes} title="{f.episodes ? 'Hide' : 'Show'} the episodes and the time axis" onclick={() => app.setFilters({ episodes: !f.episodes })}>
    <TypeMark type={EPISODE_TYPE} />Episodes{#if f.episodes}<small>{counts.get(EPISODE_TYPE) ?? 0}</small>{/if}
  </button>
  {#each shownTypes as t (t.type)}
    <button class="key" aria-pressed={f.types.has(t.type)} title="{f.types.has(t.type) ? 'Hide' : 'Show'} {t.plural}" onclick={() => toggle(t.type)}>
      <TypeMark type={t.type} />{t.label}{#if f.types.has(t.type)}<small>{counts.get(t.type) ?? 0}</small>{/if}
    </button>
  {/each}
  {#if rare.length || allOpen}
    <button class="key more" aria-expanded={allOpen} onclick={() => (allOpen = !allOpen)}>{allOpen ? "Fewer" : `${rare.length} more`}</button>
  {/if}
  <i class="gap"></i>
  <button class="key" aria-pressed={f.showRelations} title="A relation named in the podcast, for example 'wrote' or 'influenced'. A solid line was checked against the transcript." onclick={() => app.setFilters({ showRelations: !f.showRelations })}>
    <svg width="22" height="8" aria-hidden="true"><line x1="1" y1="4" x2="17" y2="4" stroke="var(--edge-relation)" stroke-width="1.6" /><path d="M16 1 L21 4 L16 7 Z" fill="var(--edge-relation)" /></svg>Named relation
  </button>
  <button class="key" aria-pressed={f.showCoDiscussed} title="Discussed in the same passages. This says nothing about how the two are related." onclick={() => app.setFilters({ showCoDiscussed: !f.showCoDiscussed })}>
    <svg width="22" height="8" aria-hidden="true"><line x1="1" y1="4" x2="21" y2="4" stroke="var(--ink-3)" stroke-width="1" /></svg>Discussed together
  </button>
</div>

<style>
  .legend { display: flex; flex-wrap: wrap; align-items: center; gap: 1px 2px; padding: 3px 8px; border-bottom: 1px solid var(--line); font-size: 12.5px; }
  .key { display: inline-flex; align-items: center; gap: 6px; border: 0; background: none; border-radius: 6px; padding: 3px 7px; cursor: pointer; color: var(--ink-2); white-space: nowrap; font-size: 12.5px; }
  .key:hover { background: var(--surface-2); color: var(--ink); }
  .key small { font-family: var(--font-data); font-size: 11.5px; color: var(--ink-3); }
  /* What is switched off stays there, crossed out and faded, to be switched on again. */
  .key[aria-pressed="false"] { color: var(--ink-3); text-decoration: line-through; }
  .key[aria-pressed="false"] :global(i), .key[aria-pressed="false"] svg { opacity: 0.3; }
  .more { color: var(--ink-3); text-decoration: underline; text-underline-offset: 3px; text-decoration-color: var(--line); }
  .gap { flex: 1; }
  /* On a phone the row stays one line high and moves sideways, the graph needs the height. */
  @media (max-width: 760px) { .legend { flex-wrap: nowrap; overflow-x: auto; scrollbar-width: none; } .legend::-webkit-scrollbar { display: none; } .gap { flex: none; width: 8px; } }
</style>
