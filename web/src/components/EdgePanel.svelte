<script lang="ts">
  import { app } from "../lib/state.svelte.ts";
  import { evidenceInView } from "../lib/filters.ts";
  import { plural } from "../lib/format.ts";
  import type { Edge } from "../lib/types.ts";
  import Quote from "./Quote.svelte";

  let { edge }: { edge: Edge } = $props();
  const idx = $derived(app.idx!);
  const a = $derived(idx.entity.get(edge.source)!);
  const b = $derived(idx.entity.get(edge.target)!);
  const evidence = $derived(evidenceInView(edge, app.vp));
  const source = $derived(edge.kind === "co_discussed"
    ? `Mentioned in the same ${plural(edge.count, "passage")}. This says nothing about how they are related.`
    : edge.status === "nli_supported"
      ? "Extracted from the transcript and checked against the passage by a second model. Still unreviewed."
      : "Extracted from the transcript by a model. Not yet checked.");
  const ranges = (p: string) => [...a.occurrences, ...b.occurrences].filter((o) => o.passage_id === p);
</script>

<p class="eyebrow">{edge.kind === "co_discussed" ? "Discussed together" : "Relation"}</p>
<h2 class="serif">
  {a.label} <span class="verb">{edge.kind === "co_discussed" ? "and" : `${edge.label}`}</span> {b.label}
</h2>
<p class="source">{source}</p>

<h3 class="section-title">Evidence in the podcast</h3>
{#each evidence as p (p)}
  <Quote passageId={p} ranges={ranges(p)} />
{:else}
  <p class="empty">No supporting passage falls within the current filters.</p>
{/each}

<div class="actions">
  <button class="btn ghost" onclick={() => app.selectEntity(a.id)}>Open {a.label}</button>
  <button class="btn ghost" onclick={() => app.selectEntity(b.id)}>Open {b.label}</button>
</div>

<style>
  h2 { font-size: 22px; font-weight: 600; margin: 4px 0 8px; text-wrap: balance; }
  .verb { color: var(--ink-3); font-weight: 400; font-style: italic; }
  .source { display: inline-block; font-size: 12px; border-radius: 8px; padding: 6px 10px; background: var(--surface-2); color: var(--ink-2); margin: 0; }
  .actions { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 12px; }
</style>
