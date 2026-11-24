<script lang="ts">
  import { app, type Answer } from "../lib/state.svelte.ts";
  import { typeVar } from "../lib/entityStyle.ts";
  import { plural } from "../lib/format.ts";
  import Quote from "./Quote.svelte";

  // The answer to a question: what was recognised in it, how those items are linked, and
  // the passages where they are discussed, best first.
  let { answer }: { answer: Answer } = $props();
  const idx = $derived(app.idx!);
  const ids = $derived(new Set([...answer.names.map((n) => n.entity.id), ...answer.topics.map((t) => t.id)]));
  const links = $derived(app.allLinks.filter((ed) => ids.has(ed.source) && ids.has(ed.target))
    .sort((a, b) => Number(b.kind === "relation") - Number(a.kind === "relation") || (b.kind === "co_discussed" ? b.count : 0) - (a.kind === "co_discussed" ? a.count : 0)));
  const label = (id: string) => idx.entity.get(id)?.label ?? id;
  const ranges = (pid: string) => [...ids].flatMap((id) => idx.entity.get(id)?.occurrences.filter((o) => o.passage_id === pid) ?? []);
</script>

<p class="eyebrow">Your question</p>
<h2 class="serif">“{answer.text}”</h2>

{#if ids.size}
  <div class="found">
    {#each answer.names as n (n.entity.id)}
      <button class="echip" onclick={() => app.selectEntity(n.entity.id)} title={n.fuzzy ? `“${n.text}” read as ${n.entity.label}` : undefined}>
        <span class="dot" style:background={typeVar(n.entity.type)}></span>{n.entity.label}{#if n.fuzzy}<small> (from “{n.text}”)</small>{/if}
      </button>
    {/each}
    {#each answer.topics as t (t.id)}
      <button class="echip meaning" onclick={() => app.selectEntity(t.id)} title="Close in meaning to words in your question ({t.score.toFixed(2)})">
        <span class="dot" style:background={typeVar(idx.entity.get(t.id)!.type)}></span>{label(t.id)}<small> by meaning</small>
      </button>
    {/each}
  </div>
{:else}
  <p class="empty">No topic of the archive is named in the question{answer.status === "done" ? ", so the passages below are the closest in meaning" : ""}.</p>
{/if}

{#if answer.status === "meaning"}<p class="note">Reading the question for its meaning…</p>
{:else if answer.status === "names"}<p class="note">Passages are ordered by the names they mention.</p>{/if}

{#if ids.size > 1}
  <h3 class="section-title">How they are linked</h3>
  {#each links as ed (ed.id)}
    <button class="link" onclick={() => app.selectEdge(ed)}>
      <b>{label(ed.source)}</b>
      <span>{ed.kind === "relation" ? `${ed.label}${ed.status === "nli_supported" ? " ✓" : ""}` : `discussed together in ${plural(ed.count, "passage")}`}</span>
      <b>{label(ed.target)}</b>
    </button>
  {:else}
    <p class="empty">Not linked directly inside the current filters. The map shows the shortest chain between them, if there is one.</p>
  {/each}
{/if}

<h3 class="section-title">Where the podcast discusses it</h3>
{#each answer.evidence as ev (ev.passageId)}
  {#if ev.ids.length}<p class="note">Mentions {ev.ids.map(label).join(", ")}</p>{/if}
  <Quote passageId={ev.passageId} ranges={ranges(ev.passageId)} />
{:else}
  <p class="empty">No passage in the current filters.</p>
{/each}

<style>
  h2 { font-size: 21px; font-weight: 600; margin: 4px 0 10px; line-height: 1.3; text-wrap: balance; }
  .found { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 8px; }
  .echip small { color: var(--ink-3); font-size: 11px; }
  .meaning { border-style: dashed; }
  .link { display: flex; flex-wrap: wrap; gap: 4px 6px; align-items: baseline; width: 100%; text-align: left; border: 0; background: none;
    padding: 6px 0; border-top: 1px solid var(--line); cursor: pointer; color: inherit; font: inherit; font-size: 13.5px; }
  .link span { color: var(--ink-3); font-size: 12.5px; }
  .link:hover b { text-decoration: underline; }
  .note { margin: 10px 0 0; }
</style>
