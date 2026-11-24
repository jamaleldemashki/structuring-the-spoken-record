<script lang="ts">
  // Details of an episode node: when it was published, what it discusses, and the way into
  // its transcript.
  import { app } from "../lib/state.svelte.ts";
  import { EPISODE_TYPE, typeVar } from "../lib/entityStyle.ts";
  import { plural } from "../lib/format.ts";

  let { id }: { id: string } = $props();
  const SHOWN = 14;
  const idx = $derived(app.idx!);
  const ep = $derived(idx.episode.get(id)!);
  const items = $derived(app.episodeItems.get(id) ?? []);
  const count = (item: string) => app.links.find((l) => l.source === item && l.target === id)?.count ?? 0;
</script>

<span class="eyebrow"><span class="dot" style:background={typeVar(EPISODE_TYPE)}></span>Episode</span>
<h2 class="serif">{ep.title}</h2>
<p class="note">{ep.code} · {ep.date ?? "date unknown"}{ep.era ? ` · ${ep.era}` : ""} · {plural(ep.passage_count, "passage")}</p>

<!-- On the map the bar over the graph has these. -->
{#if app.view !== "explore"}
  <div class="actions">
    <button class="btn" onclick={() => app.readEpisode(id)}>Read the transcript</button>
    <button class="btn ghost" onclick={() => app.showOnly([id])}>Show in the map</button>
  </div>
{/if}

<h3 class="section-title">Names {plural(items.length, "topic")}</h3>
{#each items.slice(0, SHOWN) as item (item)}
  <button class="row" onclick={() => app.selectEntity(item)}>
    <span class="dot" style:background={typeVar(idx.entity.get(item)!.type)}></span>
    <span class="name">{idx.entity.get(item)!.label}</span>
    <small>{plural(count(item), "passage")}</small>
  </button>
{:else}
  <p class="empty">No topic inside the current filters is named in this episode.</p>
{/each}

<style>
  h2 { font-size: 22px; font-weight: 600; margin: 6px 0 4px; line-height: 1.25; text-wrap: balance; }
  .note { margin: 0; }
  .actions { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
  .actions .btn { padding: 5px 12px; }
  .row { display: flex; align-items: center; gap: 8px; width: 100%; border: 0; border-top: 1px solid var(--line); background: none; padding: 6px 0;
    text-align: left; cursor: pointer; color: inherit; font: inherit; font-size: 13.5px; }
  .row:hover .name { text-decoration: underline; text-underline-offset: 3px; }
  .name { flex: 1; min-width: 0; }
  small { color: var(--ink-3); font-family: var(--font-data); font-size: 11.5px; flex: none; }
</style>
