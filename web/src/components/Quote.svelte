<script lang="ts">
  import { app } from "../lib/state.svelte.ts";
  import { formatTime, highlight } from "../lib/format.ts";

  let { passageId, ranges, showSource = true }: { passageId: string; ranges: { start: number; end: number }[]; showSource?: boolean } = $props();

  const passage = $derived(app.idx!.passage.get(passageId)!);
  const episode = $derived(app.idx!.episode.get(passage.episode_id)!);
  // The excerpt shows the mentions with some context; the full passage keeps every mark.
  let full = $state(false);
  const excerpt = $derived(passage.text ? highlight(passage.text, ranges) : []);
  const parts = $derived(full && passage.text ? highlight(passage.text, ranges, Infinity) : excerpt);
  const cut = $derived(excerpt.length > 0 && (excerpt[0].text.startsWith("…") || excerpt.at(-1)!.text.endsWith("…")));
</script>

<div class="quote">
  {#if parts.length}
    <p>{#each parts as part}{#if part.mark}<mark>{part.text}</mark>{:else}{part.text}{/if}{/each}</p>
  {:else}
    <p class="empty">Transcript text is not included in this export.</p>
  {/if}
  <div class="meta">
    <button class="play" onclick={() => app.playPassage(passageId)} aria-label="Listen from {formatTime(passage.start_s)}">▶ {formatTime(passage.start_s)}</button>
    {#if cut}
      <button class="link" aria-expanded={full} onclick={() => (full = !full)}>{full ? "Show excerpt" : "Read full passage"}</button>
    {/if}
    <button class="link" onclick={() => app.readPassage(passageId)} title="Open the episode transcript at this passage">
      {showSource ? `${episode.code} · passage ${passage.index + 1}` : "In episode"} →
    </button>
  </div>
</div>

<style>
  .quote { border-top: 1px solid var(--line); padding: 10px 0; }
  p { font-family: var(--font-read); font-size: 15px; line-height: 1.55; margin: 0 0 6px; }
  .meta { display: flex; align-items: center; flex-wrap: wrap; gap: 4px 12px; font-size: 12px; color: var(--ink-3); }
  .link { border: 0; background: none; padding: 0; cursor: pointer; color: var(--ink-2); font-size: 12px; text-decoration: underline;
    text-decoration-color: var(--line); text-underline-offset: 3px; }
  .link:hover { color: var(--ink); text-decoration-color: currentColor; }
</style>
