<script lang="ts">
  import { app } from "../lib/state.svelte.ts";
  import { formatTime, highlight } from "../lib/format.ts";
  import { player } from "../lib/player.svelte.ts";
  import { follower, runs, timedWords, wordAt } from "../lib/follow.ts";

  let { passageId, ranges, showSource = true, radius = 120 }: { passageId: string; ranges: { start: number; end: number }[]; showSource?: boolean; radius?: number } = $props();

  const passage = $derived(app.idx!.passage.get(passageId)!);
  const episode = $derived(app.idx!.episode.get(passage.episode_id)!);
  // The excerpt shows the mentions with some context; the full passage keeps every mark.
  let full = $state(false);
  // Every name in the passage is a link, as in the transcript, apart from the topic the details show.
  const names = $derived.by(() => {
    const own = app.selection?.kind === "entity" ? app.selection.id : null;
    const found: { start: number; end: number; id: string }[] = [];
    for (const id of app.idx!.passageEntities.get(passageId) ?? []) {
      const e = app.idx!.entity.get(id);
      if (!e || id === own || !app.filters!.types.has(e.type)) continue;
      for (const o of e.occurrences) if (o.passage_id === passageId) found.push({ start: o.start, end: o.end, id });
    }
    let pos = 0;
    return found.sort((a, b) => a.start - b.start || b.end - a.end).filter((n) => n.start >= pos && ((pos = n.end), true));
  });
  const excerpt = $derived(passage.text ? highlight(passage.text, ranges, radius, names) : []);
  const parts = $derived(full && passage.text ? highlight(passage.text, ranges, Infinity, names) : excerpt);
  const cut = $derived(excerpt.length > 0 && (excerpt[0].text.startsWith("…") || excerpt.at(-1)!.text.endsWith("…")));

  // While this passage is played, the whole passage is shown and the spoken word is marked,
  // wherever the quote is (details panel, answer, related discussions).
  const playing = $derived(player.passageId === passageId && !!passage.text);
  const words = $derived(playing ? timedWords(passage) : []);
  const now = $derived(playing ? wordAt(words, player.now) : -1);
  const MARK = "mark";
  // While it plays, a name stays a link and the marks stay marks. A mark that is also a name is the name.
  const isMarked = (n: { start: number; end: number }) => ranges.some((r) => r.start <= n.start && r.end >= n.end);
  const markedNames = $derived(new Set(names.filter(isMarked).map((n) => n.id)));
  const marked = $derived([...names, ...ranges.filter((r) => !names.some((n) => n.start < r.end && n.end > r.start)).map((r) => ({ ...r, id: MARK }))].sort((a, b) => a.start - b.start));
  const follow_ = (ev: MouseEvent, id: string) => { ev.preventDefault(); app.follow(() => app.selectEntity(id)); };
  let root: HTMLElement;
  const follow = follower();
  $effect(() => { if (now >= 0) follow(root.querySelector(".now")); });
</script>

<div class="quote" bind:this={root}>
  {#if playing}
    <p class="following">{#each runs(passage.text!, marked, words) as r}{#if r.id === MARK}<mark>{#each r.pieces as x}{@render word(x)}{/each}</mark>{:else if r.id}{@const id = r.id}<a class="name" href="#/map?open={id}" onclick={(ev) => follow_(ev, id)}>{#if markedNames.has(id)}<mark>{#each r.pieces as x}{@render word(x)}{/each}</mark>{:else}{#each r.pieces as x}{@render word(x)}{/each}{/if}</a>{:else}{#each r.pieces as x}{@render word(x)}{/each}{/if}{/each}</p>
  {:else if parts.length}
    <p>{#each parts as part}{#if part.id}{@const id = part.id}<a class="name" href="#/map?open={id}" onclick={(ev) => follow_(ev, id)}>{#if part.mark}<mark>{part.text}</mark>{:else}{part.text}{/if}</a>{:else if part.mark}<mark>{part.text}</mark>{:else}{part.text}{/if}{/each}</p>
  {:else}
    <p class="empty">Transcript text is not included in this export.</p>
  {/if}
  <div class="meta">
    {#if episode.audio_url}
      <button class="play" onclick={() => app.playPassage(passageId)} aria-label={passage.start_s === null ? "Listen" : `Listen from ${formatTime(passage.start_s)}`}>▶{passage.start_s === null ? "" : ` ${formatTime(passage.start_s)}`}</button>
    {/if}
    {#if cut && !playing}
      <button class="linkish" aria-expanded={full} onclick={() => (full = !full)}>{full ? "Show less" : "Read all of it"}</button>
    {/if}
    <button class="linkish" onclick={() => app.readPassage(passageId)} title="Open the transcript at this place">
      {showSource ? `${episode.code}: ${episode.title}` : "In the transcript"}
    </button>
  </div>
</div>

{#snippet word(x: { text: string; w: number })}{#if x.w >= 0}<span class="word" class:now={x.w === now} class:ahead={now >= 0 && x.w > now} role="presentation"
  onclick={() => player.seek(words[x.w].t0)}>{x.text}</span>{:else}{x.text}{/if}{/snippet}

<style>
  /* The passage being played: words still to come are a little lighter, the spoken one is marked. */
  .following .ahead { color: var(--ink-2); }
  .following .now { background: var(--accent); color: var(--accent-ink); border-radius: 3px; box-shadow: 0 0 0 2px var(--accent); }
  .following .word { cursor: pointer; }
  /* The same as names in the transcript. */
  .name { color: inherit; text-decoration: underline; text-decoration-color: var(--ink-3); text-underline-offset: 3px; }
  .name:hover { background: var(--mark); }
  .quote { border-top: 1px solid var(--line); padding: 10px 0; }
  p { font-family: var(--font-read); font-size: 15px; line-height: 1.55; margin: 0 0 6px; }
  .meta { display: flex; align-items: center; flex-wrap: wrap; gap: 4px 12px; font-size: 12.5px; color: var(--ink-3); }
  .meta .linkish { text-align: left; }
</style>
