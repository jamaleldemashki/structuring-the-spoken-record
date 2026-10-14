<script lang="ts">
  import { app } from "../lib/state.svelte.ts";
  import { player } from "../lib/player.svelte.ts";
  import { formatTime } from "../lib/format.ts";

  const passage = $derived(player.passageId ? app.idx!.passage.get(player.passageId) ?? null : null);
  const episode = $derived(passage ? app.idx!.episode.get(passage.episode_id)! : null);

  function seekTo(ev: MouseEvent) {
    const box = (ev.currentTarget as HTMLElement).getBoundingClientRect();
    player.seek(((ev.clientX - box.left) / box.width) * player.duration);
  }
</script>

{#if !app.ui.audioMin}
  <div class="bar" role="region" aria-label="Listening bar">
    <div class="inner">
      <button class="pp" onclick={() => player.toggle()} disabled={!episode?.audio_url} aria-label={player.paused ? "Play" : "Pause"}>
        {#if player.paused}<svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true"><path d="M2 1 11 6 2 11z" fill="currentColor" /></svg>
        {:else}<svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true"><path d="M2 1h3v10H2zM7 1h3v10H7z" fill="currentColor" /></svg>{/if}
      </button>
      <span class="title">
        {#if episode && passage}{episode.code} · {episode.title} · passage {passage.index + 1}
        {:else}Press ▶ on any passage to listen from that moment{/if}
      </span>
      <span class="time">{formatTime(player.now)} / {formatTime(player.duration || null)}</span>
      <button class="track" onclick={seekTo} aria-label="Seek" disabled={!player.duration}>
        <i style:width="{player.duration ? (player.now / player.duration) * 100 : 0}%"></i>
      </button>
      {#if player.failed}<span class="note">{episode && !episode.audio_url ? "This episode has no audio link." : "The audio could not be loaded."}</span>{/if}
      {#if episode?.source_url}<a class="note" href={episode.source_url} target="_blank" rel="noopener">Episode page ↗</a>{/if}
    </div>
  </div>
{/if}

<style>
  .bar { position: fixed; left: var(--rail-w); right: 0; bottom: 0; z-index: 40; background: var(--surface); border-top: 1px solid var(--line);
    padding: 10px 16px calc(10px + env(safe-area-inset-bottom, 0px)); }
  .inner { max-width: 1680px; margin-inline: auto; display: flex; align-items: center; gap: 14px; flex-wrap: wrap; }
  .pp { width: 34px; height: 34px; border-radius: 50%; border: 0; background: var(--accent); color: var(--accent-ink); display: grid; place-items: center; cursor: pointer; flex: none; }
  .pp:disabled { opacity: 0.4; cursor: default; }
  .title { font-weight: 600; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex: 1 1 220px; }
  .time { font-family: var(--font-data); font-size: 12px; color: var(--ink-2); font-variant-numeric: tabular-nums; }
  .track { flex: 2 1 160px; height: 6px; background: var(--surface-2); border-radius: 3px; position: relative; border: 0; padding: 0; cursor: pointer; }
  .track i { position: absolute; left: 0; top: 0; bottom: 0; background: var(--accent); border-radius: 3px; }
</style>
