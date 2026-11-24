<script lang="ts">
  import { app } from "../lib/state.svelte.ts";
  import { player } from "../lib/player.svelte.ts";
  import { formatTime } from "../lib/format.ts";

  const passage = $derived(player.passageId ? app.idx!.passage.get(player.passageId) ?? null : null);
  const episode = $derived(passage ? app.idx!.episode.get(passage.episode_id)! : null);

  const SKIP = 15;
  const RATES = [1, 1.25, 1.5, 1.75, 2];
  const changeRate = () => player.setRate(RATES[(RATES.indexOf(player.rate) + 1) % RATES.length]);

  function seekTo(ev: MouseEvent) {
    const box = (ev.currentTarget as HTMLElement).getBoundingClientRect();
    player.seek(((ev.clientX - box.left) / box.width) * player.duration);
  }
</script>

<!-- The bar is there while there is something to listen to, and goes away with its own button. -->
{#if app.ui.audioOpen}
  <div class="bar" role="region" aria-label="Listening bar">
    <div class="inner">
      <button class="pp" onclick={() => player.toggle()} disabled={!episode?.audio_url} aria-label={player.paused ? "Play" : "Pause"}>
        {#if player.paused}<svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true"><path d="M2 1 11 6 2 11z" fill="currentColor" /></svg>
        {:else}<svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true"><path d="M2 1h3v10H2zM7 1h3v10H7z" fill="currentColor" /></svg>{/if}
      </button>
      <button class="btn quiet" onclick={() => player.seek(player.now - SKIP)} disabled={!player.duration} aria-label="Back {SKIP} seconds" title="Back {SKIP} seconds">−{SKIP}</button>
      <button class="btn quiet" onclick={() => player.seek(player.now + SKIP)} disabled={!player.duration} aria-label="Forward {SKIP} seconds" title="Forward {SKIP} seconds">+{SKIP}</button>
      <button class="btn quiet rate" onclick={changeRate} aria-label="Speed {player.rate} times" title="Playing speed">{player.rate}x</button>
      <span class="title">
        {#if episode}
          <button class="linkish" title="Open the transcript at this place" onclick={() => app.readPassage(player.passageId!)}>{episode.code} · {episode.title}</button>
        {/if}
      </span>
      <span class="time">{formatTime(player.now)} / {formatTime(player.duration || null)}</span>
      <button class="track" onclick={seekTo} aria-label="Seek" disabled={!player.duration}>
        <i style:width="{player.duration ? (player.now / player.duration) * 100 : 0}%"></i>
      </button>
      {#if player.failed}<span class="note">{episode && !episode.audio_url ? "This episode has no audio link." : "The audio could not be loaded."}</span>{/if}
      {#if player.stopsAtEnd}<button class="btn ghost" title="Playing stops at the end of this passage. Click to go on listening." onclick={() => player.keepPlaying()}>Keep playing</button>{/if}
      {#if episode?.source_url}<a class="note page" href={episode.source_url} target="_blank" rel="noopener">Episode page ↗</a>{/if}
      <button class="close" aria-label="Stop and close the listening bar" title="Stop and close" onclick={() => { player.pause(); app.ui.audioOpen = false; }}>×</button>
    </div>
  </div>
{/if}

<style>
  .bar { position: fixed; left: 0; right: 0; bottom: var(--tabs-h, 0px); z-index: 28; min-height: var(--audio-h); background: var(--surface); border-top: 1px solid var(--line);
    padding: 8px 16px calc(8px + env(safe-area-inset-bottom, 0px)); display: flex; align-items: center; }
  .inner { flex: 1; max-width: 1680px; margin-inline: auto; display: flex; align-items: center; gap: 6px 10px; min-width: 0; }
  .pp { width: 38px; height: 38px; border-radius: 50%; border: 0; background: var(--accent); color: var(--accent-ink); display: grid; place-items: center; cursor: pointer; flex: none; }
  .pp:disabled { opacity: 0.4; cursor: default; }
  .rate { min-width: 46px; font-variant-numeric: tabular-nums; }
  .title { min-width: 0; flex: 1 1 200px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .title .linkish { font-size: 13px; font-weight: 600; color: var(--ink); text-decoration: none; max-width: 100%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .title .linkish:hover { text-decoration: underline; }
  .time { font-family: var(--font-data); font-size: 12px; color: var(--ink-2); font-variant-numeric: tabular-nums; white-space: nowrap; }
  .track { flex: 1.4 1 120px; height: 6px; background: var(--surface-2); border-radius: 3px; position: relative; border: 0; padding: 0; cursor: pointer; }
  .track i { position: absolute; left: 0; top: 0; bottom: 0; background: var(--accent); border-radius: 3px; }
  .page { white-space: nowrap; }
  @media (max-width: 760px) { .title, .page, .time { display: none; } .bar { padding-inline: 8px; } }
</style>
