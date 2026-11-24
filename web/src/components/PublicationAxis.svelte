<script lang="ts">
  // The time axis under the graph. Normally it follows panning and zooming: a graph position x is
  // drawn at x * zoom + pan. While only some episodes are shown, the graph no longer spreads over
  // time, so the axis becomes a fixed overview of the whole run instead, to pick other episodes.
  // Clicking a mark picks that episode. A bar above the axis then offers to show only those.
  import { app } from "../lib/state.svelte.ts";
  import { axisTicks, publicationScale, type PublicationScale } from "../lib/episodes.ts";
  import { episodeVisible } from "../lib/filters.ts";

  let { scale, zoom, pan, width }: { scale: PublicationScale; zoom: number; pan: number; width: number } = $props();

  /** Room left at each end of the overview, so the outer marks can be clicked. */
  const OVERVIEW_MARGIN = 36;
  const only = $derived(app.filters!.only);
  const overview = $derived(only ? publicationScale(app.ds!.episodes, Math.max(100, width - 2 * OVERVIEW_MARGIN)) : null);
  const s = $derived(overview ?? scale);
  const z = $derived(overview ? 1 : zoom), p = $derived(overview ? width / 2 : pan);
  const at = (x: number) => x * z + p;
  const ticks = $derived(axisTicks(s, -p / z, (width - p) / z, width).map((t) => ({ ...t, left: at(t.x) })));
  // Episodes outside the current choice stay on the axis, so they can be added to it.
  const marks = $derived(app.ds!.episodes.filter((e) => e.date && episodeVisible(e, { ...app.filters!, only: null }))
    .map((e) => ({ e, left: at(s.x(e.date!)) })).filter((m) => m.left >= 0 && m.left <= width));

  // Marks get narrower when many stand close, so that neighbours can still be told apart.
  const markWidth = $derived(Math.max(3, Math.min(10, width / Math.max(1, marks.length) * 0.9)));
  const codes = (ids: Set<string>) => {
    const list = [...ids].map((id) => app.idx!.episode.get(id)?.code ?? id).sort();
    return list.slice(0, 3).join(", ") + (list.length > 3 ? ` and ${list.length - 3} more` : "");
  };
  const same = $derived(!!only && only.size === app.picked.size && [...only].every((id) => app.picked.has(id)));
</script>

{#if !overview}<div class="guides" aria-hidden="true">{#each ticks as t (t.text)}<i style:left="{t.left}px"></i>{/each}</div>{/if}

{#if app.picked.size && !same}
  <div class="choice card" role="group" aria-label="Chosen episodes">
    <span><b>{codes(app.picked)}</b> picked</span>
    <button class="btn" onclick={() => app.showOnly(app.picked)}>Show only {app.picked.size === 1 ? "this" : "these"}</button>
    <button class="btn quiet" onclick={() => (app.picked = new Set())}>Clear</button>
  </div>
{/if}

<div class="axis" role="group" aria-label="Publication dates, old to new from left to right. Click an episode to pick it.">
  <i class="line" style:left="{Math.max(0, at(s.left))}px" style:right="{Math.max(0, width - at(s.right))}px"></i>
  {#each ticks as t (t.text)}<span class="tick" style:left="{t.left}px">{t.text}</span>{/each}
  {#each marks as m (m.e.id)}
    <button class="mark" class:picked={app.picked.has(m.e.id)} class:out={only && !only.has(m.e.id)} style:left="{m.left}px"
      style:--w="{markWidth}px" aria-pressed={app.picked.has(m.e.id)} title="{m.e.code}, {m.e.date}: {m.e.title}. Click to {app.picked.has(m.e.id) ? 'remove it from your choice' : 'pick it'}."
      aria-label="{m.e.code}, published {m.e.date}" onclick={() => app.pickEpisode(m.e.id)}></button>
  {/each}
</div>

<style>
  /* Behind the graph, which has a clear background: a guide for the eye, not part of the picture. */
  .guides { position: absolute; inset: 0; z-index: 0; pointer-events: none; overflow: hidden; }
  .guides i { position: absolute; top: 0; bottom: 34px; width: 1px; background: var(--line); opacity: 0.3; }
  .axis { position: absolute; left: 0; right: 0; bottom: 0; height: 34px; z-index: 3; overflow: hidden;
    background: color-mix(in srgb, var(--canvas) 92%, transparent); border-top: 1px solid var(--line); border-radius: 0 0 var(--radius) var(--radius); }
  .line { position: absolute; top: 9px; height: 2px; background: var(--ink-3); opacity: 0.45; }
  .tick { position: absolute; top: 16px; padding-left: 4px; font-family: var(--font-data); font-size: 10.5px; color: var(--ink-3); white-space: nowrap; }
  .tick::before { content: ""; position: absolute; left: 0; top: -7px; height: 16px; width: 1px; background: var(--ink-3); opacity: 0.5; }
  .mark { position: absolute; top: 5px; width: var(--w, 10px); height: 10px; margin-left: calc(var(--w, 10px) / -2); padding: 0; border-radius: 3px; border: 1.5px solid var(--canvas);
    background: var(--episode); cursor: pointer; }
  .mark:hover { transform: scale(1.35); }
  .mark.out { opacity: 0.4; }
  .mark.picked { opacity: 1; transform: scale(1.35); box-shadow: 0 0 0 2px var(--ink); }
  /* A larger area to hit than the mark itself. */
  .mark::after { content: ""; position: absolute; inset: -6px; }
  .choice { position: absolute; z-index: 6; left: 50%; transform: translateX(-50%); bottom: 42px; display: flex; align-items: center; gap: 10px; padding: 6px 8px 6px 12px;
    box-shadow: var(--shadow); font-size: 12.5px; white-space: nowrap; max-width: calc(100% - 24px); }
</style>
