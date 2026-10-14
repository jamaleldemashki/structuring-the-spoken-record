<script lang="ts">
  import { app } from "../lib/state.svelte.ts";
  import { episodeVisible } from "../lib/filters.ts";
  import { formatTime } from "../lib/format.ts";
  import { typeVar } from "../lib/entityStyle.ts";
  import Quote from "./Quote.svelte";

  const ds = $derived(app.ds!);
  const idx = $derived(app.idx!);
  const episodes = $derived([...ds.episodes].sort((a, b) => (a.date ?? "").localeCompare(b.date ?? "")));
  const current = $derived(app.episode ? idx.episode.get(app.episode) ?? null : null);
  const passages = $derived(current ? ds.passages.filter((p) => p.episode_id === current.id).sort((a, b) => a.index - b.index) : []);
  const entitiesIn = (pid: string) => [...(idx.passageEntities.get(pid) ?? [])].map((id) => idx.entity.get(id)!)
    .filter((e) => app.filters!.types.has(e.type)).sort((a, b) => b.mention_count - a.mention_count);
  const topEntities = (epId: string) => ds.entities.filter((e) => e.episode_ids.includes(epId) && app.filters!.types.has(e.type))
    .sort((a, b) => b.mention_count - a.mention_count).slice(0, 5);
  const related = $derived(current ? ds.related.filter((r) => idx.passage.get(r.source)?.episode_id === current.id && app.vp.has(r.target))
    .sort((a, b) => b.score - a.score).slice(0, 8) : []);

  // Arriving from "In episode": bring that passage into view and mark it.
  $effect(() => {
    const id = app.reading;
    if (id && current) requestAnimationFrame(() => document.getElementById(`psg-${id}`)?.scrollIntoView({ block: "center" }));
  });

  // Mark entity mentions inside the full passage text.
  function segments(pid: string, text: string) {
    const occ = ds.entities.flatMap((e) => app.filters!.types.has(e.type) ? e.occurrences.filter((o) => o.passage_id === pid).map((o) => ({ ...o, id: e.id })) : [])
      .sort((a, b) => a.start - b.start);
    const out: { text: string; id?: string }[] = [];
    let pos = 0;
    for (const o of occ) {
      if (o.start < pos) continue;
      out.push({ text: text.slice(pos, o.start) }, { text: text.slice(o.start, o.end), id: o.id });
      pos = o.end;
    }
    out.push({ text: text.slice(pos) });
    return out;
  }
</script>

<div class="cards">
  {#each episodes as ep (ep.id)}
    <button class="card ep" aria-pressed={app.episode === ep.id} class:faded={!episodeVisible(ep, app.filters!)} onclick={() => { app.episode = ep.id; app.reading = null; }}>
      <span class="code">{ep.code} · {ep.date ?? "date unknown"}{ep.era ? ` · ${ep.era}` : ""}</span>
      <b class="serif">{ep.title}</b>
      <span class="chips">{#each topEntities(ep.id) as e (e.id)}<span class="echip"><span class="dot" style:background={typeVar(e.type)}></span>{e.label}</span>{/each}</span>
    </button>
  {/each}
</div>

{#if current}
  <div class="reader">
    <article class="card read">
      <span class="note">{current.code} · {current.date ?? ""}{current.era ? ` · ${current.era}` : ""}</span>
      <h2 class="serif">{current.title}</h2>
      {#each passages as p (p.id)}
        <div class="psg" id="psg-{p.id}" class:reading={app.reading === p.id}>
          <div class="tc">{formatTime(p.start_s)}<button class="play" onclick={() => app.playPassage(p.id)} aria-label="Listen from {formatTime(p.start_s)}">▶</button></div>
          <div>
            {#if p.text}
              <p>{#each segments(p.id, p.text) as s}{#if s.id}<button class="inline" onclick={() => app.openEntity(s.id!)}>{s.text}</button>{:else}{s.text}{/if}{/each}</p>
            {/if}
            <span class="chips">{#each entitiesIn(p.id) as e (e.id)}<button class="echip" onclick={() => app.openEntity(e.id)}><span class="dot" style:background={typeVar(e.type)}></span>{e.label}</button>{/each}</span>
          </div>
        </div>
      {/each}
    </article>
    <aside class="card side">
      <h3 class="section-title">Related discussions elsewhere</h3>
      {#each related as r (r.source + r.target)}
        <p class="note">From passage {idx.passage.get(r.source)!.index + 1} · similarity {r.score.toFixed(2)}</p>
        <Quote passageId={r.target} ranges={[]} />
      {:else}
        <p class="empty">No related discussions in the current filters.</p>
      {/each}
    </aside>
  </div>
{/if}

<style>
  .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px; }
  .ep { padding: 14px; cursor: pointer; text-align: left; display: flex; flex-direction: column; gap: 6px; }
  .ep[aria-pressed="true"] { border-color: var(--accent); box-shadow: 0 0 0 1px var(--accent); }
  .ep.faded { opacity: 0.45; }
  .code { font-family: var(--font-data); font-size: 12px; color: var(--ink-3); }
  .ep b { font-size: 17px; font-weight: 600; line-height: 1.3; }
  .reader { margin-top: 14px; display: grid; grid-template-columns: minmax(0, 1fr) 320px; gap: 12px; align-items: start; }
  .read { padding: 18px 22px; min-width: 0; }
  h2 { font-size: 24px; margin: 2px 0 6px; font-weight: 600; text-wrap: balance; }
  .psg { border-top: 1px solid var(--line); padding: 14px 0; display: grid; grid-template-columns: 70px minmax(0, 1fr); gap: 12px; }
  .tc { font-family: var(--font-data); font-size: 11.5px; color: var(--ink-3); display: flex; flex-direction: column; gap: 6px; align-items: flex-start; }
  .psg p { font-family: var(--font-read); font-size: 16px; line-height: 1.65; margin: 0 0 8px; max-width: 70ch; }
  .inline { border: 0; padding: 0; background: none; font: inherit; color: inherit; cursor: pointer; text-decoration: underline;
    text-decoration-color: var(--ink-3); text-underline-offset: 3px; }
  .inline:hover { background: var(--mark); }
  .side { padding: 14px 16px; position: sticky; top: 0; max-height: 80vh; overflow: auto; }
  @media (max-width: 980px) { .reader { grid-template-columns: 1fr; } .side { position: static; } }
  @media (max-width: 520px) { .psg { grid-template-columns: 1fr; gap: 4px; } }
  .psg.reading { background: color-mix(in srgb, var(--mark) 45%, transparent); border-radius: 8px; box-shadow: 0 0 0 6px color-mix(in srgb, var(--mark) 45%, transparent); }
</style>
