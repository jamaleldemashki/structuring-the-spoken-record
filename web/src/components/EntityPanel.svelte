<script lang="ts">
  import { app } from "../lib/state.svelte.ts";
  import { entityPassagesInView } from "../lib/filters.ts";
  import { edgeRank } from "../lib/graph.ts";
  import { lifeLabel, plural } from "../lib/format.ts";
  import { typeStyle, typeVar } from "../lib/entityStyle.ts";
  import type { Edge } from "../lib/types.ts";
  import Quote from "./Quote.svelte";

  let { id }: { id: string } = $props();
  const idx = $derived(app.idx!);
  const e = $derived(idx.entity.get(id)!);
  const passages = $derived(entityPassagesInView(e, idx, app.vp));
  const byEpisode = $derived.by(() => {
    const groups = new Map<string, string[]>();
    for (const p of passages) {
      const ep = idx.passage.get(p)!.episode_id;
      if (!groups.has(ep)) groups.set(ep, []);
      groups.get(ep)!.push(p);
    }
    for (const list of groups.values()) list.sort((a, b) => idx.passage.get(a)!.index - idx.passage.get(b)!.index);
    return [...groups].sort((a, b) => (idx.episode.get(a[0])!.date ?? "").localeCompare(idx.episode.get(b[0])!.date ?? ""));
  });
  const maxCount = $derived(Math.max(1, ...byEpisode.map(([, ps]) => ps.length)));
  const connections = $derived(app.edges.filter((ed) => ed.source === id || ed.target === id).sort((a, b) => edgeRank(b) - edgeRank(a)));
  const relations = $derived(connections.filter((c) => c.kind === "relation"));
  const together = $derived(connections.filter((c) => c.kind === "co_discussed").slice(0, 8));
  const related = $derived.by(() => {
    const mine = new Set(passages), seen = new Set<string>();
    return (app.ds!.related ?? [])
      .filter((r) => mine.has(r.source) && app.vp.has(r.target) && !mine.has(r.target) && !seen.has(r.target) && seen.add(r.target))
      .sort((a, b) => b.score - a.score).slice(0, 4);
  });
  let expanded = $state<Set<string>>(new Set());

  const rangesIn = (passage: string) => e.occurrences.filter((o) => o.passage_id === passage);
  const otherEnd = (ed: Edge) => idx.entity.get(ed.source === id ? ed.target : ed.source)!;
  function direction(ed: Edge): string {
    if (ed.kind === "co_discussed") return `${plural(ed.count, "shared passage")}`;
    const mark = ed.status === "nli_supported" ? " ✓" : "";
    return ed.source === id ? `${ed.label}${mark} →` : `← ${ed.label}${mark}`;
  }
  const linkText = $derived(e.qid ? null : e.link_status === "ambiguous" ? "Wikidata: several candidates, not linked"
    : e.link_status === "nil" ? "No matching Wikidata entry" : null);
</script>

<div class="head">
  <span class="eyebrow"><span class="dot" style:background={typeVar(e.type)}></span>{typeStyle(e.type).label}</span>
  {#if app.mode === "full"}
    <button class="iconbtn" onclick={() => { app.mode = "focused"; app.path = null; }}>Show only its neighbourhood</button>
  {:else}
    <button class="iconbtn" onclick={() => { app.mode = "full"; app.path = null; }}>Back to full graph</button>
  {/if}
</div>
<h2 class="serif">{e.label}</h2>
<div class="meta">
  {#if lifeLabel(e.life)}<span>{lifeLabel(e.life)}</span>{/if}
  {#if e.qid}<a href="https://www.wikidata.org/wiki/{e.qid}" target="_blank" rel="noopener">Wikidata {e.qid} ↗</a>{:else if linkText}<span>{linkText}</span>{/if}
  <span>{plural(passages.length, "passage")} in {plural(byEpisode.length, "episode")}</span>
</div>
{#if e.aliases.length}<p class="note">Also heard as: {e.aliases.join(", ")}</p>{/if}

{#if byEpisode.length > 1}
  <h3 class="section-title">Passages per episode</h3>
  <div class="bars">
    {#each byEpisode as [ep, ps]}
      <span>{idx.episode.get(ep)!.code}</span><span class="track"><i style:width="{(ps.length / maxCount) * 100}%" style:background={typeVar(e.type)}></i></span><span class="n">{ps.length}</span>
    {/each}
  </div>
{/if}

<h3 class="section-title">Connected to</h3>
<div class="conn">
  {#each relations as ed (ed.id)}
    <button onclick={() => app.selectEdge(ed)}><span class="dot" style:background={typeVar(otherEnd(ed).type)}></span>{otherEnd(ed).label}<span class="rel">{direction(ed)}</span></button>
  {:else}
    <p class="empty">No named relations in view.</p>
  {/each}
  {#if together.length}
    <p class="note sub">Discussed together</p>
    {#each together as ed (ed.id)}
      <button onclick={() => app.selectEdge(ed)}><span class="dot hollow"></span>{otherEnd(ed).label}<span class="rel">{direction(ed)}</span></button>
    {/each}
  {/if}
</div>

<h3 class="section-title">Where it is discussed</h3>
{#each byEpisode as [ep, ps] (ep)}
  {@const episode = idx.episode.get(ep)!}
  <div class="ephead">{episode.code} · {episode.title}<span>{episode.date ?? ""}</span></div>
  {#each expanded.has(ep) ? ps : ps.slice(0, 3) as p (p)}<Quote passageId={p} ranges={rangesIn(p)} showSource={false} />{/each}
  {#if ps.length > 3 && !expanded.has(ep)}
    <button class="more" onclick={() => (expanded = new Set([...expanded, ep]))}>Show {ps.length - 3} more passages</button>
  {/if}
{:else}
  <p class="empty">No passages match the current filters.</p>
{/each}

<h3 class="section-title">Related discussions in other episodes</h3>
{#each related as r (r.target)}
  <Quote passageId={r.target} ranges={[]} />
  <p class="note score">Similarity {r.score.toFixed(2)}{r.shared_entity_ids.length ? ` · also mentions ${r.shared_entity_ids.slice(0, 3).map((x) => idx.entity.get(x)?.label).filter(Boolean).join(", ")}` : ""}</p>
{:else}
  <p class="empty">No related discussions above the similarity threshold.</p>
{/each}

<style>
  .head { display: flex; justify-content: space-between; align-items: center; gap: 8px; }
  h2 { font-size: 24px; font-weight: 600; margin: 6px 0 0; letter-spacing: -0.01em; text-wrap: balance; }
  .meta { display: flex; flex-wrap: wrap; gap: 6px 14px; font-size: 12.5px; color: var(--ink-2); margin-top: 6px; }
  .meta a { color: inherit; }
  .bars { display: grid; grid-template-columns: auto 1fr auto; gap: 4px 8px; align-items: center; font-size: 12px; }
  .track { height: 8px; background: var(--surface-2); border-radius: 4px; overflow: hidden; }
  .track i { display: block; height: 100%; border-radius: 0 4px 4px 0; }
  .n { font-family: var(--font-data); font-size: 11.5px; color: var(--ink-2); font-variant-numeric: tabular-nums; }
  .conn { display: flex; flex-direction: column; gap: 2px; }
  .conn button { display: flex; align-items: center; gap: 8px; border: 0; background: none; padding: 5px 6px; border-radius: 6px; cursor: pointer; text-align: left; width: 100%; }
  .conn button:hover { background: var(--surface-2); }
  .rel { margin-left: auto; font-size: 11.5px; color: var(--ink-3); white-space: nowrap; }
  .sub { margin: 8px 6px 2px; }
  .ephead { font-size: 12.5px; font-weight: 600; margin-top: 12px; display: flex; justify-content: space-between; gap: 8px; }
  .ephead span { font-weight: 400; color: var(--ink-3); font-family: var(--font-data); font-size: 11.5px; white-space: nowrap; }
  .more { border: 0; background: none; color: var(--ink-2); text-decoration: underline; cursor: pointer; font-size: 12px; padding: 4px 0; }
  .score { margin: -4px 0 8px; }
</style>
