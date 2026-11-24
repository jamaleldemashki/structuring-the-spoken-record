<script lang="ts">
  // The details of a topic. What the podcast says about it comes first, since that is what people
  // come for. The connections are one tab away.
  import { app } from "../lib/state.svelte.ts";
  import { entityPassagesInView, evidenceInView } from "../lib/filters.ts";
  import { edgeRank } from "../lib/graph.ts";
  import { lifeLabel, plural } from "../lib/format.ts";
  import { typeStyle } from "../lib/entityStyle.ts";
  import type { Edge } from "../lib/types.ts";
  import Quote from "./Quote.svelte";
  import TypeMark from "./TypeMark.svelte";

  let { id }: { id: string } = $props();
  const FIRST_PASSAGES = 3, FIRST_CONNECTIONS = 8;
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
    return new Map(groups);
  });
  // One bar per episode of the archive in order of publication, so a gap shows as a gap. Episodes
  // without a date come last, in the order of the archive. They still have something to say.
  const strip = $derived([...app.ds!.episodes].sort((a, b) => (a.date ? (b.date ? a.date.localeCompare(b.date) : -1) : b.date ? 1 : 0)));
  const most = $derived(Math.max(1, ...[...byEpisode.values()].map((ps) => ps.length)));
  // Clicking a bar shows only that episode's passages.
  let oneEpisode = $state<string | null>(null);
  const listed = $derived(strip.filter((ep) => byEpisode.has(ep.id) && (!oneEpisode || ep.id === oneEpisode)));

  // Every connection inside the filters, whatever the graph happens to draw.
  const connections = $derived(app.allLinks.filter((ed) => ed.source === id || ed.target === id).sort((a, b) => edgeRank(b) - edgeRank(a)));
  const relations = $derived(connections.filter((c) => c.kind === "relation"));
  // Named relations under their headings (positions, lineage, texts), in the archive's order. A
  // relation without a group goes under the general heading at the end.
  const relationGroups = $derived.by(() => {
    const order = app.idx!.site.relation_groups ?? [];
    const by = new Map<string, Edge[]>();
    for (const ed of relations) {
      const g = (ed.kind === "relation" && ed.group) || "Named relations";
      by.set(g, [...(by.get(g) ?? []), ed]);
    }
    // Inside a heading, one row per wording, read from this topic: "view on" followed by every
    // idea it has a view on, instead of one row per idea.
    return [...by].sort(([a], [b]) => (order.indexOf(a) + 1 || 1e3) - (order.indexOf(b) + 1 || 1e3))
      .map(([heading, list]) => {
        const rows = new Map<string, Edge[]>();
        for (const ed of list) rows.set(predicate(ed), [...(rows.get(predicate(ed)) ?? []), ed]);
        return [heading, [...rows]] as const;
      });
  });
  /** How many topics a relation row lists before "more". */
  const FIRST_OF_ROW = 8;
  let openRows = $state<Set<string>>(new Set());
  const allTogether = $derived(connections.filter((c) => c.kind === "co_discussed"));
  const partners = $derived(new Set(connections.map((c) => (c.source === id ? c.target : c.source))).size);
  let wholeList = $state(false);
  const together = $derived(wholeList ? allTogether : allTogether.slice(0, FIRST_CONNECTIONS));
  // Openings of episodes sound alike in every episode, so they say nothing about the topic.
  const related = $derived.by(() => {
    const mine = new Set(passages), seen = new Set<string>();
    return (app.ds!.related ?? [])
      .filter((r) => mine.has(r.source) && app.vp.has(r.target) && !mine.has(r.target) && idx.passage.get(r.target)!.index > 0 && idx.passage.get(r.source)!.index > 0 && !seen.has(r.target) && seen.add(r.target))
      .sort((a, b) => b.score - a.score).slice(0, 4);
  });
  let tab = $state<"passages" | "connections">("passages");
  let expanded = $state<Set<string>>(new Set());
  let imageFailed = $state(false);

  const rangesIn = (passage: string) => e.occurrences.filter((o) => o.passage_id === passage);
  const otherEnd = (ed: Edge) => idx.entity.get(ed.source === id ? ed.target : ed.source)!;
  // Each row reads as subject, predicate, object with this topic as the subject. When it is the
  // object of a relation, the inverse wording keeps that order ("Aquinas influenced_by Aristotle"
  // shows on Aristotle's page as "influenced Aquinas").
  function predicate(ed: Edge): string {
    if (ed.kind === "co_discussed") return "discussed with";
    const label = ed.source === id ? ed.label : ed.inverse_label ?? `${ed.label} (by)`;
    return ed.status === "nli_supported" ? `${label} ✓` : label;
  }
  const linkText = $derived(e.qid ? null : e.link_status === "ambiguous" ? "Wikidata: several candidates, not linked" : null);
</script>

<div class="title">
  {#if e.image && !imageFailed}
    <a class="portrait" href={e.image.page} target="_blank" rel="noopener" title="{e.image.file}, Wikimedia Commons (author and licence on the file page)">
      <img src={e.image.thumb} alt="Portrait of {e.label}, from Wikimedia Commons" loading="lazy"
        onerror={(ev) => { const img = ev.currentTarget as HTMLImageElement; if (img.src !== e.image!.original) img.src = e.image!.original; else imageFailed = true; }} />
    </a>
  {/if}
  <div class="who">
    <span class="eyebrow"><TypeMark type={e.type} />{typeStyle(e.type).label}</span>
    <h2 class="serif">{e.label}</h2>
    <p class="meta">
      {#if lifeLabel(e.life)}<span>{lifeLabel(e.life)}</span>{/if}
      {#if e.qid}<a href="https://www.wikidata.org/wiki/{e.qid}" target="_blank" rel="noopener">Wikidata ↗</a>{:else if linkText}<span>{linkText}</span>{/if}
    </p>
  </div>
</div>
{#if e.aliases.length}<p class="note also">Also heard as {e.aliases.slice(0, 6).join(", ")}{e.aliases.length > 6 ? ` and ${e.aliases.length - 6} more` : ""}</p>{/if}
{#if e.generic}<p class="note also">A general term rather than a name, so it is not drawn in the graph.</p>{/if}
{#if app.view !== "explore" && !e.generic}<button class="btn go" onclick={() => app.openEntity(id)}>Explore in the map</button>{/if}

<div class="tabs" role="tablist">
  <button role="tab" aria-selected={tab === "passages"} onclick={() => (tab = "passages")}>What was said <small>{passages.length}</small></button>
  <button role="tab" aria-selected={tab === "connections"} onclick={() => (tab = "connections")}>Connections <small>{partners}</small></button>
</div>

{#if tab === "passages"}
  {#if byEpisode.size > 1}
    <div class="strip" role="group" aria-label="Passages per episode, in order of publication. Click an episode to show only its passages.">
      {#each strip as ep (ep.id)}
        {@const n = byEpisode.get(ep.id)?.length ?? 0}
        <button disabled={!n} aria-pressed={oneEpisode === ep.id} title="{ep.code}: {plural(n, 'passage')}" aria-label="{ep.code}: {plural(n, 'passage')}" onclick={() => (oneEpisode = oneEpisode === ep.id ? null : ep.id)}>
          <i style:height="{n ? 12 + (n / most) * 88 : 4}%" class:none={!n}></i>
        </button>
      {/each}
    </div>
    <p class="note range">{plural(passages.length, "passage")} in {plural(byEpisode.size, "episode")}{#if oneEpisode} · <button class="linkish" onclick={() => (oneEpisode = null)}>show all episodes</button>{/if}</p>
  {/if}
  {#each listed as episode (episode.id)}
    {@const ps = byEpisode.get(episode.id)!}
    <div class="ephead"><button class="linkish strong" title="Read the transcript" onclick={() => app.readEpisode(episode.id)}>{episode.code} · {episode.title}</button><span>{episode.date ?? ""}</span></div>
    {#each expanded.has(episode.id) ? ps : ps.slice(0, FIRST_PASSAGES) as p (p)}<Quote passageId={p} ranges={rangesIn(p)} showSource={false} />{/each}
    {#if ps.length > FIRST_PASSAGES && !expanded.has(episode.id)}
      <button class="linkish more" onclick={() => (expanded = new Set([...expanded, episode.id]))}>Show {ps.length - FIRST_PASSAGES} more from this episode</button>
    {/if}
  {:else}
    <p class="empty">Nothing inside the current filters.</p>
  {/each}
  {#if related.length && !oneEpisode}
    <h3 class="section-title">Related, without the name</h3>
    {#each related as r (r.target)}<Quote passageId={r.target} ranges={[]} />{/each}
  {/if}
{:else}
  <div class="conn">
    {#each relationGroups as [heading, rows] (heading)}
      <h3 class="section-title">{heading}</h3>
      {#each rows as [wording, list] (wording)}
        {@const all = openRows.has(wording)}
        <div class="relrow">
          <span class="rel">{wording}</span>
          <span class="objs">
            {#each all ? list : list.slice(0, FIRST_OF_ROW) as ed (ed.id)}
              <button class="obj" title="The passages for {e.label} {wording.replace(' ✓', '')} {otherEnd(ed).label}" onclick={() => app.follow(() => app.selectEdge(ed))}><TypeMark type={otherEnd(ed).type} />{otherEnd(ed).label}</button>
            {/each}
            {#if list.length > FIRST_OF_ROW}
              <button class="morerow" onclick={() => { const s = new Set(openRows); if (all) s.delete(wording); else s.add(wording); openRows = s; }}>{all ? "fewer" : `+${list.length - FIRST_OF_ROW} more`}</button>
            {/if}
          </span>
        </div>
      {/each}
    {/each}
    {#if together.length}
      <h3 class="section-title">Discussed together</h3>
      {#each together as ed (ed.id)}
        <button class="co" onclick={() => app.follow(() => app.selectEdge(ed))}><span class="obj"><TypeMark type={otherEnd(ed).type} />{otherEnd(ed).label}</span><span class="n">{plural(evidenceInView(ed, app.vp).length, "passage")}</span></button>
      {/each}
      {#if allTogether.length > FIRST_CONNECTIONS}
        <button class="linkish more" onclick={() => (wholeList = !wholeList)}>{wholeList ? `Show the strongest ${FIRST_CONNECTIONS}` : `Show all ${allTogether.length}`}</button>
      {/if}
    {/if}
    {#if !partners}<p class="empty">No connections inside the current filters.</p>{/if}
  </div>
{/if}

<style>
  .title { display: flex; align-items: center; gap: 12px; padding-right: 22px; }
  .who { min-width: 0; }
  .portrait { flex: none; width: 60px; height: 60px; border-radius: 50%; overflow: hidden; border: 2px solid var(--surface-2); background: var(--surface-2); }
  .portrait img { width: 100%; height: 100%; object-fit: cover; object-position: top; display: block; }
  h2 { font-size: 23px; font-weight: 600; margin: 2px 0 0; line-height: 1.2; letter-spacing: -0.01em; text-wrap: balance; }
  .meta { display: flex; flex-wrap: wrap; gap: 2px 12px; font-size: 12.5px; color: var(--ink-2); margin: 3px 0 0; }
  .meta a { color: inherit; }
  .also { margin: 8px 0 0; }
  .go { margin-top: 10px; }
  .tabs { position: sticky; top: -16px; z-index: 2; display: flex; gap: 14px; margin: 12px -16px 10px; padding: 8px 16px 0; background: var(--surface); border-bottom: 1px solid var(--line); }
  .tabs button { border: 0; background: none; padding: 2px 0 7px; cursor: pointer; font-size: 13.5px; font-weight: 500; color: var(--ink-2); border-bottom: 2px solid transparent; margin-bottom: -1px; }
  .tabs button:hover { color: var(--ink); }
  .tabs button[aria-selected="true"] { color: var(--ink); border-bottom-color: var(--accent); }
  .tabs small { font-size: 12px; color: var(--ink-3); font-weight: 400; margin-left: 2px; }
  .strip { display: flex; align-items: flex-end; gap: 1px; height: 36px; }
  .strip button { flex: 1; min-width: 0; height: 100%; display: flex; align-items: flex-end; border: 0; padding: 0; background: none; cursor: pointer; }
  .strip button:disabled { cursor: default; }
  .strip i { display: block; width: 100%; border-radius: 1px 1px 0 0; background: var(--hue-1); }
  .strip i.none { background: var(--line); }
  .strip button:hover i:not(.none) { background: var(--ink); }
  .strip button[aria-pressed="true"] i { background: var(--ink); }
  .range { margin: 4px 0 2px; }
  .range .linkish { font-size: inherit; }
  .ephead { font-size: 12.5px; margin-top: 14px; display: flex; justify-content: space-between; align-items: baseline; gap: 8px; }
  .ephead .strong { font-weight: 600; color: var(--ink); text-align: left; text-decoration: none; }
  .ephead .strong:hover { text-decoration: underline; }
  .ephead span { color: var(--ink-3); font-family: var(--font-data); font-size: 11.5px; white-space: nowrap; }
  .more { display: block; padding: 4px 0; }
  .conn { display: flex; flex-direction: column; gap: 1px; }
  .conn .section-title { margin: 10px 0 4px; }
  /* Two columns: the predicate, then the other topic. */
  .conn > button:not(.more) { display: grid; grid-template-columns: minmax(0, 0.8fr) minmax(0, 1.2fr); align-items: baseline; gap: 10px; border: 0; background: none; padding: 5px 6px; border-radius: 6px;
    cursor: pointer; text-align: left; width: 100%; font: inherit; color: inherit; }
  .conn > button:hover { background: var(--surface-2); }
  .conn > button.co { grid-template-columns: minmax(0, 1fr) auto; }
  .rel { font-size: 12.5px; color: var(--ink-2); }
  /* A wording, then the topics it leads to, wrapping like a list of names. */
  .relrow { display: grid; grid-template-columns: minmax(0, 0.8fr) minmax(0, 1.2fr); align-items: baseline; gap: 10px; padding: 4px 6px; }
  .objs { display: flex; flex-wrap: wrap; gap: 2px 4px; min-width: 0; }
  .objs .obj { border: 0; background: none; padding: 1px 5px; margin-left: -5px; border-radius: 5px; font: inherit; color: inherit; cursor: pointer; text-align: left; }
  .objs .obj:hover { background: var(--surface-2); }
  .morerow { border: 0; background: none; padding: 1px 5px; font: inherit; font-size: 12.5px; color: var(--ink-2); cursor: pointer; text-decoration: underline; text-underline-offset: 2px; }
  .obj { display: flex; align-items: center; gap: 8px; min-width: 0; }
  .n { font-size: 12px; color: var(--ink-3); white-space: nowrap; font-variant-numeric: tabular-nums; }
</style>
