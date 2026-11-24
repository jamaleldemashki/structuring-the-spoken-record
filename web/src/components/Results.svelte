<script module lang="ts">
  // Where the page was left, so that coming back to the same search finds it as it was.
  const memory = { text: "", scroll: 0, passages: 0, episodes: 0, groups: new Set<string>() };
</script>

<script lang="ts">
  // Everything found for what was typed, on one page, for a word and for a question. A topic opens
  // in the details beside the list, so the list stays where it is. An episode opens its transcript,
  // and a passage can be played here.
  import { app } from "../lib/state.svelte.ts";
  import { typeStyle } from "../lib/entityStyle.ts";
  import { lifeLabel, plural } from "../lib/format.ts";
  import { parseQuestion } from "../lib/ask.ts";
  import { findPath } from "../lib/graph.ts";
  import type { ItemResult, PassageResult } from "../lib/results.ts";
  import Quote from "./Quote.svelte";
  import TypeMark from "./TypeMark.svelte";

  const PASSAGE_STEP = 15, ITEM_STEP = 12, EPISODE_STEP = 10;
  const results = $derived(app.results!);
  const status = $derived(app.query!.status);
  const label = (id: string) => app.idx!.entity.get(id)?.label ?? id;
  const itemCount = $derived(results.groups.reduce((n, g) => n + g.items.length, 0));
  const nothing = $derived(!itemCount && !results.closeItems.length && !results.passages.length && !results.closePassages.length && !results.episodes.length);
  // The best match gets a card of its own when its name is exactly what was typed.
  const best = $derived(results.groups.flatMap((g) => g.items)
    .find((r) => !r.entity.generic && r.entity.label.toLowerCase() === results.text.trim().toLowerCase()) ?? null);
  const others = $derived(results.groups.map((g) => ({ ...g, items: g.items.filter((r) => r !== best) })).filter((g) => g.items.length));
  // Two names in one text ask how they connect.
  const named = $derived(parseQuestion(results.text, app.names).names.map((n) => n.entity));
  const chain = $derived(named.length >= 2 ? { a: named[0], b: named[1], path: findPath(app.allLinks, named[0].id, named[1].id) } : null);

  const text = $derived(results.text);
  const returning = memory.text === app.results?.text;
  let shownPassages = $state(returning ? memory.passages : PASSAGE_STEP), shownEpisodes = $state(returning ? memory.episodes : EPISODE_STEP);
  let openGroups = $state(returning ? memory.groups : new Set<string>());
  let page: HTMLElement;
  // A new search starts at the top with the lists collapsed. The same search again stays as it was.
  $effect(() => {
    const main = page?.closest("main");
    if (memory.text === text) {
      if (main) requestAnimationFrame(() => main.scrollTo({ top: memory.scroll }));
      return;
    }
    memory.text = text;
    memory.scroll = 0;
    shownPassages = PASSAGE_STEP;
    shownEpisodes = EPISODE_STEP;
    openGroups = new Set();
    main?.scrollTo({ top: 0 });
  });
  $effect(() => { memory.passages = shownPassages; memory.episodes = shownEpisodes; memory.groups = openGroups; });
  $effect(() => {
    const main = page?.closest("main");
    if (!main) return;
    const keep = () => { memory.scroll = main.scrollTop; };
    main.addEventListener("scroll", keep, { passive: true });
    return () => main.removeEventListener("scroll", keep);
  });

  // Why it is listed, when that is not plain from the marked words.
  function why(r: PassageResult): string {
    const parts = [...r.items.map(label), ...r.terms.map((t) => `“${t}”`)];
    if (!parts.length) return "Close in meaning";
    return parts.length > 1 ? `Has ${parts.join(", ")}` : "";
  }
</script>

<div class="results" bind:this={page}>
  <header>
    <h1 class="serif">{results.text}</h1>
    <p class="summary">
      {plural(itemCount + results.closeItems.length, "topic")} · {plural(results.passages.length, "passage")} · {plural(results.episodes.length, "episode")}
      {#if status === "meaning"}<span class="status">Looking for what is close in meaning…</span>
      {:else if status === "names"}<span class="status">Showing matches by name and word.</span>{/if}
    </p>
  </header>

  {#if nothing && status !== "meaning"}
    <p class="empty">Nothing found for this inside the current filters. Try another word, or widen the filters.</p>
  {/if}

  {#if best}
    <div class="best card">
      <button class="what" onclick={() => app.selectEntity(best.entity.id)}>
        {#if best.entity.image}<img src={best.entity.image.thumb} alt="" loading="lazy" />{/if}
        <span>
          <span class="eyebrow"><TypeMark type={best.entity.type} />{typeStyle(best.entity.type).label}</span>
          <b class="serif">{best.entity.label}</b>
          <small>{[lifeLabel(best.entity.life), plural(best.passages, "passage"), plural(best.entity.episode_ids.length, "episode")].filter(Boolean).join(" · ")}</small>
        </span>
      </button>
      <button class="btn" onclick={() => app.openEntity(best.entity.id)}>Explore in the map</button>
    </div>
  {:else if itemCount > 1 && !chain?.path}
    <div><button class="btn ghost" onclick={() => app.ask(results.text)} title="The topics recognised in your text and the chains between them">Show these in the map</button></div>
  {/if}

  {#if chain}
    <section class="chain">
      <h2 class="section-title">How they connect</h2>
      {#if chain.path}
        <p class="steps">
          {#each chain.path.nodes as id, i (id)}
            <button class="echip" onclick={() => app.selectEntity(id)}><TypeMark type={app.idx!.entity.get(id)!.type} />{label(id)}</button>
            {#if i < chain.path.edges.length}
              {@const ed = chain.path.edges[i]}
              <button class="linkish" onclick={() => app.selectEdge(ed)}>{ed.kind === "relation" ? ed.label : "discussed together"}</button>
            {/if}
          {/each}
        </p>
        <button class="btn" title="The chain between them and what both are connected to" onclick={() => app.exploreTogether(chain.a.id, chain.b.id)}>See what they share</button>
      {:else}
        <p class="empty">{chain.a.label} and {chain.b.label} are not linked inside the current filters.</p>
      {/if}
    </section>
  {/if}

  {#if others.length}
    <section>
      <h2 class="section-title">Topics</h2>
      {#each others as g (g.type)}
        {@const open = openGroups.has(g.type)}
        <div class="group">
          <span class="glabel"><TypeMark type={g.type} />{g.label}</span>
          <span class="chips">
            {#each open ? g.items : g.items.slice(0, ITEM_STEP) as r (r.entity.id)}{@render item(r)}{/each}
            {#if g.items.length > ITEM_STEP}
              <button class="linkish" onclick={() => { openGroups = new Set(open ? [...openGroups].filter((t) => t !== g.type) : [...openGroups, g.type]); }}>
                {open ? "Show fewer" : `Show all ${g.items.length}`}</button>
            {/if}
          </span>
        </div>
      {/each}
    </section>
  {/if}

  {#if results.closeItems.length}
    <section>
      <h2 class="section-title">Close in meaning</h2>
      <span class="chips">{#each results.closeItems as r (r.entity.id)}{@render item(r)}{/each}</span>
    </section>
  {/if}

  <div class="columns">
    <div class="main">
      {#if results.passages.length}
        <section>
          <h2 class="section-title">What was said · {results.passages.length}</h2>
          {#each results.passages.slice(0, shownPassages) as r (r.id)}
            {#if why(r)}<p class="why">{why(r)}</p>{/if}
            <Quote passageId={r.id} ranges={r.ranges.slice(0, 2)} radius={110} />
          {/each}
          {#if results.passages.length > shownPassages}
            <button class="btn ghost more" onclick={() => (shownPassages += PASSAGE_STEP)}>Show {Math.min(PASSAGE_STEP, results.passages.length - shownPassages)} more</button>
          {/if}
        </section>
      {/if}
      {#if results.closePassages.length}
        <section>
          <h2 class="section-title">Close in meaning, without the words</h2>
          {#each results.closePassages as r (r.id)}<Quote passageId={r.id} ranges={[]} radius={110} />{/each}
        </section>
      {/if}
    </div>
    {#if results.episodes.length}
      <aside>
        <h2 class="section-title">Episodes · {results.episodes.length}</h2>
        {#each results.episodes.slice(0, shownEpisodes) as r (r.episode.id)}
          <button class="episode" onclick={() => app.readEpisode(r.episode.id)}>
            <small>{r.episode.code}{r.episode.date ? ` · ${r.episode.date}` : ""}</small>
            <b class="serif">{r.episode.title}</b>
            <small>{r.inTitle ? "In the title" : ""}{r.inTitle && r.passages ? " · " : ""}{r.passages ? plural(r.passages, "passage") : ""}</small>
          </button>
        {/each}
        {#if results.episodes.length > shownEpisodes}
          <button class="btn ghost more" onclick={() => (shownEpisodes += EPISODE_STEP)}>Show more episodes</button>
        {/if}
      </aside>
    {/if}
  </div>
</div>

{#snippet item(r: ItemResult)}
  <button class="echip" class:meaning={r.score !== undefined} onclick={() => app.selectEntity(r.entity.id)}
    title={r.score !== undefined ? "Close in meaning" : r.readAs ? `“${r.readAs}” read as ${r.entity.label}` : undefined}>
    <TypeMark type={r.entity.type} />{r.entity.label}<small>{r.passages}</small>
  </button>
{/snippet}

<style>
  .results { max-width: 1120px; margin: 0 auto; padding-block: 10px 32px; display: flex; flex-direction: column; gap: 14px; }
  header { display: flex; flex-direction: column; gap: 2px; }
  h1 { font-size: clamp(24px, 3.2vw, 32px); font-weight: 600; margin: 0; line-height: 1.2; text-wrap: balance; }
  .summary { margin: 2px 0 0; color: var(--ink-2); font-size: 13px; }
  .status { color: var(--ink-3); margin-left: 8px; }
  section .section-title { margin-top: 6px; }
  .best { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 10px 16px; padding: 10px 14px; }
  .what { display: flex; align-items: center; gap: 14px; border: 0; background: none; padding: 0; cursor: pointer; text-align: left; font: inherit; color: inherit; min-width: 0; }
  .what > span { display: flex; flex-direction: column; gap: 1px; min-width: 0; }
  .what:hover b { text-decoration: underline; text-underline-offset: 3px; text-decoration-thickness: 1px; }
  .what img { width: 52px; height: 52px; border-radius: 50%; object-fit: cover; object-position: top; background: var(--surface-2); }
  .what b { font-size: 20px; font-weight: 600; }
  .what small, .episode small { color: var(--ink-3); font-size: 12px; }
  .chain .steps { display: flex; flex-wrap: wrap; align-items: center; gap: 6px 8px; margin: 0 0 8px; }
  .group { display: grid; grid-template-columns: 120px minmax(0, 1fr); gap: 4px 12px; align-items: baseline; padding-block: 6px; border-top: 1px solid var(--line); }
  .glabel { display: inline-flex; align-items: center; gap: 7px; font-size: 13px; color: var(--ink-2); text-transform: capitalize; }
  .echip small { color: var(--ink-3); font-family: var(--font-data); font-size: 11px; }
  .meaning { border-style: dashed; }
  .columns { display: grid; grid-template-columns: minmax(0, 1fr) 300px; gap: 28px; align-items: start; }
  .why { margin: 10px 0 2px; font-size: 12.5px; color: var(--ink-3); }
  /* One line between passages, under each, so the reason above a passage stays with it. */
  .main :global(.quote) { border-top: 0; padding-top: 6px; border-bottom: 1px solid var(--line); }
  .more { margin-top: 10px; }
  .episode { display: flex; flex-direction: column; gap: 1px; width: 100%; text-align: left; border: 0; border-top: 1px solid var(--line); background: none; padding: 8px 0; cursor: pointer; color: inherit; font: inherit; }
  .episode b { font-weight: 600; font-size: 14.5px; line-height: 1.3; }
  .episode:hover b { text-decoration: underline; text-underline-offset: 3px; text-decoration-thickness: 1px; }
  aside { position: sticky; top: 0; }
  @media (max-width: 860px) {
    .columns { grid-template-columns: minmax(0, 1fr); }
    aside { position: static; grid-row: 1; }
    .group { grid-template-columns: minmax(0, 1fr); }
  }
</style>
