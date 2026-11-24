<script lang="ts">
  import { untrack } from "svelte";
  import { app } from "../lib/state.svelte.ts";
  import TypeMark from "./TypeMark.svelte";
  import { formatTime, highlight, lifeLabel } from "../lib/format.ts";
  import { searchArchive, type PassageHit } from "../lib/search.ts";
  import type { Entity, Episode } from "../lib/types.ts";

  // The home page shows a large search field, the header a normal one. With onpick the field only
  // looks for topics and hands the chosen one over (adding to an exploration, finding a connection).
  let { large = false, compact = false, placeholder, onpick }:
    { large?: boolean; compact?: boolean; placeholder?: string; onpick?: (e: Entity) => void } = $props();

  let query = $state("");
  let active = $state(0);
  // The list is open while the field is in use. The text stays when it closes.
  let open = $state(false);
  let input: HTMLInputElement;
  const mac = typeof navigator !== "undefined" && /Mac|iPhone|iPad/.test(navigator.platform);
  const shortcut = mac ? "⌘K" : "Ctrl K";

  // On the results page the field keeps what was searched for, so it can be changed.
  $effect(() => {
    if (onpick || large) return;
    if (app.view === "results" && app.query) query = untrack(() => (query === "" || !open ? app.query!.text : query));
    // Leaving the results, the text of that search is no longer what is on screen.
    else if (untrack(() => query === app.query?.text)) query = "";
  });

  const groups = $derived((app.ds && app.idx ? searchArchive(app.ds, app.idx, query) : []).filter((g) => !onpick || g.kind === "entities"));
  // Enter opens the results page, for a word and for a question alike. Choosing a row opens that row.
  const everything = $derived(!onpick && query.trim() ? { text: query.trim() } : null);
  type Row = { kind: "all"; item: { text: string } } | { kind: "entity"; item: Entity } | { kind: "episode"; item: Episode } | { kind: "passage"; item: PassageHit };
  // All results in display order, so the arrow keys move through every group.
  const rows = $derived([...(everything ? [{ kind: "all", item: everything } as Row] : []), ...groups.flatMap((g): Row[] =>
    g.kind === "entities" ? g.items.map((item) => ({ kind: "entity", item }))
    : g.kind === "episodes" ? g.items.map((item) => ({ kind: "episode", item }))
    : g.items.map((item) => ({ kind: "passage", item })))]);
  const rowIndex = (item: object) => rows.findIndex((r) => r.item === item);

  function done() {
    query = "";
    open = false;
    input.blur();
  }

  function choose(row: Row) {
    if (row.kind === "all") {
      void app.search(row.item.text);
      open = false;
      input.blur();
      return;
    }
    if (row.kind === "entity") (onpick ?? ((e: Entity) => app.openEntity(e.id)))(row.item);
    else if (row.kind === "episode") app.readEpisode(row.item.id);
    else app.readPassage(row.item.passage.id);
    done();
  }

  function onkey(ev: KeyboardEvent) {
    if (ev.key === "ArrowDown" || ev.key === "ArrowUp") {
      ev.preventDefault();
      active = (active + (ev.key === "ArrowDown" ? 1 : -1) + rows.length) % Math.max(1, rows.length);
    } else if (ev.key === "Enter" && rows[active]) choose(rows[active]);
    else if (ev.key === "Escape") { if (open) open = false; else query = ""; }
  }

  const episodes = (e: Entity) => `${e.episode_ids.length} ${e.episode_ids.length === 1 ? "episode" : "episodes"}`;
</script>

<svelte:window onkeydown={(e) => { if (!onpick && (e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") { e.preventDefault(); input.focus(); } }} />

<div class="search" class:large class:compact onfocusout={(e) => { if (!e.currentTarget.contains(e.relatedTarget as Node)) open = false; }}>
  {#if !compact}<svg width={large ? 20 : 15} height={large ? 20 : 15} viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><circle cx="11" cy="11" r="7" /><path d="m20 20-3.5-3.5" /></svg>{/if}
  <input bind:this={input} bind:value={query} oninput={() => { active = 0; open = true; }} onfocus={() => (open = true)} onkeydown={onkey} type="search"
    placeholder={placeholder ?? app.ds!.site.search_placeholder} aria-label={onpick ? placeholder : "Search the archive"} autocomplete="off" />
  {#if !onpick && !query}<kbd>{shortcut}</kbd>{/if}
  {#if open && query.trim()}
    <div class="results" role="listbox" aria-label="Search results">
      {#each groups as g (g.label)}
        <div class="group">
          <span class="glabel">{g.label}</span>
          {#if g.kind === "entities"}
            {#each g.items as e (e.id)}
              <button role="option" aria-selected={rowIndex(e) === active} class:on={rowIndex(e) === active}
                onmouseenter={() => (active = rowIndex(e))} onclick={() => choose({ kind: "entity", item: e })}>
                {#if e.image}<img src={e.image.thumb} alt="" loading="lazy" />
                {:else}<span class="mark"><TypeMark type={e.type} size={14} /></span>{/if}
                <span class="text"><b>{e.label}</b><small>{[lifeLabel(e.life), episodes(e)].filter(Boolean).join(" · ")}</small></span>
              </button>
            {/each}
          {:else if g.kind === "episodes"}
            {#each g.items as ep (ep.id)}
              <button role="option" aria-selected={rowIndex(ep) === active} class:on={rowIndex(ep) === active}
                onmouseenter={() => (active = rowIndex(ep))} onclick={() => choose({ kind: "episode", item: ep })}>
                <span class="code">{ep.code}</span>
                <span class="text"><b>{ep.title}</b><small>{ep.date ?? ""}</small></span>
              </button>
            {/each}
          {:else}
            <!-- ▶ opens the passage in its episode and plays it there, so the words can be followed. -->
            {#each g.items as hit (hit.passage.id)}
              <div class="line" class:on={rowIndex(hit) === active}>
                <button role="option" aria-selected={rowIndex(hit) === active} class="quote"
                  onmouseenter={() => (active = rowIndex(hit))} onclick={() => choose({ kind: "passage", item: hit })}>
                  <span class="text">
                    <q>{#each highlight(hit.passage.text!, [hit], 60) as s}{#if s.mark}<mark>{s.text}</mark>{:else}{s.text}{/if}{/each}</q>
                    <small>{hit.episode.code}{hit.passage.start_s === null ? "" : ` · ${formatTime(hit.passage.start_s)}`}</small>
                  </span>
                </button>
                {#if hit.episode.audio_url}<button class="play" aria-label="Listen" onclick={() => { app.readPassage(hit.passage.id); app.playPassage(hit.passage.id); done(); }}>▶</button>{/if}
              </div>
            {/each}
          {/if}
        </div>
      {/each}
      {#if everything}
        <button role="option" aria-selected={rowIndex(everything) === active} class="all" class:on={rowIndex(everything) === active}
          onmouseenter={() => (active = rowIndex(everything))} onclick={() => choose({ kind: "all", item: everything })}>
          <span>All results for <b>{everything.text}</b></span><kbd class="enter">Enter</kbd>
        </button>
      {/if}
    </div>
  {/if}
</div>

<style>
  .search { position: relative; flex: 1 1 260px; max-width: 460px; }
  input { width: 100%; padding: 8px 40px 8px 34px; border-radius: 999px; border: 1px solid var(--line); background: var(--surface); }
  .search > svg { position: absolute; left: 12px; top: 19px; transform: translateY(-50%); color: var(--ink-3); }
  kbd { position: absolute; right: 12px; top: 19px; transform: translateY(-50%); font-family: var(--font-data); font-size: 11px; color: var(--ink-3); }
  .large { max-width: 640px; width: 100%; flex: none; }
  .compact { flex: 1 1 auto; max-width: none; }
  .compact input { padding: 4px 12px; font-size: 12.5px; }
  .compact .results { left: auto; right: 0; }
  .large input { padding: 15px 54px 15px 50px; font-size: 17px; box-shadow: var(--shadow); border-color: transparent; }
  .large > svg { left: 18px; top: 28px; }
  .large kbd { right: 18px; top: 28px; }
  .results { position: absolute; z-index: 30; top: calc(100% + 6px); left: 0; right: 0; background: var(--surface);
    border: 1px solid var(--line); border-radius: var(--radius); box-shadow: var(--shadow); padding: 6px; max-height: 70vh; overflow: auto;
    min-width: min(420px, 90vw); }
  .group + .group { border-top: 1px solid var(--line); margin-top: 4px; padding-top: 4px; }
  .glabel { display: flex; align-items: center; gap: 6px; padding: 6px 8px 2px; font-size: 10.5px; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.08em; color: var(--ink-3); }
  button[role="option"] { display: flex; width: 100%; align-items: center; gap: 10px; padding: 6px 8px; border: 0; background: none;
    border-radius: 6px; text-align: left; cursor: pointer; color: inherit; font: inherit; }
  button.on, .line.on { background: var(--surface-2); }
  img, .mark { width: 28px; height: 28px; flex: none; }
  img { border-radius: 50%; object-fit: cover; background: var(--surface-2); }
  .mark { display: grid; place-items: center; }
  .text { display: flex; flex-direction: column; min-width: 0; }
  .text b { font-weight: 500; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  small { color: var(--ink-3); font-size: 12px; }
  .code { flex: none; width: 64px; font-family: var(--font-data); font-size: 11px; color: var(--ink-3); }
  .line { display: flex; align-items: center; border-radius: 6px; }
  .quote q { font-family: var(--font-read); font-size: 13.5px; line-height: 1.45; color: var(--ink-2); }
  mark { background: var(--mark); color: var(--ink); border-radius: 2px; }
  .play { flex: none; border: 1px solid var(--line); background: var(--surface); border-radius: 999px; width: 28px; height: 28px;
    margin-right: 6px; cursor: pointer; font-size: 10px; color: var(--ink-2); }
  .play:hover { color: var(--ink); border-color: var(--ink-3); }
  .results .all { position: sticky; bottom: -6px; display: flex; align-items: center; justify-content: space-between; gap: 10px; width: 100%; margin-top: 6px; padding: 9px 10px;
    border: 0; border-top: 1px solid var(--line); background: var(--surface); color: inherit; font: inherit; cursor: pointer; text-align: left; }
  .results .all.on { background: var(--surface-2); }
  .results .all b { font-weight: 600; }
  .enter { position: static; transform: none; border: 1px solid var(--line); border-radius: 4px; padding: 0 5px; }
</style>
