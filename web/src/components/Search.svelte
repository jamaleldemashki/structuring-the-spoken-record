<script lang="ts">
  import { app } from "../lib/state.svelte.ts";
  import { typeStyle, typeVar } from "../lib/entityStyle.ts";
  import type { Entity } from "../lib/types.ts";

  let query = $state("");
  let active = $state(0);
  let input: HTMLInputElement;

  const hits = $derived.by((): Entity[] => {
    const q = query.trim().toLowerCase();
    if (!q || !app.ds) return [];
    return app.ds.entities
      .filter((e) => e.label.toLowerCase().includes(q) || e.aliases.some((a) => a.toLowerCase().includes(q)))
      .sort((a, b) => Number(b.label.toLowerCase().startsWith(q)) - Number(a.label.toLowerCase().startsWith(q)) || b.mention_count - a.mention_count)
      .slice(0, 8);
  });

  function choose(e: Entity) {
    app.openEntity(e.id);
    query = "";
    input.blur();
  }

  function onkey(ev: KeyboardEvent) {
    if (ev.key === "ArrowDown" || ev.key === "ArrowUp") {
      ev.preventDefault();
      active = (active + (ev.key === "ArrowDown" ? 1 : -1) + hits.length) % Math.max(1, hits.length);
    } else if (ev.key === "Enter" && hits[active]) choose(hits[active]);
    else if (ev.key === "Escape") query = "";
  }
</script>

<svelte:window onkeydown={(e) => { if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") { e.preventDefault(); input.focus(); } }} />

<div class="search">
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><circle cx="11" cy="11" r="7" /><path d="m20 20-3.5-3.5" /></svg>
  <input bind:this={input} bind:value={query} oninput={() => (active = 0)} onkeydown={onkey} type="search"
    placeholder={app.ds!.site.search_placeholder} aria-label="Search the archive" autocomplete="off" />
  <kbd>⌘K</kbd>
  {#if query.trim()}
    <div class="results" role="listbox">
      {#each hits as e, i (e.id)}
        <button role="option" aria-selected={i === active} class:on={i === active} onclick={() => choose(e)}>
          <span class="dot" style:background={typeVar(e.type)}></span>{e.label}<small>{typeStyle(e.type).label} · {e.mention_count}</small>
        </button>
      {:else}
        <p class="empty">No person, concept or work matches “{query.trim()}”.</p>
      {/each}
    </div>
  {/if}
</div>

<style>
  .search { position: relative; flex: 1 1 260px; max-width: 460px; }
  input { width: 100%; padding: 8px 40px 8px 34px; border-radius: 999px; border: 1px solid var(--line); background: var(--surface); }
  svg { position: absolute; left: 12px; top: 50%; transform: translateY(-50%); color: var(--ink-3); }
  kbd { position: absolute; right: 12px; top: 50%; transform: translateY(-50%); font-family: var(--font-data); font-size: 11px; color: var(--ink-3); }
  .results { position: absolute; z-index: 30; top: calc(100% + 6px); left: 0; right: 0; background: var(--surface);
    border: 1px solid var(--line); border-radius: var(--radius); box-shadow: var(--shadow); padding: 6px; }
  .results button { display: flex; width: 100%; align-items: center; gap: 10px; padding: 7px 8px; border: 0; background: none; border-radius: 6px; text-align: left; cursor: pointer; }
  .results button.on, .results button:hover { background: var(--surface-2); }
  small { margin-left: auto; color: var(--ink-3); font-family: var(--font-data); font-size: 11px; }
  .empty { margin: 6px 8px; }
</style>
