<script lang="ts">
  // What two topics have in common. The shared topics come first, the ones discussed most with both
  // at the top. Their own passages together are one tab away.
  import { app } from "../lib/state.svelte.ts";
  import { sharedPartners } from "../lib/scene.ts";
  import { typeStyle } from "../lib/entityStyle.ts";
  import { evidenceInView } from "../lib/filters.ts";
  import { plural } from "../lib/format.ts";
  import Quote from "./Quote.svelte";
  import TopicLink from "./TopicLink.svelte";
  import TypeMark from "./TypeMark.svelte";

  let { a, b }: { a: string; b: string } = $props();
  const FIRST_ROWS = 30;
  const idx = $derived(app.idx!);
  const ea = $derived(idx.entity.get(a)!), eb = $derived(idx.entity.get(b)!);
  const shared = $derived(sharedPartners(a, b, app.ranked));
  const types = $derived([...new Set(shared.map((s) => idx.entity.get(s.id)!.type))]);
  let only = $state<string | null>(null);
  let whole = $state(false);
  let tab = $state<"shared" | "said">("shared");
  const listed = $derived(shared.filter((s) => !only || idx.entity.get(s.id)!.type === only));
  const rows = $derived(whole ? listed : listed.slice(0, FIRST_ROWS));
  const shown = $derived(new Set(app.scene?.nodes ?? []));
  // Their passages together, whichever kind of link holds them.
  const said = $derived([...new Set(app.allLinks.filter((ed) => (ed.source === a && ed.target === b) || (ed.source === b && ed.target === a))
    .flatMap((ed) => evidenceInView(ed, app.vp)))].sort());
  const ranges = (p: string) => [...ea.occurrences, ...eb.occurrences].filter((o) => o.passage_id === p);
</script>

<p class="eyebrow">Together</p>
<h2 class="serif"><TopicLink id={a} /> and <TopicLink id={b} /></h2>
<p class="note lead">{shared.length ? `${plural(shared.length, "topic")} connected to both. The ones discussed most with both come first.` : "Nothing inside the current filters is connected to both."}</p>

<div class="tabs" role="tablist">
  <button role="tab" aria-selected={tab === "shared"} onclick={() => (tab = "shared")}>What they share <small>{shared.length}</small></button>
  <button role="tab" aria-selected={tab === "said"} onclick={() => (tab = "said")}>Said together <small>{said.length}</small></button>
</div>

{#if tab === "shared"}
  {#if types.length > 1}
    <div class="chips kinds" role="group" aria-label="Which types to list">
      <button class="chip" aria-pressed={!only} onclick={() => (only = null)}>All</button>
      {#each types as t (t)}
        <button class="chip" aria-pressed={only === t} onclick={() => (only = only === t ? null : t)}><TypeMark type={t} />{typeStyle(t).plural}</button>
      {/each}
    </div>
  {/if}
  {#if listed.length}
    <div class="head"><span></span><span title="Passages with {ea.label}">{ea.label}</span><span title="Passages with {eb.label}">{eb.label}</span></div>
    {#each rows as s (s.id)}
      {@const e = idx.entity.get(s.id)!}
      <button class="row" class:off={!shown.has(s.id)} title={shown.has(s.id) ? "On the map" : "Not on the map yet"} onclick={() => app.follow(() => app.selectEntity(s.id))}>
        <span class="name"><TypeMark type={e.type} />{e.label}</span><span class="n">{s.a}</span><span class="n">{s.b}</span>
      </button>
    {/each}
    {#if listed.length > FIRST_ROWS}<button class="linkish more" onclick={() => (whole = !whole)}>{whole ? `Show the first ${FIRST_ROWS}` : `Show all ${listed.length}`}</button>{/if}
  {/if}
{:else}
  {#each said as p (p)}
    <Quote passageId={p} ranges={ranges(p)} />
  {:else}
    <p class="empty">They are not named in the same passage inside the current filters.</p>
  {/each}
{/if}

<style>
  h2 { font-size: 22px; font-weight: 600; margin: 4px 0 6px; line-height: 1.25; text-wrap: balance; padding-right: 20px; }
  .lead { margin: 0; }
  .tabs { position: sticky; top: -16px; z-index: 2; display: flex; gap: 14px; margin: 12px -16px 10px; padding: 8px 16px 0; background: var(--surface); border-bottom: 1px solid var(--line); }
  .tabs button { border: 0; background: none; padding: 2px 0 7px; cursor: pointer; font-size: 13.5px; font-weight: 500; color: var(--ink-2); border-bottom: 2px solid transparent; margin-bottom: -1px; }
  .tabs button:hover { color: var(--ink); }
  .tabs button[aria-selected="true"] { color: var(--ink); border-bottom-color: var(--accent); }
  .tabs small { font-size: 12px; color: var(--ink-3); font-weight: 400; margin-left: 2px; }
  .kinds { margin-bottom: 8px; }
  .head, .row { display: grid; grid-template-columns: minmax(0, 1fr) 64px 64px; gap: 8px; align-items: baseline; }
  .head { font-size: 11.5px; color: var(--ink-3); padding: 0 6px 3px; }
  .head span { text-align: right; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .row { width: 100%; border: 0; border-top: 1px solid var(--line); background: none; padding: 5px 6px; cursor: pointer; text-align: left; font: inherit; color: inherit; font-size: 13.5px; }
  .row:hover { background: var(--surface-2); }
  .row.off .name { color: var(--ink-3); }
  .name { display: flex; align-items: center; gap: 8px; min-width: 0; }
  .n { font-family: var(--font-data); font-size: 12px; color: var(--ink-2); text-align: right; font-variant-numeric: tabular-nums; }
  .more { display: block; padding: 8px 0; }
</style>
