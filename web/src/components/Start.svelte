<script lang="ts">
  // The first page: a search field, the three ways into the archive, and what it holds most of.
  // The whole map stays one click away instead of being the first thing shown.
  import { app } from "../lib/state.svelte.ts";
  import { allTypes, typeStyle, typeVar } from "../lib/entityStyle.ts";
  import { lifeLabel, plural } from "../lib/format.ts";
  import { mostDiscussed, suggestions, type Suggestion } from "../lib/search.ts";
  import { formatMonth } from "../lib/timeAxis.ts";
  import { monthRange } from "../lib/filters.ts";
  import Search from "./Search.svelte";
  import TypeMark from "./TypeMark.svelte";

  const ds = $derived(app.ds!);
  const lead = $derived(ds.site.lifespan_type ? typeStyle(ds.site.lifespan_type) : null);
  const people = $derived(lead ? mostDiscussed(ds, lead.type, 8) : []);
  // The first default type that is not people, usually concepts.
  const ideaType = $derived(allTypes().find((t) => t.type !== lead?.type && t.visibleByDefault) ?? null);
  const ideas = $derived(ideaType ? mostDiscussed(ds, ideaType.type, 10) : []);
  // Every other default type gets a short list further down.
  const otherTypes = $derived(allTypes().filter((t) => t.visibleByDefault && t.type !== lead?.type && t.type !== ideaType?.type));
  const examples = $derived(suggestions(ds, app.edges));
  const [first, last] = $derived(monthRange(ds));
  const datedPeople = $derived(lead ? ds.entities.filter((e) => e.type === lead.type && e.life).length : 0);
  const eras = $derived([...new Set(ds.episodes.map((e) => e.era ?? "Unknown"))].sort());

  // Each question starts from the default filters, so an earlier window can't hide its answer.
  function ask(s: Suggestion) {
    app.resetFilters();
    app.path = null;
    if (s.kind === "connection") app.connect(s.path.nodes[0], s.path.nodes.at(-1)!);
    else if (s.kind === "period") {
      // The whole map with a lifespan window draws exactly the people of that period.
      app.setFilters({ lived: [s.preset.start, s.preset.end], livedMatch: "within" });
      app.wholeMap();
    } else app.openEntity(s.entity.id);
  }
</script>

<div class="start">
  <section class="hero">
    <h1 class="serif">Who or what do you want to explore?</h1>
    <p class="lede">Search everything said in {plural(ds.summary.episodes, "episode")} of {ds.site.title}, then read it, listen to it and follow how it connects.</p>
    <Search large />
  </section>

  <nav class="ways" aria-label="Ways into the archive">
    <button class="way card" onclick={() => app.wholeMap()}>
      <b>Map</b><span>How {app.topicCount.toLocaleString("en")} topics connect</span>
    </button>
    <button class="way card" onclick={() => (app.view = "episodes")}>
      <b>Episodes</b><span>Read and listen to all {ds.summary.episodes}, {formatMonth(first)} to {formatMonth(last)}</span>
    </button>
    {#if datedPeople}
      <button class="way card" onclick={() => (app.view = "timeline")}>
        <b>Timeline</b><span>When the {datedPeople} {lead!.plural} with known dates lived</span>
      </button>
    {/if}
  </nav>

  {#if people.length}
    <section>
      <h2 class="section-title">Most discussed {lead!.plural}</h2>
      <div class="people">
        {#each people as e (e.id)}
          <button class="person" onclick={() => app.openEntity(e.id)}>
            {#if e.image}<img src={e.image.thumb} alt="" loading="lazy" />
            {:else}<span class="initial" style:background={typeVar(e.type)}>{e.label[0]}</span>{/if}
            <b>{e.label}</b>
            <small>{lifeLabel(e.life) ?? plural(e.episode_ids.length, "episode")}</small>
          </button>
        {/each}
      </div>
    </section>
  {/if}

  {#if ideas.length}
    <section>
      <h2 class="section-title">Most discussed {ideaType!.plural}</h2>
      <div class="chips">
        {#each ideas as e (e.id)}
          <button class="chip big" onclick={() => app.openEntity(e.id)}><TypeMark type={e.type} />{e.label}</button>
        {/each}
      </div>
    </section>
  {/if}

  {#if examples.length}
    <section>
      <h2 class="section-title">Or start from a question</h2>
      <div class="questions">
        {#each examples as s (s.text)}
          <button class="serif" onclick={() => ask(s)}>{s.text}</button>
        {/each}
      </div>
    </section>
  {/if}

  {#if otherTypes.length}
    <div class="lists">
      {#each otherTypes as t (t.type)}
        {@const rows = mostDiscussed(ds, t.type, 6)}
        {@const most = Math.max(1, ...rows.map((e) => e.mention_count))}
        <section>
          <h2 class="section-title">Most discussed {t.plural}</h2>
          <div class="toplist">
            {#each rows as e (e.id)}
              <button title={e.label} onclick={() => app.openEntity(e.id)}>{e.label}</button>
              <span class="track"><i style:width="{(e.mention_count / most) * 100}%" style:background={typeVar(t.type)}></i></span>
            {/each}
          </div>
        </section>
      {/each}
    </div>
  {/if}

  {#if eras.length > 1 && ds.site.era_label}
    <section>
      <h2 class="section-title">Episodes per {ds.site.era_label.toLowerCase()}</h2>
      <div class="toplist">
        {#each eras as era}
          {@const n = ds.episodes.filter((e) => (e.era ?? "Unknown") === era).length}
          <span>{era}</span><span class="track"><i style:width="{(n / ds.episodes.length) * 100}%" style:background="var(--accent)"></i></span>
        {/each}
      </div>
    </section>
  {/if}
</div>

<style>
  .start { max-width: 920px; margin: 0 auto; padding-block: clamp(10px, 4vh, 44px) 32px; display: flex; flex-direction: column; gap: 24px; }
  .hero { display: flex; flex-direction: column; align-items: center; text-align: center; gap: 10px; }
  h1 { font-size: clamp(26px, 4vw, 36px); font-weight: 600; margin: 0; letter-spacing: -0.01em; text-wrap: balance; }
  .lede { margin: 0 0 10px; color: var(--ink-2); max-width: 58ch; text-wrap: balance; }
  section .section-title { margin-top: 0; }
  .ways { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 10px; }
  .way { display: flex; flex-direction: column; gap: 2px; padding: 12px 14px; cursor: pointer; text-align: left; color: inherit; font: inherit; transition: border-color 0.15s; }
  .way:hover { border-color: var(--ink-3); }
  .way b { font-family: var(--font-read); font-size: 17px; font-weight: 600; }
  .way span { font-size: 13px; color: var(--ink-2); line-height: 1.35; }
  .people { display: grid; grid-template-columns: repeat(auto-fill, minmax(100px, 1fr)); gap: 10px; }
  .person { display: flex; flex-direction: column; align-items: center; gap: 6px; padding: 12px 8px 10px; border: 1px solid var(--line);
    background: var(--surface); border-radius: var(--radius); cursor: pointer; color: inherit; font: inherit; text-align: center; transition: border-color 0.15s; }
  .person:hover { border-color: var(--ink-3); }
  .person img, .initial { width: 60px; height: 60px; border-radius: 50%; }
  .person img { object-fit: cover; object-position: top; background: var(--surface-2); }
  .initial { display: grid; place-items: center; color: #fff; font-size: 24px; font-weight: 600; }
  .person b { font-weight: 600; font-size: 13.5px; line-height: 1.25; }
  .person small { font-size: 11.5px; color: var(--ink-3); }
  .big { padding: 5px 12px; font-size: 13.5px; }
  .questions { display: flex; flex-direction: column; }
  .questions button { text-align: left; border: 0; border-top: 1px solid var(--line); background: none; padding: 9px 2px; cursor: pointer; color: inherit; font-size: 16px; }
  .questions button:hover { text-decoration: underline; text-underline-offset: 3px; text-decoration-thickness: 1px; }
  .lists { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 24px; }
  .toplist { display: grid; grid-template-columns: minmax(0, 1fr) 100px; gap: 6px 10px; align-items: center; font-size: 13.5px; }
  .toplist button { border: 0; background: none; text-align: left; padding: 0; cursor: pointer; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .toplist button:hover { text-decoration: underline; text-underline-offset: 3px; }
  .track { height: 6px; background: var(--surface-2); border-radius: 3px; overflow: hidden; }
  .track i { display: block; height: 100%; border-radius: 0 3px 3px 0; }
</style>
