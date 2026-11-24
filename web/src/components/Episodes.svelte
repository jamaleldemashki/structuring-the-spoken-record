<script lang="ts">
  import { app } from "../lib/state.svelte.ts";
  import { episodeVisible } from "../lib/filters.ts";
  import { formatTime } from "../lib/format.ts";
  import { formatMonth } from "../lib/timeAxis.ts";
  import { player } from "../lib/player.svelte.ts";
  import { follower, runs, timedWords, wordAt } from "../lib/follow.ts";
  import BackToResults from "./BackToResults.svelte";
  import Quote from "./Quote.svelte";
  import TypeMark from "./TypeMark.svelte";

  const ds = $derived(app.ds!);
  const idx = $derived(app.idx!);
  const episodes = $derived([...ds.episodes].sort((a, b) => (a.date ?? "").localeCompare(b.date ?? "")));
  const current = $derived(app.episode ? idx.episode.get(app.episode) ?? null : null);
  const passages = $derived(current ? ds.passages.filter((p) => p.episode_id === current.id).sort((a, b) => a.index - b.index) : []);
  // The items the open episode discusses most, shown in its header.
  const topEntities = $derived(current ? ds.entities.filter((e) => e.episode_ids.includes(current.id) && !e.generic && app.filters!.types.has(e.type))
    .sort((a, b) => b.mention_count - a.mention_count).slice(0, 8) : []);
  // The passage in the middle of the screen, and what is close to it in other episodes. Openings
  // of episodes are left out, since they sound alike in every episode.
  let readingNow = $state<string | null>(null);
  const related = $derived(readingNow ? ds.related.filter((r) => r.source === readingNow && app.vp.has(r.target) && idx.passage.get(r.target)!.index > 0 && idx.passage.get(r.source)!.index > 0)
    .sort((a, b) => b.score - a.score).slice(0, 3) : []);
  const readingStart = $derived(readingNow ? idx.passage.get(readingNow)?.start_s ?? null : null);
  const readingAt = $derived(readingStart === null ? "" : formatTime(readingStart));
  // An archive without this kind of finding has no column for it, and text and times can be missing too.
  const hasRelated = $derived(ds.related.length > 0);
  const hasTimes = $derived(passages.some((p) => p.start_s !== null));
  const hasText = $derived(passages.some((p) => p.text));

  // The strip of episodes: one row, old to new from left to right, however many there are. The
  // slider above it moves along the row, and the row keeps only what is needed to recognise
  // an episode. What it discusses is in the header of the open episode.
  let strip: HTMLDivElement;
  let reader = $state<HTMLDivElement>();
  let stripHeight = $state(0);
  let find = $state("");
  // How far along the row we are, from 0 to 1000.
  let along = $state(0);
  const shown = $derived.by(() => {
    const q = find.trim().toLowerCase();
    return q ? episodes.filter((ep) => `${ep.code} ${ep.title} ${ep.date ?? ""}`.toLowerCase().includes(q)) : episodes;
  });
  const middle = $derived(shown[Math.round((along / 1000) * (shown.length - 1))]);
  const motion = () => (matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth") as ScrollBehavior;
  const moveAlong = (x: number) => { strip.scrollLeft = (x / 1000) * (strip.scrollWidth - strip.clientWidth); };
  const readPosition = () => {
    const room = strip.scrollWidth - strip.clientWidth;
    along = room > 0 ? (strip.scrollLeft / room) * 1000 : 0;
  };

  // Clicking an episode opens it and brings its text up under the strip.
  function open(id: string) {
    app.episode = id;
    app.reading = null;
    requestAnimationFrame(() => reader?.scrollIntoView({ behavior: motion(), block: "start" }));
  }

  // Keep the open episode's card in view of the row, wherever it was opened from. Only the row
  // moves here, not the page.
  $effect(() => {
    const id = app.episode;
    void shown;
    requestAnimationFrame(() => {
      const card = strip?.querySelector<HTMLElement>(`[data-episode="${CSS.escape(id ?? "")}"]`);
      if (!card) return;
      const left = card.offsetLeft - strip.offsetLeft, right = left + card.offsetWidth;
      if (left < strip.scrollLeft || right > strip.scrollLeft + strip.clientWidth)
        strip.scrollTo({ left: left - (strip.clientWidth - card.offsetWidth) / 2, behavior: motion() });
      readPosition();
    });
  });

  // A thin band across the middle of the screen: whichever passage crosses it is being read. Until
  // one does, that is the first passage of the episode.
  $effect(() => {
    readingNow = passages[0]?.id ?? null;
    const seen = new IntersectionObserver((entries) => {
      for (const en of entries) if (en.isIntersecting) readingNow = en.target.id.slice(4);
    }, { rootMargin: "-45% 0px -50% 0px" });
    for (const el of document.querySelectorAll(".read .psg")) seen.observe(el);
    return () => seen.disconnect();
  });

  // The passage being listened to is the one being read, and so it is marked too.
  $effect(() => {
    const id = player.passageId;
    if (id && passages.some((p) => p.id === id)) readingNow = id;
  });

  // Arriving from a quote: bring that passage into view and mark it.
  $effect(() => {
    const id = app.reading;
    if (id && current) requestAnimationFrame(() => document.getElementById(`psg-${id}`)?.scrollIntoView({ block: "center" }));
  });

  // Mentions per passage of the open episode, collected once instead of once per passage.
  const mentionsIn = $derived.by(() => {
    const out = new Map<string, { start: number; end: number; id: string }[]>(passages.map((p) => [p.id, []]));
    for (const e of ds.entities) {
      if (!app.filters!.types.has(e.type)) continue;
      for (const o of e.occurrences) out.get(o.passage_id)?.push({ start: o.start, end: o.end, id: e.id });
    }
    for (const list of out.values()) list.sort((a, b) => a.start - b.start);
    return out;
  });

  // Following the audio: the words of the passage being played, with their estimated times,
  // and the word spoken now. Playing stops at the end of the passage, so only that passage
  // is followed.
  const words = $derived(new Map(passages.map((p) => [p.id, timedWords(p)])));
  const spoken = $derived.by(() => {
    const pid = player.passageId;
    if (!pid || !words.has(pid)) return null;
    const w = wordAt(words.get(pid)!, player.now);
    return w >= 0 ? { pid, w } : null;
  });

  const follow = follower();
  $effect(() => { if (spoken) follow(document.querySelector(".read .now")); });

  // Clicking a word in the passage being played jumps the audio there.
  function seekWord(pid: string, w: number) {
    const word = words.get(pid)?.[w];
    if (word) player.seek(word.t0);
  }
</script>

<div class="episodes" style:--strip-h="{stripHeight}px">
<nav class="strip" aria-label="Episodes" bind:clientHeight={stripHeight}>
  <div class="bar">
    <BackToResults />
    <input class="find" type="search" bind:value={find} placeholder="Find an episode by number, title or date" aria-label="Find an episode" />
    <span class="note count">{shown.length === episodes.length ? `${episodes.length} episodes` : `${shown.length} of ${episodes.length}`}</span>
    {#if shown.length > 1}
      <span class="scrub">
        <span class="note">{shown[0].date ? formatMonth(shown[0].date.slice(0, 7)) : ""}</span>
        <input type="range" min="0" max="1000" value={along} oninput={(e) => moveAlong(Number(e.currentTarget.value))}
          aria-label="Move along the episodes" aria-valuetext={middle?.code} />
        <span class="note">{shown.at(-1)!.date ? formatMonth(shown.at(-1)!.date!.slice(0, 7)) : ""}</span>
      </span>
    {/if}
  </div>
  <div class="row" bind:this={strip} onscroll={readPosition}>
    {#each shown as ep (ep.id)}
      <button class="card ep" data-episode={ep.id} aria-pressed={app.episode === ep.id} class:faded={!episodeVisible(ep, app.filters!)} title={ep.title} onclick={() => open(ep.id)}>
        <span class="code">{ep.code}{ep.date ? ` · ${ep.date}` : ""}</span>
        <b class="serif">{ep.title}</b>
      </button>
    {:else}
      <p class="empty">No episode matches “{find.trim()}”.</p>
    {/each}
  </div>
</nav>

{#if current}
  <div class="reader" class:alone={!hasRelated} bind:this={reader}>
    <article class="card read">
      <span class="note">{[current.code, current.date, current.era].filter(Boolean).join(" · ")}</span>
      <h2 class="serif">{current.title}</h2>
      <div class="acts">
        {#if current.audio_url}<button class="btn" onclick={() => app.listen(current.id)}>▶ Listen from the start</button>{/if}
        <button class="btn ghost" onclick={() => app.showOnly([current.id])}>Show in the map</button>
      </div>
      {#if topEntities.length}
        <p class="about"><span class="note">Discusses most</span>
          <span class="chips">{#each topEntities as e (e.id)}<button class="echip" onclick={() => app.selectEntity(e.id)}><TypeMark type={e.type} />{e.label}</button>{/each}</span></p>
      {/if}
      {#if !hasText}<p class="empty">The transcript text is not included in this archive.</p>{/if}
      {#each hasText ? passages : [] as p (p.id)}
        <!-- A click marks a passage too, which also reaches the last ones, that never get to the middle of the screen. -->
        <div class="psg" id="psg-{p.id}" class:reading={app.reading === p.id} class:current={readingNow === p.id} class:plain={!hasTimes && !current.audio_url} role="presentation" onclick={() => (readingNow = p.id)}>
          {#if hasTimes || current.audio_url}
            <div class="tc">
              {#if current.audio_url}<button class="play" onclick={() => app.playPassage(p.id)} aria-label={p.start_s === null ? "Listen" : `Listen from ${formatTime(p.start_s)}`} title="Listen from here">▶{p.start_s === null ? "" : ` ${formatTime(p.start_s)}`}</button>
              {:else if p.start_s !== null}<span class="time">{formatTime(p.start_s)}</span>{/if}
            </div>
          {/if}
          {#if p.text}
            {@const now = spoken?.pid === p.id ? spoken.w : -1}
            <!-- Words are only cut into pieces in the passage being played, the others stay plain text.
                 A name is a real link, so it wraps with the text and punctuation stays next to it. -->
            <p class:following={now >= 0}>{#each runs(p.text, mentionsIn.get(p.id) ?? [], now >= 0 ? words.get(p.id)! : []) as r}{#if r.id}<a class="inline" href="#/episodes/{current.id}?open={r.id}" onclick={(ev) => { ev.preventDefault(); app.selectEntity(r.id!); }}>{#each r.pieces as x}{#if x.w >= 0}<span class:now={x.w === now} class:ahead={x.w > now}>{x.text}</span>{:else}{x.text}{/if}{/each}</a>{:else}{#each r.pieces as x}{#if x.w >= 0}<span class="word" class:now={x.w === now} class:ahead={x.w > now} role="presentation" onclick={() => seekWord(p.id, x.w)}>{x.text}</span>{:else}{x.text}{/if}{/each}{/if}{/each}</p>
          {/if}
        </div>
      {/each}
    </article>
    {#if hasRelated}
    <aside class="side">
      <h3 class="section-title">Related to what you are reading{#if readingAt} <button class="at" title="Go to this passage" onclick={() => document.getElementById(`psg-${readingNow}`)?.scrollIntoView({ block: "center", behavior: motion() })}>{readingAt}</button>{/if}</h3>
      {#each related as r (r.source + r.target)}
        <Quote passageId={r.target} ranges={[]} radius={90} />
      {:else}
        <p class="empty">Nothing close to this part in other episodes.</p>
      {/each}
    </aside>
    {/if}
  </div>
{/if}
</div>

<style>
  /* The strip stays at the top while reading, so another episode is always one click away. */
  .strip { position: sticky; top: -8px; z-index: 6; background: var(--bg); padding-block: 2px 8px; display: flex; flex-direction: column; gap: 6px; }
  .bar { display: flex; flex-wrap: wrap; align-items: center; gap: 4px 12px; }
  .find { flex: 0 1 300px; min-width: 0; padding: 4px 12px; border-radius: 999px; border: 1px solid var(--line); background: var(--surface); font-size: 13px; }
  .count { white-space: nowrap; }
  .scrub { flex: 1 1 280px; display: flex; align-items: center; gap: 10px; min-width: 0; }
  .scrub input { flex: 1; min-width: 0; accent-color: var(--accent); }
  .scrub .note { white-space: nowrap; font-family: var(--font-data); font-size: 11.5px; }
  .row { display: flex; gap: 8px; overflow-x: auto; scroll-snap-type: x proximity; scrollbar-width: none; padding: 2px; }
  .row::-webkit-scrollbar { display: none; }
  .ep { flex: none; width: 212px; padding: 6px 10px; cursor: pointer; text-align: left; display: flex; flex-direction: column; gap: 2px; scroll-snap-align: start; color: inherit; font: inherit; }
  .ep:hover { border-color: var(--ink-3); }
  .ep[aria-pressed="true"] { border-color: var(--accent); box-shadow: 0 0 0 1px var(--accent); }
  .ep.faded { opacity: 0.45; }
  .code { font-family: var(--font-data); font-size: 11.5px; color: var(--ink-3); white-space: nowrap; }
  .ep b { font-size: 13.5px; font-weight: 600; line-height: 1.25; display: -webkit-box; -webkit-line-clamp: 2; line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; min-height: 2.5em; }
  .reader { margin-top: 6px; display: grid; grid-template-columns: minmax(0, 1fr) 300px; gap: 20px; align-items: start; scroll-margin-top: calc(var(--strip-h) - 4px); }
  .reader.alone { grid-template-columns: minmax(0, 1fr); }
  .read { padding: 18px 22px; min-width: 0; }
  .time { font-family: var(--font-data); font-size: 12px; color: var(--ink-3); }
  .psg.plain { grid-template-columns: minmax(0, 1fr); }
  h2 { font-size: 24px; margin: 2px 0 8px; font-weight: 600; text-wrap: balance; }
  .acts { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 10px; }
  .about { display: flex; flex-wrap: wrap; align-items: center; gap: 6px 10px; margin: 2px 0 12px; }
  /* A passage reaches from edge to edge of the card, so its mark does too. */
  .psg { border-top: 1px solid var(--line); margin-inline: -22px; padding: 14px 22px; display: grid; grid-template-columns: 84px minmax(0, 1fr); gap: 10px; transition: background-color 0.2s; }
  /* The passage the side column is about. The same bar stands on the side column, which ties the two together. */
  .psg.current { background: color-mix(in srgb, var(--surface-2) 55%, transparent); box-shadow: inset 3px 0 0 var(--accent); }
  .tc { padding-top: 2px; }
  .psg p { font-family: var(--font-read); font-size: 16.5px; line-height: 1.65; margin: 0; }
  .inline { color: inherit; cursor: pointer; text-decoration: underline; text-decoration-color: var(--ink-3); text-underline-offset: 3px; }
  .inline:hover { background: var(--mark); }
  .side { position: sticky; top: calc(var(--strip-h) + 2px); max-height: calc(100vh - var(--strip-h) - 80px); overflow: auto; }
  .side { border-left: 3px solid var(--accent); padding-left: 14px; }
  .side .section-title { margin-top: 4px; }
  .at { border: 1px solid var(--line); background: var(--surface); border-radius: 999px; padding: 0 7px; margin-left: 2px; cursor: pointer; font-family: var(--font-data); font-size: 11.5px; color: var(--ink); letter-spacing: 0; }
  .at:hover { border-color: var(--ink-3); }
  @media (max-width: 1080px) { .reader { grid-template-columns: minmax(0, 1fr); } .side { display: none; } }
  /* On a low window the strip would take most of the screen, so it scrolls away. */
  @media (max-height: 620px) { .strip { position: static; } .reader { scroll-margin-top: 0; } .side { top: 0; } }
  @media (max-width: 520px) { .psg { grid-template-columns: minmax(0, 1fr); gap: 6px; } .read { padding: 12px 14px; } .psg { margin-inline: -14px; padding-inline: 14px; } }
  /* The passage being played: words still to come are a little lighter, the spoken one is marked. */
  .following .ahead { color: var(--ink-2); }
  .following .now { background: var(--accent); color: var(--accent-ink); border-radius: 3px; box-shadow: 0 0 0 2px var(--accent); }
  .following .word { cursor: pointer; }
  .following .word:hover { text-decoration: underline; text-decoration-color: var(--ink-3); }
  .psg.reading, .psg.reading.current { background: color-mix(in srgb, var(--mark) 45%, transparent); border-radius: 8px; box-shadow: 0 0 0 6px color-mix(in srgb, var(--mark) 45%, transparent); }
</style>
