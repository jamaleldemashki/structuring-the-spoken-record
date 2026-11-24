<script lang="ts">
  // A line between a topic and an episode: the passages of that episode that name the topic.
  import { app } from "../lib/state.svelte.ts";
  import { plural } from "../lib/format.ts";
  import type { EpisodeLink } from "../lib/episodes.ts";
  import Quote from "./Quote.svelte";
  import TopicLink from "./TopicLink.svelte";

  let { link }: { link: EpisodeLink } = $props();
  const item = $derived(app.idx!.entity.get(link.source)!);
  const ep = $derived(app.idx!.episode.get(link.target)!);
  const ranges = (p: string) => item.occurrences.filter((o) => o.passage_id === p);
</script>

<p class="eyebrow">Named in an episode</p>
<h2 class="serif"><TopicLink id={item.id} /> <span class="verb">in</span> <TopicLink id={ep.id} /></h2>
<p class="note">{ep.title} · {plural(link.count, "passage")}</p>
<h3 class="section-title">Where it is named</h3>
{#each link.evidence as p (p)}<Quote passageId={p} ranges={ranges(p)} showSource={false} />{/each}

<style>
  h2 { font-size: 22px; font-weight: 600; margin: 4px 0 4px; text-wrap: balance; }
  .verb { color: var(--ink-3); font-weight: 400; font-style: italic; }
  .note { margin: 0; }
</style>
