<script lang="ts">
  // The name of a topic or an episode as a link. It opens that one in the details, and their Back
  // button leads to what was there before. A real link, so it wraps with a heading and can open in a new tab.
  import { app } from "../lib/state.svelte.ts";

  let { id }: { id: string } = $props();
  const episode = $derived(app.idx!.episode.get(id));
  const name = $derived(episode?.code ?? app.idx!.entity.get(id)?.label ?? id);
  function open(ev: MouseEvent) {
    ev.preventDefault();
    app.follow(() => (episode ? app.selectEpisode(id) : app.selectEntity(id)));
  }
</script>

<a href="#/map?open={id}" title="Open {name}" onclick={open}>{name}</a>

<style>
  a { color: inherit; text-decoration: underline; text-decoration-color: var(--ink-3); text-decoration-thickness: 1px; text-underline-offset: 3px; }
  a:hover { text-decoration-color: currentColor; background: var(--mark); }
</style>
