<script lang="ts">
  import { app } from "../lib/state.svelte.ts";
  import EntityPanel from "./EntityPanel.svelte";
  import EdgePanel from "./EdgePanel.svelte";
  import PathPanel from "./PathPanel.svelte";
  import AnswerPanel from "./AnswerPanel.svelte";
  import EpisodePanel from "./EpisodePanel.svelte";
  import MentionPanel from "./MentionPanel.svelte";
  import PairPanel from "./PairPanel.svelte";
  import type { Selection } from "../lib/state.svelte.ts";

  // What the Back button of the details says.
  function trailName(s: Selection): string {
    if (s.kind === "entity") return app.idx!.entity.get(s.id)?.label ?? "the topic";
    if (s.kind === "episode") return app.idx!.episode.get(s.id)?.code ?? "the episode";
    if (s.kind === "pair") return `${app.idx!.entity.get(s.a)?.label} and ${app.idx!.entity.get(s.b)?.label}`;
    if (s.kind === "mention") return `${app.idx!.entity.get(s.link.source)?.label} in ${app.idx!.episode.get(s.link.target)?.code}`;
    if (s.kind === "edge") return "the connection";
    if (s.kind === "path") return "the connection";
    if (s.kind === "question") return "your question";
    return "the details";
  }
</script>

{#if app.panelTrail.length}
  <button class="back" onclick={() => app.panelBack()}>‹ Back to {trailName(app.panelTrail.at(-1)!)}</button>
{:else if app.path && app.selection?.kind !== "path"}
  <button class="back" onclick={() => (app.selection = { kind: "path", path: app.path! })}>‹ Back to the connection</button>
{:else if app.question && app.selection?.kind !== "question"}
  <!-- After clicking an item of the answer, the way back to it. -->
  <button class="back" onclick={() => (app.selection = { kind: "question" })}>← Back to your question</button>
{/if}
{#if app.selection?.kind === "question" && app.question}
  <AnswerPanel answer={app.question} />
{:else if app.selection?.kind === "entity"}
  {#key app.selection.id}<EntityPanel id={app.selection.id} />{/key}
{:else if app.selection?.kind === "pair"}
  {#key app.selection.a + app.selection.b}<PairPanel a={app.selection.a} b={app.selection.b} />{/key}
{:else if app.selection?.kind === "edge"}
  <EdgePanel edge={app.selection.edge} />
{:else if app.selection?.kind === "episode"}
  {#key app.selection.id}<EpisodePanel id={app.selection.id} />{/key}
{:else if app.selection?.kind === "mention"}
  <MentionPanel link={app.selection.link} />
{:else if app.selection?.kind === "path"}
  <PathPanel path={app.selection.path} />
{:else if app.selection?.kind === "nopath"}
  <p class="eyebrow">Connection</p>
  <h2 class="serif">No connection found</h2>
  <p class="empty">{app.idx!.entity.get(app.selection.from)?.label} and {app.idx!.entity.get(app.selection.to)?.label} are not linked by any chain
    of connections inside the current filters. A wider time window or more types may connect them.</p>
{/if}

<style>
  .back { border: 0; background: none; padding: 0 0 10px; cursor: pointer; color: var(--ink-2); font: inherit; font-size: 12.5px; }
  .back:hover { color: var(--ink); }
  h2 { font-size: 24px; font-weight: 600; margin: 4px 0 8px; }
</style>
