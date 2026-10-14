<script lang="ts">
  import { app } from "../lib/state.svelte.ts";
  import type { PathResult } from "../lib/graph.ts";
  import { typeVar } from "../lib/entityStyle.ts";

  let { path }: { path: PathResult } = $props();
  const label = (id: string) => app.idx!.entity.get(id)!.label;
  const type = (id: string) => app.idx!.entity.get(id)!.type;
</script>

<p class="eyebrow">Connection</p>
<h2 class="serif">How {label(path.nodes[0])} is connected to {label(path.nodes.at(-1)!)}</h2>
<p class="note">The shortest chain of links between the two, using what is in view. Named relations are preferred over “discussed together” links. Click a step to read its passages.</p>

<ol class="chain">
  {#each path.edges as ed, i (ed.id)}
    {@const from = path.nodes[i]}
    {@const to = path.nodes[i + 1]}
    <li>
      <button onclick={() => app.selectEdge(ed)}>
        <span class="node"><span class="dot" style:background={typeVar(type(from))}></span>{label(from)}</span>
        <span class="link">{ed.kind === "co_discussed" ? `discussed together (${ed.count})` : ed.source === from ? `${ed.label} →` : `← ${ed.label}`}</span>
        <span class="node"><span class="dot" style:background={typeVar(type(to))}></span>{label(to)}</span>
      </button>
    </li>
  {/each}
</ol>
<button class="btn ghost" onclick={() => { app.path = null; app.selection = { kind: "entity", id: path.nodes[0] }; }}>Clear connection</button>

<style>
  h2 { font-size: 22px; font-weight: 600; margin: 4px 0 8px; text-wrap: balance; }
  .chain { list-style: none; padding: 0; margin: 12px 0; display: flex; flex-direction: column; gap: 6px; }
  .chain button { width: 100%; text-align: left; border: 1px solid var(--line); background: var(--bg); border-radius: 8px; padding: 8px 10px; cursor: pointer; display: flex; flex-wrap: wrap; gap: 4px 8px; align-items: center; }
  .chain button:hover { border-color: var(--ink-3); }
  .node { display: inline-flex; gap: 6px; align-items: center; font-weight: 500; }
  .link { color: var(--ink-3); font-size: 12px; font-style: italic; }
</style>
