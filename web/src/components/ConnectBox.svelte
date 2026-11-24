<script lang="ts">
  // Finding the chain between two topics. Both ends are found by typing, since scrolling through
  // a list of 1,300 names is no way to find one.
  import { app } from "../lib/state.svelte.ts";
  import Search from "./Search.svelte";
  import TypeMark from "./TypeMark.svelte";

  let to = $state<string | null>(null);
  const entity = (id: string) => app.idx!.entity.get(id)!;
</script>

<div class="box card" role="dialog" aria-label="Find a connection">
  <header><strong>Find a connection</strong><button class="close" aria-label="Close" title="Close (Esc)" onclick={() => (app.ui.connectOpen = false)}>×</button></header>
  <p class="note">The shortest chain between two topics: a person, a relation, another person, a shared passage, an idea.</p>
  {#each [["From", app.connectFrom, (id: string | null) => (app.connectFrom = id)], ["To", to, (id: string | null) => (to = id)]] as const as [name, value, set] (name)}
    <div class="end">
      <span>{name}</span>
      {#if value}
        <button class="chip" title="Choose another" onclick={() => set(null)}><TypeMark type={entity(value).type} />{entity(value).label}<span aria-hidden="true">×</span></button>
      {:else}
        <Search compact placeholder="Type a name" onpick={(e) => set(e.id)} />
      {/if}
    </div>
  {/each}
  <button class="btn find" disabled={!app.connectFrom || !to || app.connectFrom === to} onclick={() => app.connect(app.connectFrom!, to!)}>Show the connection</button>
</div>

<style>
  .box { position: absolute; z-index: 7; top: 8px; right: 8px; width: 320px; max-width: calc(100% - 16px); padding: 8px 12px 12px; box-shadow: var(--shadow); display: flex; flex-direction: column; gap: 8px; }
  header { display: flex; align-items: center; justify-content: space-between; }
  header strong { font-size: 14px; }
  .note { margin: 0; }
  .end { display: grid; grid-template-columns: 40px minmax(0, 1fr); align-items: center; gap: 8px; font-size: 13px; color: var(--ink-2); }
  .end .chip { justify-self: start; max-width: 100%; }
  .find { width: 100%; }
</style>
