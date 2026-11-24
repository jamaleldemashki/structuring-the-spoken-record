<script lang="ts">
  // The way back to what was searched for. It stands first on the top row of a page, where a way
  // back is expected, and is there once a search led to this page or the user went on from it.
  import { app } from "../lib/state.svelte.ts";
</script>

{#if app.query}
  <button class="back" title="Back to the results for: {app.query.text}" onclick={() => app.backToResults()}>‹ Back to results</button>
{/if}

<style>
  .back { flex: none; border: 0; background: none; padding: 3px 8px 3px 0; margin-right: 2px; border-right: 1px solid var(--line); cursor: pointer; font-size: 13px; font-weight: 500; color: var(--ink-2); white-space: nowrap; }
  .back:hover { color: var(--ink); text-decoration: underline; text-underline-offset: 3px; }
</style>
