<script lang="ts">
  // The mark of a topic type or of an episode, as it looks in the graph: a filled circle, a ring
  // or the episode tag. The same mark is used wherever a type is named, next to its name.
  import { typeStyle, typeVar } from "../lib/entityStyle.ts";

  let { type, size = 11 }: { type: string; size?: number } = $props();
</script>

{#if typeStyle(type).ring}
  <i class="ellipse ring" style:border-color={typeVar(type)} style:width="{size}px" style:height="{size}px" aria-hidden="true"></i>
{:else}
  <i class={typeStyle(type).shape} style:background={typeVar(type)} style:width="{size}px" style:height="{size}px" aria-hidden="true"></i>
{/if}

<style>
  i { display: inline-block; flex: none; border-radius: 2px; }
  .ellipse { border-radius: 50%; }
  .ring { box-sizing: border-box; border: 2px solid; background: none; }
  .tag { clip-path: polygon(0 0, 70% 0, 100% 50%, 70% 100%, 0 100%); }
</style>
