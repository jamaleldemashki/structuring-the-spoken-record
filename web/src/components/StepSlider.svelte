<script lang="ts">
  // A slider that shows its steps: a small line at every step with its number under it. The handle is
  // drawn here and not by the browser, so that it stands exactly on the lines. On a very narrow
  // screen the lines stay, and only the ends and the middle have a number.
  import { SHARE_STEP as STEP } from "../lib/strength.ts";

  let { value, label, oninput }: { value: number; label: string; oninput: (percent: number) => void } = $props();

  const marks = Array.from({ length: 100 / STEP + 1 }, (_, i) => i * STEP);
  // The handle stands on the nearest step. The number shown bold is that one.
  const atStep = $derived(Math.round(value / STEP) * STEP);
</script>

<div class="scale" style:--at="{atStep}">
  <input type="range" min="0" max="100" step={STEP} value={atStep} aria-label={label} aria-valuetext="{atStep}%" oninput={(e) => oninput(Number(e.currentTarget.value))} />
  <div class="marks" aria-hidden="true">
    {#each marks as m (m)}
      <span class="mark" class:on={m === atStep} class:main={m % 50 === 0} style:--m="{m}"><i></i><b>{m}</b></span>
    {/each}
  </div>
</div>

<style>
  /* The middle of the handle moves over the width less the handle itself. Half of it on each side. */
  .scale { --handle: 14px; width: 320px; max-width: 100%; container-type: inline-size; }
  input { display: block; width: 100%; height: 16px; margin: 0; padding: 0; background: none; appearance: none; -webkit-appearance: none; cursor: pointer; }
  input::-webkit-slider-runnable-track { height: 4px; border-radius: 2px; background: linear-gradient(to right, var(--accent) calc(var(--at) * 1%), var(--line) calc(var(--at) * 1%)); }
  input::-moz-range-track { height: 4px; border-radius: 2px; background: linear-gradient(to right, var(--accent) calc(var(--at) * 1%), var(--line) calc(var(--at) * 1%)); }
  input::-webkit-slider-thumb { -webkit-appearance: none; width: var(--handle); height: var(--handle); margin-top: -5px; border-radius: 50%; background: var(--accent); border: 2px solid var(--surface); box-shadow: 0 0 0 1px var(--accent); }
  input::-moz-range-thumb { width: var(--handle); height: var(--handle); box-sizing: border-box; border-radius: 50%; background: var(--accent); border: 2px solid var(--surface); box-shadow: 0 0 0 1px var(--accent); }
  .marks { position: relative; height: 19px; }
  .mark { position: absolute; top: 0; left: calc(var(--handle) / 2 + (100% - var(--handle)) * var(--m) / 100); transform: translateX(-50%); display: flex; flex-direction: column; align-items: center; pointer-events: none; }
  .mark i { display: block; width: 1px; height: 6px; background: var(--ink-3); }
  .mark b { font-family: var(--font-data); font-size: 10px; font-weight: 400; line-height: 12px; color: var(--ink-3); font-variant-numeric: tabular-nums; }
  .mark.on b { color: var(--ink); font-weight: 700; }
  .mark.on i { background: var(--ink); opacity: 1; }
  /* The number at the handle always stays, whatever else has no room. */
  @container (max-width: 190px) { .mark:not(.main):not(.on) b { display: none; } }
</style>
