<script lang="ts">
  // What can be done with what is selected, on top of the graph. Double-click and
  // right-click are shortcuts for these, not the only way to them. The slider works on this one
  // topic and leaves all others as they are, so the graph only changes around what is looked at.
  //
  // It waits as a small icon in the corner, so it covers no nodes, and opens when the pointer is over
  // it. It stays open while the slider is dragged, wherever the pointer goes. Without a pointer that
  // can hover, a tap opens it and the keyboard opens it on reaching it.
  import { app } from "../lib/state.svelte.ts";
  import { typeStyle } from "../lib/entityStyle.ts";
  import { pairKey } from "../lib/scene.ts";
  import { stepOf } from "../lib/strength.ts";
  import StepSlider from "./StepSlider.svelte";
  import TypeMark from "./TypeMark.svelte";

  const sel = $derived(app.selection);
  const id = $derived(sel?.kind === "entity" || sel?.kind === "episode" ? sel.id : null);
  const isEpisode = $derived(sel?.kind === "episode");
  const entity = $derived(id && !isEpisode ? app.idx!.entity.get(id) ?? null : null);
  const episode = $derived(id && isEpisode ? app.idx!.episode.get(id) ?? null : null);
  const target = $derived(app.shareTarget);
  const now = $derived(target ? app.shareOf(target) : null);
  const alone = $derived(!!id && app.filters!.only?.size === 1 && app.filters!.only.has(id));
  const isRoot = $derived(app.exploring && app.steps[0]?.id === id);
  const noun = $derived(isEpisode ? "topics" : "connections");
  // Whether this topic is in the picture that can grow: a connection or an exploration.
  const inPicture = $derived(!!id && (app.path ? app.path.nodes.includes(id) : app.exploring ? !!app.scene?.nodes.includes(id) : false));
  const growing = $derived(!!app.path || app.exploring);
  // Two topics explored together have their own slider, over what they share.
  const pair = $derived(sel?.kind === "pair" ? sel : null);
  const pairNow = $derived(pair ? app.pairShare(pairKey(pair.a, pair.b)) : null);
  const name = (id: string) => app.idx!.entity.get(id)?.label ?? id;
  const root = $derived(app.exploring ? app.steps[0]?.id ?? null : null);
  // What the slider shows, said in words. The percentage is only named when it is exact: a topic that
  // has no share of its own shows what the calm map chose for it, which need not be its strongest.
  function strongest(share: number): string {
    if (share === 0) return "None drawn";
    return share > 0 && Math.abs(share * 100 - stepOf(share)) < 0.01 ? `Strongest ${stepOf(share)}%` : "Strongest first";
  }

  /** How long it stays open after the pointer has left, so a small slip does not close it. */
  const CLOSE_AFTER = 300;
  let open = $state(false);
  let box = $state<HTMLDivElement>();
  let over = false, dragging = false, lastPointer = "mouse", closing = 0;
  const hasSlider = $derived(!!target && !!now && now.total > 0 && (!growing || inPicture));
  const small = $derived(pairNow?.total ? `${pairNow.shown}/${pairNow.total}` : !pair && hasSlider ? `${now!.shown}/${now!.total}` : "");
  const title = $derived(pair ? `${name(pair.a)} and ${name(pair.b)}` : entity?.label ?? episode?.code ?? "");

  function closeSoon() {
    clearTimeout(closing);
    closing = window.setTimeout(() => { if (!over && !dragging) open = false; }, CLOSE_AFTER);
  }
  function enter(ev: PointerEvent) {
    if (ev.pointerType === "touch") return;
    over = true;
    clearTimeout(closing);
    open = true;
  }
  function leave(ev: PointerEvent) {
    if (ev.pointerType === "touch") return;
    over = false;
    closeSoon();
  }
  function down(ev: PointerEvent) {
    lastPointer = ev.pointerType;
    dragging = true;
  }
  // A drag can end anywhere. A tap outside closes it on a touch screen, where nothing leaves it.
  function windowUp() {
    if (dragging) { dragging = false; if (!over && lastPointer !== "touch") closeSoon(); }
  }
  function windowDown(ev: PointerEvent) {
    if (ev.pointerType === "touch" && open && !box?.contains(ev.target as Node)) open = false;
  }
  // The keyboard opens it on reaching it. A tap also gives focus, and that is left to the click.
  function focusIn(ev: FocusEvent) {
    if ((ev.target as Element).matches(":focus-visible")) open = true;
  }
  function focusOut(ev: FocusEvent) {
    if (!over && !box?.contains(ev.relatedTarget as Node)) closeSoon();
  }
  function toggle(ev: MouseEvent) {
    // With a mouse it is already open from hovering. A click must not close it under the pointer.
    if (ev.detail === 0 || lastPointer === "touch") open = !open;
  }
  // Something else selected starts closed again, unless the pointer is on it right now.
  $effect(() => { void sel; if (!over) open = false; });
</script>

<svelte:window onpointerup={windowUp} onpointercancel={windowUp} onpointerdown={windowDown} />

{#if title && (pair ? pairNow : entity || episode)}
  <div class="pop card" class:open bind:this={box} onpointerenter={enter} onpointerleave={leave} onpointerdown={down} onfocusin={focusIn} onfocusout={focusOut} role="group">
    <button class="icon" aria-expanded={open} aria-label="{title}: {small ? `${small} drawn, ` : ""}slider and actions" title={open ? undefined : `${title}: slider and actions`} onclick={toggle}>
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><path d="M4 7h9M19 7h1M4 17h1M11 17h9" /><circle cx="16" cy="7" r="2.5" /><circle cx="8" cy="17" r="2.5" /></svg>
      {#if small && !open}<span>{small}</span>{/if}
    </button>
    {#if open}
    {#if pair && pairNow}
      {@const key = pairKey(pair.a, pair.b)}
      <div class="body" role="toolbar" aria-label="What {name(pair.a)} and {name(pair.b)} share">
        <span class="what"><b>{name(pair.a)} and {name(pair.b)}</b></span>
        {#if pairNow.total}
          <div class="amount" title="Draws the strongest of what both are connected to: discussed most with both of them. Select one of the two for what only it has.">
            <StepSlider value={pairNow.share * 100} label="Percent of their shared topics to draw" oninput={(p) => app.setShare(key, p / 100)} />
            <span class="reading"><b>{strongest(pairNow.share)}</b><span class="count">{pairNow.shown} of {pairNow.total} shared topics</span></span>
          </div>
          {#if pairNow.changed}<button class="btn quiet" onclick={() => app.setShare(key, null)}>Reset</button>{/if}
        {/if}
      </div>
    {:else if id && (entity || episode)}
      <div class="body" role="toolbar" aria-label="Actions for {entity?.label ?? episode?.code}">
        <span class="what">
          <TypeMark type={entity?.type ?? "__episode"} />
          <b title={entity?.label ?? episode?.title}>{entity?.label ?? episode?.code}</b>
          {#if entity}<span class="sr">{typeStyle(entity.type).label}</span>{/if}
        </span>
        {#if target && now && hasSlider}
          <div class="amount" title="Draws the strongest of its {noun}: named relations first, then what is discussed together most. Only this one changes.">
            <StepSlider value={now.share * 100} label="Percent of its {noun} to draw" oninput={(p) => app.setShare(target, p / 100)} />
            <span class="reading"><b>{strongest(now.own ? now.share : -1)}</b><span class="count">{now.shown} of {now.total} {noun}</span></span>
          </div>
          {#if now.changed}<button class="btn quiet" onclick={() => app.setShare(target, null)}>Reset</button>{/if}
        {/if}
        <span class="acts">
          {#if entity}
            {#if !growing}
              <button class="btn" onclick={() => app.openEntity(id!)}>Explore from here</button>
            {:else if !inPicture}
              <button class="btn" title="Adds it to what is on screen, with the chain that links it" onclick={() => app.addToScene(id!, 0)}>Add to the map</button>
            {/if}
            {#if root && root !== id}<button class="btn ghost" title="What {name(root)} and {entity.label} are both connected to" onclick={() => app.exploreTogether(root!, id!)}>Shared with {name(root)}</button>{/if}
            <button class="btn ghost" onclick={() => { app.connectFrom = id; app.ui.connectOpen = true; }}>Find a connection</button>
            {#if growing && !isRoot}<button class="btn quiet" title="Leave what is on screen and start again from this topic" onclick={() => app.openEntity(id!)}>Start new here</button>{/if}
            {#if app.exploring && inPicture && !isRoot}<button class="btn quiet" onclick={() => app.hide(id!)}>Hide</button>{/if}
          {:else if episode}
            <button class="btn" onclick={() => app.readEpisode(id!)}>Read the transcript</button>
            {#if alone}<button class="btn ghost" onclick={() => app.showAllEpisodes()}>Show all episodes</button>
            {:else}<button class="btn ghost" onclick={() => app.showOnly([id!])}>Show only this</button>{/if}
            <button class="btn quiet" aria-pressed={app.picked.has(id!)} onclick={() => app.pickEpisode(id!)}>{app.picked.has(id!) ? "Picked" : "Pick to compare"}</button>
          {/if}
        </span>
      </div>
    {/if}

    {/if}
  </div>
{/if}

<style>
  /* In the corner, where it covers least. It only grows while it is in use. */
  .pop { position: absolute; z-index: 6; top: 8px; left: 8px; max-width: min(680px, calc(100% - 16px)); display: flex; align-items: flex-start; gap: 4px; padding: 3px; box-shadow: var(--shadow); font-size: 13px; }
  .icon { flex: none; display: inline-flex; align-items: center; gap: 6px; min-height: 28px; padding: 3px 7px; border: 0; border-radius: 6px; background: none; color: var(--ink-2); cursor: pointer; }
  .icon span { font-family: var(--font-data); font-size: 12px; font-variant-numeric: tabular-nums; }
  .pop:not(.open) .icon:hover, .icon:focus-visible { background: var(--surface-2); color: var(--ink); }
  .open .icon { color: var(--ink); }
  .body { min-width: 0; display: flex; flex-wrap: wrap; align-items: center; gap: 4px 12px; padding: 2px 6px 3px 0; animation: pop-open 0.12s ease-out; }
  @keyframes pop-open { from { opacity: 0; transform: translateY(-3px); } }
  .what { display: inline-flex; align-items: center; gap: 7px; min-width: 0; min-height: 28px; }
  .what b { font-weight: 600; max-width: 200px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .sr { position: absolute; width: 1px; height: 1px; overflow: hidden; clip-path: inset(50%); }
  /* The scale needs the width, so it has a row of its own under the name and the buttons. */
  .amount { order: 5; flex-basis: 100%; display: flex; flex-wrap: wrap; align-items: flex-start; gap: 2px 12px; min-width: 0; }
  .reading { display: flex; flex-direction: column; line-height: 1.3; font-size: 12.5px; }
  .reading b { font-weight: 600; white-space: nowrap; }
  .count { font-size: 12.5px; color: var(--ink-2); white-space: nowrap; font-variant-numeric: tabular-nums; }
  .acts { display: inline-flex; flex-wrap: wrap; gap: 4px; }
  .acts [aria-pressed="true"] { color: var(--ink); font-weight: 600; }
  /* On a phone the details cover the lower part of the screen, so it sits just above them. */
  @media (max-width: 760px) {
    .pop { position: fixed; z-index: 26; top: calc(38vh - var(--bottom-room, 0px) - 44px); max-width: calc(100vw - 16px); }
  }
</style>
