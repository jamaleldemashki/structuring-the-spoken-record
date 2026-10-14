<script lang="ts">
  import { app, type View } from "../lib/state.svelte.ts";
  import Search from "./Search.svelte";

  const views: [View, string][] = [["explore", "Explore"], ["timeline", "Timeline"], ["episodes", "Episodes"], ["overview", "Overview"]];

  type Theme = "system" | "light" | "dark";
  let theme = $state<Theme>(readTheme());
  function readTheme(): Theme {
    try { return (localStorage.getItem("theme") as Theme) || "system"; } catch { return "system"; }
  }
  $effect(() => {
    if (theme === "system") delete document.documentElement.dataset.theme;
    else document.documentElement.dataset.theme = theme;
    try { localStorage.setItem("theme", theme); } catch { /* storage unavailable */ }
  });
  const nextTheme: Record<Theme, Theme> = { system: "light", light: "dark", dark: "system" };
</script>

<header>
  <div class="brand"><b>{app.ds!.site.title}</b>{#if app.ds!.site.subtitle}<span>{app.ds!.site.subtitle}</span>{/if}</div>
  <Search />
  <nav class="tabs" aria-label="Views">
    {#each views as [id, label]}
      <button aria-current={app.view === id ? "page" : undefined} class:on={app.view === id} onclick={() => (app.view = id)}>{label}</button>
    {/each}
  </nav>
  <div class="right">
    <button class="iconbtn" title="Theme: {theme}" onclick={() => (theme = nextTheme[theme])}>
      {theme === "dark" ? "Dark" : theme === "light" ? "Light" : "Auto"}
    </button>
  </div>
</header>

<style>
  header { display: flex; align-items: center; gap: 12px 16px; flex-wrap: wrap; padding-block: 12px 8px; }
  .brand { display: flex; align-items: baseline; gap: 8px; }
  .brand b { font-family: var(--font-read); font-size: 21px; font-weight: 600; letter-spacing: -0.01em; }
  .brand span { font-size: 12px; color: var(--ink-3); }
  .tabs { display: flex; gap: 2px; background: var(--surface-2); padding: 3px; border-radius: 999px; }
  .tabs button { border: 0; background: none; padding: 6px 14px; border-radius: 999px; cursor: pointer; color: var(--ink-2); font-weight: 500; }
  .tabs button.on { background: var(--surface); color: var(--ink); box-shadow: 0 1px 2px rgba(15, 20, 30, 0.12); }
  .right { margin-left: auto; display: flex; gap: 8px; }
</style>
