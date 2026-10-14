import { buildIndex, loadDataset } from "./data.ts";
import { defaultFilters, entityVisible, livedSelection, visibleEdges, visiblePassages, type Filters } from "./filters.ts";
import { loadSettings, saveSettings } from "./forces.ts";
import { player } from "./player.svelte.ts";
import type { Mode, PathResult } from "./graph.ts";
import type { Dataset, Edge, Index } from "./types.ts";

export type View = "explore" | "timeline" | "episodes" | "overview";
export type Selection =
  | { kind: "entity"; id: string }
  | { kind: "edge"; edge: Edge }
  | { kind: "path"; path: PathResult }
  | { kind: "nopath"; from: string; to: string };

/** All UI state lives here. Filters are replaced, not mutated, so derived values update. */
class AppState {
  ds = $state.raw<Dataset | null>(null);
  idx = $derived(this.ds ? buildIndex(this.ds) : null);
  filters = $state.raw<Filters | null>(null);
  vp = $derived(this.ds && this.idx && this.filters ? visiblePassages(this.ds, this.idx, this.filters) : new Set<string>());
  edges = $derived(this.ds && this.idx && this.filters ? visibleEdges(this.ds, this.idx, this.vp, this.filters) : []);
  /** Entities a lifespan window selects, drawn even without links. */
  livedIds = $derived(this.ds && this.idx && this.filters ? livedSelection(this.ds, this.idx, this.vp, this.filters) : []);
  /** The selected item, if the current filters let it through. */
  visibleFocus = $derived.by(() => {
    const e = this.focus ? this.idx?.entity.get(this.focus) : undefined;
    return e && this.filters && entityVisible(e, this.idx!, this.vp, this.filters) ? e.id : null;
  });

  view = $state<View>("explore");
  mode = $state<Mode>("full");
  depth = $state(1);
  focus = $state<string | null>(null);
  selection = $state.raw<Selection | null>(null);
  path = $state.raw<PathResult | null>(null);
  episode = $state<string | null>(null);
  /** Passage to scroll to and mark when the Episodes view opens. */
  reading = $state<string | null>(null);
  /** Each bar can be hidden on its own. The listening bar starts hidden until something plays. */
  ui = $state({ headerOpen: true, filtersOpen: false, toolbarOpen: true, panelOpen: false, legendOpen: false,
    settingsOpen: false, audioMin: true });
  /** Bars to bring back when leaving the maximised graph. null when not maximised. */
  restoreUi = $state.raw<Partial<AppState["ui"]> | null>(null);
  graph = $state(loadSettings());

  constructor() {
    $effect.root(() => { $effect(() => saveSettings($state.snapshot(this.graph))); });
  }

  async load(base?: string) {
    const ds = await loadDataset(base);
    this.filters = defaultFilters(ds);
    this.ds = ds;
    // Nothing is selected at start, the app opens on the full graph.
    this.episode = ds.episodes[0]?.id ?? null;
  }

  /** Plays a passage from its start time. Must run inside the click that asked for it. */
  playPassage(id: string) {
    const p = this.idx!.passage.get(id)!, ep = this.idx!.episode.get(p.episode_id)!;
    this.ui.audioMin = false;
    if (!ep.audio_url) {
      player.passageId = id;
      player.failed = true;
      return;
    }
    player.playFrom(id, { url: ep.audio_url, start: p.start_s ?? 0 });
  }

  /** Opens a passage in its episode's full transcript. */
  readPassage(id: string) {
    this.episode = this.idx!.passage.get(id)!.episode_id;
    this.reading = id;
    this.view = "episodes";
  }

  /** Hides every bar around the graph, or brings back the ones that were open. */
  toggleMaximized() {
    if (this.restoreUi) {
      Object.assign(this.ui, this.restoreUi);
      this.restoreUi = null;
      return;
    }
    const { headerOpen, filtersOpen, toolbarOpen, panelOpen, legendOpen, settingsOpen, audioMin } = this.ui;
    this.restoreUi = { headerOpen, filtersOpen, toolbarOpen, panelOpen, legendOpen, settingsOpen, audioMin };
    Object.assign(this.ui, { headerOpen: false, filtersOpen: false, toolbarOpen: false, panelOpen: false,
      legendOpen: false, settingsOpen: false, audioMin: true });
  }

  setFilters(patch: Partial<Filters>) {
    this.filters = { ...this.filters!, ...patch };
  }

  resetFilters() {
    if (this.ds) this.filters = defaultFilters(this.ds);
  }

  selectEntity(id: string) {
    this.focus = id;
    this.path = null;
    this.selection = { kind: "entity", id };
    this.ui.panelOpen = true;
  }

  selectEdge(edge: Edge) {
    this.selection = { kind: "edge", edge };
    this.ui.panelOpen = true;
  }

  /**
   * Picking an item by name (search, lists, transcript chips) focuses the graph on it.
   * Clicking a node only selects it, so the full graph stays put while exploring.
   */
  openEntity(id: string) {
    this.selectEntity(id);
    this.mode = "focused";
    this.view = "explore";
  }
}

export const app = new AppState();
