import { buildIndex, loadDataset } from "./data.ts";
import { defaultFilters, entityVisible, linksInView, livedActive, livedSelection, visiblePassages, type Filters } from "./filters.ts";
import { loadSettings, saveSettings } from "./forces.ts";
import { player } from "./player.svelte.ts";
import type { Mode, PathResult } from "./graph.ts";
import { hiddenItems, pairKey, replay, sceneEdges, sharedPartners, type Step } from "./scene.ts";
import { applyShares, drawnLinks, hiddenLinks, nextStep, portionOf, rankPartners, SHARE_STEP } from "./strength.ts";
import { readRoute, samePage, writeRoute, type Route } from "./route.ts";
import { findPath } from "./graph.ts";
import { axisLength, episodeLinks, itemPositions, itemsOfEpisodes, publicationScale, type EpisodeLink } from "./episodes.ts";
import { findAll, type Meaning } from "./results.ts";
import { nameIndex, parseQuestion, rankEvidence, topicsByMeaning, type Evidence, type NameHit } from "./ask.ts";
import { loadIndex, loadModel } from "./meaning.ts";
import { passageAt, timedWords } from "./follow.ts";
import type { Dataset, Edge, Index } from "./types.ts";

export type View = "start" | "explore" | "results" | "timeline" | "episodes";
/** How much of the archive the whole map draws. */
export type Detail = "main" | "more" | "all";
export const DETAIL_SIZE: Record<Exclude<Detail, "all">, number> = { main: 150, more: 450 };
export type Selection =
  | { kind: "entity"; id: string }
  | { kind: "edge"; edge: Edge }
  | { kind: "episode"; id: string }
  | { kind: "mention"; link: EpisodeLink }
  | { kind: "pair"; a: string; b: string }
  | { kind: "path"; path: PathResult }
  | { kind: "nopath"; from: string; to: string }
  | { kind: "question" };

/** A question asked in the search field and what was found for it. */
export interface Answer {
  text: string;
  names: NameHit[];
  /** Items found by meaning for the words that named nothing, with their similarity. */
  topics: { id: string; score: number }[];
  evidence: Evidence[];
  /** "meaning" while the model loads, "names" when it could not be used. */
  status: "meaning" | "done" | "names";
}

/** Two items explored together start at the first slider step that draws at least this many shared topics. */
const TOGETHER_START = 15;
/** How close in meaning a word of the question must be to a topic's name to show that topic. */
const TOPIC_FLOOR = 0.6;

/** All UI state lives here. Filters are replaced, not mutated, so derived values update. */
class AppState {
  ds = $state.raw<Dataset | null>(null);
  idx = $derived(this.ds ? buildIndex(this.ds) : null);
  filters = $state.raw<Filters | null>(null);
  vp = $derived(this.ds && this.idx && this.filters ? visiblePassages(this.ds, this.idx, this.filters) : new Set<string>());
  /** Every link inside the filters. */
  allLinks = $derived(this.ds && this.idx && this.filters ? linksInView(this.ds, this.idx, this.vp, this.filters) : []);
  /** The links of the calm starting picture (strength.ts). */
  edges = $derived(this.filters ? drawnLinks(this.allLinks, this.filters.share) : []);
  /** Per item its partners, strongest first. */
  ranked = $derived(rankPartners(this.allLinks, (id) => this.idx?.entity.get(id)?.mention_count ?? 0));
  /** Shares chosen for single items on the whole map. An exploration keeps them in its steps. */
  mapShares = $state.raw(new Map<string, number>());
  /** The starting picture with those choices applied. */
  drawn = $derived(applyShares(this.edges, this.ranked, this.mapShares));
  /** One link per item and episode that names it. Kept apart from `edges`, see episodes.ts. */
  links = $derived(this.ds && this.idx && this.filters ? episodeLinks(this.ds, this.idx, this.vp, this.filters) : []);
  // Not every podcast has everything. What is missing is left out of the interface.
  /** Whether there is a lifespan to show, which the Timeline and the 3D view are made of. */
  hasLifespans = $derived(!!this.ds?.site.lifespan_type && this.ds.summary.lifespans === "enriched" && this.ds.entities.some((e) => e.life));
  /** Whether episodes spread over more than one month, so that a publication window means something. */
  hasPublicationRange = $derived(this.ds ? new Set(this.ds.episodes.flatMap((e) => (e.date ? [e.date.slice(0, 7)] : []))).size > 1 : false);
  /** The whole map opens with what is discussed most. More is one click away. */
  detail = $state<Detail>("main");
  /**
   * The topics the whole map draws at this detail, or null for all of them. With a lifespan window
   * or chosen episodes the filters have already made the choice, so nothing is left out then.
   */
  mapTopics = $derived.by(() => {
    if (!this.filters || this.detail === "all" || this.filters.only || livedActive(this.filters)) return null;
    const passages = new Map<string, number>();
    for (const l of this.links) passages.set(l.source, (passages.get(l.source) ?? 0) + l.count);
    const top = [...passages].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0])).slice(0, DETAIL_SIZE[this.detail]);
    return new Set(top.map(([id]) => id));
  });
  /** That choice, plus the selected topic and what the user asked to see more of. */
  mapShown = $derived.by(() => {
    if (!this.mapTopics) return null;
    const shown = new Set(this.mapTopics);
    if (this.visibleFocus) shown.add(this.visibleFocus);
    for (const id of this.mapShares.keys()) {
      shown.add(id);
      for (const e of this.drawn) if (e.source === id || e.target === id) { shown.add(e.source); shown.add(e.target); }
    }
    return shown;
  });
  /** What the whole map draws. */
  mapEdges = $derived(this.mapShown ? this.drawn.filter((e) => this.mapShown!.has(e.source) && this.mapShown!.has(e.target)) : this.drawn);
  mapMentions = $derived(this.mapShown ? this.links.filter((l) => this.mapShown!.has(l.source)) : this.links);
  /** How many topics there are to draw at all. */
  topicCount = $derived(new Set(this.links.map((l) => l.source)).size);
  episodeItems = $derived(this.idx ? itemsOfEpisodes(this.links, this.idx) : new Map<string, string[]>());
  /** Entities a lifespan window selects, drawn even without links. */
  livedIds = $derived(this.ds && this.idx && this.filters ? livedSelection(this.ds, this.idx, this.vp, this.filters) : []);
  /** The selected item, if the current filters let it through. */
  visibleFocus = $derived.by(() => {
    const e = this.focus ? this.idx?.entity.get(this.focus) : undefined;
    return e && this.filters && entityVisible(e, this.idx!, this.vp, this.filters) ? e.id : null;
  });

  view = $state<View>("start");
  mode = $state<Mode>("full");
  /** The exploration as the user built it, step by step. Empty when there is none. */
  steps = $state.raw<Step[]>([]);
  /** What the steps give over the visible edges, as drawn in "focused" mode. */
  scene = $derived.by(() => {
    if (!this.steps.length || !this.idx || !this.filters) return null;
    const replayed = replay(this.steps, { links: this.allLinks, ranked: this.ranked, share: this.filters.share, episodeItems: this.episodeItems });
    // The first item can lie outside a filter set later; the rest come from visible edges.
    replayed.nodes = replayed.nodes.filter((id) => entityVisible(this.idx!.entity.get(id)!, this.idx!, this.vp, this.filters!));
    return { ...replayed, edges: sceneEdges(replayed, this.edges, this.ranked) };
  });
  /** Per item how many of its links are not drawn, per episode how many items are not shown. */
  more = $derived(this.mode === "focused" && this.scene
    ? new Map([...hiddenLinks(this.allLinks, this.scene.edges), ...hiddenItems(this.scene, this.episodeItems)])
    : hiddenLinks(this.allLinks, this.mapEdges));
  /** The publication axis for the current mode and where each item sits on it. */
  publication = $derived(this.ds ? publicationScale(this.ds.episodes,
    axisLength(this.mode, this.ds.episodes.length, this.mapTopics?.size ?? this.ds.entities.filter((e) => !e.generic).length)) : null);
  itemTimes = $derived(this.publication && this.idx ? itemPositions(this.links, this.idx, this.publication) : new Map<string, number>());
  focus = $state<string | null>(null);
  selection = $state.raw<Selection | null>(null);
  path = $state.raw<PathResult | null>(null);
  episode = $state<string | null>(null);
  question = $state.raw<Answer | null>(null);
  /** What was typed for the results page, and its vector once the model has read it. */
  query = $state.raw<{ text: string; meaning: Meaning | null; status: "meaning" | "done" | "names" } | null>(null);
  #searched = 0;
  names = $derived(this.ds ? nameIndex(this.ds) : new Map());
  /** The results page. It is worked out from the query, so it follows the filters. */
  results = $derived(this.query && this.ds && this.idx && this.filters
    ? findAll(this.ds, this.idx, this.vp, this.names, this.query.text, this.query.meaning) : null);
  #asked = 0;
  /** Passage to scroll to and mark when the Episodes view opens. */
  reading = $state<string | null>(null);
  /** What is open around the content. `full` gives the graph the whole window and changes nothing else. */
  ui = $state({ filtersOpen: false, panelOpen: false, connectOpen: false, full: false, audioOpen: false, threeD: false });
  /** What the details showed before, for their Back button. */
  panelTrail = $state.raw<Selection[]>([]);
  // The connection on screen was asked for inside an exploration, which it can go back to.
  #connectedInExploration = false;
  /** The first end of a connection, when it was asked for from a topic. */
  connectFrom = $state<string | null>(null);
  /** Episodes picked on the time axis, not yet shown on their own. */
  picked = $state.raw(new Set<string>());
  graph = $state(loadSettings());

  constructor() {
    $effect.root(() => {
      $effect(() => saveSettings($state.snapshot(this.graph)));
      // Playing on past a passage: the passage being listened to follows the audio, so the words
      // stay marked and the listening bar stays right from one passage into the next.
      $effect(() => {
        const id = player.passageId;
        if (!id || player.paused || player.stopsAtEnd || !this.idx) return;
        const list = this.#spokenStarts(this.idx.passage.get(id)?.episode_id ?? "");
        const here = list.ids[passageAt(list.starts, player.now)];
        if (here && here !== id) player.passageId = here;
      });
      // The address follows the state. A new page goes into the history, a selection only updates it.
      $effect(() => {
        if (!this.ds) return;
        const next = writeRoute(this.route);
        const now = location.hash || "#/";
        if (next === now) return;
        if (samePage(next, now)) history.replaceState(null, "", next);
        else history.pushState(null, "", next);
      });
    });
    // The state follows the address when the browser goes back or forward.
    if (typeof window !== "undefined") window.addEventListener("popstate", () => { if (this.ds) this.applyRoute(readRoute(location.hash)); });
  }

  /** Per episode its passages in order, with the time of each one's first word. Kept once worked out. */
  #starts = new Map<string, { ids: string[]; starts: number[] }>();
  #spokenStarts(episode: string) {
    let list = this.#starts.get(episode);
    if (!list) {
      const passages = this.ds!.passages.filter((p) => p.episode_id === episode).sort((a, b) => a.index - b.index);
      list = { ids: passages.map((p) => p.id), starts: passages.map((p) => timedWords(p)[0]?.t0 ?? p.start_s ?? 0) };
      this.#starts.set(episode, list);
    }
    return list;
  }

  /** The address of what is on screen. */
  get route(): Route {
    const sel = this.selection;
    const open = sel?.kind === "entity" || sel?.kind === "episode" ? sel.id : null;
    switch (this.view) {
      case "results": return this.query ? { view: "results", text: this.query.text, open } : { view: "start", open: null };
      case "episodes": return { view: "episodes", episode: this.episode, passage: this.reading, open };
      case "timeline": return { view: "timeline", open };
      case "explore":
        if (this.path) return { view: "connection", from: this.path.nodes[0], to: this.path.nodes.at(-1)!, open: null };
        if (sel?.kind === "nopath") return { view: "connection", from: sel.from, to: sel.to, open: null };
        return { view: "explore", root: this.mode === "focused" ? this.steps[0]?.id ?? null : null, open,
          with: this.mode === "focused" ? this.#togetherWith() : null };
      default: return { view: "start", open: null };
    }
  }

  /** The second topic of a pair whose first is where the exploration began, for the address. */
  #togetherWith() {
    const root = this.steps[0]?.id;
    for (const s of this.steps) if (s.kind === "together" && (s.a === root || s.b === root)) return s.a === root ? s.b : s.a;
    return null;
  }

  /** Shows what an address says. Ids that are not in the data are ignored. */
  applyRoute(r: Route) {
    const idx = this.idx!;
    const known = (id: string | null) => !!id && (idx.entity.has(id) || idx.episode.has(id));
    if (r.view === "connection") {
      if (idx.entity.has(r.from) && idx.entity.has(r.to)) this.connect(r.from, r.to);
      this.view = "explore";
      this.mode = "full";
      return;
    }
    this.path = null;
    if (r.view === "results") {
      this.view = "results";
      if (this.query?.text !== r.text) void this.search(r.text);
    } else if (r.view === "episodes") {
      this.view = "episodes";
      if (r.episode && idx.episode.has(r.episode)) this.episode = r.episode;
      this.reading = r.passage && idx.passage.has(r.passage) ? r.passage : null;
    } else if (r.view === "explore") {
      this.view = "explore";
      if (r.root && idx.entity.has(r.root)) {
        if (r.with && idx.entity.has(r.with)) {
          if (this.steps[0]?.id !== r.root) { this.steps = []; this.mode = "full"; }
          this.exploreTogether(r.root, r.with);
        } else if (this.steps[0]?.id !== r.root) this.steps = [{ kind: "start", id: r.root }];
        this.mode = "focused";
      } else this.mode = "full";
    } else this.view = r.view === "timeline" && !this.hasLifespans ? "start" : r.view;
    if (known(r.open)) {
      if (idx.episode.has(r.open!)) this.selectEpisode(r.open!);
      else this.selectEntity(r.open!);
    } else if (!(r.view === "explore" && r.with)) this.closePanel();
  }

  async load(base?: string) {
    const ds = await loadDataset(base);
    this.filters = defaultFilters(ds);
    this.ds = ds;
    this.episode = ds.episodes[0]?.id ?? null;
    if (location.hash.length > 2) this.applyRoute(readRoute(location.hash));
  }

  /** Plays a passage from its start to its end. Must run inside the click that asked for it. */
  playPassage(id: string) {
    const p = this.idx!.passage.get(id)!, ep = this.idx!.episode.get(p.episode_id)!;
    this.ui.audioOpen = true;
    if (!ep.audio_url) {
      player.passageId = id;
      player.failed = true;
      return;
    }
    // Start at the passage's first word and stop after its last one. Whisper segments can
    // begin in the previous passage or run on into the next, so their times would play some
    // of the neighbours too. The start is a quarter second early, the word times are estimates.
    const words = timedWords(p);
    const start = words.length ? Math.max(0, words[0].t0 - 0.25) : p.start_s ?? 0;
    player.playFrom(id, { url: ep.audio_url, start, end: words.at(-1)?.t1 ?? p.end_s });
  }

  /** Opens a passage in its episode's full transcript. */
  readPassage(id: string) {
    this.episode = this.idx!.passage.get(id)!.episode_id;
    this.reading = id;
    this.view = "episodes";
  }

  /** A new publication window or series replaces a hand-picked set of episodes. */
  setFilters(patch: Partial<Filters>) {
    const next = { ...this.filters!, ...patch };
    if ((patch.pub || patch.series) && !patch.only) { next.only = null; this.picked = new Set(); }
    this.filters = next;
  }

  resetFilters() {
    if (this.ds) this.filters = defaultFilters(this.ds);
    this.picked = new Set();
  }

  pickEpisode(id: string) {
    const next = new Set(this.picked);
    if (!next.delete(id)) next.add(id);
    this.picked = next;
  }

  /** The graph with only these episodes, each with what it names around it. */
  showOnly(ids: Iterable<string>) {
    const only = new Set(ids);
    this.setFilters({ only, episodes: true });
    this.picked = new Set(only);
    this.path = null;
    this.mode = "full";
    this.view = "explore";
    if (only.size === 1) this.selectEpisode([...only][0]);
  }

  showAllEpisodes() {
    this.setFilters({ only: null });
    this.picked = new Set();
  }

  /** Shows a topic's details. What the map shows stays as it is, a connection too. */
  selectEntity(id: string) {
    this.focus = id;
    this.panelTrail = [];
    this.selection = { kind: "entity", id };
    this.ui.panelOpen = true;
  }

  selectEpisode(id: string) {
    this.panelTrail = [];
    this.selection = { kind: "episode", id };
    this.ui.panelOpen = true;
  }

  /** Moves on inside the details, so that their Back button leads to what was there before. */
  follow(go: () => void) {
    const from = this.selection, trail = this.panelTrail;
    go();
    if (from) this.panelTrail = [...trail, from];
  }

  /** One step back inside the details. */
  panelBack() {
    const previous = this.panelTrail.at(-1);
    if (!previous) return;
    this.panelTrail = this.panelTrail.slice(0, -1);
    this.selection = previous;
    this.focus = previous.kind === "entity" ? previous.id : null;
  }

  selectMention(link: EpisodeLink) {
    this.selection = { kind: "mention", link };
    this.ui.panelOpen = true;
  }

  /** Lets go of what is selected, which closes the details. A connection stays until it is cleared. */
  closePanel() {
    this.panelTrail = [];
    // Inside a connection, letting go of one of its topics leads back to the connection.
    if (this.path && this.selection?.kind !== "path") {
      this.selection = { kind: "path", path: this.path };
      this.focus = null;
      return;
    }
    this.ui.panelOpen = false;
    this.selection = null;
    this.focus = null;
    this.question = null;
  }

  /** The chain between two topics, in the whole map, with its steps in the details. */
  connect(from: string, to: string) {
    const path = findPath(this.allLinks, from, to);
    if (this.exploring) this.#connectedInExploration = true;
    else if (!this.path) this.#connectedInExploration = false;
    this.path = path;
    this.selection = path ? { kind: "path", path } : { kind: "nopath", from, to };
    this.ui.panelOpen = true;
    this.ui.connectOpen = false;
    this.view = "explore";
    this.mode = "full";
  }

  clearConnection() {
    this.path = null;
    this.closePanel();
    if (this.#connectedInExploration) this.mode = "focused";
  }

  /**
   * Turns the connection on screen into an exploration, with all its topics and links, to grow it
   * from there. A connection found inside an exploration joins that exploration. `around` also
   * brings what its two ends share, otherwise only what the user asks for next.
   */
  continueConnection(around = false) {
    const path = this.path;
    if (!path) return;
    const kept = this.#connectedInExploration ? this.steps : [];
    const seen = new Set(kept.flatMap((s) => (s.kind === "start" || s.kind === "add" ? [s.id] : [])));
    const added = path.nodes.filter((id) => !seen.has(id));
    this.steps = kept.length
      ? [...kept, ...added.map((id): Step => ({ kind: "add", id, share: 0 }))]
      : added.map((id, i): Step => ({ kind: i ? "add" : "start", id, share: 0 }));
    this.path = null;
    this.mode = "focused";
    this.view = "explore";
    if (around) this.exploreTogether(path.nodes[0], path.nodes.at(-1)!);
    else if (this.selection?.kind === "path") this.closePanel();
  }

  /**
   * Explores two topics together: what they both are connected to comes first, with both its links.
   * Joins the exploration under way, otherwise starts one. Asking again selects the pair.
   */
  exploreTogether(a: string, b: string) {
    this.path = null;
    const id = pairKey(a, b), exploring = this.mode === "focused" && this.steps.length > 0;
    if (!exploring || !this.steps.some((s) => s.kind === "together" && s.id === id)) {
      // On a line of the slider, so that what it says is exact from the start.
      const total = sharedPartners(a, b, this.ranked).length;
      let share = SHARE_STEP / 100;
      while (share < 1 && portionOf(share, total) < TOGETHER_START) share = nextStep(share);
      const step: Step = { kind: "together", id, a, b, share };
      this.steps = exploring ? [...this.steps, step] : [{ kind: "start", id: a, share: 0 }, step];
    }
    this.mode = "focused";
    this.view = "explore";
    this.selectPair(a, b);
  }

  selectPair(a: string, b: string) {
    this.panelTrail = [];
    this.focus = null;
    this.selection = { kind: "pair", a, b };
    this.ui.panelOpen = true;
  }

  /** How much of what a pair shares is drawn. `total` is how many topics both are connected to. */
  pairShare(id: string) {
    const [a, b] = id.split("|");
    const total = sharedPartners(a, b, this.ranked).length;
    const share = this.scene?.shares.get(id) ?? 0;
    return { total, share, shown: portionOf(share, total), changed: this.steps.some((s) => s.kind === "share" && s.id === id && s.share !== null) };
  }

  /** Listens to a whole episode, not stopping after its first passage. */
  listen(id: string) {
    const first = this.ds!.passages.filter((p) => p.episode_id === id).sort((a, b) => a.index - b.index)[0];
    if (!first) return;
    this.playPassage(first.id);
    player.keepPlaying();
  }

  /** Opens an episode's transcript. */
  readEpisode(id: string) {
    this.episode = id;
    this.reading = null;
    this.view = "episodes";
  }

  selectEdge(edge: Edge) {
    this.panelTrail = [];
    this.selection = { kind: "edge", edge };
    this.ui.panelOpen = true;
  }

  /**
   * Picking an item by name (search, lists, transcript chips) starts a new exploration
   * from it. Clicking a node only selects it, so the view stays put.
   */
  openEntity(id: string) {
    this.question = null;
    this.path = null;
    // An item of a type that is switched off would open an empty graph.
    const type = this.idx!.entity.get(id)!.type;
    if (!this.filters!.types.has(type)) this.setFilters({ types: new Set([...this.filters!.types, type]) });
    this.selectEntity(id);
    this.steps = [{ kind: "start", id }];
    this.mode = "focused";
    this.view = "explore";
  }

  /** The item the link control works on: a selected item, or an episode inside an exploration. */
  get shareTarget(): string | null {
    const sel = this.selection;
    if (!sel) return null;
    if (sel.kind === "entity") return sel.id;
    return sel.kind === "episode" && this.exploring ? sel.id : null;
  }

  get exploring() {
    return this.mode === "focused" && this.steps.length > 0;
  }

  /**
   * How much of one item is shown: its own share if it has one, otherwise what the picture
   * happens to show of it. `total` counts its partners, or for an episode the items it names.
   */
  shareOf(id: string) {
    if (this.path) {
      // In a connection, what is shown of a topic is its neighbours in the chain.
      const total = this.ranked.get(id)?.length ?? 0;
      const shown = this.path.edges.filter((e) => e.source === id || e.target === id).length;
      return { total, shown, share: total ? shown / total : 0, changed: false, own: false };
    }
    const total = this.idx?.episode.has(id) ? this.episodeItems.get(id)?.length ?? 0 : this.ranked.get(id)?.length ?? 0;
    const hiddenNow = Math.min(total, this.more.get(id) ?? 0);
    const own = (this.exploring ? this.scene?.shares : this.mapShares)?.get(id);
    const changed = this.exploring ? this.steps.some((s) => s.kind === "share" && s.id === id && s.share !== null) && own !== undefined : own !== undefined;
    return { total, shown: total - hiddenNow, share: own ?? (total ? (total - hiddenNow) / total : 0), changed, own: own !== undefined };
  }

  /** Chooses how much of this one item is shown. Dragging the slider counts as one step for Back. */
  setShare(id: string, share: number | null) {
    // Asking for more from inside a connection continues from that connection.
    if (this.path) this.continueConnection();
    if (this.exploring) {
      const last = this.steps.at(-1);
      const earlier = last?.kind === "share" && last.id === id ? this.steps.slice(0, -1) : this.steps;
      this.steps = [...earlier, { kind: "share", id, share }];
      return;
    }
    const next = new Map(this.mapShares);
    if (share === null) next.delete(id);
    else next.set(id, share);
    this.mapShares = next;
  }

  /** Selects an item and moves its slider one step on: what double-clicking does. */
  moreOf(id: string) {
    if (this.idx!.episode.has(id)) this.selectEpisode(id);
    else this.selectEntity(id);
    if (this.idx!.episode.has(id) && !this.exploring) return;
    this.setShare(id, nextStep(this.shareOf(id).share));
  }

  /** Adds an item to the current exploration, with the chain that links it, if any. */
  addToScene(id: string, share?: number) {
    this.path = null;
    if (!this.steps.length) return this.openEntity(id);
    this.steps = [...this.steps, { kind: "add", id, share }];
    this.selectEntity(id);
    this.mode = "focused";
  }

  hide(id: string) {
    if (this.steps.length) this.steps = [...this.steps, { kind: "hide", id }];
    if (this.focus === id) { this.focus = null; this.selection = null; }
  }

  /** Undoes the last step. There is nothing to undo on the first one. */
  back() {
    if (this.steps.length > 1) this.steps = this.steps.slice(0, -1);
  }

  /**
   * Answers a question. The items named in it are shown at once; the model for meaning loads
   * meanwhile and then adds topics and puts the passages in order.
   */
  async ask(text: string) {
    const run = ++this.#asked;
    const parsed = parseQuestion(text, this.names);
    const ids = parsed.names.map((n) => n.entity.id);
    this.question = { text, names: parsed.names, topics: [], status: "meaning",
      evidence: ids.length ? rankEvidence(this.idx!, ids, this.vp, null, null) : [] };
    this.#show(ids);
    const index = await loadIndex();
    const embed = index && (await loadModel(index.model));
    if (run !== this.#asked) return;
    const vectors = index && embed ? await embed([text, ...parsed.rest]).catch(() => null) : null;
    if (run !== this.#asked) return;
    if (!index || !vectors) { this.question = { ...this.question, status: "names" }; return; }
    const [q, ...rest] = vectors;
    const topics = topicsByMeaning(rest, index.topics, TOPIC_FLOOR).filter((t) => !ids.includes(t.id) && this.idx!.entity.has(t.id)).slice(0, 2);
    const all = [...ids, ...topics.map((t) => t.id)];
    this.question = { ...this.question, topics, status: "done", evidence: rankEvidence(this.idx!, all, this.vp, q, index.passages) };
    this.#show(all);
  }

  /**
   * Opens the results page for what was typed. What is found by name and by word shows at
   * once; the model for meaning loads meanwhile and then adds what is close in meaning.
   */
  async search(text: string) {
    const run = ++this.#searched;
    this.query = { text, meaning: null, status: "meaning" };
    this.view = "results";
    const parsed = parseQuestion(text, this.names);
    const index = await loadIndex();
    const embed = index && (await loadModel(index.model));
    if (run !== this.#searched) return;
    const vectors = index && embed ? await embed([text, ...parsed.rest]).catch(() => null) : null;
    if (run !== this.#searched) return;
    if (!index || !vectors) { this.query = { text, meaning: null, status: "names" }; return; }
    this.query = { text, status: "done", meaning: { vector: vectors[0], passages: index.passages,
      topics: topicsByMeaning(vectors, index.topics, TOPIC_FLOOR, 8) } };
  }

  /** An exploration with these items and the chains between them, the answer in the side panel. */
  #show(ids: string[]) {
    if (ids.length) {
      // A single item still gets its neighbours, otherwise there would be nothing to see.
      const share = ids.length > 1 ? 0 : undefined;
      this.steps = [{ kind: "start", id: ids[0], share }, ...ids.slice(1).map((id): Step => ({ kind: "add", id, share }))];
      this.focus = ids[0];
      this.mode = "focused";
    }
    this.selection = { kind: "question" };
    this.ui.panelOpen = true;
    this.view = "explore";
  }

  startOver() {
    this.question = null;
    this.steps = [];
    this.mapShares = new Map();
    this.focus = null;
    this.selection = null;
    this.path = null;
    this.mode = "full";
    this.view = "start";
  }

  /** Back to the results of the last search, as they were. The search does not run again. */
  backToResults() {
    this.path = null;
    this.closePanel();
    this.view = "results";
  }

  /** Leaves an exploration for the whole map. The exploration is kept, to come back to. */
  wholeMap() {
    this.path = null;
    this.mode = "full";
    this.view = "explore";
  }

  /** The Map page: the exploration if one is under way, otherwise the whole map. */
  wholeMapOrExploration() {
    if (this.view === "explore") this.wholeMap();
    else this.view = "explore";
  }

  /** Back to the exploration left for the whole map. */
  resumeExploration() {
    this.mode = "focused";
    this.view = "explore";
  }
}

export const app = new AppState();
