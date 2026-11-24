// Run with: npm test  (Node's built-in test runner strips the TypeScript types).
import { test } from "node:test";
import assert from "node:assert/strict";
import { buildIndex } from "./data.ts";
import { defaultFilters, edgeVisible, episodeVisible, linksInView, livedOk, livedSelection, visiblePassages, visibleEdges } from "./filters.ts";
import { buildGraph, findPath } from "./graph.ts";
import { formatTime, highlight, lifeLabel, lifeRange } from "./format.ts";
import { configureTypes, typeStyle } from "./entityStyle.ts";
import { lifeAxis, monthLabels, monthsBetween, withoutOutliers } from "./timeAxis.ts";
import { readRoute, samePage, writeRoute, type Route } from "./route.ts";
import { DEFAULT_SETTINGS, makeSimulation, screenDiameter, seed, separate, settle, setLargestTopic, SMALLEST_NODE, type SimNode } from "./forces.ts";
import { searchArchive, suggestions } from "./search.ts";
import { pairKey, replay, sceneEdges, sharedPartners, type Step } from "./scene.ts";
import { placeInTime, timeScale } from "./timeScale.ts";
import { passageAt, runs, timedWords, wordAt } from "./follow.ts";
import { cosine, decodeVectors, distance, nameIndex, parseQuestion, rankEvidence, topicsByMeaning } from "./ask.ts";
import { applyShares, drawnLinks, hiddenLinks, nextStep, quota, rankPartners, stepOf, weighted } from "./strength.ts";
import { wheelFactor } from "./zoom.ts";
import { chooseLabels } from "./labels.ts";
import { axisTicks, episodeLinks, itemPositions, itemsOfEpisodes, publicationScale, selectionPositions, episodeMarks } from "./episodes.ts";
import { findAll, termPattern } from "./results.ts";
import type { CoEdge, Dataset, Edge, Entity, Site } from "./types.ts";

const site: Site = {
  schema_version: 2, archive_id: "test", title: "Test archive", subtitle: "", series_label: "Series", era_label: null,
  entity_types: [
    { type: "Person", label: "Person", plural: "people", visible_by_default: true },
    { type: "Concept", label: "Concept", plural: "concepts", visible_by_default: true },
    { type: "Work", label: "Work", plural: "works", visible_by_default: true },
    { type: "Place", label: "Place", plural: "places", visible_by_default: false },
  ],
  lifespan_type: "Person", lived_presets: [], search_placeholder: "Search",
};
configureTypes(site);

const ent = (id: string, type: Entity["type"], passages: string[], life?: Entity["life"]): Entity => ({
  id, label: id, type, aliases: [], qid: null, link_status: "not_attempted", mention_count: passages.length,
  episode_ids: [], resolution_status: "provisional", life,
  occurrences: passages.map((p) => ({ passage_id: p, start: 0, end: 1 })),
});

function dataset(): Dataset {
  return {
    site,
    episodes: [
      { id: "e1", code: "A 1", title: "One", date: "2015-01-01", series: "A", era: null, audio_url: null, source_url: null, passage_count: 2 },
      { id: "e2", code: "B 2", title: "Two", date: "2020-01-01", series: "B", era: null, audio_url: null, source_url: null, passage_count: 1 },
    ],
    passages: [
      { id: "p1", episode_id: "e1", index: 0, start_s: 0, end_s: 10, word_count: 5, text: "Aquinas read Aristotle." },
      { id: "p2", episode_id: "e1", index: 1, start_s: 10, end_s: 20, word_count: 5 },
      { id: "p3", episode_id: "e2", index: 0, start_s: 0, end_s: 10, word_count: 5 },
    ],
    entities: [ent("aq", "Person", ["p1", "p2"], { start: 1225, end: 1274, living: false, approximate: true, basis: "birth_death" }), ent("ar", "Person", ["p1", "p3"], { start: -384, end: -322, living: false, approximate: false, basis: "birth_death" }),
      ent("soul", "Concept", ["p2", "p3"]), ent("anon", "Person", ["p3"])],
    edges: [
      { id: "r1", kind: "relation", source: "aq", target: "ar", predicate: "influenced_by", label: "influenced by",
        status: "nli_supported", confidence: 0.9, evidence: ["p1"] },
      { id: "c1", kind: "co_discussed", source: "ar", target: "soul", count: 1, pmi: 1, evidence: ["p3"] },
    ],
    related: [],
    summary: { episodes: 2, passages: 3, entities: 4, relation_edges: 1, co_discussed_edges: 1, related_passages: 0,
      relation_status: "nli_supported", discovery_method: null, lifespans: "enriched", dated_entities: 2 },
  };
}

test("publication window hides passages and therefore edges whose only evidence is outside it", () => {
  const ds = dataset(), idx = buildIndex(ds), f = defaultFilters(ds);
  assert.deepEqual(f.pub, ["2015-01", "2020-01"]);
  f.pub = ["2015-01", "2016-12"];
  const vp = visiblePassages(ds, idx, f);
  assert.deepEqual([...vp].sort(), ["p1", "p2"]);
  assert.equal(edgeVisible(ds.edges[0], idx, vp, f), true);
  assert.equal(edgeVisible(ds.edges[1], idx, vp, f), false);
});

test("a passage counts less for each pair the more items it names", () => {
  const ds = dataset();
  ds.entities = [...ds.entities, ent("x", "Person", ["p1"]), ent("y", "Person", ["p1"])];
  const co = (id: string, a: string, b: string, evidence: string[]) =>
    ({ id, kind: "co_discussed", source: a, target: b, count: evidence.length, pmi: 0, evidence }) as Dataset["edges"][number];
  ds.edges = [co("c1", "aq", "x", ["p1"]), co("c2", "ar", "y", ["p1"]), co("c3", "ar", "soul", ["p3"])];
  const idx = buildIndex(ds);
  assert.equal(idx.passageWeight.get("p1"), 1 / 3, "aq, ar, x and y are named in p1");
  assert.equal(idx.passageWeight.get("p3"), 1, "only Aristotle and the soul in p3");
  const f = defaultFilters(ds);
  f.pub = ["2015-01", "2016-12"];
  const narrow = visiblePassages(ds, idx, f);
  assert.deepEqual(visibleEdges(ds, idx, narrow, f).filter((e) => e.kind === "co_discussed").map((e) => e.id), ["c1", "c2"],
    "a link whose passages are all outside the filters is not drawn");
});

test("every item gets the same share of its links, and a hub stays inside its quota", () => {
  const co = (a: string, b: string, weight: number) =>
    ({ id: `${a}${b}`, kind: "co_discussed", source: a, target: b, count: 1, pmi: 0, evidence: [], weight }) as CoEdge;
  // A hub with 10 partners, each of which also knows a partner of its own.
  const links = Array.from({ length: 10 }, (_, i) => [co("hub", `p${i}`, 1), co(`p${i}`, `q${i}`, 1)]).flat();
  const degree = (edges: { source: string; target: string }[], id: string) => edges.filter((e) => e.source === id || e.target === id).length;
  const at20 = drawnLinks(links, 0.2);
  assert.equal(degree(at20, "hub"), 2, "20% of 10");
  assert.equal(quota(0.2, 10), 2);
  assert.equal(quota(0.2, 2), 1, "at least one");
  assert.ok(new Set(at20.flatMap((e) => [e.source, e.target])).size === 21, "nobody falls off the map");
  assert.equal(drawnLinks(links, 1).length, links.length, "100% draws every link");
  // The badge is what is left.
  assert.equal(hiddenLinks(links, at20).get("hub"), 8);
  assert.equal(hiddenLinks(links, links).get("hub"), 0);
});

test("stronger links are drawn first and named relations always", () => {
  const co = (id: string, a: string, b: string, weight: number) =>
    ({ id, kind: "co_discussed", source: a, target: b, count: 1, pmi: 0, evidence: [], weight }) as CoEdge;
  const rel = { id: "r", kind: "relation", source: "a", target: "z", predicate: "x", label: "x", status: "nli_supported", confidence: 1, evidence: [] } as Edge;
  const links = [co("ab", "a", "b", 0.2), co("ac", "a", "c", 1), co("ad", "a", "d", 0.5), rel];
  // Quota of a is 2 of 4: the relation and then the strongest link. b and d are not left without one.
  const ids = drawnLinks(links, 0.5).map((e) => e.id);
  assert.ok(ids.includes("r") && ids.includes("ac"));
  // A pair with a relation and shared passages takes one place of the quota, not two.
  const both = [rel, co("az", "a", "z", 1), co("ab", "a", "b", 1)];
  assert.equal(drawnLinks(both, 1).length, 3);
  assert.deepEqual(new Set(drawnLinks(links, 0.5).flatMap((e) => [e.source, e.target])), new Set(["a", "b", "c", "d", "z"]));
});

test("the wheel zooms by the same rule on every device, and a slow turn still moves the view", () => {
  const notch = wheelFactor({ deltaY: -100, deltaMode: 0, ctrlKey: false }, 500);
  assert.ok(notch > 1.3 && notch < 1.7, "one mouse notch is a clear step");
  assert.ok(Math.abs(wheelFactor({ deltaY: 100, deltaMode: 0, ctrlKey: false }, 500) * notch - 1) < 1e-9, "out undoes in");
  assert.ok(wheelFactor({ deltaY: -3, deltaMode: 1, ctrlKey: false }, 500) > 1.3, "lines are converted to pixels");
  assert.ok(wheelFactor({ deltaY: -4, deltaMode: 0, ctrlKey: false }, 500) > 1.15, "a small step after a pause counts as a notch");
  assert.ok(wheelFactor({ deltaY: -4, deltaMode: 0, ctrlKey: false }, 16) < 1.03, "inside a trackpad gesture it stays small");
  assert.ok(wheelFactor({ deltaY: -5000, deltaMode: 0, ctrlKey: false }, 16) < 2.7, "one event can't jump");
  assert.equal(wheelFactor({ deltaY: 0, deltaMode: 0, ctrlKey: false }, 16), 1);
});

test("a hand-picked set of episodes replaces the publication window", () => {
  const ds = dataset(), idx = buildIndex(ds), f = { ...defaultFilters(ds), pub: ["2015-01", "2015-01"] as [string, string], only: new Set(["e2"]) };
  assert.equal(episodeVisible(ds.episodes[1], f), true);
  assert.equal(episodeVisible(ds.episodes[0], f), false);
  assert.deepEqual([...visiblePassages(ds, idx, f)], ["p3"]);
  const links = episodeLinks(ds, idx, visiblePassages(ds, idx, f), f);
  assert.ok(links.length && links.every((l) => l.target === "e2"));
});

test("chosen episodes stand evenly apart in order of publication, a single one in the middle", () => {
  const eps = dataset().episodes;
  assert.equal(selectionPositions([eps[0]], 100).x(eps[0].date!), 0);
  const two = selectionPositions(eps, 200);
  assert.equal(two.x("2015-01-01"), -100);
  assert.equal(two.x("2020-01-01"), 100);
});

test("lifespan window: born and died within by default, overlap on request, undated hidden unless asked", () => {
  const ds = dataset(), f = defaultFilters(ds);
  f.lived = [1200, 1300];
  assert.equal(livedOk(ds.entities[0], f, "Person"), true, "Aquinas 1225-1274 lies inside");
  assert.equal(livedOk(ds.entities[1], f, "Person"), false);
  assert.equal(livedOk(ds.entities[2], f, "Person"), true, "non-people are left to visibleEdges");
  assert.equal(livedOk(ds.entities[3], f, "Person"), false, "undated people are hidden by default");
  f.undated = true;
  assert.equal(livedOk(ds.entities[3], f, "Person"), true);
  f.lived = [1250, 1300];
  assert.equal(livedOk(ds.entities[0], f, "Person"), false, "born before the window");
  f.livedMatch = "overlap";
  assert.equal(livedOk(ds.entities[0], f, "Person"), true, "but alive during it");
});

test("lifespan window keeps other items only when they connect to a selected person", () => {
  const ds = dataset();
  ds.edges.push({ id: "c2", kind: "co_discussed", source: "soul", target: "virtue", count: 1, pmi: 1, evidence: ["p2"] },
    { id: "c3", kind: "co_discussed", source: "aq", target: "soul", count: 1, pmi: 1, evidence: ["p2"] });
  ds.entities.push(ent("virtue", "Concept", ["p2"]));
  const idx2 = buildIndex(ds), f = { ...defaultFilters(ds), share: 1 };
  const vp = visiblePassages(ds, idx2, f);
  assert.deepEqual(visibleEdges(ds, idx2, vp, f).map((e) => e.id).sort(), ["c1", "c2", "c3", "r1"]);
  f.lived = [1200, 1300];
  // Aristotle is outside, so r1 and c1 go; soul stays through Aquinas, and virtue through soul is dropped.
  assert.deepEqual(visibleEdges(ds, idx2, vp, f).map((e) => e.id), ["c3"]);
});

test("path prefers the shortest chain and reports it in order", () => {
  const ds = dataset(), idx = buildIndex(ds), f = defaultFilters(ds);
  const edges = visibleEdges(ds, idx, visiblePassages(ds, idx, f), f);
  const path = findPath(edges, "aq", "soul")!;
  assert.deepEqual(path.nodes, ["aq", "ar", "soul"]);
  assert.deepEqual(path.edges.map((e) => e.id), ["r1", "c1"]);
  assert.equal(findPath(edges, "aq", "anon"), null);
});

test("an exploration grows one step at a time, and Back is replaying fewer steps", () => {
  const ds = dataset(), idx = buildIndex(ds), f = defaultFilters(ds);
  const vp = visiblePassages(ds, idx, f), edges = visibleEdges(ds, idx, vp, f);
  const ranked = rankPartners(edges);
  const input = { links: edges, ranked, share: 1 };
  const start = [{ kind: "start", id: "aq" }] as Step[];
  const scene = replay(start, input);
  assert.deepEqual(scene.nodes, ["aq", "ar"]);
  const more = hiddenLinks(edges, sceneEdges(scene, edges, ranked));
  assert.equal(more.get("aq"), 0);
  assert.equal(more.get("ar"), 1, "its link to soul is not drawn yet");
  const grown = [...start, { kind: "share", id: "ar", share: 1 }] as Step[];
  assert.deepEqual(replay(grown, input).nodes, ["aq", "ar", "soul"]);
  assert.deepEqual(replay(grown.slice(0, -1), input).nodes, ["aq", "ar"]);
  // A hidden item stays hidden when its neighbour shows all its links.
  assert.deepEqual(replay([...start, { kind: "hide", id: "ar" }, { kind: "share", id: "aq", share: 1 }] as Step[], input).nodes, ["aq"]);
});

test("the share of one item changes only that item, and turning it down takes its partners away", () => {
  const co = (a: string, b: string, weight: number) => ({ id: `${a}${b}`, kind: "co_discussed", source: a, target: b, count: 1, pmi: 0, evidence: [], weight }) as CoEdge;
  const links = [co("a", "b", 4), co("a", "c", 3), co("a", "d", 2), co("a", "e", 1), co("x", "y", 1), co("x", "z", 1)];
  const ranked = rankPartners(links);
  const input = { links, ranked, share: 0.25 };
  assert.deepEqual(replay([{ kind: "start", id: "a" }], input).nodes, ["a", "b"], "a quarter of its four partners");
  const half = [{ kind: "start", id: "a" }, { kind: "share", id: "a", share: 0.5 }] as Step[];
  assert.deepEqual(replay(half, input).nodes, ["a", "b", "c"], "the strongest come first");
  assert.deepEqual(replay([...half, { kind: "share", id: "a", share: null }], input).nodes, ["a", "b"], "back to where it started");
  assert.deepEqual(replay([{ kind: "start", id: "a" }, { kind: "share", id: "a", share: 0 }], input).nodes, ["a"]);
  // On the whole map: the links of other items are not touched.
  const base = [links[0], links[4]];
  const ids = (shares: [string, number][]) => applyShares(base, ranked, new Map(shares)).map((e) => e.id).sort();
  assert.deepEqual(ids([["a", 0.75]]), ["ab", "ac", "ad", "xy"]);
  assert.deepEqual(ids([["a", 0]]), ["xy"]);
  assert.deepEqual(ids([]), ["ab", "xy"]);
});

test("partners are ranked with named relations first, and a pair counts once", () => {
  const co = (b: string, weight: number) => ({ id: `a${b}`, kind: "co_discussed", source: "a", target: b, count: 1, pmi: 0, evidence: [], weight }) as CoEdge;
  const rel = { id: "r", kind: "relation", source: "a", target: "z", predicate: "x", label: "x", status: "nli_supported", confidence: 1, evidence: [] } as Edge;
  const ranked = rankPartners([co("b", 5), co("z", 1), rel], (id) => (id === "b" ? 9 : 0)).get("a")!;
  assert.deepEqual(ranked.map((p) => p.id), ["z", "b"]);
  assert.equal(ranked[0].edges.length, 2);
});

test("adding an item brings the chain that links it to what is shown", () => {
  const ds = dataset(), idx = buildIndex(ds), f = defaultFilters(ds);
  const vp = visiblePassages(ds, idx, f), edges = visibleEdges(ds, idx, vp, f);
  const ranked = rankPartners(edges);
  const scene = replay([{ kind: "start", id: "soul", share: 0 }, { kind: "add", id: "aq", share: 0 }] as Step[], { links: edges, ranked, share: 1 });
  assert.deepEqual(scene.nodes, ["soul", "aq", "ar"]);
  assert.deepEqual(scene.tree.map((e) => e.id).sort(), ["c1", "r1"]);
  const g = buildGraph(edges, idx, vp, { mode: "focused", focus: "aq", path: null,
    scene: { nodes: scene.nodes, edges: sceneEdges(scene, edges, ranked) } });
  assert.deepEqual(g.nodes.map((n) => n.id), ["aq", "ar", "soul"]);
  assert.equal(g.edges.find((e) => e.id === "r1")!.label, "influenced by ✓");
  assert.ok(g.edges.find((e) => e.id === "r1")!.showLabel, "named relations are labelled in an exploration");
});

test("formatting helpers", () => {
  assert.equal(formatTime(252), "04:12");
  assert.equal(formatTime(3725), "1:02:05");
  assert.equal(lifeLabel({ start: -384, end: -322, living: false, approximate: false, basis: "birth_death" }), "384 BCE – 322 BCE");
  assert.equal(lifeLabel({ start: 1225, end: 1274, living: false, approximate: true, basis: "birth_death" }), "c. 1225 – 1274");
  assert.equal(lifeLabel({ start: 1946, end: null, living: true, approximate: false, basis: "birth_death" }), "born 1946");
  assert.equal(lifeLabel({ start: 401, end: 500, living: false, approximate: true, basis: "floruit" }), "active c. 401 – 500");
  assert.deepEqual(lifeRange({ start: 1946, end: null, living: true, approximate: false, basis: "birth_death" }, 2026), [1946, 2026]);
  assert.deepEqual(lifeRange({ start: 1033, end: null, living: false, approximate: false, basis: "birth_death" }, 2026), [1033, 1033]);
  const text = "Long ago Aquinas read Aristotle in Paris at length.";
  const parts = highlight(text, [{ start: 22, end: 31 }, { start: 9, end: 16 }], 6);
  assert.deepEqual(parts.filter((p) => p.mark).map((p) => p.text), ["Aquinas", "Aristotle"]);
  assert.ok(parts[0].text.startsWith("…") && parts.at(-1)!.text.endsWith("…"));
  const whole = highlight(text, [{ start: 22, end: 31 }], Infinity);
  assert.equal(whole.map((p) => p.text).join(""), text, "the full passage is the exact text, without ellipses");
  assert.deepEqual(whole.filter((p) => p.mark).map((p) => p.text), ["Aristotle"]);
});

test("force layout keeps every pair of nodes apart by at least their collision radii", () => {
  const nodes: SimNode[] = Array.from({ length: 30 }, (_, i) => ({ id: `n${i}`, r: 40 }));
  const links = nodes.slice(1).map((n, i) => ({ id: `l${i}`, source: "n0", target: n.id }));
  assert.equal(seed(nodes, links, new Map([["n0", { x: 5, y: 5 }]])), 29 / 30);
  settle(makeSimulation(nodes, links, DEFAULT_SETTINGS).alpha(1), 600);
  let worst = Infinity;
  for (const a of nodes) for (const b of nodes) if (a !== b)
    worst = Math.min(worst, Math.hypot(a.x! - b.x!, a.y! - b.y!) / (a.r + b.r));
  assert.ok(worst > 0.9, `closest pair at ${worst.toFixed(2)} of their combined radii`);
});

test("a lifespan window draws exactly the people it selects, linked or not, and never the stale selection", () => {
  // The reported case: a window around Plato only. Plato has no link strong enough to draw,
  // and the previously selected Aquinas lies outside the window.
  const ds = dataset();
  ds.entities.push(ent("plato", "Person", ["p3"], { start: -430, end: -341, living: false, approximate: true, basis: "birth_death" }));
  const idx = buildIndex(ds), f = { ...defaultFilters(ds), lived: [-432, -340] as [number, number] };
  const vp = visiblePassages(ds, idx, f);
  const edges = visibleEdges(ds, idx, vp, f);
  const always = livedSelection(ds, idx, vp, f);
  assert.deepEqual(always, ["plato"]);
  const g = buildGraph(edges, idx, vp, { mode: "full", focus: null, path: null, always });
  assert.deepEqual(g.nodes.map((n) => n.id), ["plato"]);
});

test("entity types are styled by their configured position, so any archive's types work", () => {
  assert.equal(typeStyle("Person").token, "hue-1");
  assert.equal(typeStyle("Work").token, "hue-3");
  assert.ok(site.entity_types.every((t) => typeStyle(t.type).shape === "ellipse"), "every topic is round");
  assert.deepEqual(site.entity_types.map((t) => typeStyle(t.type).ring), site.entity_types.map((_, i) => i >= 4), "rings from the fifth type on");
  assert.equal(typeStyle("Instrument").label, "Instrument", "unknown types fall back to a neutral style");
  assert.equal(typeStyle("Instrument").ring, true);
  assert.equal(defaultFilters(dataset()).types.has("Place"), false);
});

test("time axes fit the data and never label past the last publication year", () => {
  const axis = lifeAxis([[-430, -341], [1225, 1274]]);
  assert.ok(axis.lo <= -430 && axis.hi >= 1274 && axis.ticks.length <= 11);
  assert.deepEqual(lifeAxis([[1900, 1950]]).ticks.slice(0, 2), [1890, 1900]);
  // Eleven months on a wide axis get month labels, a long run only year labels.
  assert.deepEqual(monthLabels("2015-06", "2016-05", 800).map((l) => l.text).slice(0, 3), ["Jun 2015", "Jul 2015", "Aug 2015"]);
  assert.deepEqual(monthLabels("2015-06", "2016-05", 300).map((l) => l.text), ["Jun 2015", "Sep 2015", "Dec 2015", "Mar 2016"]);
  assert.deepEqual(monthLabels("2000-01", "2025-12", 300).map((l) => l.text), ["2000", "2005", "2010", "2015", "2020", "2025"]);
  assert.deepEqual(monthsBetween("2015-11", "2016-02"), ["2015-11", "2015-12", "2016-01", "2016-02"]);
});

test("search groups results by type, then episodes, then lines in the transcripts", () => {
  const ds = dataset(), idx = buildIndex(ds);
  ds.entities[3].generic = true;
  const groups = searchArchive(ds, idx, "a");
  assert.deepEqual(groups.map((g) => g.label), ["people", "episodes"], "one letter does not search the transcripts");
  assert.deepEqual(groups[0].items.map((e) => (e as Entity).id), ["aq", "ar", "anon"], "general words go last");
  const lines = searchArchive(ds, idx, "aristotle").find((g) => g.kind === "passages");
  assert.ok(lines && lines.kind === "passages");
  assert.deepEqual([lines.items[0].passage.id, lines.items[0].start, lines.items[0].end], ["p1", 13, 22]);
  assert.deepEqual(searchArchive(ds, idx, "  "), []);
});

test("start page questions are worked out from the data, so each one has an answer", () => {
  const ds = dataset();
  const found = suggestions(ds, ds.edges);
  assert.deepEqual(found.map((s) => s.kind), ["connection", "topic"], "no period question without presets");
  assert.equal(found[0].text, "How is aq connected to ar?");
  assert.equal(found[1].text, "What does the podcast say about soul?");
});

test("the 3D time axis keeps time in order, gives dense periods more room and clamps outliers", () => {
  // Many people around 1250, a few in antiquity, one far-off outlier.
  const years = [-400, -350, 400, 1220, 1230, 1240, 1250, 1255, 1260, 1270, 1280, -4000];
  const scale = timeScale(years, 1000, [-600, 2100]);
  const heights = [-500, 0, 500, 1000, 1200, 1250, 1300, 1500].map(scale.height);
  assert.ok(heights.every((h, i) => i === 0 || h > heights[i - 1]), "later is always higher");
  const linear = (a: number, b: number) => ((b - a) / (scale.hi - scale.lo)) * 1000;
  assert.ok(scale.height(1300) - scale.height(1200) > 2 * linear(1200, 1300), "the dense 13th century is stretched");
  assert.equal(scale.height(-4000), scale.height(-600), "the outlier sits at the bottom end");
  for (let i = 1; i < scale.ticks.length; i++) assert.ok(scale.ticks[i].y - scale.ticks[i - 1].y >= 40, "labels never crowd");
});

test("undated items take the time of the people they are linked to", () => {
  const edges = [{ source: "aq", target: "soul" }, { source: "ar", target: "soul" }, { source: "soul", target: "summa" }];
  const { year, inferred } = placeInTime(["aq", "ar", "soul", "summa", "alone"], edges, new Map([["aq", 1250], ["ar", -350]]));
  assert.equal(year.get("soul"), 450);
  assert.equal(year.get("summa"), 450, "found in a second round, through the concept");
  assert.equal(year.get("alone"), null);
  assert.deepEqual([...inferred].sort(), ["soul", "summa"]);
});

test("words get times inside their segment, also when the segment crosses the passage start", () => {
  // The first segment began 10 characters before this passage, so only its last part is here.
  const text = "soul is form. Then the body";
  const words = timedWords({ text, timing: [[-10, 13, 0, 23], [13, 27, 23, 37]] });
  assert.deepEqual(words.map((w) => text.slice(w.start, w.end)), ["soul", "is", "form.", "Then", "the", "body"]);
  assert.equal(words[0].t0, 10, "one second per character, ten characters were in the previous passage");
  assert.ok(words.every((w, i) => i === 0 || w.t0 >= words[i - 1].t0), "times only go forward");
  assert.equal(wordAt(words, 9), -1);
  assert.equal(text.slice(words[wordAt(words, 24)].start, words[wordAt(words, 24)].end), "Then");
  assert.equal(wordAt(words, 40), -1);
});

test("mentions stay one piece and keep their words", () => {
  const text = "Then Thomas Aquinas wrote";
  const words = timedWords({ text, timing: [[0, 25, 0, 25]] });
  const out = runs(text, [{ start: 5, end: 19, id: "aq" }], words);
  assert.deepEqual(out.map((r) => r.id ?? null), [null, "aq", null]);
  assert.equal(out.map((r) => r.pieces.map((p) => p.text).join("")).join(""), text, "no text is lost");
  assert.deepEqual(out[1].pieces.filter((p) => p.w >= 0).map((p) => p.text), ["Thomas", "Aquinas"]);
  assert.deepEqual(runs(text, [], []), [{ id: undefined, pieces: [{ text, w: -1 }] }], "without timing the text stays whole");
});

test("a question is read for names, with longer names first and small typos forgiven", () => {
  const ds = dataset();
  ds.entities = [{ ...ds.entities[0], label: "Thomas Aquinas", aliases: ["Aquinas"] }, { ...ds.entities[1], label: "Aristotle" },
    { ...ds.entities[2], label: "soul" }, { ...ds.entities[3], label: "Thomas", mention_count: 9 }, { ...ent("gen", "Concept", ["p1"]), label: "relation", generic: true }];
  const names = nameIndex(ds);
  const q = parseQuestion("Is there any relation between Thomas Auinas and Aristotle when it comes to the soul?", names);
  assert.deepEqual(q.names.map((n) => n.entity.id), ["aq", "ar", "soul"], "the full name wins over the more mentioned first name");
  assert.equal(q.names[0].fuzzy, true);
  assert.deepEqual(q.rest, [], "framing words are not sent to the model");
  assert.deepEqual(parseQuestion("what did Aristotle say about the mind", names).rest, ["mind"]);
  assert.equal(parseQuestion("Thomas and the soul", names).names[0].entity.id, "anon", "on its own the first name is the other Thomas");
  assert.equal(distance("auinas", "aquinas", 1), 1);
  assert.equal(distance("plato", "pluto", 0), 1, "short words get no typo allowance, they stop at once");
});

test("evidence mentioning more of the asked items comes first, then the closest in meaning", () => {
  const ds = dataset(), idx = buildIndex(ds);
  const all = new Set(ds.passages.map((p) => p.id));
  const vec = (a: number, b: number) => Int8Array.from([a, b]);
  const passages = new Map([["p1", vec(127, 0)], ["p2", vec(0, 127)], ["p3", vec(90, 90)]]);
  const q = Float32Array.from([0, 1]);
  assert.deepEqual(rankEvidence(idx, ["aq", "ar"], all, q, passages).map((e) => e.passageId), ["p1", "p2", "p3"]);
  assert.deepEqual(rankEvidence(idx, ["soul"], all, q, passages).map((e) => e.passageId), ["p2", "p3"]);
  assert.deepEqual(rankEvidence(idx, [], all, q, passages).map((e) => e.passageId), ["p2", "p3", "p1"], "no names: meaning alone");
  assert.deepEqual(rankEvidence(idx, [], all, null, null), [], "no names and no model: nothing to rank by");
  const topics = { ids: ["soul", "ar"], vectors: [vec(0, 127), vec(127, 0)] };
  assert.deepEqual(topicsByMeaning([q], topics, 0.5).map((t) => t.id), ["soul"]);
  const [v] = decodeVectors(btoa(String.fromCharCode(127, 0)), 2);
  assert.equal(cosine(Float32Array.from([1, 0]), v), 1);
});

test("a question shows only the asked items and the chain between them", () => {
  const ds = dataset();
  const scene = replay([{ kind: "start", id: "aq", share: 0 }, { kind: "add", id: "soul", share: 0 }], { links: ds.edges, ranked: rankPartners(ds.edges), share: 1 });
  assert.deepEqual(scene.nodes, ["aq", "soul", "ar"], "Aristotle is the link between Aquinas and the soul");
});

test("every item gets one link per episode that names it, inside the filters", () => {
  const ds = dataset(), idx = buildIndex(ds), f = defaultFilters(ds);
  const links = episodeLinks(ds, idx, visiblePassages(ds, idx, f), f);
  // Aristotle is named in both episodes, Aquinas twice in the first one.
  const of = (id: string) => links.filter((l) => l.source === id).map((l) => [l.target, l.count]).sort();
  assert.deepEqual(of("ar"), [["e1", 1], ["e2", 1]]);
  assert.deepEqual(of("aq"), [["e1", 2]]);
  assert.deepEqual(itemsOfEpisodes(links, idx).get("e1")?.[0], "aq");

  const narrowed = { ...f, series: new Set(["A"]) };
  assert.deepEqual(episodeLinks(ds, idx, visiblePassages(ds, idx, narrowed), narrowed).filter((l) => l.source === "ar").map((l) => l.target), ["e1"]);
});

test("episode nodes are drawn with their links, and can be hidden", () => {
  const ds = dataset(), idx = buildIndex(ds), f = defaultFilters(ds), vp = visiblePassages(ds, idx, f);
  const links = episodeLinks(ds, idx, vp, f), edges = visibleEdges(ds, idx, vp, f);
  const base = { mode: "full" as const, focus: null, path: null };
  const shown = buildGraph(edges, idx, vp, { ...base, episodes: { links, show: true, everyItem: true } });
  assert.deepEqual(shown.nodes.filter((n) => n.episode).map((n) => n.label), ["A 1", "B 2"]);
  assert.equal(shown.edges.filter((e) => e.kind === "mentioned_in").length, links.length);
  const hidden = buildGraph(edges, idx, vp, { ...base, episodes: { links, show: false, everyItem: true } });
  assert.equal(hidden.nodes.some((n) => n.episode), false);
  assert.equal(hidden.edges.some((e) => e.kind === "mentioned_in"), false);
});

test("a chain between two items never runs through an episode", () => {
  const ds = dataset(), idx = buildIndex(ds), f = defaultFilters(ds), vp = visiblePassages(ds, idx, f);
  const edges = visibleEdges(ds, idx, vp, f);
  assert.equal(edges.some((e) => idx.episode.has(e.source) || idx.episode.has(e.target)), false);
  const path = findPath(edges, "aq", "ar");
  const g = buildGraph(edges, idx, vp, { mode: "full", focus: null, path, episodes: { links: episodeLinks(ds, idx, vp, f), show: true, everyItem: true } });
  assert.equal(g.nodes.some((n) => n.episode), false);
});

test("the publication axis runs from old to new and keeps gaps in publishing", () => {
  const eps = dataset().episodes;
  const scale = publicationScale([...eps, { ...eps[0], id: "e3", date: "2015-02-01" }], 200)!;
  assert.ok(scale.x("2015-01-01") < scale.x("2015-02-01") && scale.x("2015-02-01") < scale.x("2020-01-01"));
  assert.ok(Math.abs(scale.x("2015-01-01") + 100) < 1e-9);
  assert.ok(Math.abs(scale.x("2020-01-01") - 100) < 1e-9);
  // One month of five years, so well under the even spacing of equal steps.
  assert.ok(scale.x("2015-02-01") < -90);
  assert.equal(scale.dateAt(scale.x("2015-02-01")), "2015-02-01");
  assert.equal(publicationScale([eps[0]], 100), null);
});

test("an item sits at the average date of its episodes, weighted by passages", () => {
  const ds = dataset(), idx = buildIndex(ds), f = defaultFilters(ds), vp = visiblePassages(ds, idx, f);
  const scale = publicationScale(ds.episodes, 100)!;
  const at = itemPositions(episodeLinks(ds, idx, vp, f), idx, scale);
  assert.equal(at.get("aq"), scale.x("2015-01-01"));
  assert.ok(Math.abs(at.get("ar")!) < 1e-9);
});

test("axis labels stay on the axis and fit the room", () => {
  const scale = publicationScale(dataset().episodes, 1000)!;
  const ticks = axisTicks(scale, scale.left - 500, scale.right + 500, 1000);
  assert.ok(ticks.length > 1 && ticks.length < 16);
  assert.ok(ticks.every((t) => t.x >= scale.left && t.x <= scale.right));
  assert.deepEqual(axisTicks(scale, scale.right + 10, scale.right + 20, 500), []);
});

test("a share for an episode brings in its strongest items", () => {
  const ds = dataset(), idx = buildIndex(ds), f = defaultFilters(ds), vp = visiblePassages(ds, idx, f);
  const links = episodeLinks(ds, idx, vp, f), items = itemsOfEpisodes(links, idx), edges = visibleEdges(ds, idx, vp, f);
  const input = { links: edges, ranked: rankPartners(edges), share: 0, episodeItems: items };
  const scene = replay([{ kind: "start", id: "aq" }, { kind: "share", id: "e2", share: 0.1 }], input);
  assert.equal(scene.nodes.length, 2);
  assert.ok(items.get("e2")!.includes(scene.nodes[1]));
});

test("the results page finds names, words with their endings and keeps them apart", () => {
  const ds = dataset();
  ds.passages[0].text = "Love and lovers, said Aquinas. Clover is not it.";
  ds.passages[2].text = "Nothing here.";
  const idx = buildIndex(ds), f = defaultFilters(ds), vp = visiblePassages(ds, idx, f);
  const names = nameIndex(ds);
  const word = findAll(ds, idx, vp, names, "love", null);
  assert.deepEqual(word.passages.map((p) => p.id), ["p1"]);
  // "Clover" is not a match, the word has to start with the query.
  assert.deepEqual(word.passages[0].ranges.map((r) => ds.passages[0].text!.slice(r.start, r.end)), ["Love", "lovers"]);
  assert.deepEqual(word.episodes.map((e) => e.episode.id), ["e1"]);
  assert.equal(word.groups.length, 0);

  const named = findAll(ds, idx, vp, names, "aq", null);
  assert.deepEqual(named.groups.flatMap((g) => g.items.map((r) => r.entity.id)), ["aq"]);
  assert.deepEqual(named.passages.map((p) => p.id), ["p1", "p2"]);
});

test("a question on the results page counts its items and leaves the rest to meaning", () => {
  const ds = dataset();
  const idx = buildIndex(ds), f = defaultFilters(ds), vp = visiblePassages(ds, idx, f);
  const vector = Float32Array.from([1, 0]);
  const meaning = { vector, topics: [{ id: "soul", score: 0.7 }],
    passages: new Map([["p1", Int8Array.from([0, 100])], ["p2", Int8Array.from([0, 100])], ["p3", Int8Array.from([100, 0])]]) };
  const r = findAll(ds, idx, vp, nameIndex(ds), "what did aq and ar think?", meaning);
  assert.deepEqual(r.passages[0].id, "p1");
  assert.deepEqual(r.passages[0].items.sort(), ["aq", "ar"]);
  assert.deepEqual(r.closeItems.map((c) => c.entity.id), ["soul"]);
  assert.ok(termPattern("love").test("Beloved love"));
});

test("after a layout no node lies on another, and pinned nodes and boxes are respected", () => {
  const nodes: SimNode[] = Array.from({ length: 200 }, (_, i) => ({ id: `n${i}`, r: 10, x: i % 7, y: i % 5, box: [-500, 500, 10, 500] as [number, number, number, number] }));
  nodes.push({ id: "pin", r: 30, x: 0, y: 30, fx: 0, fy: 30 });
  assert.equal(separate(nodes), true);
  for (const a of nodes) for (const b of nodes) {
    if (a !== b) assert.ok(Math.hypot(a.x! - b.x!, a.y! - b.y!) >= a.r + b.r - 0.5, `${a.id} and ${b.id} overlap`);
  }
  assert.equal(nodes.at(-1)!.x, 0, "a pinned node stays");
  assert.ok(nodes.slice(0, 200).every((n) => n.y! >= 10), "nodes stay in their box");
  assert.equal(separate(nodes), false, "nothing left to do");
  // Two pinned items give way to each other, an episode at its date never does.
  const pinned: SimNode[] = [{ id: "a", r: 10, x: 0, y: 50, fx: 0, fy: 50, box: [-99, 99, 0, 99] }, { id: "b", r: 10, x: 5, y: 50, fx: 5, fy: 50, box: [-99, 99, 0, 99] },
    { id: "e", r: 10, x: 0, y: 60, fx: 0, fy: 60 }];
  separate(pinned);
  assert.deepEqual([pinned[2].x, pinned[2].y], [0, 60]);
  assert.ok(Math.hypot(pinned[0].x! - pinned[1].x!, pinned[0].y! - pinned[1].y!) >= 20 - 0.5);
  assert.equal(pinned[0].fx, pinned[0].x, "a moved pin stays pinned where it went");
  // keep overrides r when it is set.
  const wide: SimNode[] = [{ id: "a", r: 50, keep: 5, x: 0, y: 0 }, { id: "b", r: 50, keep: 5, x: 20, y: 0 }];
  assert.equal(separate(wide), false);
});

test("zoom changes the distance between nodes more than their size", () => {
  const size = (zoom: number) => screenDiameter(4, 1, false, zoom);
  assert.equal(size(1), 17.4);
  assert.equal(size(0.25), 8.7, "zooming out to a quarter only halves it");
  assert.equal(screenDiameter(0, 1, false, 0.01), SMALLEST_NODE, "never below the minimum");
  assert.ok(size(4) < 2 * size(1), "zooming in opens up the space more than it grows the node");
  assert.ok(screenDiameter(100, 1, false, 0.1) > 2 * SMALLEST_NODE, "much discussed items stay larger");
});

test("a long archive does not get larger nodes than a short one", () => {
  const size = (passages: number) => screenDiameter(passages, 1, false, 1);
  setLargestTopic(143);
  const short = size(143);
  assert.equal(short, 9 + Math.sqrt(143) * 4.2, "a short archive keeps its sizes as they are");
  setLargestTopic(150);
  const atLimit = size(150);
  setLargestTopic(3000);
  assert.ok(Math.abs(size(3000) - atLimit) < 1e-9, "the main topic of a long archive is drawn as one with 150 passages");
  assert.ok(size(3000) < 1.05 * short);
  setLargestTopic(1);
});

test("labels are written by importance and never touch each other", () => {
  const c = (id: string, x: number, priority: number) => ({ id, x, y: 100, r: 5, text: "Thomas Aquinas", font: 11, priority });
  // Two at almost the same place: only the important one. A far one is free, one off screen is not.
  const chosen = chooseLabels([c("a", 100, 1), c("b", 110, 9), c("far", 400, 1), c("off", 900, 99)], 800, 600);
  assert.deepEqual([...chosen].sort(), ["b", "far"]);
  // With more room, both can be written.
  assert.equal(chooseLabels([c("a", 100, 1), c("b", 300, 9)], 800, 600).size, 2);
});

test("every view has an address that leads back to it", () => {
  const routes: Route[] = [
    { view: "start", open: null },
    { view: "results", text: "how is Plato linked to Aristotle?", open: "ce-1" },
    { view: "explore", root: null, open: null },
    { view: "explore", root: "ce-1", open: "ce-2" },
    { view: "explore", root: "ce-1", with: "ce-3", open: null },
    { view: "explore", root: "ce-1", with: "ce-3", open: "ce-2" },
    { view: "connection", from: "ce-1", to: "ce-2", open: null },
    { view: "episodes", episode: "ep-1", passage: "p-1", open: null },
    { view: "episodes", episode: null, passage: null, open: null },
    { view: "timeline", open: "ce-9" },
  ];
  for (const r of routes) assert.deepEqual(readRoute(writeRoute(r)), r);
});

test("an address that makes no sense gives the home page", () => {
  for (const hash of ["", "#", "#/nonsense/x", "#/search", "#/connection/one"]) assert.equal(readRoute(hash).view, "start");
});

test("only a change of page goes into the history", () => {
  assert.ok(samePage("#/map", "#/map?open=ce-1"));
  assert.ok(!samePage("#/map", "#/map/ce-1"));
});

test("a lifespan far outside the rest does not set the scale", () => {
  const medieval: [number, number][] = Array.from({ length: 30 }, (_, i) => [900 + i * 20, 960 + i * 20]);
  const ranges: [number, number][] = [[-4000, -3070], [-428, -348], ...medieval];
  const kept = withoutOutliers(ranges);
  assert.ok(!kept.some((r) => r[0] === -4000));
  // Someone merely older than the rest is not an outlier.
  const spread: [number, number][] = Array.from({ length: 30 }, (_, i) => [-500 + i * 80, -440 + i * 80]);
  assert.equal(withoutOutliers(spread).length, 30);
  // Too few to say what the rest is.
  assert.equal(withoutOutliers([[-4000, -3000], [1200, 1250]]).length, 2);
});

test("a node wedged between pinned ones finds a free spot", () => {
  // A row of pinned nodes against the edge of the box, with a free node right on top of one of them.
  const box: [number, number, number, number] = [0, 1000, 100, 2000];
  const wall: SimNode[] = Array.from({ length: 21 }, (_, i) => ({ id: `w${i}`, r: 20, keep: 20, x: i * 45, y: 100, fx: i * 45, fy: 100, box }));
  const free: SimNode = { id: "new", r: 20, keep: 20, x: 450, y: 100, box };
  separate([...wall, free], 50);
  const clear = wall.every((w) => Math.hypot(w.x! - free.x!, w.y! - free.y!) >= 39.9);
  assert.ok(clear);
  assert.ok(free.y! >= 100);
  // The pinned row stays where it is, apart from what the free node's first pushes can do.
  assert.ok(wall.every((w, i) => w.x === i * 45 && w.y === 100));
});

test("the passage being spoken follows the time", () => {
  const starts = [0.5, 100, 178];
  assert.equal(passageAt(starts, 0), -1);
  assert.equal(passageAt(starts, 0.5), 0);
  assert.equal(passageAt(starts, 99.9), 0);
  assert.equal(passageAt(starts, 100), 1);
  assert.equal(passageAt(starts, 5000), 2);
  assert.equal(passageAt([], 5), -1);
});

test("names in an excerpt carry their topic, whether they are marked or not", () => {
  const text = "Thomas read Aristotle and Plato on the soul.";
  const parts = highlight(text, [{ start: 12, end: 21 }], Infinity, [{ start: 0, end: 6, id: "t" }, { start: 12, end: 21, id: "a" }, { start: 26, end: 31, id: "p" }]);
  assert.equal(parts.map((p) => p.text).join(""), text);
  assert.deepEqual(parts.filter((p) => p.id).map((p) => [p.text, p.id, p.mark]), [["Thomas", "t", false], ["Aristotle", "a", true], ["Plato", "p", false]]);
  // Without names nothing changes.
  assert.ok(highlight(text, [{ start: 12, end: 21 }], Infinity).every((p) => !("id" in p)));
});

test("a connection kept as an exploration holds its topics and links, and only grows on request", () => {
  const link = (id: string, source: string, target: string) => ({ id, kind: "relation", source, target, label: "x", status: "nli_supported", evidence: ["p"] }) as unknown as Edge;
  const links = [link("ab", "a", "b"), link("bc", "b", "c"), link("cd", "c", "d"), link("ce", "c", "e")];
  const ranked = rankPartners(links, () => 1);
  // A chain a, b, c with shares of nothing: exactly the chain.
  const chain: Step[] = [{ kind: "start", id: "a", share: 0 }, { kind: "add", id: "b", share: 0 }, { kind: "add", id: "c", share: 0 }];
  const input = { links, ranked, share: 0.5 };
  const first = replay(chain, input);
  assert.deepEqual([...first.nodes].sort(), ["a", "b", "c"]);
  assert.deepEqual(first.tree.map((e) => e.id).sort(), ["ab", "bc"]);
  // More of one end adds to it, and everything stays.
  const grown = replay([...chain, { kind: "share", id: "c", share: 1 }], input);
  assert.deepEqual([...grown.nodes].sort(), ["a", "b", "c", "d", "e"]);
  assert.deepEqual(grown.tree.map((e) => e.id).sort(), ["ab", "bc"]);
});

test("a link's count is of the passages in view", () => {
  const ed = { id: "x", kind: "co_discussed", source: "a", target: "b", count: 3, pmi: 1, evidence: ["p1", "p2", "p3"] } as CoEdge;
  const [inView] = weighted([ed], new Set(["p1", "p3"]), new Map([["p1", 1], ["p2", 1], ["p3", 1]]));
  assert.equal(inView.count, 2);
});

test("the dots of a cut excerpt stay outside a linked name", () => {
  const text = "One two three four five Aristotle six seven eight nine ten.";
  // The window starts right at the name, which is not the marked word.
  const parts = highlight(text, [{ start: 34, end: 37 }], 14, [{ start: 24, end: 33, id: "a" }]);
  assert.deepEqual(parts.slice(0, 2).map((p) => p.text), ["…", "Aristotle"]);
  assert.equal(parts[1].id, "a");
});

test("two topics together bring what they share first, with both its links", () => {
  const co = (id: string, source: string, target: string, count: number) =>
    ({ id, kind: "co_discussed", source, target, count, pmi: 1, weight: count, evidence: Array.from({ length: count }, (_, i) => `p${i}`) }) as unknown as Edge;
  const links = [co("ab", "a", "b", 50),
    // Strong with one only: this tops the list of a, and is not what they have in common.
    co("a-solo", "a", "solo", 40),
    // Balanced and shared.
    co("a-virtue", "a", "virtue", 7), co("b-virtue", "b", "virtue", 9),
    // Shared, but nearly all from one side.
    co("a-lop", "a", "lop", 30), co("b-lop", "b", "lop", 1)];
  const ranked = rankPartners(links, () => 1);
  assert.deepEqual(sharedPartners("a", "b", ranked).map((s) => [s.id, s.a, s.b]), [["virtue", 7, 9], ["lop", 30, 1]]);
  const steps: Step[] = [{ kind: "start", id: "a", share: 0 }, { kind: "together", id: pairKey("a", "b"), a: "a", b: "b", share: 0.5 }];
  const scene = replay(steps, { links, ranked, share: 0.1 });
  assert.deepEqual([...scene.nodes].sort(), ["a", "b", "virtue"]);
  assert.deepEqual(scene.tree.map((e) => e.id).sort(), ["a-virtue", "ab", "b-virtue"]);
  // All the way up means all that is shared. What only one of them has does not come with it.
  const all = replay([...steps, { kind: "share", id: pairKey("a", "b"), share: 1 }], { links, ranked, share: 0.1 });
  assert.deepEqual([...all.nodes].sort(), ["a", "b", "lop", "virtue"]);
  // Undoing the slider gives the same picture as before.
  assert.deepEqual(replay(steps, { links, ranked, share: 0.1 }).nodes, scene.nodes);
});

test("show more goes one step of the slider, from wherever the handle stands", () => {
  assert.equal(nextStep(0.1), 0.2);
  assert.equal(nextStep(0.3), 0.4);
  // Shares that did not come from the slider: the handle stands on the nearest step.
  assert.equal(stepOf(0.02), 0);
  assert.equal(nextStep(0.02), 0.1);
  assert.equal(stepOf(0.07), 10);
  assert.equal(nextStep(0.07), 0.2);
  assert.equal(nextStep(0.95), 1);
  assert.equal(nextStep(1), 1);
  // Adding up tenths must not skip or repeat a step.
  let share = 0;
  const seen = [];
  for (let i = 0; i < 10; i++) { share = nextStep(share); seen.push(stepOf(share)); }
  assert.deepEqual(seen, [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]);
});

test("episode marks stay apart however many episodes a podcast has", () => {
  const WIDTH = 880;
  // This archive: full size on two rows.
  const few = episodeMarks(25, WIDTH);
  assert.equal(few.size, 22);
  assert.equal(few.rows, 2);
  for (const scale of [0.6, 1, 1.8]) for (const count of [2, 25, 80, 120, 300, 712, 2000]) {
    const { size, rows, rowGap } = episodeMarks(count, WIDTH, scale);
    const pitch = WIDTH / (count - 1);
    // As drawn: the size setting and the border. The view may end up 10% further out than planned.
    const drawn = size * scale + 3, out = 0.9;
    assert.ok(pitch * rows * out >= drawn, `${count} at ${scale}: same row`);
    assert.ok(rowGap * out >= drawn, `${count} at ${scale}: between rows`);
    assert.ok(size >= 7, `${count} at ${scale}: still visible`);
  }
});
