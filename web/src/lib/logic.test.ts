// Run with: npm test  (Node's built-in test runner strips the TypeScript types).
import { test } from "node:test";
import assert from "node:assert/strict";
import { buildIndex } from "./data.ts";
import { defaultFilters, edgeVisible, livedOk, livedSelection, visiblePassages, visibleEdges } from "./filters.ts";
import { buildGraph, findPath } from "./graph.ts";
import { formatTime, highlight, lifeLabel, lifeRange } from "./format.ts";
import { configureTypes, typeStyle } from "./entityStyle.ts";
import { lifeAxis, yearLabels } from "./timeAxis.ts";
import { collisionRadius, DEFAULT_SETTINGS, makeSimulation, seed, settle, type SimNode } from "./forces.ts";
import type { Dataset, Entity, Site } from "./types.ts";

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
  assert.deepEqual(f.pub, [2015, 2020]);
  f.pub = [2015, 2016];
  const vp = visiblePassages(ds, idx, f);
  assert.deepEqual([...vp].sort(), ["p1", "p2"]);
  f.minShared = 1;
  assert.equal(edgeVisible(ds.edges[0], idx, vp, f), true);
  assert.equal(edgeVisible(ds.edges[1], idx, vp, f), false);
});

test("shared-passage threshold counts only passages inside the current filters", () => {
  const ds = dataset(), idx = buildIndex(ds), f = defaultFilters(ds);
  const co = { ...ds.edges[1], count: 2, evidence: ["p2", "p3"], source: "aq", target: "soul" } as Dataset["edges"][number];
  assert.equal(f.minShared, 2);
  assert.equal(edgeVisible(co, idx, visiblePassages(ds, idx, f), f), true);
  f.pub = [2015, 2016];
  assert.equal(edgeVisible(co, idx, visiblePassages(ds, idx, f), f), false, "only one shared passage left in view");
  f.minShared = 1;
  assert.equal(edgeVisible(co, idx, visiblePassages(ds, idx, f), f), true);
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
  const idx2 = buildIndex(ds), f = { ...defaultFilters(ds), minShared: 1 };
  const vp = visiblePassages(ds, idx2, f);
  assert.deepEqual(visibleEdges(ds, idx2, vp, f).map((e) => e.id).sort(), ["c1", "c2", "c3", "r1"]);
  f.lived = [1200, 1300];
  // Aristotle is outside, so r1 and c1 go; soul stays through Aquinas, and virtue through soul is dropped.
  assert.deepEqual(visibleEdges(ds, idx2, vp, f).map((e) => e.id), ["c3"]);
});

test("path prefers the shortest chain and reports it in order", () => {
  const ds = dataset(), idx = buildIndex(ds), f = { ...defaultFilters(ds), minShared: 1 };
  const edges = visibleEdges(ds, idx, visiblePassages(ds, idx, f), f);
  const path = findPath(edges, "aq", "soul")!;
  assert.deepEqual(path.nodes, ["aq", "ar", "soul"]);
  assert.deepEqual(path.edges.map((e) => e.id), ["r1", "c1"]);
  assert.equal(findPath(edges, "aq", "anon"), null);
});

test("focused graph keeps the focus and labels only its named relations", () => {
  const ds = dataset(), idx = buildIndex(ds), f = defaultFilters(ds);
  const vp = visiblePassages(ds, idx, f);
  const g = buildGraph(visibleEdges(ds, idx, vp, f), idx, vp, { mode: "focused", focus: "aq", depth: 1, path: null, coPerNode: 5 });
  assert.deepEqual(g.nodes.map((n) => n.id), ["aq", "ar"]);
  assert.equal(g.edges[0].showLabel, true);
  assert.equal(g.edges[0].label, "influenced by ✓");
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
  const nodes: SimNode[] = Array.from({ length: 30 }, (_, i) => ({ id: `n${i}`, r: collisionRadius(`Philosopher ${i}`, i % 7, DEFAULT_SETTINGS) }));
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
  const g = buildGraph(edges, idx, vp, { mode: "full", focus: null, depth: 1, path: null, coPerNode: 5, always });
  assert.deepEqual(g.nodes.map((n) => n.id), ["plato"]);
});

test("entity types are styled by their configured position, so any archive's types work", () => {
  assert.equal(typeStyle("Person").token, "hue-1");
  assert.equal(typeStyle("Work").token, "hue-3");
  assert.equal(typeStyle("Place").token, "neutral-1");
  assert.equal(new Set(site.entity_types.map((t) => typeStyle(t.type).shape)).size, 4, "every type has its own shape");
  assert.equal(typeStyle("Instrument").label, "Instrument", "unknown types fall back to a neutral style");
  assert.equal(defaultFilters(dataset()).types.has("Place"), false);
});

test("time axes fit the data and never label past the last publication year", () => {
  const axis = lifeAxis([[-430, -341], [1225, 1274]]);
  assert.ok(axis.lo <= -430 && axis.hi >= 1274 && axis.ticks.length <= 11);
  assert.deepEqual(lifeAxis([[1900, 1950]]).ticks.slice(0, 2), [1890, 1900]);
  assert.deepEqual(yearLabels(2015, 2020, 600), [2015, 2016, 2017, 2018, 2019, 2020]);
  assert.deepEqual(yearLabels(2000, 2025, 300), [2000, 2003, 2006, 2009, 2012, 2015, 2018, 2021, 2024]);
});
