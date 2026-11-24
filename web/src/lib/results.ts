// The results page: everything the archive holds for what was typed, for one word ("love") as
// well as for a question.
//
// Three kinds of finding are kept apart, so it is always clear why something is listed:
//   by name     items whose name or other spelling contains the words
//   by word     passages that contain the words or name the items of a question
//   by meaning  items and passages close to it in the sentence model, even without the word
// Nothing is generated. Each finding opens the item or the passage itself.
import { cosine, isQuestion, parseQuestion, words } from "./ask.ts";
import { allTypes } from "./entityStyle.ts";
import type { Dataset, Entity, Episode, Index, Occurrence } from "./types.ts";

export interface Meaning {
  /** The vector of the whole text. */
  vector: Float32Array;
  passages: Map<string, Int8Array>;
  /** Items close in meaning to the text or its leftover words, best first. */
  topics: { id: string; score: number }[];
}

export interface ItemResult {
  entity: Entity;
  /** Passages inside the current filters. */
  passages: number;
  /** Present when the item was found by meaning. */
  score?: number;
  /** The words that named it when they were a near miss, for example a typo. */
  readAs?: string;
}

export interface ItemGroup {
  type: string;
  label: string;
  items: ItemResult[];
}

export interface PassageResult {
  id: string;
  /** Items of the query named in the passage. */
  items: string[];
  /** Words of the query found in the passage. */
  terms: string[];
  /** What to mark in the text. */
  ranges: { start: number; end: number }[];
  score: number;
}

export interface EpisodeResult {
  episode: Episode;
  passages: number;
  inTitle: boolean;
}

export interface Results {
  text: string;
  groups: ItemGroup[];
  closeItems: ItemResult[];
  passages: PassageResult[];
  closePassages: PassageResult[];
  episodes: EpisodeResult[];
}

/** Below this a passage is not listed as close in meaning. Queries the archive has nothing on ("pizza", "quantum computing") reach about 0.3 at best, real topics go above it. */
export const PASSAGE_FLOOR = 0.33;
const CLOSE_PASSAGES = 10;
/** Shorter words find too much by accident, "of" or "is" for example. */
const SHORTEST_TERM = 3;

/** Finds a word or phrase and its endings: "love" also finds "loves" and "lover". */
export function termPattern(term: string): RegExp {
  const escaped = term.replace(/[.*+?^${}()|[\]\\]/g, "\\$&").replace(/ /g, "\\s+");
  return new RegExp(`\\b${escaped}[\\p{L}\\p{N}]*`, "giu");
}

/** Sorted ranges without overlaps. A mention and a word match are often the same characters. */
function separate(ranges: { start: number; end: number }[]) {
  const out: { start: number; end: number }[] = [];
  for (const r of [...ranges].sort((a, b) => a.start - b.start || b.end - a.end))
    if (!out.length || r.start >= out.at(-1)!.end) out.push(r);
  return out;
}

export function findAll(ds: Dataset, idx: Index, vp: Set<string>, names: Map<string, Entity[]>, text: string,
  meaning: Meaning | null): Results {
  const parsed = parseQuestion(text, names);
  const asked = isQuestion(text);
  const whole = words(text).join(" ");
  // A question is searched by its content words, a short query also as one phrase.
  const terms = [...new Set([...(asked ? [] : [whole]), ...parsed.rest.flatMap((r) => r.split(" "))])]
    .filter((t) => t.length >= SHORTEST_TERM);
  const patterns = terms.map((t) => [t, termPattern(t)] as const);

  const inView = (e: Entity) => [...(idx.entityPassages.get(e.id) ?? [])].filter((p) => vp.has(p)).length;
  const named = new Map(parsed.names.map((n) => [n.entity.id, n]));
  // Items by name. A question only counts the names recognised in it, not every name that
  // happens to contain one of its words.
  const wholeName = asked || whole.length < SHORTEST_TERM ? null : termPattern(whole);
  const byName = new Map<string, ItemResult>();
  for (const e of ds.entities) {
    const hit = named.get(e.id);
    wholeName && (wholeName.lastIndex = 0);
    if (hit || (wholeName && [e.label, ...e.aliases].some((name) => name.search(wholeName) >= 0)))
      byName.set(e.id, { entity: e, passages: inView(e), readAs: hit?.fuzzy ? hit.text : undefined });
  }
  const exact = (e: Entity) => Number(words(e.label).join(" ") !== whole);
  const groups = allTypes().map((t) => ({ type: t.type, label: t.plural, items: [...byName.values()]
    .filter((r) => r.entity.type === t.type && r.passages > 0)
    .sort((a, b) => Number(Boolean(a.entity.generic)) - Number(Boolean(b.entity.generic)) || exact(a.entity) - exact(b.entity)
      || b.passages - a.passages || a.entity.label.localeCompare(b.entity.label)) })).filter((g) => g.items.length);

  // Passages by word, and by the items a question names.
  const occurrences = new Map<string, { id: string; o: Occurrence }[]>();
  for (const id of named.keys()) {
    for (const o of idx.entity.get(id)!.occurrences) occurrences.set(o.passage_id, [...(occurrences.get(o.passage_id) ?? []), { id, o }]);
  }
  const passages: PassageResult[] = [];
  const others: { id: string; score: number }[] = [];
  for (const p of ds.passages) {
    if (!vp.has(p.id)) continue;
    const v = meaning?.passages.get(p.id);
    const score = meaning && v ? cosine(meaning.vector, v) : 0;
    const ranges: { start: number; end: number }[] = [];
    const found: string[] = [];
    for (const [term, pattern] of p.text ? patterns : []) {
      const hits = [...p.text!.matchAll(pattern)];
      // The word that named an item is listed as the item, not a second time as a word.
      if (hits.length && !parsed.names.some((n) => n.text === term)) found.push(term);
      for (const m of hits) ranges.push({ start: m.index, end: m.index + m[0].length });
    }
    const mentioned = occurrences.get(p.id) ?? [];
    if (found.length || mentioned.length) {
      ranges.push(...mentioned.map(({ o }) => ({ start: o.start, end: o.end })));
      passages.push({ id: p.id, items: [...new Set(mentioned.map((m) => m.id))], terms: found, ranges: separate(ranges), score });
    } else if (score >= PASSAGE_FLOOR) others.push({ id: p.id, score });
  }
  // More of the query inside one passage counts first, then closeness in meaning.
  const covered = (r: PassageResult) => r.items.length + r.terms.length;
  passages.sort((a, b) => covered(b) - covered(a) || b.score - a.score || b.ranges.length - a.ranges.length || a.id.localeCompare(b.id));
  const closePassages = others.sort((a, b) => b.score - a.score || a.id.localeCompare(b.id)).slice(0, CLOSE_PASSAGES)
    .map(({ id, score }) => ({ id, items: [], terms: [], ranges: [], score }));

  const closeItems = (meaning?.topics ?? []).filter((t) => !byName.has(t.id) && idx.entity.has(t.id))
    .map((t) => { const entity = idx.entity.get(t.id)!; return { entity, passages: inView(entity), score: t.score }; })
    .filter((r) => r.passages > 0);

  const counts = new Map<string, number>();
  for (const r of passages) {
    const ep = idx.passage.get(r.id)!.episode_id;
    counts.set(ep, (counts.get(ep) ?? 0) + 1);
  }
  const titled = (ep: Episode) => patterns.some(([, pattern]) => ep.title.search(pattern) >= 0);
  const episodes = ds.episodes.map((episode) => ({ episode, passages: counts.get(episode.id) ?? 0, inTitle: titled(episode) }))
    .filter((r) => (r.passages || r.inTitle) && ds.passages.some((p) => p.episode_id === r.episode.id && vp.has(p.id)))
    .sort((a, b) => Number(b.inTitle) - Number(a.inTitle) || b.passages - a.passages || (a.episode.date ?? "").localeCompare(b.episode.date ?? ""));

  return { text, groups, closeItems, passages, closePassages, episodes };
}
