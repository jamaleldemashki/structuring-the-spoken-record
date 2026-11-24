// Asking the archive a question, for example "Is there any relation between Thomas Aquinas
// and Aristotle when it comes to the soul?".
//
// The question is read in two ways. Names are found by matching the words of the question
// against the names of the entities, allowing a small typo ("Auinas"), because for names
// that is more reliable than any vector. What is left ("when it comes to") is not a name;
// those words are compared by meaning with the names of concepts, schools and works, so a
// word that is not in the archive ("mind") can still lead to one that is ("intellect").
// The whole question is then compared with the passages to put the best evidence first.
// Nothing is generated: the answer is the items, their links and the passages.
import type { Dataset, Entity, Index } from "./types.ts";

export interface NameHit {
  entity: Entity;
  /** The words of the question that named it. */
  text: string;
  /** True when the words were a near miss (a typo) rather than the exact name. */
  fuzzy: boolean;
}

export interface Parsed {
  names: NameHit[];
  /** Runs of content words that named nothing. They are compared by meaning. */
  rest: string[];
}

// Small words of English that never name anything on their own.
const STOP = new Set(("a an the and or but of to in on at by for with from about into over as is are was were be been being " +
  "do does did done has have had there here this that these those it its they them their he him his she her we us our you " +
  "your i me my what which who whom whose when where why how any some all each every no not than then so if whether can could " +
  "would should will shall may might must very more most also just only between among comes come came").split(" "));
// Words that frame a question but are not its subject ("the relation between A and B").
const FRAME = new Set(("relation relations relationship relationships connection connections connected link links linked " +
  "difference differences differ compare compared comparison similar similarity similarities related relate tell show find " +
  "anything something think thought say says said view views discuss discussed discussion talk talks talked mention mentioned " +
  "according regarding concerning respect matter question").split(" "));

export function words(text: string): string[] {
  return text.toLowerCase().normalize("NFKD").replace(/[̀-ͯ]/g, "").match(/[a-z0-9]+(?:['’][a-z]+)?/g) ?? [];
}

const content = (w: string) => !STOP.has(w) && !FRAME.has(w);

/** Edit distance with swaps, stopped early once it is over `max`. */
export function distance(a: string, b: string, max: number): number {
  if (Math.abs(a.length - b.length) > max) return max + 1;
  let prev2: number[] = [], prev = Array.from({ length: b.length + 1 }, (_, j) => j);
  for (let i = 1; i <= a.length; i++) {
    const row = [i];
    let best = i;
    for (let j = 1; j <= b.length; j++) {
      let d = Math.min(prev[j] + 1, row[j - 1] + 1, prev[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
      if (i > 1 && j > 1 && a[i - 1] === b[j - 2] && a[i - 2] === b[j - 1]) d = Math.min(d, prev2[j - 2] + 1);
      row.push(d);
      best = Math.min(best, d);
    }
    if (best > max) return max + 1;
    [prev2, prev] = [prev, row];
  }
  return prev[b.length];
}

/** Typos allowed for a name of this length: none for short names, where a typo is another word. */
const allowed = (length: number) => (length >= 9 ? 2 : length >= 5 ? 1 : 0);

/**
 * The names of every item that can be shown, as word lists. An item is reachable by its
 * label and every other spelling the podcast used. General words ("philosopher") are left
 * out, they have no links to show.
 */
export function nameIndex(ds: Dataset): Map<string, Entity[]> {
  const out = new Map<string, Entity[]>();
  for (const e of ds.entities) {
    if (e.generic) continue;
    for (const name of new Set([e.label, ...e.aliases])) {
      const key = words(name).filter((w, i, all) => !(i === 0 && w === "the" && all.length > 1)).join(" ");
      if (!key || words(name).every((w) => STOP.has(w))) continue;
      out.set(key, [...(out.get(key) ?? []), e]);
    }
  }
  // When one spelling fits several items, the one the podcast talks about most comes first.
  for (const list of out.values()) list.sort((a, b) => b.mention_count - a.mention_count);
  return out;
}

/**
 * Names in the question, longest first so "Thomas Aquinas" wins over "Thomas", and the
 * content words left over. Each item is found once.
 */
export function parseQuestion(question: string, names: Map<string, Entity[]>): Parsed {
  const ws = words(question);
  const used = new Array(ws.length).fill(false);
  const found = new Map<string, NameHit>();
  const byLength = new Map<number, string[]>();
  for (const key of names.keys()) {
    const n = key.split(" ").length;
    byLength.set(n, [...(byLength.get(n) ?? []), key]);
  }
  for (let n = 4; n >= 1; n--) {
    for (let i = 0; i + n <= ws.length; i++) {
      if (used.slice(i, i + n).some(Boolean)) continue;
      const gram = ws.slice(i, i + n);
      // A name has to start and end with a content word: "of Hales" or "the soul of" are not names.
      if (!content(gram[0]) || !content(gram[n - 1])) continue;
      const key = gram.join(" ");
      let hit = names.get(key)?.[0];
      let fuzzy = false;
      if (!hit && allowed(key.length)) {
        let best = allowed(key.length) + 1;
        for (const other of byLength.get(n) ?? []) {
          const d = distance(key, other, best - 1);
          if (d < best) { best = d; hit = names.get(other)![0]; fuzzy = true; }
        }
      }
      if (!hit || found.has(hit.id)) continue;
      found.set(hit.id, { entity: hit, text: key, fuzzy });
      used.fill(true, i, i + n);
    }
  }
  const rest: string[] = [];
  let run: string[] = [];
  ws.forEach((w, i) => {
    if (!used[i] && content(w)) run.push(w);
    else if (run.length) { rest.push(run.join(" ")); run = []; }
  });
  if (run.length) rest.push(run.join(" "));
  return { names: [...found.values()], rest };
}

/** Vectors as stored in search.json: int8 rows, base64 encoded. */
export function decodeVectors(data: string, dimension: number): Int8Array[] {
  const bytes = Uint8Array.from(atob(data), (c) => c.charCodeAt(0));
  const all = new Int8Array(bytes.buffer);
  const out: Int8Array[] = [];
  for (let i = 0; i + dimension <= all.length; i += dimension) out.push(all.subarray(i, i + dimension));
  return out;
}

/** Cosine of a normalized float vector with a stored int8 one. */
export function cosine(q: ArrayLike<number>, v: Int8Array): number {
  let dot = 0, norm = 0;
  for (let i = 0; i < v.length; i++) { dot += q[i] * v[i]; norm += v[i] * v[i]; }
  return norm ? dot / Math.sqrt(norm) : 0;
}

/** Items to show for the words that named nothing: the closest topic names by meaning. */
export function topicsByMeaning(vectors: Float32Array[], topics: { ids: string[]; vectors: Int8Array[] },
  floor: number, perPhrase = 1): { id: string; score: number }[] {
  const out = new Map<string, number>();
  for (const q of vectors) {
    topics.ids.map((id, i) => ({ id, score: cosine(q, topics.vectors[i]) }))
      .filter((t) => t.score >= floor).sort((a, b) => b.score - a.score).slice(0, perPhrase)
      .forEach((t) => out.set(t.id, Math.max(out.get(t.id) ?? 0, t.score)));
  }
  return [...out].map(([id, score]) => ({ id, score })).sort((a, b) => b.score - a.score);
}

export interface Evidence {
  passageId: string;
  /** The asked-about items this passage mentions. */
  ids: string[];
  score: number;
}

/**
 * Passages that mention the items of the question, the ones that mention more of them
 * first, and among those the ones closest to the question in meaning. Without a question
 * vector (the model could not load) the order is by how many items are mentioned. With no
 * items, every passage is ranked by meaning.
 */
export function rankEvidence(idx: Index, ids: string[], visible: Set<string>, question: Float32Array | null,
  passages: Map<string, Int8Array> | null, n = 6): Evidence[] {
  const wanted = new Set(ids);
  const out: Evidence[] = [];
  for (const [pid, inside] of idx.passageEntities) {
    if (!visible.has(pid)) continue;
    const hit = [...inside].filter((id) => wanted.has(id));
    // A question that names nothing is answered by meaning alone, over all passages.
    if (!hit.length && (wanted.size || !question)) continue;
    const v = passages?.get(pid);
    out.push({ passageId: pid, ids: hit, score: question && v ? cosine(question, v) : 0 });
  }
  // One item more counts for more than any difference in meaning.
  return out.sort((a, b) => b.ids.length - a.ids.length || b.score - a.score || a.passageId.localeCompare(b.passageId)).slice(0, n);
}

/** A question reads as a question, not as a name: several words, or a question mark. */
export const isQuestion = (text: string) => text.includes("?") || words(text).length >= 4;
