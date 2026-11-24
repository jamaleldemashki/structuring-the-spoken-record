// Which links are drawn: one percentage, the same for every item.
//
// A fixed number per item treats items unfairly: Thomas Aquinas has almost 500 partners,
// someone named twice has ten. So the slider gives every item the same share of its own links.
// At 20%, Aquinas may show about 96 of his 482 and a rare item 2 of its 10.
//
// 1. Strength. A shared passage counts 1 / (n - 1), where n is the number of items named in it
//    (Newman 2001, weighting of co-authorship). Two thinkers discussed in a passage of their
//    own count fully; 24 names listed in a Q&A answer count very little each.
// 2. Quota. Every item may show ceil(share x its links), at least one.
// 3. Choice. Links are taken strongest first, and one is drawn only if both of its items still
//    have room. So no item goes over its quota, and the number on the slider stays true for
//    hubs too. Keeping a link when just one end wants it was tried first: 361 small items
//    each had Aquinas as their strongest link, so he kept 375 of 482 links at 10%.
// 4. Ties are common, most pairs share one passage. Among them the pair with more links in total
//    comes first, which lets a hub reach its quota before its partners run out of room.
// 5. An item left with no link at all gets its strongest one, by preference to a partner that
//    still has room, so nothing that has links falls off the map.
//
// Named relations are always drawn and use up quota first. Everything is worked out from the
// passages in view, so the choice follows the filters.
import type { CoEdge, Edge, Index } from "./types.ts";

/** How much one shared passage counts for each pair of items named in it. */
export function passageWeights(idx: Index, edges: Edge[]): Map<string, number> {
  // Only items that can be "discussed together" count towards how crowded a passage is.
  const partners = new Set<string>();
  for (const ed of edges) if (ed.kind === "co_discussed") partners.add(ed.source).add(ed.target);
  const out = new Map<string, number>();
  for (const [pid, inside] of idx.passageEntities) {
    let n = 0;
    for (const id of inside) if (partners.has(id)) n++;
    if (n > 1) out.set(pid, 1 / (n - 1));
  }
  return out;
}

/** The weight of a link from the passages in view. */
export function linkWeight(ed: CoEdge, visible: Set<string>, weights: Map<string, number>): number {
  let w = 0;
  for (const pid of ed.evidence) if (visible.has(pid)) w += weights.get(pid) ?? 0;
  return w;
}

/** The "discussed together" links with their weight from the passages in view. */
export function weighted(links: CoEdge[], visible: Set<string>, weights: Map<string, number>): CoEdge[] {
  // The count is of the passages in view too, so a link says the same on every screen under a filter.
  return links.map((ed) => ({ ...ed, weight: linkWeight(ed, visible, weights), count: ed.evidence.filter((p) => visible.has(p)).length })).filter((ed) => ed.weight > 0);
}

/** Per item the other items it has a link to. Two links to one partner count once. */
export function partnersOf(links: Edge[]): Map<string, Set<string>> {
  const out = new Map<string, Set<string>>();
  const add = (a: string, b: string) => { if (!out.has(a)) out.set(a, new Set()); out.get(a)!.add(b); };
  for (const ed of links) { add(ed.source, ed.target); add(ed.target, ed.source); }
  return out;
}

/** How many links of an item the share allows, at least one. */
export const quota = (share: number, links: number) => Math.max(1, Math.ceil(share * links - 1e-9));

/** The links to draw for this share (see the top of this file). Links must carry their weight. */
export function drawnLinks(links: Edge[], share: number): Edge[] {
  const partners = partnersOf(links);
  const degree = (id: string) => partners.get(id)?.size ?? 0;
  const room = new Map([...partners.keys()].map((id) => [id, quota(share, degree(id))]));
  const take = (id: string) => room.set(id, room.get(id)! - 1);
  const drawn: Edge[] = [];
  const linked = new Set<string>();
  // Quota counts partners. A pair with a named relation and shared passages uses it only once.
  const pairKey = (ed: Edge) => (ed.source < ed.target ? `${ed.source}|${ed.target}` : `${ed.target}|${ed.source}`);
  const drawnPairs = new Set<string>();
  const draw = (ed: Edge) => {
    drawn.push(ed);
    linked.add(ed.source).add(ed.target);
    if (drawnPairs.has(pairKey(ed))) return;
    drawnPairs.add(pairKey(ed));
    take(ed.source);
    take(ed.target);
  };

  for (const ed of links) if (ed.kind === "relation") draw(ed);
  const pairs = links.filter((ed): ed is CoEdge => ed.kind === "co_discussed");
  const total = (ed: CoEdge) => degree(ed.source) + degree(ed.target);
  pairs.sort((a, b) => b.weight! - a.weight! || total(b) - total(a) || a.id.localeCompare(b.id));
  const left: CoEdge[] = [];
  for (const ed of pairs) {
    if (drawnPairs.has(pairKey(ed)) || (room.get(ed.source)! > 0 && room.get(ed.target)! > 0)) draw(ed);
    else left.push(ed);
  }
  // Items still without a link: first a partner with room, otherwise the strongest there is.
  for (const mustHaveRoom of [true, false]) {
    for (const ed of left) {
      const [a, b] = [ed.source, ed.target];
      if (linked.has(a) && linked.has(b)) continue;
      const other = linked.has(a) ? a : linked.has(b) ? b : null;
      if (mustHaveRoom && other !== null && room.get(other)! <= 0) continue;
      draw(ed);
    }
  }
  return drawn;
}

/** Per item how many of its links are not drawn. */
export function hiddenLinks(links: Edge[], drawn: Edge[]): Map<string, number> {
  const all = partnersOf(links), shown = partnersOf(drawn);
  return new Map([...all].map(([id, set]) => [id, set.size - (shown.get(id)?.size ?? 0)]));
}

// One item at a time. The percentage above is only the calm starting picture. Selecting an item
// gives a choice for that item alone: it then shows exactly its strongest share of partners, and
// the links of all other items stay as they are, so the rest of the graph does not change.

export interface RankedPartner {
  id: string;
  edges: Edge[];
}

/** Per item its partners, strongest first: named relations, then weight, then the more discussed partner. */
export function rankPartners(links: Edge[], mentions: (id: string) => number = () => 0): Map<string, RankedPartner[]> {
  const score = (ed: Edge) => (ed.kind === "relation" ? Infinity : ed.weight ?? ed.count);
  const out = new Map<string, Map<string, RankedPartner & { score: number }>>();
  const add = (a: string, b: string, ed: Edge) => {
    if (!out.has(a)) out.set(a, new Map());
    const row = out.get(a)!.get(b);
    if (row) { row.edges.push(ed); row.score = Math.max(row.score, score(ed)); }
    else out.get(a)!.set(b, { id: b, edges: [ed], score: score(ed) });
  };
  for (const ed of links) { add(ed.source, ed.target, ed); add(ed.target, ed.source, ed); }
  return new Map([...out].map(([id, partners]) => [id, [...partners.values()]
    .sort((x, y) => y.score - x.score || mentions(y.id) - mentions(x.id) || x.id.localeCompare(y.id))]));
}

/** One step of the sliders, in percent. Double-click and the menu move by the same step. */
export const SHARE_STEP = 10;

/** The step a share stands on, in percent. A share between two steps counts as the nearest. */
export const stepOf = (share: number) => Math.round((share * 100) / SHARE_STEP) * SHARE_STEP;

/** The share one step further on, whichever way the current one came about. */
export const nextStep = (share: number) => Math.min(1, (stepOf(share) + SHARE_STEP) / 100);

/** How many of n a share gives. Unlike the quota, an item the user turned down can show none. */
export const portionOf = (share: number, n: number) => (share <= 0 ? 0 : quota(share, n));

/**
 * The links to draw once some items have their own share. A link of such an item is drawn only
 * if it is among the strongest share of that item, or of its other end if that has one too.
 * Links between other items stay as they were.
 */
export function applyShares(base: Edge[], ranked: Map<string, RankedPartner[]>, shares: Map<string, number>): Edge[] {
  if (!shares.size) return base;
  const chosen = new Map<string, Edge>();
  for (const [id, share] of shares) {
    const partners = ranked.get(id) ?? [];
    for (const p of partners.slice(0, portionOf(share, partners.length))) for (const ed of p.edges) chosen.set(ed.id, ed);
  }
  const kept = base.filter((ed) => !shares.has(ed.source) && !shares.has(ed.target));
  return [...kept, ...chosen.values()];
}
