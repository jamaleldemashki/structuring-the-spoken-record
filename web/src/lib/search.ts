// Search over the archive and the suggestions on the start page. Both only read the
// exported data, so they work the same for any podcast.
import { allTypes } from "./entityStyle.ts";
import { findPath, type PathResult } from "./graph.ts";
import { lifeRange } from "./format.ts";
import type { Dataset, Edge, Entity, Episode, Index, Passage, TimePreset } from "./types.ts";

export interface EntityGroup {
  kind: "entities";
  label: string;
  type: string;
  items: Entity[];
}

export interface EpisodeGroup {
  kind: "episodes";
  label: string;
  items: Episode[];
}

export interface PassageHit {
  passage: Passage;
  episode: Episode;
  /** Where the query was found, relative to the passage text. */
  start: number;
  end: number;
}

export interface PassageGroup {
  kind: "passages";
  label: string;
  items: PassageHit[];
}

export type ResultGroup = EntityGroup | EpisodeGroup | PassageGroup;

const PER_TYPE = 4;
const PER_GROUP = 3;
const MIN_TRANSCRIPT_QUERY = 3;

const matches = (text: string, q: string) => text.toLowerCase().includes(q);

/**
 * Results grouped by kind: people, concepts, works and the other types in the archive's
 * order, then episode titles, then lines in the transcripts. Names that start with the
 * query come first, then the ones the podcast mentions most. General words ("philosopher")
 * go last.
 */
export function searchArchive(ds: Dataset, idx: Index, query: string): ResultGroup[] {
  const q = query.trim().toLowerCase();
  if (!q) return [];
  const groups: ResultGroup[] = [];
  const found = ds.entities.filter((e) => matches(e.label, q) || e.aliases.some((a) => matches(a, q)));
  const rank = (e: Entity) => [Number(Boolean(e.generic)), Number(!e.label.toLowerCase().startsWith(q)), -e.mention_count];
  for (const t of allTypes()) {
    const items = found.filter((e) => e.type === t.type)
      .sort((a, b) => { const [x, y] = [rank(a), rank(b)]; return x[0] - y[0] || x[1] - y[1] || x[2] - y[2]; })
      .slice(0, PER_TYPE);
    if (items.length) groups.push({ kind: "entities", label: t.plural, type: t.type, items });
  }
  const episodes = ds.episodes.filter((ep) => matches(ep.title, q) || matches(ep.code, q)).slice(0, PER_GROUP);
  if (episodes.length) groups.push({ kind: "episodes", label: "episodes", items: episodes });
  if (q.length >= MIN_TRANSCRIPT_QUERY) {
    const hits: PassageHit[] = [];
    for (const p of ds.passages) {
      const at = p.text?.toLowerCase().indexOf(q) ?? -1;
      if (at >= 0) hits.push({ passage: p, episode: idx.episode.get(p.episode_id)!, start: at, end: at + q.length });
      if (hits.length === PER_GROUP) break;
    }
    if (hits.length) groups.push({ kind: "passages", label: "in the transcripts", items: hits });
  }
  return groups;
}

/** The most mentioned items of one type, leaving out general words. */
export function mostDiscussed(ds: Dataset, type: string, n: number): Entity[] {
  return ds.entities.filter((e) => e.type === type && !e.generic)
    .sort((a, b) => b.mention_count - a.mention_count || a.label.localeCompare(b.label)).slice(0, n);
}

export type Suggestion =
  | { kind: "connection"; text: string; path: PathResult }
  | { kind: "period"; text: string; preset: TimePreset }
  | { kind: "topic"; text: string; entity: Entity };

/**
 * Example questions for the start page, each worked out from the data so it always has
 * an answer: a chain between two of the most discussed people, the period most of the
 * dated people lived in, and the most discussed idea.
 */
export function suggestions(ds: Dataset, edges: Edge[]): Suggestion[] {
  const out: Suggestion[] = [];
  const lead = ds.site.lifespan_type;
  if (lead) {
    const people = mostDiscussed(ds, lead, 8);
    // Prefer a chain with one step in between: it shows more than a direct link.
    let best: PathResult | null = null;
    for (let i = 0; i < people.length && !(best && best.edges.length === 2); i++) {
      for (let j = i + 1; j < people.length; j++) {
        const path = findPath(edges, people[i].id, people[j].id);
        if (path && (!best || Math.abs(path.edges.length - 2) < Math.abs(best.edges.length - 2))) best = path;
        if (best?.edges.length === 2) break;
      }
    }
    if (best) {
      const name = (id: string) => ds.entities.find((e) => e.id === id)!.label;
      out.push({ kind: "connection", text: `How is ${name(best.nodes[0])} connected to ${name(best.nodes.at(-1)!)}?`, path: best });
    }
    const dated = ds.entities.filter((e) => e.type === lead && e.life);
    const inside = (p: TimePreset) => dated.filter((e) => { const [a, b] = lifeRange(e.life!); return a >= p.start && b <= p.end; }).length;
    const period = [...ds.site.lived_presets].sort((a, b) => inside(b) - inside(a))[0];
    const plural = allTypes().find((t) => t.type === lead)?.plural ?? "people";
    if (period && inside(period)) out.push({ kind: "period", text: `Which ${plural} of the ${period.label} period come up?`, preset: period });
  }
  const topicType = allTypes().find((t) => t.type !== lead && t.visibleByDefault)?.type;
  const topic = topicType ? mostDiscussed(ds, topicType, 1)[0] : undefined;
  if (topic) out.push({ kind: "topic", text: `What does the podcast say about ${topic.label}?`, entity: topic });
  return out;
}
