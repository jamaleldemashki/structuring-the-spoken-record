// Shapes of the files written by the pipeline's export-web stage (src/spoken_record/web_export.py).
// The app is generic: everything archive-specific arrives in site.json.

/** Data version this app reads. Must match the export's schema_version. */
export const SCHEMA_VERSION = 2;

/** Entity types are whatever the archive defines. site.json lists them in display order. */
export type EntityType = string;

export interface TypeDisplay {
  type: EntityType;
  label: string;
  plural: string;
  visible_by_default: boolean;
}

export interface TimePreset {
  label: string;
  start: number;
  end: number;
}

export interface Site {
  schema_version: number;
  archive_id: string;
  title: string;
  subtitle: string;
  /** Null when the archive has no such episode grouping. */
  series_label: string | null;
  era_label: string | null;
  entity_types: TypeDisplay[];
  /** The type whose lifespans drive the lived filter and timeline. null turns both off. */
  lifespan_type: EntityType | null;
  lived_presets: TimePreset[];
  search_placeholder: string;
  /** Default share of each item's "discussed together" weight that is drawn (0 to 1). */
  co_discussed_share?: number;
  /** Headings that group the named relations of a topic, in this order. */
  relation_groups?: string[];
}

export interface Episode {
  id: string;
  code: string;
  title: string;
  date: string | null;
  series: string | null;
  era: string | null;
  audio_url: string | null;
  source_url: string | null;
  passage_count: number;
}

export interface Passage {
  id: string;
  episode_id: string;
  index: number;
  start_s: number | null;
  end_s: number | null;
  word_count: number;
  text?: string;
  /** The transcript segments: character range in the text (can reach past either end) and times. */
  timing?: [number, number, number, number][];
}

export interface Occurrence {
  passage_id: string;
  start: number;
  end: number;
}

/** Widest plausible lifespan from Wikidata. Negative years are BCE. */
export interface Life {
  start: number | null;
  /** Null when the person is living or the end is unknown. */
  end: number | null;
  living: boolean;
  approximate: boolean;
  /** "floruit" means only active years are known. */
  basis: "birth_death" | "floruit" | "mixed";
}

export interface Entity {
  id: string;
  label: string;
  type: EntityType;
  aliases: string[];
  qid: string | null;
  link_status: "not_attempted" | "linked" | "nil" | "ambiguous";
  mention_count: number;
  episode_ids: string[];
  resolution_status: string;
  occurrences: Occurrence[];
  /** Present only when the enrichment stage found dates. */
  life?: Life;
  /** Wikidata image of the item, hosted on Wikimedia Commons. */
  image?: CommonsImage;
  /** A general word ("philosopher") rather than a name. It has no edges in the graph. */
  generic?: boolean;
}

export interface CommonsImage {
  file: string;
  /** 250 px wide thumbnail. */
  thumb: string;
  original: string;
  /** File page with author and licence. */
  page: string;
}

export interface RelationEdge {
  id: string;
  kind: "relation";
  source: string;
  target: string;
  predicate: string;
  label: string;
  /** The relation read from its object ("influenced" for influenced_by). Older exports lack it. */
  inverse_label?: string;
  /** The heading it is listed under, for example "Positions". Without one it is a named relation. */
  group?: string;
  status: "nli_supported" | "unverified";
  confidence: number;
  evidence: string[];
}

export interface CoEdge {
  id: string;
  kind: "co_discussed";
  source: string;
  target: string;
  count: number;
  pmi: number;
  evidence: string[];
  /** Weight from the passages in view, set by the app when it picks the links (strength.ts). */
  weight?: number;
}

export type Edge = RelationEdge | CoEdge;

export interface Related {
  source: string;
  target: string;
  score: number;
  rank: number;
  shared_entity_ids: string[];
}

export interface Summary {
  episodes: number;
  passages: number;
  entities: number;
  relation_edges: number;
  co_discussed_edges: number;
  related_passages: number;
  relation_status: "nli_supported" | "unverified" | "none";
  discovery_method: string | null;
  lifespans: "not_enriched" | "enriched";
  dated_entities?: number;
}

export interface Dataset {
  site: Site;
  episodes: Episode[];
  passages: Passage[];
  entities: Entity[];
  edges: Edge[];
  related: Related[];
  summary: Summary;
}

/** Lookup tables built once after loading. */
export interface Index {
  site: Site;
  episode: Map<string, Episode>;
  passage: Map<string, Passage>;
  entity: Map<string, Entity>;
  /** Distinct passages per entity. */
  entityPassages: Map<string, Set<string>>;
  /** Entities mentioned per passage. */
  passageEntities: Map<string, Set<string>>;
  /** Visible-or-not edges touching each entity. */
  edgesOf: Map<string, Edge[]>;
  /** How much one shared passage counts for each pair named in it (strength.ts). */
  passageWeight: Map<string, number>;
}
