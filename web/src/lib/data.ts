import { configureTypes } from "./entityStyle.ts";
import { SCHEMA_VERSION, type Dataset, type Edge, type Index } from "./types.ts";

const FILES = ["site", "episodes", "passages", "entities", "edges", "related", "summary"] as const;

/**
 * Where the archive's data lives. One build can serve several podcasts side by side:
 * ?archive=<id> reads ./data/<id>/, otherwise ./data/.
 */
export function dataBase(search = location.search): string {
  const id = new URLSearchParams(search).get("archive");
  return id && /^[a-z0-9][a-z0-9-]{0,63}$/.test(id) ? `./data/${id}` : "./data";
}

/** Fetch the export files, refusing data written for a different contract version. */
export async function loadDataset(base = dataBase()): Promise<Dataset> {
  const parts = await Promise.all(
    FILES.map(async (name) => {
      const response = await fetch(`${base}/${name}.json`);
      if (!response.ok) throw new Error(`Could not load ${name}.json (${response.status}). Run "npm run data" with a current export first.`);
      return [name, await response.json()] as const;
    }),
  );
  const ds = Object.fromEntries(parts) as unknown as Dataset;
  if (ds.site.schema_version !== SCHEMA_VERSION)
    throw new Error(`This data was exported with schema version ${ds.site.schema_version}, but the app reads version ${SCHEMA_VERSION}. Re-run the export-web stage.`);
  configureTypes(ds.site);
  return ds;
}

export function buildIndex(ds: Dataset): Index {
  const entityPassages = new Map<string, Set<string>>();
  const passageEntities = new Map<string, Set<string>>();
  for (const e of ds.entities) {
    const set = new Set(e.occurrences.map((o) => o.passage_id));
    entityPassages.set(e.id, set);
    for (const p of set) {
      if (!passageEntities.has(p)) passageEntities.set(p, new Set());
      passageEntities.get(p)!.add(e.id);
    }
  }
  const edgesOf = new Map<string, Edge[]>();
  for (const ed of ds.edges) {
    for (const id of [ed.source, ed.target]) {
      if (!edgesOf.has(id)) edgesOf.set(id, []);
      edgesOf.get(id)!.push(ed);
    }
  }
  return {
    site: ds.site,
    episode: new Map(ds.episodes.map((e) => [e.id, e])),
    passage: new Map(ds.passages.map((p) => [p.id, p])),
    entity: new Map(ds.entities.map((e) => [e.id, e])),
    entityPassages,
    passageEntities,
    edgesOf,
  };
}
