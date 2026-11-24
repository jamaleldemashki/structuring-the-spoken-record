// The sentence model in the browser, loaded the first time someone asks a question.
//
// search.json names the model and its revision. The model files are served with the app
// from models/ (npm run models puts them there) and the build bundles the ONNX runtime, so
// the app doesn't depend on other servers while it runs. If anything fails, asking still works by
// names alone, only without ranking by meaning.
import { decodeVectors } from "./ask.ts";
import { dataBase } from "./data.ts";
// The CPU build of the ONNX runtime (see vite.config.ts). Importing the files here makes the
// build ship them; otherwise transformers.js would fetch them from a CDN.
import ortWasm from "onnxruntime-web/ort-wasm-simd-threaded.wasm?url";
import ortModule from "onnxruntime-web/ort-wasm-simd-threaded.mjs?url";

export interface SearchIndex {
  model: { repo_id: string; revision: string; dtype: "q8" | "fp32"; max_tokens: number };
  dimension: number;
  passages: Map<string, Int8Array>;
  topics: { ids: string[]; vectors: Int8Array[] };
}

interface RawIndex {
  model: SearchIndex["model"];
  dimension: number;
  passages: { ids: string[]; data: string };
  topics: { ids: string[]; data: string };
}

type Embed = (texts: string[]) => Promise<Float32Array[]>;

let index: Promise<SearchIndex | null> | null = null;
let embedder: Promise<Embed | null> | null = null;

/** The vectors of the archive, or null when the export has none. */
export function loadIndex(base = dataBase()): Promise<SearchIndex | null> {
  index ??= fetch(`${base}/search.json`).then(async (r) => {
    if (!r.ok) return null;
    const raw = (await r.json()) as RawIndex;
    const passages = decodeVectors(raw.passages.data, raw.dimension);
    return { model: raw.model, dimension: raw.dimension,
      passages: new Map(raw.passages.ids.map((id, i) => [id, passages[i]])),
      topics: { ids: raw.topics.ids, vectors: decodeVectors(raw.topics.data, raw.dimension) } };
  }).catch(() => null);
  return index;
}

/** A function that turns texts into normalized vectors, or null when the model can't load. */
export function loadModel(model: SearchIndex["model"]): Promise<Embed | null> {
  embedder ??= (async () => {
    try {
      const { env, pipeline } = await import("@huggingface/transformers");
      // Only the copy served with the app, never the Hugging Face servers.
      env.allowLocalModels = true;
      env.allowRemoteModels = false;
      // The path from the site root, worked out from the page. A relative path would be
      // resolved against the library's own module, and a full URL is not treated as local.
      env.localModelPath = new URL("models/", location.href).pathname;
      if (env.backends.onnx.wasm) env.backends.onnx.wasm.wasmPaths = {
        wasm: new URL(ortWasm, location.href).href, mjs: new URL(ortModule, location.href).href };
      const extract = await pipeline("feature-extraction", model.repo_id, { dtype: model.dtype, revision: model.revision });
      return async (texts: string[]) => {
        const out = await extract(texts, { pooling: "mean", normalize: true });
        const flat = out.data as Float32Array, dim = out.dims[1];
        return texts.map((_, i) => flat.slice(i * dim, (i + 1) * dim));
      };
    } catch (err) {
      console.warn("The sentence model could not be loaded, questions use names only.", err);
      return null;
    }
  })();
  return embedder;
}
