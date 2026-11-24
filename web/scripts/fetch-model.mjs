// Put the sentence model for questions next to the app.
//
//   npm run models                     model named in public/data/search.json
//   npm run models -- <search.json>    model named in that file
//
// The model is downloaded once at the revision search.json names and kept in
// public/models/. It is served with the app, so the browser never fetches it from another
// server. The ONNX runtime it needs is bundled by the build (see vite.config.ts).
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const index = resolve(process.argv[2] ?? join(here, "../public/data/search.json"));
if (!existsSync(index)) throw new Error(`${index} not found. Run "npm run data" with an export that has search.json first.`);
const { model } = JSON.parse(readFileSync(index, "utf8"));

// transformers.js names the int8 weights model_quantized.onnx.
const weights = model.dtype === "q8" ? "onnx/model_quantized.onnx" : "onnx/model.onnx";
const files = ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", weights];
const target = resolve(here, "../public/models", model.repo_id);
for (const file of files) {
  const out = join(target, file);
  const stamp = `${out}.revision`;
  if (existsSync(out) && existsSync(stamp) && readFileSync(stamp, "utf8") === model.revision) continue;
  const url = `https://huggingface.co/${model.repo_id}/resolve/${model.revision}/${file}`;
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: ${response.status}`);
  mkdirSync(dirname(out), { recursive: true });
  writeFileSync(out, Buffer.from(await response.arrayBuffer()));
  writeFileSync(stamp, model.revision);
  console.log(`Downloaded ${model.repo_id}/${file}`);
}

console.log(`Model ${model.repo_id}@${model.revision.slice(0, 7)} is in public/models/`);
