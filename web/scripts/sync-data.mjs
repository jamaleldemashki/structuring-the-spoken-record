// Copy one validated web export into public/data for dev and build.
//
//   npm run data                                  newest export -> public/data
//   npm run data -- <export folder>               that export   -> public/data
//   npm run data -- <export folder> <id>          that export   -> public/data/<id>  (open the app with ?archive=<id>)
//
// The file list and checksums come from the export's manifest, so the app always gets a
// complete and unchanged export in the data version it reads.
import { createHash } from "node:crypto";
import { cpSync, existsSync, mkdirSync, readdirSync, readFileSync, rmSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const SCHEMA_VERSION = 2; // keep in step with SCHEMA_VERSION in src/lib/types.ts
const here = dirname(fileURLToPath(import.meta.url));
const exports = resolve(here, "../../export");

const readManifest = (dir) => JSON.parse(readFileSync(join(dir, "manifest.json"), "utf8"));
const usable = (m) => m.stage === "export_web" && m.status === "complete" && m.schema_version === SCHEMA_VERSION;

function newestExport() {
  const runs = readdirSync(exports, { withFileTypes: true })
    .filter((d) => d.isDirectory() && existsSync(join(exports, d.name, "manifest.json")))
    .map((d) => ({ dir: join(exports, d.name), manifest: readManifest(join(exports, d.name)) }))
    .filter((r) => usable(r.manifest))
    .sort((a, b) => a.manifest.started_utc.localeCompare(b.manifest.started_utc));
  if (!runs.length) throw new Error(`No completed schema-${SCHEMA_VERSION} web export in ${exports}. Run spoken-record update first.`);
  return runs.at(-1).dir;
}

// Accept the run folder or, as before, its data/ subfolder.
let source = process.argv[2] ? resolve(process.argv[2]) : newestExport();
if (!existsSync(join(source, "manifest.json")) && existsSync(join(source, "..", "manifest.json"))) source = resolve(source, "..");
const manifest = readManifest(source);
if (!usable(manifest)) throw new Error(`${source} is not a complete schema-${SCHEMA_VERSION} web export (found schema ${manifest.schema_version}).`);

const archive = process.argv[3];
if (archive !== undefined && !/^[a-z0-9][a-z0-9-]{0,63}$/.test(archive)) throw new Error("archive id: lowercase letters, digits and dashes");
const target = resolve(here, "../public/data", archive ?? "");

const files = Object.keys(manifest.artifact_sha256).map((p) => p.replace(/^data\//, ""));
for (const f of files) {
  const digest = createHash("sha256").update(readFileSync(join(source, "data", f))).digest("hex");
  if (digest !== manifest.artifact_sha256[`data/${f}`]) throw new Error(`data/${f} does not match its manifest checksum`);
}
// Only replace this archive's files, other archives under public/data stay.
for (const f of files) rmSync(join(target, f), { force: true });
mkdirSync(target, { recursive: true });
for (const f of files) cpSync(join(source, "data", f), join(target, f));
console.log(`Copied ${files.length} checked files from ${source} to ${target}`);
