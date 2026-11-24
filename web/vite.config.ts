import { defineConfig } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";

// Relative base so the static build works from any folder on any web server.
export default defineConfig({
  plugins: [svelte()],
  base: "./",
  // The sentence model for questions is small enough for the CPU. transformers.js asks for
  // the WebGPU build of the ONNX runtime, whose files are twice as large (28 MB, over the
  // file limit of some static hosts), so the plain CPU build (14 MB) is used instead.
  resolve: { alias: [{ find: /^onnxruntime-web\/webgpu$/, replacement: "onnxruntime-web/wasm" }] },
  build: {
    rollupOptions: { output: { manualChunks: { graph: ["cytoscape"] } } },
    chunkSizeWarningLimit: 600,
  },
  // The graph layout runs in a module worker (src/lib/layout.worker.ts).
  worker: { format: "es" },
});
