import { defineConfig } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";

// Relative base so the static build works from any folder on any web server.
export default defineConfig({
  plugins: [svelte()],
  base: "./",
  build: {
    rollupOptions: { output: { manualChunks: { graph: ["cytoscape", "d3-force"] } } },
    chunkSizeWarningLimit: 600,
  },
});
