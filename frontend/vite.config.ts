import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// `base: "./"` keeps the static build hostable under any sub-path (GitHub Pages, Vercel).
// In API mode, the dev server proxies /api to `optlab serve` on :8000.
export default defineConfig({
  base: "./",
  plugins: [react(), tailwindcss()],
  server: { proxy: { "/api": "http://127.0.0.1:8000" } },
  build: { chunkSizeWarningLimit: 5000 },
});
