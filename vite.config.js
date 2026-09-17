import { defineConfig } from 'vite';

// The benchmark harness runs entirely in the browser. Playwright drives real
// Chrome against this dev server so every measurement reflects genuine in-tab
// WebGPU execution (device selection, quantization, download size).
//
// `root: 'src'` keeps the HTML entry next to the modules it imports.
//
// BENCH_NO_WATCH=1 disables the file watcher. This matters for measurement
// validity: an HMR reload mid-run destroys the page's execution context and
// silently kills multi-minute model runs. It also avoids EBUSY crashes when a
// scratch file is locked on Windows.
const noWatch = process.env.BENCH_NO_WATCH === '1';

export default defineConfig({
  root: 'src',
  server: {
    host: '127.0.0.1',
    port: 5179,
    strictPort: true,
    // Fixtures live in the repo root, outside Vite's root.
    fs: { allow: ['..'] },
    watch: noWatch
      ? null
      : {
          // Subagent scratch dirs (.foo.tmpdir) get locked mid-write on Windows
          // and crash chokidar with EBUSY. They are never part of the app.
          ignored: ['**/*.tmpdir/**', '**/.probe/**', '**/fixtures/**', '**/results/**'],
        },
  },
  build: {
    target: 'esnext',
    outDir: '../dist-web',
    emptyOutDir: true,
  },
  optimizeDeps: {
    // transformers.js ships its own wasm/ort assets; prebundling breaks them.
    exclude: ['@huggingface/transformers'],
  },
  worker: { format: 'es' },
});

