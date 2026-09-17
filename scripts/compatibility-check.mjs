/**
 * Compatibility check for every candidate in the model registry.
 *
 * Adding a model to a benchmark lineup is not just a question of download size:
 * the architecture has to be recognized by the installed transformers.js, the
 * processor has to accept the standard call shape, and the dtype has to produce
 * coherent output rather than degenerate repetition. Each of those has silently
 * burned a run in this project, so they are asserted here instead.
 *
 * This is the fast gate. It does not run the full benchmark.
 *
 *   node scripts/compatibility-check.mjs
 */
import { chromium } from 'playwright-core';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { MODELS, EXCLUDED_MODELS } from '../src/models.js';

const CHROME = 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';
const HARNESS_URL = process.env.HARNESS_URL ?? 'http://127.0.0.1:5179/';

// Read the fixture in Node. Vite's root is `src`, so `/fixtures/...` is NOT
// served to the page — fetching it there returns index.html and yields
// "Unexpected token '<'".
const manifest = JSON.parse(await readFile(path.join('fixtures', 'eval', 'manifest.json'), 'utf8'));
const entry = manifest.entries[0];
const fixtureB64 = (await readFile(path.join(process.cwd(), entry.image))).toString('base64');
const fixtureMime = entry.ext === 'jpg' ? 'image/jpeg' : 'image/png';
console.log(`probe fixture: ${entry.id} (${Math.round(fixtureB64.length * 0.75 / 1024)}KB, ${fixtureMime})\n`);

const browser = await chromium.launch({
  executablePath: CHROME,
  headless: true,
  args: ['--no-sandbox', '--disable-gpu-sandbox', '--enable-unsafe-webgpu', '--no-first-run', '--disable-crash-reporter'],
});
const page = await browser.newPage();
page.on('pageerror', (e) => console.log(`[pageerror] ${e.message}`));
await page.goto(HARNESS_URL, { waitUntil: 'load' });
await page.waitForFunction('window.__VLM_BENCH_READY__ === true || window.__VLM_BENCH_ERROR__', null, { timeout: 60_000 });

const ping = await page.evaluate('window.__VLM_BENCH__.ping()');
console.log(`WebGPU: ${ping.gpu?.available ? 'available' : 'UNAVAILABLE'} (${ping.gpu?.vendor ?? '?'} ${ping.gpu?.architecture ?? '?'}, shader-f16=${ping.gpu?.shaderF16})`);
console.log(`transformers.js runtime loaded in-tab; checking ${MODELS.length} benchmarked + ${EXCLUDED_MODELS.length} excluded candidates\n`);

const rows = [];
for (const model of MODELS) {
  const r = await page.evaluate(
    async ({ repo, dtype, fixtureB64, fixtureMime }) => {
      const out = { repo, dtype };
      try {
        const load = await window.__VLM_BENCH__.load({ repo, dtype, device: 'webgpu' });
        out.loadMs = load.loadMs;
        out.downloadBytes = load.downloadBytes;
        out.device = load.device;
      } catch (err) {
        out.loadError = String(err?.message ?? err).slice(0, 300);
        return out;
      }
      // A tiny text-only generation would not exercise the vision path, so this
      // deliberately runs a real fixture image.
      try {
        const bin = atob(fixtureB64);
        const bytes = new Uint8Array(bin.length);
        for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
        const res = await window.__VLM_BENCH__.run({
          imageBytes: bytes,
          imageMime: fixtureMime,
          prompt: 'What is the total amount on this receipt? Answer with just the number.',
          maxNewTokens: 64,
          doSample: false,
        });
        out.text = (res.text ?? '').slice(0, 160);
        out.genMs = res.metrics?.generateMs;
      } catch (err) {
        out.runError = String(err?.message ?? err).slice(0, 300);
      } finally {
        try { await window.__VLM_BENCH__.unload(); } catch { /* ignore */ }
      }
      return out;
    },
    { repo: model.repo, dtype: model.dtype, fixtureB64, fixtureMime },
  );

  const degenerate = /^(\s*[-–"']?\s*(if|-1|what|the|a|,)\b[\s,.'"-]*){6,}/i.test(r.text ?? '');
  const status = r.loadError
    ? 'LOAD FAILED'
    : r.runError
      ? 'RUN FAILED'
      : degenerate
        ? 'DEGENERATE'
        : 'OK';
  rows.push({ ...r, status, params: model.params });
  console.log(
    `${status.padEnd(12)} ${model.repo.padEnd(46)} ${model.dtype.padEnd(7)} ` +
    `${r.downloadBytes ? (r.downloadBytes / 1048576).toFixed(0) + 'MB' : '--'}`.padEnd(70) +
    `${r.text ? ` -> ${JSON.stringify(r.text)}` : r.loadError || r.runError ? ` -> ${r.loadError ?? r.runError}` : ''}`,
  );
}

console.log('\n=== SUMMARY ===');
const ok = rows.filter((r) => r.status === 'OK');
const bad = rows.filter((r) => r.status !== 'OK');
console.log(`usable: ${ok.length}/${rows.length}`);
if (bad.length) {
  for (const b of bad) console.log(`  UNUSABLE ${b.repo} [${b.dtype}]: ${b.status} — ${b.loadError ?? b.runError ?? 'degenerate output'}`);
}

console.log('\n=== DOCUMENTED EXCLUSIONS (not re-tested here) ===');
for (const e of EXCLUDED_MODELS) {
  console.log(`  ${e.repo} (${e.params}, ${e.sizeMB ?? '?'}MB, ${e.license}) — ${e.reason}`);
}

await browser.close();
process.exit(bad.length ? 1 : 0);
