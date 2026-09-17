/**
 * Smoke-probe a model + dtype pair before committing to a full benchmark.
 *
 * Full runs cost many minutes per model, so this answers the cheap gating
 * questions first: does the architecture load in transformers.js at all, does
 * the dtype produce coherent output, and does the output survive JSON
 * extraction? A model that fails here should not be added to the lineup.
 *
 *   node scripts/probe-model.mjs <repo> <dtype> [fixtureId]
 */
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { chromium } from 'playwright-core';
import { extractJson, scoreReceipt } from '../src/schema.js';
import { PROMPT_V1 } from '../src/prompts.js';

const CHROME = 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';
const HARNESS_URL = process.env.HARNESS_URL ?? 'http://127.0.0.1:5179/';

const repo = process.argv[2];
const dtype = process.argv[3] ?? 'q4';
const fixtureId = process.argv[4] ?? 'eval-000000';
if (!repo) {
  console.error('usage: node scripts/probe-model.mjs <repo> <dtype> [fixtureId]');
  process.exit(1);
}

const split = 'eval';
const manifest = JSON.parse(await readFile(path.join('fixtures', split, 'manifest.json'), 'utf8'));
const entry = manifest.entries.find((e) => e.id === fixtureId);
if (!entry) throw new Error(`fixture ${fixtureId} not found`);
const b64 = (await readFile(path.join(process.cwd(), entry.image))).toString('base64');
const expected = JSON.parse(await readFile(path.join(process.cwd(), entry.label), 'utf8'));

const browser = await chromium.launch({
  executablePath: CHROME,
  headless: true,
  args: ['--no-sandbox', '--disable-gpu-sandbox', '--enable-unsafe-webgpu', '--no-first-run', '--disable-crash-reporter'],
});
const page = await browser.newPage();
page.on('pageerror', (e) => console.log(`[pageerror] ${e.message}`));
// Surface worker diagnostics; without this, worker-side failures are opaque.
page.on('console', (m) => {
  const t = m.text();
  if (t.includes('[worker:debug]') || m.type() === 'error') console.log(`[page] ${t.slice(0, 400)}`);
});

await page.goto(HARNESS_URL, { waitUntil: 'load' });
await page.waitForFunction('window.__VLM_BENCH_READY__ === true || window.__VLM_BENCH_ERROR__', null, { timeout: 60_000 });

console.log(`repo=${repo} dtype=${dtype} fixture=${fixtureId}`);
console.log(`expected: ${JSON.stringify({ merchant: expected.merchant, date: expected.date, total: expected.total, currency: expected.currency })}\n`);

// 1. Load
const load = await page.evaluate(
  async ({ repo, dtype }) => {
    try {
      const r = await window.__VLM_BENCH__.load({ repo, dtype, device: 'webgpu' });
      return { ok: true, ...r };
    } catch (err) {
      return { ok: false, error: String(err?.message ?? err).slice(0, 600) };
    }
  },
  { repo, dtype },
);

if (!load.ok) {
  console.log(`LOAD FAILED: ${load.error}`);
  await browser.close();
  process.exit(2);
}
console.log(`LOADED in ${(load.loadMs / 1000).toFixed(1)}s device=${load.device} dtype=${load.dtype} files=${load.fileCount} bytes=${(load.downloadBytes / 1048576).toFixed(0)}MB`);
if (load.device !== 'webgpu') console.log(`WARNING: fell back to ${load.device}`);

// 2. Run
const run = await page.evaluate(
  async ({ b64, mime, prompt }) => {
    const bin = atob(b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    try {
      const r = await window.__VLM_BENCH__.run({
        imageBytes: bytes, imageMime: mime, prompt, maxNewTokens: 768, doSample: false,
      });
      return { ok: true, text: r.text, metrics: r.metrics };
    } catch (err) {
      // Include the worker stack and step trail so processor/generation failures
      // are traceable across the postMessage boundary.
      return {
        ok: false,
        error: `${err?.message ?? err}\n${String(err?.stack ?? '')}`.slice(0, 1500),
        diagnostics: err?.diagnostics ?? null,
      };
    }
  },
  { b64, mime: entry.ext === 'jpg' ? 'image/jpeg' : 'image/png', prompt: PROMPT_V1 },
);

if (!run.ok) {
  console.log(`\nRUN FAILED: ${run.error}`);
  if (run.diagnostics) console.log(`worker steps: ${JSON.stringify(run.diagnostics, null, 2)}`);
  await browser.close();
  process.exit(3);
}
const parsed = extractJson(run.text);
const score = scoreReceipt({ raw: run.text, parsed: parsed.value, expected, parseStrategy: parsed.strategy, locale: entry.locale });

console.log(`gen=${((run.metrics.generateMs ?? 0) / 1000).toFixed(1)}s tokens=${run.metrics.outputTokens} inTok=${run.metrics.inputTokens}`);
console.log(`\n--- RAW OUTPUT ---\n${(run.text || '(empty)').slice(0, 1200)}\n--- END ---`);
console.log(`\nparse=${parsed.strategy}${parsed.error ? ` (${parsed.error})` : ''}`);
console.log(`parsed=${JSON.stringify(parsed.value)?.slice(0, 400)}`);
console.log(`accuracy=${score.accuracy.toFixed(2)} fields=${JSON.stringify(Object.fromEntries(Object.entries(score.fieldScores).map(([k, v]) => [k, +v.toFixed(2)])))}`);

// Coherence gates. Two distinct failure modes have already burned real runs:
//   1. Degenerate repetition ("if if if ...") from a broken quantization.
//   2. The model reciting the prompt's schema back instead of reading the image
//      (granite-docling does exactly this - it is not an instruction follower).
const text = run.text ?? '';
const degenerate = /^(\s*[-–"']?\s*(if|-1|what|the|a|,)\b[\s,.'"-]*){6,}/i.test(text) || text.length < 12;
const recitesSchema = /['"]?(merchant|total|currency|line_items)['"]?\s*:\s*(string|number|array)\b/i.test(text);
const verdict = degenerate
  ? 'DEGENERATE - broken quantization, do not use'
  : recitesSchema
    ? 'RECITES SCHEMA - model echoed the prompt instead of reading the image'
    : 'coherent';
console.log(`\nVERDICT: ${verdict}`);
console.log(`  degenerateRepetition=${degenerate} recitesSchema=${recitesSchema}`);

await browser.close();
