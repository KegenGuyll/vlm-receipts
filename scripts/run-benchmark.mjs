/**
 * Benchmark runner.
 *
 * Drives real Chrome against the Vite harness and evaluates each candidate model
 * on the receipt fixture set, entirely in-browser on WebGPU. Produces a JSON
 * result file plus a human-readable report.
 *
 * Design notes that matter for validity:
 *  - One browser context per model, closed afterwards, so a leaky model cannot
 *    contaminate the next model's VRAM measurements.
 *  - The model cache is warm by default. Cold-start download size is still
 *    reported from the load phase, but latency numbers are steady-state, which
 *    is what a user actually experiences after first launch.
 *  - Scoring happens in Node against the on-disk ground truth, never in the
 *    page, so a model cannot influence how it is graded.
 *
 *   node scripts/run-benchmark.mjs [--models repo1,repo2] [--limit N]
 *                                  [--dtype q4] [--prompt v1] [--out results/run.json]
 */
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import path from 'node:path';
import { chromium } from 'playwright-core';
import { extractJson, scoreReceipt, summarize } from '../src/schema.js';
import { PROMPTS, DEFAULT_PROMPT } from '../src/prompts.js';
import { MODELS, getModel } from '../src/models.js';

const CHROME = 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';
const HARNESS_URL = process.env.HARNESS_URL ?? 'http://127.0.0.1:5179/';

function arg(name, fallback) {
  const i = process.argv.indexOf(`--${name}`);
  return i !== -1 && process.argv[i + 1] ? process.argv[i + 1] : fallback;
}

const split = arg('split', 'eval');
const limit = Number(arg('limit', '12'));
const dtypeOverride = arg('dtype', null);
const promptId = arg('prompt', DEFAULT_PROMPT);
const outputPath = arg('out', path.join('results', `bench-${new Date().toISOString().replace(/[:.]/g, '-')}.json`));
const maxNewTokens = Number(arg('maxNewTokens', '320'));
const receiptTimeoutMs = Number(arg('receiptTimeoutMs', '180000'));
// Constrained decoding is opt-in via --constrain (or --no-constrain to force off)
// so both configurations can be measured with the same harness.
const constrain = !process.argv.includes('--no-constrain');
const schemaName = arg('schema', 'full');

const requestedModels = arg('models', null);
const plan = requestedModels
  ? requestedModels.split(',').map((r) => getModel(r.trim()))
  : MODELS;

const fixtureDir = path.join(process.cwd(), 'fixtures', split);
const manifestPath = path.join(fixtureDir, 'manifest.json');
if (!existsSync(manifestPath)) {
  console.error(`No fixture manifest at ${manifestPath}. Run scripts/fetch-fixtures.mjs first.`);
  process.exit(1);
}

const manifest = JSON.parse(await readFile(manifestPath, 'utf8'));
const fixtures = manifest.entries.slice(0, limit);

if (!fixtures.length) {
  console.error('Fixture manifest is empty.');
  process.exit(1);
}

const prompt = PROMPTS[promptId];
if (!prompt) {
  console.error(`Unknown prompt variant "${promptId}". Options: ${Object.keys(PROMPTS).join(', ')}`);
  process.exit(1);
}

console.log('='.repeat(78));
console.log('VLM receipt benchmark');
console.log('='.repeat(78));
console.log(`harness   : ${HARNESS_URL}`);
console.log(`split     : ${split}  (${fixtures.length} receipts of ${manifest.count})`);
console.log(`prompt    : ${promptId} — ${prompt.description}`);
console.log(`constrain : ${constrain ? `json_schema(${schemaName})` : 'off'}`);
console.log(`models    : ${plan.length}`);
plan.forEach((m) => console.log(`            - ${m.repo} [${dtypeOverride ?? m.dtype}]`));
console.log(`maxTokens : ${maxNewTokens}`);
console.log('');

// Preload fixture images once; each model replays the identical byte stream.
console.log('[bench] loading fixture images…');
const cases = [];
for (const entry of fixtures) {
  const imgAbs = path.join(process.cwd(), entry.image);
  const labelAbs = path.join(process.cwd(), entry.label);
  const bytes = await readFile(imgAbs);
  const expected = JSON.parse(await readFile(labelAbs, 'utf8'));
  cases.push({
    id: entry.id,
    bytes,
    b64: bytes.toString('base64'),
    mime: entry.ext === 'jpg' ? 'image/jpeg' : 'image/png',
    expected,
    meta: {
      locale: entry.locale,
      degradations: entry.degradations,
      bytes: bytes.length,
      nItems: entry.nItems,
    },
  });
}
console.log(`[bench] ${cases.length} cases ready (avg ${(cases.reduce((a, c) => a + c.bytes.length, 0) / cases.length / 1024).toFixed(0)} KB)\n`);

const browser = await chromium.launch({
  executablePath: CHROME,
  headless: true,
  args: [
    '--no-sandbox',
    '--disable-gpu-sandbox',
    '--enable-unsafe-webgpu',
    '--no-first-run',
    '--no-default-browser-check',
    '--disable-crash-reporter',
    '--disable-breakpad',
  ],
});

/** @type {any[]} */
const runs = [];

for (const model of plan) {
  const dtype = dtypeOverride ?? model.dtype;
  console.log('─'.repeat(78));
  console.log(`[bench] MODEL ${model.repo}  (${model.params}, ${dtype})`);
  console.log('─'.repeat(78));

  const context = await browser.newContext();
  const page = await context.newPage();
  const pageErrors = [];
  page.on('pageerror', (e) => pageErrors.push(String(e.message)));
  page.on('console', (m) => {
    const t = m.text();
    if (m.type() === 'error' && !/powerPreference/i.test(t)) pageErrors.push(t);
  });

  const record = {
    model: model.repo,
    label: model.id,
    params: model.params,
    paramClass: model.paramClass,
    dtype,
    prompt: promptId,
    constrain,
    schemaName: constrain ? schemaName : null,
    status: 'pending',
    load: null,
    gpu: null,
    results: [],
    summary: null,
    error: null,
  };

  try {
    await page.goto(HARNESS_URL, { waitUntil: 'load', timeout: 60_000 });
    await page.waitForFunction(
      'window.__VLM_BENCH_READY__ === true || window.__VLM_BENCH_ERROR__',
      null,
      { timeout: 60_000 },
    );
    const bootErr = await page.evaluate('window.__VLM_BENCH_ERROR__ ?? null');
    if (bootErr) throw new Error(`harness boot failed: ${bootErr}`);

    const info = await page.evaluate('window.__VLM_BENCH__.ping()');
    record.gpu = info.gpu;
    record.hardwareConcurrency = info.hardwareConcurrency;
    record.deviceMemory = info.deviceMemory;

    // ---- load ----
    console.log(`[bench] loading…`);
    const load = await page.evaluate(
      async ({ repo, dtype, onProgressEvery }) => {
        let lastLogged = -1;
        const res = await window.__VLM_BENCH__.load({
          repo,
          dtype,
          device: 'webgpu',
          onProgress: (msg) => {
            if (msg.type === 'progress' && msg.stage === 'download' && msg.totalBytes) {
              const pct = Math.round((msg.loadedBytes / msg.totalBytes) * 100);
              if (pct >= lastLogged + onProgressEvery) {
                lastLogged = pct;
                window.__VLM_PROGRESS__ = { pct, loadedBytes: msg.loadedBytes, totalBytes: msg.totalBytes };
              }
            }
          },
        });
        return res;
      },
      { repo: model.repo, dtype, onProgressEvery: 25 },
    );
    record.load = load;
    console.log(
      `[bench] loaded in ${(load.loadMs / 1000).toFixed(1)}s ` +
      `(device=${load.device}, dtype=${load.dtype}, files=${load.fileCount}, ${(load.downloadBytes / 1048576).toFixed(1)}MB)`,
    );
    if (load.device !== 'webgpu') {
      console.log(`[bench] WARNING: fell back to ${load.device}; results are not representative`);
    }

    // ---- run each receipt ----
    const pageResult = await page.evaluate(
      async ({ cases, promptText, maxNewTokens, receiptTimeoutMs, constrain, schemaName }) => {
        const out = [];
        for (const c of cases) {
          const started = performance.now();
          let record;
          try {
            const bin = atob(c.b64);
            const bytes = new Uint8Array(bin.length);
            for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);

            const run = await Promise.race([
              window.__VLM_BENCH__.run({
                imageBytes: bytes,
                imageMime: c.mime,
                prompt: promptText,
                maxNewTokens,
                doSample: false,
                constrain,
                constraintType: 'json_schema',
                schemaName,
              }),
              new Promise((_, reject) =>
                setTimeout(() => reject(new Error('receipt timeout')), receiptTimeoutMs),
              ),
            ]);
            record = { id: c.id, ok: true, text: run.text, metrics: run.metrics, textSource: run.textSource ?? 'worker' };
          } catch (err) {
            record = { id: c.id, ok: false, error: String(err?.message ?? err), wallMs: performance.now() - started };
          }
          record.wallMs = performance.now() - started;
          out.push(record);
          window.__VLM_PROGRESS__ = { receipt: c.id, done: out.length, total: cases.length };
        }
        return out;
      },
      {
        cases: cases.map((c) => ({ id: c.id, b64: c.b64, mime: c.mime })),
        promptText: prompt.text,
        maxNewTokens,
        receiptTimeoutMs,
        constrain,
        schemaName,
      },
    );

    // ---- score in Node ----
    for (let i = 0; i < pageResult.length; i++) {
      const raw = pageResult[i];
      const c = cases[i];
      const parsed = raw.ok ? extractJson(raw.text) : { value: null, strategy: 'no-output' };
      const score = scoreReceipt({
        raw: raw.text ?? '',
        parsed: parsed.value,
        expected: c.expected,
        parseStrategy: parsed.strategy,
        // The receipt's locale disambiguates numeric dates (03/04/2021).
        locale: c.meta.locale,
      });
      record.results.push({
        id: c.id,
        meta: c.meta,
        ok: !!raw.ok,
        error: raw.error ?? null,
        text: raw.text ?? '',
        parsed: parsed.value ?? null,
        parseError: parsed.error ?? null,
        score,
        metrics: raw.metrics ?? null,
        wallMs: raw.wallMs,
      });
      const mark = score.parsedOk ? (score.exactMatch ? '✓' : '~') : '✗';
      process.stdout.write(
        `\r[bench] ${String(i + 1).padStart(3)}/${cases.length} ${mark} acc=${score.accuracy.toFixed(2)} ` +
        `gen=${raw.metrics?.generateMs ? (raw.metrics.generateMs / 1000).toFixed(1) + 's' : '--'}   `,
      );
    }
    console.log('');
    record.summary = summarize(record.results);
    record.status = 'ok';

    console.log(
      `[bench] accuracy=${(record.summary.accuracy * 100).toFixed(1)}% ` +
      `json=${(record.summary.jsonParseRate * 100).toFixed(0)}% ` +
      `exact=${(record.summary.exactMatchRate * 100).toFixed(0)}% ` +
      `p50=${record.summary.latencyMs.p50 ? (record.summary.latencyMs.p50 / 1000).toFixed(1) + 's' : '--'} ` +
      `p90=${record.summary.latencyMs.p90 ? (record.summary.latencyMs.p90 / 1000).toFixed(1) + 's' : '--'}`,
    );
    console.log(
      `[bench] per-field: ${Object.entries(record.summary.fieldAccuracy)
        .map(([k, v]) => `${k}=${(v * 100).toFixed(0)}%`)
        .join(' ')}`,
    );
  } catch (err) {
    record.status = 'failed';
    record.error = String(err?.message ?? err);
    console.log(`[bench] FAILED: ${record.error}`);
  } finally {
    if (pageErrors.length) record.pageErrors = pageErrors.slice(0, 10);
    try {
      await page.evaluate('window.__VLM_BENCH__.unload()');
    } catch { /* page may be gone */ }
    await context.close();
  }

  runs.push(record);
}

await browser.close();

// ---------------------------------------------------------------- report ----
const ranked = [...runs].sort((a, b) => {
  const ax = a.summary?.accuracy ?? -1;
  const bx = b.summary?.accuracy ?? -1;
  if (bx !== ax) return bx - ax;
  return (a.summary?.latencyMs?.p50 ?? Infinity) - (b.summary?.latencyMs?.p50 ?? Infinity);
});

const output = {
  generatedAt: new Date().toISOString(),
  harness: HARNESS_URL,
  dataset: manifest.dataset,
  datasetRevision: manifest.revision ?? null,
  fixturesSource: manifest.source ?? 'datasets-server-rows',
  split,
  prompt: { id: prompt.id, description: prompt.description, text: prompt.text },
  constrain,
  schemaName: constrain ? schemaName : null,
  maxNewTokens,
  fixtureCount: cases.length,
  gpu: runs[0]?.gpu ?? null,
  runs: ranked,
};

await mkdir(path.dirname(outputPath), { recursive: true });
await writeFile(outputPath, JSON.stringify(output, null, 2));

console.log(`\n${'='.repeat(78)}`);
console.log('RANKING (by headline accuracy, then latency)');
console.log('='.repeat(78));
console.log(
  'model'.padEnd(42) + 'acc'.padStart(7) + 'json'.padStart(7) + 'exact'.padStart(7) +
  'p50'.padStart(8) + 'DL'.padStart(8),
);
for (const r of ranked) {
  if (r.status !== 'ok') {
    console.log(`${r.model.padEnd(42)} FAILED  ${String(r.error).slice(0, 60)}`);
    continue;
  }
  const s = r.summary;
  console.log(
    r.model.padEnd(42) +
    `${(s.accuracy * 100).toFixed(1)}%`.padStart(7) +
    `${(s.jsonParseRate * 100).toFixed(0)}%`.padStart(7) +
    `${(s.exactMatchRate * 100).toFixed(0)}%`.padStart(7) +
    `${s.latencyMs.p50 ? (s.latencyMs.p50 / 1000).toFixed(1) + 's' : '--'}`.padStart(8) +
    `${(r.load ? r.load.downloadBytes / 1048576 : 0).toFixed(0) + 'MB'}`.padStart(8),
  );
}
console.log(`\n[bench] wrote ${outputPath}`);
