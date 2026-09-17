/**
 * A/B test: unconstrained prompting vs JSON-schema constrained decoding.
 *
 * The 256M models fail on FORMAT, not recognition — they read merchant/date/
 * prices correctly, then emit invented nested objects and invalid numbers. This
 * measures how much of that gap schema-constrained decoding closes, and what it
 * costs in latency. It is the decisive experiment for whether small models are
 * viable for this app at all.
 *
 *   node scripts/test-constrained.mjs [repoId] [--limit 6] [--prompt v1]
 */
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { chromium } from 'playwright-core';
import { extractJson, scoreReceipt, FIELDS } from '../src/schema.js';
import { PROMPTS } from '../src/prompts.js';

const CHROME = 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';
const HARNESS_URL = process.env.HARNESS_URL ?? 'http://127.0.0.1:5179/';

function arg(name, fallback) {
  const i = process.argv.indexOf(`--${name}`);
  return i !== -1 && process.argv[i + 1] ? process.argv[i + 1] : fallback;
}
const positional = process.argv[2] && !process.argv[2].startsWith('--') ? process.argv[2] : null;
const repo = positional ?? 'HuggingFaceTB/SmolVLM-256M-Instruct';
const dtype = arg('dtype', 'q4');
const limit = Number(arg('limit', '6'));
const promptId = arg('prompt', 'v1');
const split = arg('split', 'eval');
const maxNewTokens = Number(arg('maxNewTokens', '320'));

/**
 * Arms to compare. Key order is `total` first in both schemas, and the scalar
 * schema deliberately omits `line_items`: a variable-length array inside a
 * grammar is what pushes generation past the token ceiling, and truncation costs
 * every field, not just the items.
 */
const ARMS = [
  { name: 'unconstrained', constrain: false, prompt: promptId },
  { name: 'schema-full', constrain: true, schema: 'full', prompt: promptId },
  { name: 'schema-scalars', constrain: true, schema: 'scalars', prompt: promptId },
];

const manifest = JSON.parse(await readFile(path.join('fixtures', split, 'manifest.json'), 'utf8'));
const entries = manifest.entries.slice(0, limit);
const cases = [];
for (const e of entries) {
  cases.push({
    id: e.id,
    b64: (await readFile(path.join(process.cwd(), e.image))).toString('base64'),
    mime: e.ext === 'jpg' ? 'image/jpeg' : 'image/png',
    expected: JSON.parse(await readFile(path.join(process.cwd(), e.label), 'utf8')),
    locale: e.locale,
  });
}

const browser = await chromium.launch({
  executablePath: CHROME,
  headless: true,
  args: ['--no-sandbox', '--disable-gpu-sandbox', '--enable-unsafe-webgpu', '--no-first-run', '--disable-crash-reporter'],
});
const page = await browser.newPage();
page.on('pageerror', (e) => console.log(`[pageerror] ${e.message}`));
await page.goto(HARNESS_URL, { waitUntil: 'load' });
await page.waitForFunction('window.__VLM_BENCH_READY__ === true || window.__VLM_BENCH_ERROR__', null, { timeout: 60_000 });

console.log(`repo=${repo} dtype=${dtype} prompt=${promptId} cases=${cases.length} maxNewTokens=${maxNewTokens}\n`);
const load = await page.evaluate(({ repo, dtype }) => window.__VLM_BENCH__.load({ repo, dtype, device: 'webgpu' }), { repo, dtype });
console.log(`loaded in ${(load.loadMs / 1000).toFixed(1)}s\n`);

const summary = [];

for (const arm of ARMS) {
  const promptText = PROMPTS[arm.prompt].text;
  const t0 = Date.now();
  const outputs = await page.evaluate(
    async ({ cases, promptText, constrain, schemaName, maxNewTokens }) => {
      const out = [];
      for (const c of cases) {
        const bin = atob(c.b64);
        const bytes = new Uint8Array(bin.length);
        for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
        try {
          const r = await window.__VLM_BENCH__.run({
            imageBytes: bytes,
            imageMime: c.mime,
            prompt: promptText,
            maxNewTokens,
            doSample: false,
            constrain,
            constraintType: 'json_schema',
            schemaName,
          });
          out.push({ id: c.id, ok: true, text: r.text, metrics: r.metrics });
        } catch (err) {
          out.push({ id: c.id, ok: false, error: String(err?.message ?? err) });
        }
      }
      return out;
    },
    {
      cases: cases.map((c) => ({ id: c.id, b64: c.b64, mime: c.mime })),
      promptText,
      constrain: !!arm.constrain,
      schemaName: arm.schema,
      maxNewTokens,
    },
  );

  const scores = [];
  const fieldTotals = Object.fromEntries(FIELDS.map((f) => [f, 0]));
  let parseOk = 0;
  let genMs = 0;
  console.log('='.repeat(78));
  console.log(`ARM: ${arm.name}${arm.constrain ? ` (schema=${arm.schema})` : ''}`);
  console.log('='.repeat(78));

  for (let i = 0; i < outputs.length; i++) {
    const o = outputs[i];
    const c = cases[i];
    if (!o.ok) { console.log(`  ${c.id}: ERROR ${o.error}`); continue; }
    const parsed = extractJson(o.text);
    const score = scoreReceipt({ raw: o.text, parsed: parsed.value, expected: c.expected, parseStrategy: parsed.strategy, locale: c.locale });
    scores.push(score.accuracy);
    if (score.parsedOk) parseOk++;
    genMs += o.metrics?.generateMs ?? 0;
    for (const f of FIELDS) fieldTotals[f] += score.fieldScores[f];
    console.log(
      `  ${c.id} acc=${score.accuracy.toFixed(2)} parse=${parsed.strategy.padEnd(14)} ` +
      `gen=${o.metrics?.generateMs ? (o.metrics.generateMs / 1000).toFixed(1) + 's' : '--'} tok=${o.metrics?.outputTokens ?? '?'}`,
    );
    console.log(`     out: ${JSON.stringify((o.text ?? '').slice(0, 200))}`);
  }

  const mean = scores.length ? scores.reduce((a, b) => a + b, 0) / scores.length : 0;
  const row = {
    arm: arm.name,
    accuracy: mean,
    parseRate: outputs.length ? parseOk / outputs.length : 0,
    avgGenMs: outputs.length ? genMs / outputs.length : 0,
    fields: Object.fromEntries(FIELDS.map((f) => [f, outputs.length ? fieldTotals[f] / outputs.length : 0])),
  };
  summary.push(row);
  console.log(`  -> accuracy=${(mean * 100).toFixed(1)}%  jsonParse=${(row.parseRate * 100).toFixed(0)}%  avgGen=${(row.avgGenMs / 1000).toFixed(1)}s  (${((Date.now() - t0) / 1000).toFixed(0)}s wall)\n`);
}

console.log('='.repeat(78));
console.log('SUMMARY');
console.log('='.repeat(78));
console.log('arm'.padEnd(18) + 'acc'.padStart(8) + 'json'.padStart(8) + 'avgGen'.padStart(9) + '  ' + FIELDS.map((f) => f.slice(0, 6).padStart(7)).join(''));
for (const r of summary) {
  console.log(
    r.arm.padEnd(18) +
    `${(r.accuracy * 100).toFixed(1)}%`.padStart(8) +
    `${(r.parseRate * 100).toFixed(0)}%`.padStart(8) +
    `${(r.avgGenMs / 1000).toFixed(1)}s`.padStart(9) + '  ' +
    FIELDS.map((f) => `${(r.fields[f] * 100).toFixed(0)}%`.padStart(7)).join(''),
  );
}

await browser.close();
