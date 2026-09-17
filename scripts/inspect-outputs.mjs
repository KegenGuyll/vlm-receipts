/**
 * Inspect raw model output against ground truth, per prompt variant.
 *
 * Aggregate accuracy alone doesn't tell us whether a tiny model fails because it
 * cannot read the receipt or because it never emits the required keys. This
 * prints the actual generated text next to the expected values so prompt and
 * schema problems are distinguishable from OCR limits.
 *
 *   node scripts/inspect-outputs.mjs [repoId] [--limit 4] [--maxNewTokens 320]
 */
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { chromium } from 'playwright-core';
import { extractJson, scoreReceipt } from '../src/schema.js';
import { PROMPTS } from '../src/prompts.js';

const CHROME = 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';
const HARNESS_URL = 'http://127.0.0.1:5179/';

function arg(name, fallback) {
  const i = process.argv.indexOf(`--${name}`);
  return i !== -1 && process.argv[i + 1] ? process.argv[i + 1] : fallback;
}
const positionalRepo = process.argv[2] && !process.argv[2].startsWith('--') ? process.argv[2] : null;
const repo = positionalRepo ?? 'HuggingFaceTB/SmolVLM-256M-Instruct';
const dtype = arg('dtype', 'q4');
const limit = Number(arg('limit', '4'));
const maxNewTokens = Number(arg('maxNewTokens', '320'));
const split = arg('split', 'eval');

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
    degradations: e.degradations,
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

console.log(`model=${repo} dtype=${dtype} cases=${cases.length} maxNewTokens=${maxNewTokens}\n`);
await page.evaluate(({ repo, dtype }) => window.__VLM_BENCH__.load({ repo, dtype, device: 'webgpu' }), { repo, dtype });

for (const [pid, prompt] of Object.entries(PROMPTS)) {
  console.log('='.repeat(90));
  console.log(`PROMPT ${pid} — ${prompt.description}`);
  console.log('='.repeat(90));

  const outputs = await page.evaluate(
    async ({ cases, promptText, maxNewTokens }) => {
      const out = [];
      for (const c of cases) {
        const bin = atob(c.b64);
        const bytes = new Uint8Array(bin.length);
        for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
        const r = await window.__VLM_BENCH__.run({
          imageBytes: bytes,
          imageMime: c.mime,
          prompt: promptText,
          maxNewTokens,
          doSample: false,
        });
        out.push({ id: c.id, text: r.text, metrics: r.metrics });
      }
      return out;
    },
    { cases: cases.map((c) => ({ id: c.id, b64: c.b64, mime: c.mime })), promptText: prompt.text, maxNewTokens },
  );

  const scores = [];
  for (let i = 0; i < outputs.length; i++) {
    const o = outputs[i];
    const c = cases[i];
    const parsed = extractJson(o.text);
    const score = scoreReceipt({ raw: o.text, parsed: parsed.value, expected: c.expected, parseStrategy: parsed.strategy, locale: c.locale });
    scores.push(score.accuracy);
    console.log(`\n--- ${c.id} (${c.locale}; ${c.degradations}) ---`);
    console.log(`expected: ${JSON.stringify({ merchant: c.expected.merchant, date: c.expected.date, total: c.expected.total, currency: c.expected.currency, tax: c.expected.tax })}`);
    console.log(`raw (${o.text?.length ?? 0} chars, ${o.metrics?.outputTokens ?? '?'} tok, ${o.metrics?.generateMs?.toFixed(0)}ms, strategy=${parsed.strategy}):`);
    console.log(o.text ? o.text.slice(0, 900) : '(EMPTY)');
    if (parsed.value) console.log(`parsed: ${JSON.stringify(parsed.value).slice(0, 400)}`);
    console.log(`accuracy=${score.accuracy.toFixed(2)} fields=${JSON.stringify(Object.fromEntries(Object.entries(score.fieldScores).map(([k, v]) => [k, Number(v.toFixed(2))])))}`);
  }
  console.log(`\n>>> ${pid} mean accuracy: ${(scores.reduce((a, b) => a + b, 0) / scores.length * 100).toFixed(1)}%\n`);
}

await browser.close();
