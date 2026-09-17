/**
 * Dump raw model output for a single (model, receipt) pair.
 *
 * Used to diagnose why a configuration scored badly — aggregate metrics cannot
 * distinguish "the model read it wrong" from "the output was truncated" or "the
 * grammar constraint did not apply".
 *
 *   node scripts/dump-output.mjs <repo> <fixtureId> [--constrain] [--dtype q4]
 *                                [--prompt v1] [--maxNewTokens 320]
 */
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { chromium } from 'playwright-core';
import { extractJson, scoreReceipt } from '../src/schema.js';
import { PROMPTS } from '../src/prompts.js';

const CHROME = 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';
const HARNESS_URL = process.env.HARNESS_URL ?? 'http://127.0.0.1:5179/';

function arg(name, fallback) {
  const i = process.argv.indexOf(`--${name}`);
  return i !== -1 && process.argv[i + 1] ? process.argv[i + 1] : fallback;
}
const positional = process.argv.slice(2).filter((a) => !a.startsWith('--') && !a.includes('.json'));
const repo = positional[0] ?? 'HuggingFaceTB/SmolVLM-500M-Instruct';
const fixtureId = positional[1] ?? 'eval-000000';
const dtype = arg('dtype', 'q4');
const promptId = arg('prompt', 'v1');
const maxNewTokens = Number(arg('maxNewTokens', '320'));
const constrain = process.argv.includes('--constrain');
const split = arg('split', 'eval');

const manifest = JSON.parse(await readFile(path.join('fixtures', split, 'manifest.json'), 'utf8'));
const entry = manifest.entries.find((e) => e.id === fixtureId);
if (!entry) throw new Error(`fixture ${fixtureId} not in manifest`);

const b64 = (await readFile(path.join(process.cwd(), entry.image))).toString('base64');
const expected = JSON.parse(await readFile(path.join(process.cwd(), entry.label), 'utf8'));

const browser = await chromium.launch({
  executablePath: CHROME,
  headless: true,
  args: ['--no-sandbox', '--disable-gpu-sandbox', '--enable-unsafe-webgpu', '--no-first-run', '--disable-crash-reporter'],
});
const page = await browser.newPage();
page.on('pageerror', (e) => console.log(`[pageerror] ${e.message}`));
await page.goto(HARNESS_URL, { waitUntil: 'load' });
await page.waitForFunction('window.__VLM_BENCH_READY__ === true || window.__VLM_BENCH_ERROR__', null, { timeout: 60_000 });

console.log(`repo=${repo} dtype=${dtype} fixture=${fixtureId} locale=${entry.locale}`);
console.log(`degradations=${entry.degradations}`);
console.log(`constrain=${constrain} prompt=${promptId} maxNewTokens=${maxNewTokens}\n`);
console.log(`EXPECTED: ${JSON.stringify({ merchant: expected.merchant, date: expected.date, total: expected.total, currency: expected.currency, tax: expected.tax })}\n`);

await page.evaluate(({ repo, dtype }) => window.__VLM_BENCH__.load({ repo, dtype, device: 'webgpu' }), { repo, dtype });

const r = await page.evaluate(
  async ({ b64, mime, promptText, maxNewTokens, constrain }) => {
    const bin = atob(b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    try {
      const run = await window.__VLM_BENCH__.run({
        imageBytes: bytes,
        imageMime: mime,
        prompt: promptText,
        maxNewTokens,
        doSample: false,
        constrain,
        constraintType: 'json_schema',
        schemaName: 'full',
      });
      return { ok: true, text: run.text, metrics: run.metrics };
    } catch (err) {
      return { ok: false, error: String(err?.message ?? err) };
    }
  },
  { b64, mime: entry.ext === 'jpg' ? 'image/jpeg' : 'image/png', promptText: PROMPTS[promptId].text, maxNewTokens, constrain },
);

if (!r.ok) {
  console.log(`ERROR: ${r.error}`);
} else {
  console.log(`metrics: ${JSON.stringify(r.metrics, null, 2)}\n`);
  console.log(`RAW OUTPUT (${r.text.length} chars):`);
  console.log('-----');
  console.log(r.text);
  console.log('-----');
  const parsed = extractJson(r.text);
  console.log(`\nparse strategy: ${parsed.strategy}${parsed.error ? ` (${parsed.error})` : ''}`);
  console.log(`parsed: ${JSON.stringify(parsed.value)}`);
  const score = scoreReceipt({ raw: r.text, parsed: parsed.value, expected, parseStrategy: parsed.strategy, locale: entry.locale });
  console.log(`accuracy=${score.accuracy.toFixed(2)} fields=${JSON.stringify(Object.fromEntries(Object.entries(score.fieldScores).map(([k, v]) => [k, +v.toFixed(2)])))}`);
}

await browser.close();
