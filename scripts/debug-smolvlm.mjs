/**
 * Diagnose why SmolVLM-256M produced degenerate output ("if if if …").
 *
 * Runs a ladder of controlled generations inside the real browser tab so we can
 * separate three possible causes: a broken chat template, a quantization that
 * destroyed the weights, or simply a prompt the model can't follow.
 *
 *   node scripts/debug-smolvlm.mjs [repoId]
 */
import { readFile } from 'node:fs/promises';
import { chromium } from 'playwright-core';

const CHROME = 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';
const HARNESS_URL = 'http://127.0.0.1:5179/';
const repo = process.argv[2] ?? 'HuggingFaceTB/SmolVLM-256M-Instruct';

const browser = await chromium.launch({
  executablePath: CHROME,
  headless: true,
  args: ['--no-sandbox', '--disable-gpu-sandbox', '--enable-unsafe-webgpu', '--no-first-run', '--disable-crash-reporter'],
});
const page = await browser.newPage();
page.on('console', (m) => {
  const t = m.text();
  if (!/vite|Download the React/i.test(t)) console.log(`[page:${m.type()}] ${t}`);
});
page.on('pageerror', (e) => console.log(`[pageerror] ${e.message}`));

await page.goto(HARNESS_URL, { waitUntil: 'load' });
// Load the debug module (same origin, so Vite serves it).
await page.evaluate(() => import('/debug-api.js'));
await page.waitForFunction('window.__VLM_DEBUG_READY__ === true', null, { timeout: 30_000 });

const info = await page.evaluate(({ repo }) => window.__VLM_DEBUG__.load({ repo }), { repo });
console.log('=== MODEL INFO ===');
console.log(`repo:            ${info.repo}`);
console.log(`processorClass:  ${info.processorClass}`);
console.log(`modelClass:      ${info.modelClass}`);
console.log(`hasChatTemplate: ${info.hasChatTemplate}`);
console.log(`chatTemplate:\n${info.chatTemplate}`);

const results = await page.evaluate(async () => {
  const D = window.__VLM_DEBUG__;
  const receipt = await D.makeSyntheticReceipt();
  const out = { image: { width: receipt.width, height: receipt.height, channels: receipt.channels }, attempts: [] };

  const ladder = [
    ['A: template-inspect', 'What is the total on this receipt?', { inspect: true, maxNewTokens: 8 }],
    ['B: caption', 'Describe this image in one sentence.', {}],
    ['C: plain question', 'What is the total amount on this receipt? Answer with just the number.', {}],
    ['D: with rep-penalty', 'What is the total amount on this receipt? Answer with just the number.', { generateKwargs: { repetition_penalty: 1.1 } }],
    ['E: json prompt', 'Extract this receipt as JSON with keys merchant, date, total, currency. Reply with JSON only.', {}],
  ];

  for (const [label, prompt, opts = {}] of ladder) {
    try {
      const r = await D.attempt({
        prompt,
        image: receipt.image,
        inspectTemplate: opts.inspect,
        maxNewTokens: opts.maxNewTokens ?? 64,
        generateKwargs: opts.generateKwargs ?? {},
      });
      out.attempts.push({ label, prompt, ...r });
    } catch (err) {
      out.attempts.push({ label, prompt, error: String(err?.message ?? err) });
    }
  }
  return out;
});

console.log('\n=== IMAGE ===');
console.log(JSON.stringify(results.image));

for (const a of results.attempts) {
  console.log(`\n=== ${a.label} ===`);
  console.log(`prompt: ${a.prompt}`);
  if (a.error) { console.log(`ERROR: ${a.error}`); continue; }
  console.log(`inputTokens=${a.inputTokens} imageTokens=${a.imageTokenCount} pixelValues=${JSON.stringify(a.pixelValuesDims)} outputDims=${JSON.stringify(a.outputDims)} ms=${a.ms}`);
  if (a.template) console.log(`template: ${JSON.stringify(a.template)}`);
  console.log(`decoded: ${JSON.stringify(a.decoded)}`);
}

await browser.close();
