/**
 * Smoke test: prove a real small VLM loads and reads a receipt image inside a
 * browser tab on WebGPU.
 *
 * This is the go/no-go gate for the whole approach. It intentionally does the
 * minimum: one model, one image, one prompt — then prints the raw output so we
 * can judge prompt quality before building the full corpus.
 *
 *   node scripts/smoke-test.mjs [repoId] [imagePath]
 */
import { readFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import path from 'node:path';
import { chromium } from 'playwright-core';
import { PROMPT_V1 } from '../src/prompts.js';

const CHROME = 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';
const HARNESS_URL = process.env.HARNESS_URL ?? 'http://127.0.0.1:5179/';

const repo = process.argv[2] ?? 'HuggingFaceTB/SmolVLM-256M-Instruct';
const imagePath = process.argv[3] ?? path.join(process.cwd(), 'fixtures', 'smoke', 'receipt.png');

if (!existsSync(imagePath)) {
  console.error(`No image at ${imagePath}. Fetch a fixture first.`);
  process.exit(1);
}

const imageBytes = new Uint8Array(await readFile(imagePath));
const mime = imagePath.endsWith('.jpg') || imagePath.endsWith('.jpeg') ? 'image/jpeg' : 'image/png';
console.log(`[smoke] model=${repo}`);
console.log(`[smoke] image=${imagePath} (${(imageBytes.length / 1024).toFixed(0)} KB)`);

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
    // Keep the model cache alive across runs.
    '--disable-features=CalculateNativeWinOcclusion',
  ],
});

const context = await browser.newContext();
const page = await context.newPage();

page.on('console', (m) => {
  const t = m.text();
  if (/error|warn|fail/i.test(t)) console.log(`[page:${m.type()}] ${t}`);
});
page.on('pageerror', (e) => console.log(`[pageerror] ${e.message}`));

console.log('[smoke] navigating…');
await page.goto(HARNESS_URL, { waitUntil: 'load', timeout: 60_000 });
await page.waitForFunction('window.__VLM_BENCH_READY__ === true || window.__VLM_BENCH_ERROR__', null, { timeout: 60_000 });

const err = await page.evaluate('window.__VLM_BENCH_ERROR__ ?? null');
if (err) {
  console.error(`[smoke] harness failed to boot: ${err}`);
  await browser.close();
  process.exit(1);
}

const ping = await page.evaluate('window.__VLM_BENCH__.ping()');
console.log(`[smoke] gpu=${JSON.stringify(ping.gpu)}`);

console.log('[smoke] loading model (this downloads weights on first run)…');
const loadStarted = Date.now();
const load = await page.evaluate(
  async ({ repo }) => {
    const t0 = performance.now();
    const res = await window.__VLM_BENCH__.load({
      repo,
      dtype: 'q4f16',
      device: 'webgpu',
      onProgress: (msg) => {
        if (msg.type === 'progress' && msg.stage === 'download' && msg.totalBytes) {
          const pct = Math.round((msg.loadedBytes / msg.totalBytes) * 100);
          if (pct !== window.__lastPct) {
            window.__lastPct = pct;
            window.__VLM_LOG__?.(`  download ${pct}% (${(msg.loadedBytes / 1048576).toFixed(0)}MB)`, 'dim');
          }
        }
      },
    });
    return { ...res, wallMs: performance.now() - t0 };
  },
  { repo },
);

console.log(`[smoke] loaded in ${(load.loadMs / 1000).toFixed(1)}s (device=${load.device}, dtype=${load.dtype}, files=${load.fileCount}, bytes=${(load.downloadBytes / 1048576).toFixed(1)}MB)`);

console.log('[smoke] running inference…');
const b64 = Buffer.from(imageBytes).toString('base64');
const run = await page.evaluate(
  async ({ b64, prompt }) => {
    const bin = atob(b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return await window.__VLM_BENCH__.run({
      imageBytes: bytes,
      imageMime: 'image/png',
      prompt,
      maxNewTokens: 256,
      doSample: false,
    });
  },
  { b64, prompt: PROMPT_V1 },
);

console.log('\n===== RAW MODEL OUTPUT =====');
console.log(run.text || '(empty)');
console.log('===== END OUTPUT =====\n');
console.log(`[smoke] metrics: ${JSON.stringify(run.metrics, null, 2)}`);
console.log(`[smoke] streamedTokens=${run.streamedTokens} rawDims=${JSON.stringify(run.rawDims)} textSource=${run.textSource ?? 'worker'}`);
console.log(`[smoke] total wall time: ${((Date.now() - loadStarted) / 1000).toFixed(1)}s`);

await browser.close();
