/**
 * Cross-model quantization sweep.
 *
 * Established that SmolVLM-256M emits degenerate text under q4f16/fp16 but is
 * correct under q4/uint8/int8 — a defect in those ONNX weight files, not a
 * prompt or template problem. This runs the same fixed OCR question across every
 * candidate model and dtype so we can pick a working precision per model before
 * spending time on the full benchmark.
 *
 *   node scripts/sweep-quantizations.mjs
 */
import { chromium } from 'playwright-core';

const CHROME = 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';
const HARNESS_URL = process.env.HARNESS_URL ?? 'http://127.0.0.1:5179/';
const PROMPT = 'What is the total amount on this receipt? Answer with just the number.';

/** Models worth testing, with the dtypes worth testing for each. */
const PLAN = [
  { repo: 'HuggingFaceTB/SmolVLM-256M-Instruct', dtypes: ['q4', 'uint8'] },
  { repo: 'HuggingFaceTB/SmolVLM2-256M-Video-Instruct', dtypes: ['q4', 'uint8'] },
  { repo: 'HuggingFaceTB/SmolVLM-500M-Instruct', dtypes: ['q4', 'uint8'] },
  { repo: 'HuggingFaceTB/SmolVLM2-500M-Video-Instruct', dtypes: ['q4', 'uint8'] },
];

const browser = await chromium.launch({
  executablePath: CHROME,
  headless: true,
  args: ['--no-sandbox', '--disable-gpu-sandbox', '--enable-unsafe-webgpu', '--no-first-run', '--disable-crash-reporter'],
});
const page = await browser.newPage();
page.on('pageerror', (e) => console.log(`[pageerror] ${e.message}`));
page.on('console', (m) => {
  const t = m.text();
  if (!/vite|powerPreference/i.test(t)) console.log(`[page:${m.type()}] ${t.slice(0, 300)}`);
});

await page.goto(HARNESS_URL, { waitUntil: 'load' });
await page.evaluate(() => import('/debug-api.js'));
await page.waitForFunction('window.__VLM_DEBUG_READY__ === true', null, { timeout: 30_000 });

console.log(`prompt: ${PROMPT}\n`);
const seen = [];

for (const { repo, dtypes } of PLAN) {
  for (const dtype of dtypes) {
    const rec = await page.evaluate(
      async ({ repo, dtype, prompt }) => {
        const out = { repo, dtype };
        const t0 = performance.now();
        try {
          const info = await window.__VLM_DEBUG__.load({ repo, dtype });
          out.loadMs = Math.round(performance.now() - t0);
          out.modelClass = info.modelClass;
          const receipt = await window.__VLM_DEBUG__.makeSyntheticReceipt();
          const r = await window.__VLM_DEBUG__.attempt({ prompt, image: receipt.image, maxNewTokens: 32 });
          out.genMs = r.ms;
          out.pixelValues = r.pixelValuesDims;
          out.decoded = r.decoded;
        } catch (err) {
          out.loadMs = Math.round(performance.now() - t0);
          out.error = String(err?.message ?? err).slice(0, 400);
        }
        return out;
      },
      { repo, dtype, prompt: PROMPT },
    );
    seen.push(rec);
    const verdict = rec.error ? `ERROR ${rec.error}` : JSON.stringify(rec.decoded);
    console.log(`${repo.padEnd(46)} ${dtype.padEnd(7)} load=${String(rec.loadMs).padStart(6)}ms gen=${String(rec.genMs ?? '-').padStart(6)}ms -> ${verdict}`);
  }
}

console.log('\n=== SUMMARY (expect 6.25) ===');
for (const r of seen) {
  const ok = typeof r.decoded === 'string' && r.decoded.includes('6.25');
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${r.repo} [${r.dtype}] -> ${JSON.stringify(r.decoded ?? r.error).slice(0, 120)}`);
}

await browser.close();
