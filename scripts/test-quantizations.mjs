/**
 * Determine whether SmolVLM's degenerate output is a quantization artifact.
 *
 * The chat template, image token count, and tile dimensions all check out, so
 * the failure is either in the ONNX weights for a given dtype or in the model
 * itself. This reloads the same repo across every available quantization with a
 * fixed prompt and compares.
 *
 *   node scripts/test-quantizations.mjs [repoId]
 */
import { chromium } from 'playwright-core';

const CHROME = 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';
const HARNESS_URL = 'http://127.0.0.1:5179/';
const repo = process.argv[2] ?? 'HuggingFaceTB/SmolVLM-256M-Instruct';
const DTYPES = ['q4f16', 'fp16', 'q4', 'uint8', 'int8'];

const browser = await chromium.launch({
  executablePath: CHROME,
  headless: true,
  args: ['--no-sandbox', '--disable-gpu-sandbox', '--enable-unsafe-webgpu', '--no-first-run', '--disable-crash-reporter'],
});
const page = await browser.newPage();
page.on('pageerror', (e) => console.log(`[pageerror] ${e.message}`));
page.on('console', (m) => {
  const t = m.text();
  if (!/vite|powerPreference/i.test(t)) console.log(`[page:${m.type()}] ${t}`);
});

await page.goto(HARNESS_URL, { waitUntil: 'load' });
await page.evaluate(() => import('/debug-api.js'));
await page.waitForFunction('window.__VLM_DEBUG_READY__ === true', null, { timeout: 30_000 });

console.log(`repo: ${repo}`);
console.log(`dtypes: ${DTYPES.join(', ')}\n`);

const rows = await page.evaluate(
  async ({ repo, dtypes }) => {
    const out = [];
    for (const dtype of dtypes) {
      const rec = { dtype };
      const t0 = performance.now();
      try {
        const info = await window.__VLM_DEBUG__.load({ repo, dtype });
        rec.loadMs = Math.round(performance.now() - t0);
        rec.modelClass = info.modelClass;

        const receipt = await window.__VLM_DEBUG__.makeSyntheticReceipt();
        const r = await window.__VLM_DEBUG__.attempt({
          prompt: 'What is the total amount on this receipt? Answer with just the number.',
          image: receipt.image,
          maxNewTokens: 48,
        });
        rec.ms = r.ms;
        rec.pixelValues = r.pixelValuesDims;
        rec.decoded = r.decoded;
      } catch (err) {
        rec.error = String(err?.message ?? err).slice(0, 300);
        rec.loadMs = Math.round(performance.now() - t0);
      }
      out.push(rec);
    }
    return out;
  },
  { repo, dtypes: DTYPES },
);

for (const r of rows) {
  console.log(`--- dtype=${r.dtype} (load ${r.loadMs}ms) ---`);
  if (r.error) { console.log(`  ERROR: ${r.error}`); continue; }
  console.log(`  pixelValues=${JSON.stringify(r.pixelValues)} genMs=${r.ms}`);
  console.log(`  decoded: ${JSON.stringify(r.decoded)}`);
}

await browser.close();
