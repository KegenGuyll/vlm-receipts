/**
 * Trace the Lfm2VlProcessor failure to its exact origin.
 *
 * Calling with images-first order still throws, so the fault is inside the image
 * processor rather than the argument order. This walks the pipeline in
 * isolation and captures full stack traces.
 *
 *   node scripts/trace-lfm2.mjs [repo]
 */
import { chromium } from 'playwright-core';

const CHROME = 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';
const HARNESS_URL = process.env.HARNESS_URL ?? 'http://127.0.0.1:5179/';
const repo = process.argv[2] ?? 'onnx-community/LFM2.5-VL-450M-ONNX';

const browser = await chromium.launch({
  executablePath: CHROME,
  headless: true,
  args: ['--no-sandbox', '--disable-gpu-sandbox', '--enable-unsafe-webgpu', '--no-first-run', '--disable-crash-reporter'],
});
const page = await browser.newPage();
page.on('pageerror', (e) => console.log(`[pageerror] ${e.message}`));
await page.goto(HARNESS_URL, { waitUntil: 'load' });
await page.evaluate(() => import('/debug-api.js'));
await page.waitForFunction('window.__VLM_DEBUG_READY__ === true', null, { timeout: 30_000 });

const out = await page.evaluate(async ({ repo }) => {
  const D = window.__VLM_DEBUG__;
  const res = {};
  const processor = await D.tf.AutoProcessor.from_pretrained(repo);
  const receipt = await D.makeSyntheticReceipt();
  res.imageDims = [receipt.image.width, receipt.image.height];

  const vis = processor.image_processor ?? processor.components?.find?.((c) => c.constructor?.name?.includes('Image'));
  res.imageProcessorClass = vis?.constructor?.name ?? null;

  // 1. Image processor alone.
  try {
    const ip = await vis([receipt.image], { return_row_col_info: true });
    res.imageProcessorKeys = Object.keys(ip);
    res.imageProcessorDims = Object.fromEntries(
      Object.entries(ip).map(([k, v]) => [k, v?.dims ?? (Array.isArray(v) ? `array(${v.length})` : typeof v)]),
    );
  } catch (err) {
    res.imageProcessorError = String(err?.message ?? err);
    res.imageProcessorStack = String(err?.stack ?? '').split('\n').slice(0, 6).join('\n');
  }

  // 2. Full processor, images-first.
  const text = processor.apply_chat_template(
    [{ role: 'user', content: [{ type: 'image' }, { type: 'text', text: 'What is the total?' }] }],
    { add_generation_prompt: true },
  );
  try {
    const inputs = await processor([receipt.image], text);
    res.imagesFirstKeys = Object.keys(inputs);
    res.imagesFirstDims = Object.fromEntries(Object.entries(inputs).map(([k, v]) => [k, v?.dims ?? typeof v]));
  } catch (err) {
    res.imagesFirstError = String(err?.message ?? err);
    res.imagesFirstStack = String(err?.stack ?? '').split('\n').slice(0, 8).join('\n');
  }

  // 3. Text-only tokenization, to confirm the tokenizer is fine.
  try {
    const tok = processor.tokenizer(text);
    res.tokenizerDims = Object.fromEntries(Object.entries(tok).map(([k, v]) => [k, v?.dims ?? typeof v]));
  } catch (err) {
    res.tokenizerError = String(err?.message ?? err);
  }

  return res;
}, { repo });

console.log(JSON.stringify(out, null, 2));
await browser.close();
