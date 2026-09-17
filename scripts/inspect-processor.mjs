/**
 * Inspect an unfamiliar VLM's processor to learn what call shape it needs.
 *
 * LFM2.5-VL fails in the shared image-text-to-text path with "undefined is not
 * iterable", which means its processor expects different arguments than SmolVLM's
 * Idefics3 processor. This prints the class, config, and the exact error so the
 * correct call shape can be derived rather than guessed.
 *
 *   node scripts/inspect-processor.mjs <repo>
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

const report = await page.evaluate(async ({ repo }) => {
  const D = window.__VLM_DEBUG__;
  const out = { repo, steps: [] };

  const processor = await D.tf.AutoProcessor.from_pretrained(repo);
  out.processorClass = processor.constructor?.name;
  out.hasChatTemplate = !!processor.tokenizer?.chat_template;
  out.chatTemplate = String(processor.tokenizer?.chat_template ?? '').slice(0, 800);
  out.processorKeys = Object.keys(processor);
  out.imageProcessorClass = processor.image_processor?.constructor?.name ?? null;
  out.imageProcessorConfig = processor.image_processor?.config ?? null;

  const receipt = await D.makeSyntheticReceipt();

  // Try the documented chat-template shape first.
  const messages = [{ role: 'user', content: [{ type: 'image' }, { type: 'text', text: 'What is the total on this receipt?' }] }];
  try {
    const text = processor.apply_chat_template(messages, { add_generation_prompt: true });
    out.chatTemplateText = text;
    try {
      const inputs = await processor(text, [receipt.image]);
      out.chatShapeInputs = Object.fromEntries(Object.entries(inputs).map(([k, v]) => [k, v?.dims ?? typeof v]));
    } catch (err) {
      out.chatShapeError = String(err?.message ?? err);
    }
  } catch (err) {
    out.chatTemplateError = String(err?.message ?? err);
  }

  // Try the "text + images" alternative shape some processors expect.
  try {
    const inputs = await processor('What is the total on this receipt?', [receipt.image]);
    out.altShapeInputs = Object.fromEntries(Object.entries(inputs).map(([k, v]) => [k, v?.dims ?? typeof v]));
  } catch (err) {
    out.altShapeError = String(err?.message ?? err);
  }

  return out;
}, { repo });

console.log(JSON.stringify(report, null, 2));
await browser.close();
