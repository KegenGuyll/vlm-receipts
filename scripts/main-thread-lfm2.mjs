/**
 * Reproduce the LFM2.5-VL failure on the MAIN thread so the stack trace survives.
 *
 * Worker stack traces do not cross postMessage, so the failure has been
 * undebuggable from the runner. This drives the raw transformers.js classes
 * directly from the page, where errors are complete.
 *
 *   node scripts/main-thread-lfm2.mjs [repo]
 */
import { chromium } from 'playwright-core';
import { readFile } from 'node:fs/promises';
import path from 'node:path';

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

// Feed the real fixture so the worker's exact image path is reproduced.
const manifest = JSON.parse(await readFile(path.join('fixtures', 'eval', 'manifest.json'), 'utf8'));
const entry = manifest.entries[0];
await page.evaluate(
  async ({ b64, mime }) => {
    window.__FIXTURE_B64__ = b64;
    window.__FIXTURE_MIME__ = mime;
  },
  {
    b64: (await readFile(path.join(process.cwd(), entry.image))).toString('base64'),
    mime: entry.ext === 'jpg' ? 'image/jpeg' : 'image/png',
  },
);

const out = await page.evaluate(async ({ repo }) => {
  const D = window.__VLM_DEBUG__;
  const res = { repo };

  const processor = await D.tf.AutoProcessor.from_pretrained(repo);
  const model = await D.tf.AutoModelForImageTextToText.from_pretrained(repo, {
    dtype: 'q4f16',
    device: 'webgpu',
  });
  res.modelClass = model.constructor?.name;

  const receipt = await D.makeSyntheticReceipt();
  const messages = [{ role: 'user', content: [{ type: 'image' }, { type: 'text', text: 'What is the total amount on this receipt?' }] }];
  const text = processor.apply_chat_template(messages, { add_generation_prompt: true });

  // Images-first, per Lfm2VlProcessor._call(images, text, kwargs).
  const inputs = await processor([receipt.image], text);
  res.inputKeys = Object.keys(inputs);
  res.inputShapes = Object.fromEntries(Object.entries(inputs).map(([k, v]) => [k, v?.dims ?? typeof v]));

  try {
    const output = await model.generate({ ...inputs, max_new_tokens: 64, do_sample: false });
    const ids = output.slice(null, [inputs.input_ids.dims[1], null]);
    res.decoded = processor.tokenizer.batch_decode(ids, { skip_special_tokens: true })[0];
    res.outputDims = output.dims;
  } catch (err) {
    res.generateError = String(err?.message ?? err);
    res.generateStack = String(err?.stack ?? '').split('\n').slice(0, 14);
  }

  // Hypothesis: the worker's TextStreamer is what breaks this model. Reproduce
  // the worker's exact streaming configuration.
  try {
    let streamed = '';
    const streamer = new D.tf.TextStreamer(processor.tokenizer, {
      skip_prompt: true,
      skip_special_tokens: true,
      callback_function: (tok) => { streamed += tok; },
    });
    const out2 = await model.generate({ ...inputs, max_new_tokens: 32, do_sample: false, streamer });
    res.streamerOk = true;
    res.streamed = streamed.slice(0, 200);
    res.streamerDims = out2.dims;
  } catch (err) {
    res.streamerError = String(err?.message ?? err);
    res.streamerStack = String(err?.stack ?? '').split('\n').slice(0, 14);
  }
  // Hypothesis 2: the failure is specific to the REAL fixture image (JPEG),
  // which may take a different tiling path than a synthetic canvas PNG. Accept an
  // optional base64 fixture and run the worker's exact image construction.
  if (window.__FIXTURE_B64__) {
    const bin = atob(window.__FIXTURE_B64__);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    const blob = new Blob([bytes], { type: window.__FIXTURE_MIME__ || 'image/jpeg' });
    const rawImage = await D.tf.RawImage.fromBlob(blob);
    res.fixtureImageDims = [rawImage.width, rawImage.height, rawImage.channels];
    res.fixtureBytes = bytes.length;
    try {
      const finputs = await processor([rawImage], text);
      res.fixtureInputShapes = Object.fromEntries(
        Object.entries(finputs).map(([k, v]) => [k, v?.dims ?? (Array.isArray(v) ? `array(${v.length})` : typeof v)]),
      );
      const fout = await model.generate({ ...finputs, max_new_tokens: 48, do_sample: false });
      const fids = fout.slice(null, [finputs.input_ids.dims[1], null]);
      res.fixtureDecoded = processor.tokenizer.batch_decode(fids, { skip_special_tokens: true })[0];
    } catch (err) {
      res.fixtureError = String(err?.message ?? err);
      res.fixtureStack = String(err?.stack ?? '').split('\n').slice(0, 14);
    }
  }
  return res;
}, { repo });

console.log(JSON.stringify(out, null, 2));
await browser.close();
