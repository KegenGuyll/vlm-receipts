/**
 * Browser inference worker.
 *
 * Runs the vision-language model entirely inside a Web Worker so that (a) the
 * page stays responsive, and (b) disposing a model actually frees its WebGPU
 * buffers between candidates — critical when benchmarking several models in one
 * session, since VRAM is the binding constraint.
 *
 * Runs on WebGPU with a WASM fallback. Protocol is request/response over
 * postMessage, with progress events streamed back during download and load.
 */
import {
  AutoProcessor,
  AutoModelForImageTextToText,
  RawImage,
  env,
  TextStreamer,
} from '@huggingface/transformers';
import { StructuredOutputProcessor } from '@huggingface/transformers-structured-output';
import { SCHEMAS } from './receipt-schema.js';

// Let transformers.js use the Cache API so repeated runs don't re-download
// hundreds of MB. Benchmarks must report a cold-start number separately, so the
// runner clears/inspects the cache rather than relying on this being empty.
env.allowLocalModels = false;
env.useBrowserCache = true;

/** @type {{ processor: any, model: any, repo: string, device: string, dtype: string } | null} */
let loaded = null;

/** Serialize work: WebGPU cannot usefully run two models at once. */
let queue = Promise.resolve();
function enqueue(fn) {
  const run = queue.then(fn, fn);
  // Swallow rejections on the chain so one failure doesn't poison the queue;
  // the caller still sees the rejection via the returned promise.
  queue = run.then(
    () => undefined,
    () => undefined,
  );
  return run;
}

function post(msg) {
  self.postMessage(msg);
}

/**
 * Make a value safe to structured-clone across the worker boundary.
 *
 * Tensors and other class instances are not cloneable, and a failed clone
 * throws inside postMessage with a message that points at the caller rather than
 * the real fault — which previously made a successful generation look like a
 * crash. Anything non-primitive is reduced to a description.
 */
function sanitize(value, depth = 0) {
  if (value == null || typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') {
    return value;
  }
  if (depth > 4) return '[truncated]';
  if (Array.isArray(value)) return value.map((v) => sanitize(v, depth + 1));
  if (typeof value === 'object') {
    const out = {};
    for (const [k, v] of Object.entries(value)) {
      if (v == null || typeof v === 'string' || typeof v === 'number' || typeof v === 'boolean') out[k] = v;
      else if (Array.isArray(v)) out[k] = sanitize(v, depth + 1);
      else if (typeof v === 'object') out[k] = sanitize(v, depth + 1);
      else out[k] = String(v);
    }
    return out;
  }
  return String(value);
}

/** Best-effort enumeration of WebGPU adapter limits for the report. */
async function gpuInfo() {
  if (!navigator.gpu) return { available: false };
  try {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) return { available: false, reason: 'no adapter' };
    const info = adapter.info ?? (adapter.requestAdapterInfo ? await adapter.requestAdapterInfo() : {});
    return {
      available: true,
      vendor: info.vendor ?? null,
      architecture: info.architecture ?? null,
      description: info.description ?? null,
      features: [...adapter.features].sort(),
      shaderF16: adapter.features.has('shader-f16'),
      maxBufferSize: adapter.limits?.maxBufferSize ?? null,
    };
  } catch (err) {
    return { available: false, reason: String(err?.message ?? err) };
  }
}

/**
 * Load a model. Reports download progress so the runner can separate
 * network-bound cold start from compute-bound warm inference.
 */
async function loadModel({ repo, dtype, device, revision }) {
  const info = await gpuInfo();
  post({ type: 'gpu-info', gpu: info });

  const effectiveDevice = device === 'webgpu' && !info.available ? 'wasm' : device;
  const started = performance.now();
  const files = new Map();

  const progress_callback = (p) => {
    if (p.status === 'progress' && p.file) {
      files.set(p.file, { loaded: p.loaded ?? 0, total: p.total ?? 0 });
      let loadedBytes = 0;
      let totalBytes = 0;
      for (const f of files.values()) {
        loadedBytes += f.loaded;
        totalBytes += f.total;
      }
      post({
        type: 'progress',
        stage: 'download',
        file: p.file,
        percent: p.progress ?? null,
        loadedBytes,
        totalBytes,
      });
    } else {
      post({ type: 'progress', stage: p.status ?? 'unknown', file: p.file ?? null });
    }
  };

  const processor = await AutoProcessor.from_pretrained(repo, { progress_callback, revision });
  const model = await AutoModelForImageTextToText.from_pretrained(repo, {
    dtype,
    device: effectiveDevice,
    progress_callback,
    revision,
  });

  const downloadBytes = [...files.values()].reduce((a, f) => a + f.loaded, 0);
  const loadMs = performance.now() - started;

  loaded = { processor, model, repo, device: effectiveDevice, dtype };
  return { loadMs, downloadBytes, device: effectiveDevice, dtype, fileCount: files.size };
}

/**
 * Invoke a multimodal processor with the argument order that model family needs.
 *
 * transformers.js calls processors generically as `processor(text, images)`,
 * which matches Idefics3/SmolVLM and Qwen-VL. `Lfm2VlProcessor` instead declares
 * `_call(images, text, kwargs)` — see `src/models/lfm2_vl/processing_lfm2_vl.js` —
 * so the generic order hands a string to the image processor, which throws
 * "undefined is not iterable".
 *
 * Rather than keying off `constructor.name` (which is unreliable: what
 * `AutoProcessor.from_pretrained` returns is not always a direct instance of the
 * processor class, so the name check silently failed for LFM2.5-VL), attempt one
 * order and fall back to the other on throw. The wrong order fails fast before
 * any expensive work, so the cost of a failed attempt is negligible, and this
 * needs no per-family knowledge at all.
 *
 * @param {any} processor
 * @param {string} text rendered chat template
 * @param {any} image RawImage | null
 */
async function callProcessor(processor, text, image) {
  const imagesArg = image ? [image] : null;
  const cls = processor?.constructor?.name ?? 'unknown';

  if (!imagesArg) {
    return await processor(text, null);
  }

  // Order 1: (text, images) — SmolVLM / Idefics3 / Qwen-VL.
  try {
    return await processor(text, imagesArg);
  } catch (firstErr) {
    // Order 2: (images, text) — LFM2-VL.
    try {
      const inputs = await processor(imagesArg, text);
      post({ type: 'debug', stage: 'processor-order', processorClass: cls, used: 'images-first' });
      return inputs;
    } catch (secondErr) {
      // Report both failures: the first is usually the informative one.
      throw new Error(
        `processor rejected both argument orders. class=${cls}. ` +
        `(text,images): ${firstErr?.message ?? firstErr}; (images,text): ${secondErr?.message ?? secondErr}`,
      );
    }
  }
}

/** Release the model and drop references so WebGPU memory can be reclaimed. */
async function unloadModel() {  if (!loaded) return { unloaded: false };
  try {
    await loaded.model?.dispose?.();
  } catch { /* best effort */ }
  loaded = null;
  // Nudge the GC where exposed; harmless if not.
  try { globalThis.gc?.(); } catch { /* ignore */ }
  return { unloaded: true };
}

async function runReceipt({
  imageBytes,
  imageMime,
  prompt,
  maxNewTokens,
  doSample,
  temperature,
  constrain,
  constraintType = 'json_schema',
  schemaName = 'full',
}) {
  if (!loaded) throw new Error('no model loaded');
  const { processor, model } = loaded;

  // Recorded as returned data, not log events: worker stack traces do not
  // survive postMessage, so an opaque failure needs a step-by-step trail.
  const diag = { processorClass: processor?.constructor?.name ?? null, steps: [] };
  const mark = (name, extra) => diag.steps.push({ name, ms: Math.round(performance.now() - t0), ...(extra ?? {}) });

  const t0 = performance.now();
  // Pass the original encoded bytes: decoding happens once, and no re-encode is
  // needed. `RawImage` handles both clean PNGs and degraded JPEGs.
  const image = imageBytes
    ? await RawImage.fromBlob(new Blob([imageBytes], { type: imageMime || 'image/png' }))
    : null;
  mark('decode-image', { width: image?.width ?? null, height: image?.height ?? null, channels: image?.channels ?? null });

  const messages = [
    {
      role: 'user',
      content: [
        { type: 'image' },
        { type: 'text', text: prompt },
      ],
    },
  ];

  let text;
  try {
    text = processor.apply_chat_template(messages, { add_generation_prompt: true });
  } catch (err) {
    const e = new Error(`apply_chat_template failed: ${err?.message ?? err}`);
    e.diagnostics = diag;
    throw e;
  }
  mark('chat-template', { textLength: typeof text === 'string' ? text.length : null });

  const tPre = performance.now();
  let inputs;
  try {
    inputs = await callProcessor(processor, text, image, model);
  } catch (err) {
    const e = new Error(`processor failed: ${err?.message ?? err}`);
    e.diagnostics = diag;
    throw e;
  }
  mark('process', { keys: Object.keys(inputs ?? {}) });
  const tPost = performance.now();

  let outputText = '';
  let tokenCount = 0;
  const streamer = new TextStreamer(processor.tokenizer, {
    skip_prompt: true,
    skip_special_tokens: true,
    callback_function: (tok) => {
      outputText += tok;
      tokenCount++;
      post({ type: 'stream', token: tok });
    },
  });

  // MEASURED CAVEAT: `StructuredOutputProcessor extends LogitsProcessorList`, but
  // the generation loop invokes `logits_processor` as a `Callable`. The object is
  // therefore accepted and then silently ignored — output is identical whether
  // the schema has 5 properties or 6, and `line_items` appears even when the
  // schema forbids it. So this does NOT constrain the grammar; treat it as an
  // unverified path. What it does do is force `repetition_penalty` to 1.0, and
  // that alone is worth measurable accuracy on the 256M tier.
  //
  // The honest name for this switch is "disable the repetition penalty".
  let logits_processor;
  let constraintMs = null;
  if (constrain) {
    const tc0 = performance.now();
    logits_processor = new StructuredOutputProcessor(
      processor.tokenizer,
      constraintType === 'json_object'
        ? { type: 'json_object' }
        : { type: 'json_schema', json_schema: SCHEMAS[schemaName] ?? SCHEMAS.full },
    );
    constraintMs = performance.now() - tc0;
  }

  const tGen0 = performance.now();
  let output;
  try {
    output = await model.generate({
      ...inputs,
      max_new_tokens: maxNewTokens ?? 256,
      do_sample: doSample ?? false,
      ...(doSample ? { temperature: temperature ?? 0.2 } : {}),
      // Repetition penalty fights any grammar that legitimately repeats tokens
      // (several `"amount"` keys), so it is disabled in this mode. It is NOT
      // disabled in the default path, where it is harmless.
      repetition_penalty: constrain ? 1.0 : 1.1,
      ...(logits_processor ? { logits_processor } : {}),
      streamer,
    });
  } catch (err) {
    const e = new Error(`generate failed: ${err?.message ?? err}`);
    e.diagnostics = diag;
    throw e;
  }
  mark('generate');
  const tGen1 = performance.now();

  // Prefer the streamer text; fall back to decoding the raw ids.
  let finalText = outputText.trim();
  if (!finalText && output) {
    try {
      const ids = output.slice(null, [inputs.input_ids.dims[1], null]);
      finalText = processor.tokenizer.batch_decode(ids, { skip_special_tokens: true })[0]?.trim() ?? '';
    } catch { /* leave empty */ }
  }

  const generateMs = tGen1 - tGen0;
  const metrics = {
    preprocessMs: tPre - t0,
    processorMs: tPost - tPre,
    generateMs,
    totalMs: tGen1 - t0,
    outputTokens: tokenCount || null,
    tokensPerSecond: generateMs > 0 && tokenCount ? (tokenCount / generateMs) * 1000 : null,
    inputTokens: inputs?.input_ids?.dims?.[1] ?? null,
    imageTiles: inputs?.pixel_values?.dims?.[0] ?? null,
    constrained: !!constrain,
    constraintBuildMs: constraintMs,
    diagnostics: diag,
  };

  return {
    text: finalText,
    metrics,
    streamedTokens: tokenCount,
    rawDims: output?.dims ?? null,
    constrained: !!constrain,
  };
}

self.addEventListener('message', (event) => {
  const { id, op, payload } = event.data ?? {};
  if (!id || !op) return;

  const handlers = {
    'gpu-info': () => gpuInfo(),
    load: () => loadModel(payload),
    unload: () => unloadModel(),
    run: () => runReceipt(payload),
    ping: () => ({ pong: true }),
  };

  const handler = handlers[op];
  if (!handler) {
    self.postMessage({ id, ok: false, error: `unknown op: ${op}` });
    return;
  }

  enqueue(async () => {
    try {
      const result = await handler();
      self.postMessage({ id, ok: true, result });
    } catch (err) {
      // Worker stack traces do NOT survive postMessage (they come back as a bare
      // message pointing at the main-thread handler), so log the full stack here
      // where it is still real, and pass structured diagnostics as data.
      const stack = String(err?.stack ?? '');
      const diagnostics = sanitize(err?.diagnostics ?? null);
      console.error(`[vlm-worker] ${op} failed: ${err?.message ?? err}\n${stack}`);
      self.postMessage({
        id,
        ok: false,
        error: String(err?.message ?? err),
        // Legacy field kept for compatibility; the authoritative trace is above.
        stack,
        diagnostics,
      });
    }
  });
});

post({ type: 'ready' });
