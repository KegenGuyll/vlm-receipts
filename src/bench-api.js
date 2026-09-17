/**
 * Page-side benchmark API.
 *
 * Exposed as `window.__VLM_BENCH__` so the Node runner can drive real in-tab
 * inference through Playwright. Deliberately thin: all model work happens in the
 * worker, this only brokers requests, forwards progress, and owns the worker
 * lifecycle.
 */
import { MODELS } from './models.js';

const worker = new Worker(new URL('./vlm-worker.js', import.meta.url), { type: 'module' });

let nextId = 1;
const pending = new Map();
/** Progress/stream listeners, keyed by request id. */
const listeners = new Map();

worker.addEventListener('message', (event) => {
  const msg = event.data ?? {};

  if (msg.type === 'ready') return;

  if (msg.type === 'progress' || msg.type === 'stream' || msg.type === 'gpu-info' || msg.type === 'debug') {
    // Broadcast unsolicited events (gpu-info, debug) and route request-scoped ones.
    if (msg.type === 'debug') console.log('[worker:debug]', JSON.stringify(msg));
    for (const fn of listeners.values()) {
      try { fn(msg); } catch { /* listener errors must not break inference */ }
    }
    return;
  }
  if (msg.id == null) return;
  const entry = pending.get(msg.id);
  if (!entry) return;
  pending.delete(msg.id);
  listeners.delete(msg.id);
  try {
    if (msg.ok) entry.resolve(msg.result);
    else {
      // Preserve the worker-side stack: worker errors otherwise surface as a bare
      // message across the postMessage boundary, which makes them undebuggable.
      const err = new Error(msg.error || 'worker error');
      if (msg.stack) err.stack = `${msg.error}\n--- worker stack ---\n${msg.stack}`;
      if (msg.diagnostics) err.diagnostics = msg.diagnostics;
      entry.reject(err);
    }
  } catch (err) {
    // Surface the real failure instead of letting it look like a worker error.
    err.message = `[bench-api dispatch] ${err.message}; msg.ok=${msg.ok} keys=${Object.keys(msg).join(',')} resultType=${typeof msg.result}`;
    entry.reject(err);
  }
});

worker.addEventListener('error', (event) => {
  for (const entry of pending.values()) {
    entry.reject(new Error(`worker crash: ${event.message ?? 'unknown'}`));
  }
  pending.clear();
  listeners.clear();
});

function invoke(op, payload, onEvent) {
  const id = nextId++;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    if (onEvent) listeners.set(id, onEvent);
    worker.postMessage({ id, op, payload });
  });
}

/** Collect streamed tokens so a silent streamer can't lose the output. */
function makeEventSink(onProgress) {
  const streamed = [];
  return {
    streamed,
    handler: (msg) => {
      if (msg.type === 'stream' && msg.token) streamed.push(msg.token);
      if (onProgress) onProgress(msg);
    },
  };
}

const cacheNames = async () => (typeof caches === 'undefined' ? [] : await caches.keys());

window.__VLM_BENCH__ = {
  models: MODELS,

  /** Confirm the worker booted and report what the page can see. */
  async ping() {
    const [pong, gpu] = await Promise.all([
      invoke('ping', {}),
      invoke('gpu-info', {}),
    ]);
    return {
      pong,
      gpu,
      crossOriginIsolated: self.crossOriginIsolated ?? false,
      caches: await cacheNames(),
      userAgent: navigator.userAgent,
      hardwareConcurrency: navigator.hardwareConcurrency,
      deviceMemory: navigator.deviceMemory ?? null,
    };
  },

  gpuInfo() {
    return invoke('gpu-info', {});
  },

  async cacheStatus() {
    const names = await cacheNames();
    const detail = [];
    for (const name of names) {
      const cache = await caches.open(name);
      const keys = await cache.keys();
      let bytes = 0;
      for (const req of keys) {
        const res = await cache.match(req);
        const len = res?.headers?.get('content-length');
        if (len) bytes += Number(len);
      }
      detail.push({ name, entries: keys.length, bytes });
    }
    return detail;
  },

  async clearCache() {
    const names = await cacheNames();
    for (const name of names) await caches.delete(name);
    return { cleared: names };
  },

  /**
   * Load a model on WebGPU. `repo` must be one of the registry entries.
   * Returns load timing plus accumulated download bytes.
   */
  load({ repo, dtype, device = 'webgpu', revision, onProgress }) {
    return invoke('load', { repo, dtype, device, revision }, (msg) => {
      if (msg.type === 'progress' && onProgress) onProgress(msg);
      if (msg.type === 'gpu-info' && onProgress) onProgress(msg);
    });
  },

  unload() {
    return invoke('unload', {});
  },

  /**
   * Run one receipt image.
   * `constrain` enables JSON-schema constrained decoding, which is the fix for
   * tiny models inventing their own output structure.
   * @param {{ imageBytes: Uint8Array, imageMime: string, prompt: string,
   *           constrain?: boolean, constraintType?: 'json_schema'|'json_object' }} args
   */
  async run({
    imageBytes,
    imageMime,
    prompt,
    maxNewTokens = 256,
    doSample = false,
    temperature,
    constrain = false,
    constraintType = 'json_schema',
    onProgress,
  }) {
    const sink = makeEventSink(onProgress);
    const result = await invoke(
      'run',
      { imageBytes, imageMime, prompt, maxNewTokens, doSample, temperature, constrain, constraintType },
      sink.handler,
    );
    // Fall back to tokens captured on this side if the worker streamer was quiet.
    if (!result.text && sink.streamed.length) {
      result.text = sink.streamed.join('').trim();
      result.textSource = 'main-thread-stream';
    }
    return result;
  },

  MODELS,
};

window.__VLM_BENCH_READY__ = true;

// Surface the API becoming available to a waiting driver.
self.dispatchEvent(new Event('vlm-bench-ready'));
