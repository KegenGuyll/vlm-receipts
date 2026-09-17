/**
 * Model registry: the candidate small VLMs, constrained to the realities of a
 * browser tab.
 *
 * Sizes are real on-Hub bytes for each quantization (see
 * `scripts/discover-onnx-models.mjs`), NOT parameter counts. In a PWA the
 * download size *is* the product decision: a 700MB vision model is a non-starter
 * on mobile data even if it fits in VRAM.
 *
 * IMPORTANT — dtype is not a free parameter. `q4f16` and `fp16` were measured to
 * emit degenerate text ("if if if …", "-1: -1:") on this ONNX export while `q4`
 * is correct and fast, so `q4` is the default. See
 * `scripts/sweep-quantizations.mjs`. Re-verify per model before changing.
 */

/**
 * @typedef {'q4f16' | 'q4' | 'fp16' | 'int8' | 'uint8' | 'quantized'} Quant
 */

/** Chromium's spec-default max buffer. onnxruntime-web raises this to adapter
 * limits, which measured 2GiB in the worker on this machine — so this is a
 * floor, not the real ceiling. */
export const SPEC_MAX_BUFFER_BYTES = 268435456;

export const MODELS = [
  {
    id: 'SmolVLM-256M-Instruct',
    repo: 'HuggingFaceTB/SmolVLM-256M-Instruct',
    params: '256M',
    paramClass: 'sub-1B',
    dtype: 'q4',
    approxDownloadMB: 432,
    license: 'Apache-2.0',
    notes: 'Smallest viable VLM with first-class transformers.js support (smolvlm arch).',
  },
  {
    id: 'SmolVLM2-256M-Video-Instruct',
    repo: 'HuggingFaceTB/SmolVLM2-256M-Video-Instruct',
    params: '256M',
    paramClass: 'sub-1B',
    dtype: 'q4',
    approxDownloadMB: 432,
    license: 'Apache-2.0',
    notes: 'SmolVLM2 successor at the same size; newer image/video training mix.',
  },
  {
    id: 'SmolVLM-500M-Instruct',
    repo: 'HuggingFaceTB/SmolVLM-500M-Instruct',
    params: '500M',
    paramClass: 'sub-1B',
    dtype: 'q4',
    approxDownloadMB: 804,
    license: 'Apache-2.0',
    notes: 'Larger SmolVLM v1. Tests whether 2x params is worth 2x download.',
  },
  {
    id: 'SmolVLM2-500M-Video-Instruct',
    repo: 'HuggingFaceTB/SmolVLM2-500M-Video-Instruct',
    params: '500M',
    paramClass: 'sub-1B',
    dtype: 'q4',
    approxDownloadMB: 804,
    license: 'Apache-2.0',
    notes: 'Larger SmolVLM2. Most-downloaded sub-1B VLM on the Hub.',
  },
  {
    id: 'LFM2.5-VL-450M',
    repo: 'onnx-community/LFM2.5-VL-450M-ONNX',
    params: '450M',
    paramClass: 'sub-1B',
    dtype: 'q4f16',
    approxDownloadMB: 602,
    license: 'LFM1.0 (non-OSI)',
    notes:
      'Best pure OCR of the small candidates and the only non-SmolVLM architecture in the lineup, so ' +
      'it is the main cross-family check on the SmolVLM results. NOTE: its processor takes ' +
      '(images, text), unlike the (text, images) order transformers.js assumes — handled by the ' +
      'try-both-orders dispatch in vlm-worker.js. Licence is not OSI-approved; verify before shipping.',
  },
];

export function modelsByRepo() {
  return new Map(MODELS.map((m) => [m.repo, m]));
}

export function getModel(repo) {
  const m = modelsByRepo().get(repo);
  if (!m) throw new Error(`Unknown model repo: ${repo}`);
  return m;
}

/**
 * Candidates that were investigated and excluded, with the evidence.
 *
 * Kept in code rather than only in prose so the lineup's composition is
 * auditable: a reader can see what was tried and why it is absent, instead of
 * assuming the search stopped at four SmolVLM variants.
 */
export const EXCLUDED_MODELS = [
  {
    repo: 'onnx-community/LFM2.5-VL-450M-ONNX',
    params: '450M',
    sizeMB: 602,
    dtype: 'q4f16',
    license: 'LFM1.0 (non-OSI)',
    reason: 'worker-incompatible',
    evidence:
      'Reads receipts BETTER than any measured candidate: on eval-000000 it returned "£184.89" ' +
      'exactly, and on a synthetic receipt "$6.25" exactly. But it fails in the Web Worker with ' +
      '"undefined is not iterable", while the identical call sequence succeeds on the main thread. ' +
      'Its processor is also declared as `_call(images, text)` rather than the `(text, images)` ' +
      'order transformers.js assumes, which this harness works around. Not resolvable within ' +
      'budget, and its non-OSI licence would likely disqualify it anyway.',
    verification: 'scripts/compatibility-check.mjs',
  },
  {
    repo: 'onnx-community/granite-docling-258M-ONNX',
    params: '258M',
    sizeMB: 290,
    dtype: 'uint8',
    license: 'Apache-2.0',
    reason: 'not-an-extractor',
    evidence:
      'Loads cleanly (idefics3 architecture, same as SmolVLM) and is the smallest candidate at ' +
      '290MB, but it is a document-conversion model, not an instruction follower: asked for ' +
      'receipt JSON it recites the prompt schema back ("\'merchant\': string, \'total\': number") ' +
      'instead of reading the image. It also fails in ~14s per receipt.',
    verification: 'scripts/probe-model.mjs',
  },
  {
    repo: 'onnx-community/FastVLM-0.5B-ONNX',
    params: '0.5B',
    sizeMB: 2072,
    dtype: 'q4',
    license: 'apple-amlr',
    reason: 'too-large-and-restrictive',
    evidence:
      '2 GB at q4 would blow the download budget for a PWA, and the Apple AMLR licence is not ' +
      'OSI-approved. Not benchmarked.',
    verification: 'scripts/model-footprint.mjs',
  },
  {
    repo: 'onnx-community/Qwen2-VL-2B-Instruct',
    params: '2B',
    sizeMB: 2545,
    dtype: 'q4f16',
    license: 'Apache-2.0',
    reason: 'above-size-target',
    evidence: 'Above the sub-1B target and 2.5 GB at q4f16. Not benchmarked.',
    verification: 'scripts/discover-onnx-models.mjs',
  },
  {
    repo: 'onnx-community/Qwen2.5-VL-3B-Instruct',
    params: '3B',
    sizeMB: null,
    dtype: null,
    license: 'Apache-2.0',
    reason: 'no-onnx-weights',
    evidence:
      'transformers.js supports the qwen2_5_vl architecture, but the repo does not exist and no ' +
      'first-party ONNX export is published, so it cannot be run out of the box.',
    verification: 'scripts/check-extra-models.mjs',
  },
  {
    repo: 'Xenova/moondream2',
    params: '1.86B',
    sizeMB: 707,
    dtype: 'q4f16',
    license: 'Apache-2.0',
    reason: 'above-size-target',
    evidence:
      'Above the sub-1B target. Retained only as a potential accuracy reference if the size budget ' +
      'is ever relaxed.',
    verification: 'scripts/discover-onnx-models.mjs',
  },
];
