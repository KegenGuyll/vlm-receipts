/**
 * Debug/inspection entry point.
 *
 * Exists so diagnostics can reach the actual transformers.js classes (Vite only
 * serves modules under `src`), while keeping the production harness in
 * `bench-api.js` free of test scaffolding.
 *
 * Imported on demand by diagnostic scripts; never part of the normal bench path.
 */
import {
  AutoProcessor,
  AutoModelForImageTextToText,
  RawImage,
  env,
  TextStreamer,
  Tensor,
} from '@huggingface/transformers';
import { StructuredOutputProcessor } from '@huggingface/transformers-structured-output';
import { SCHEMAS } from './receipt-schema.js';

const state = { processor: null, model: null, repo: null };

window.__VLM_DEBUG__ = {
  tf: { AutoProcessor, AutoModelForImageTextToText, RawImage, env, TextStreamer, Tensor },

  async load({ repo, dtype = 'q4f16', device = 'webgpu' }) {
    env.allowLocalModels = false;
    env.useBrowserCache = true;
    const processor = await AutoProcessor.from_pretrained(repo);
    const model = await AutoModelForImageTextToText.from_pretrained(repo, { dtype, device });
    state.processor = processor;
    state.model = model;
    state.repo = repo;
    return {
      repo,
      processorClass: processor.constructor?.name,
      modelClass: model.constructor?.name,
      hasChatTemplate: !!processor.tokenizer?.chat_template,
      chatTemplate: String(processor.tokenizer?.chat_template ?? ''),
    };
  },

  /** Render a receipt onto a canvas so diagnostics need no fixture files. */
  async makeSyntheticReceipt() {
    const canvas = new OffscreenCanvas(420, 280);
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, 420, 280);
    ctx.fillStyle = '#000';
    ctx.font = '18px monospace';
    const lines = [
      'FRESH MART',
      '12 OAK AVENUE',
      '2024-03-12',
      'Coffee          3.50',
      'Bagel           2.25',
      'TOTAL           6.25',
    ];
    lines.forEach((l, i) => ctx.fillText(l, 16, 36 + i * 36));
    const blob = await canvas.convertToBlob({ type: 'image/png' });
    const bytes = new Uint8Array(await blob.arrayBuffer());
    const image = await RawImage.fromBlob(blob);
    return { bytes, image, width: image.width, height: image.height, channels: image.channels };
  },

  /**
   * One controlled generation attempt with full visibility into the inputs.
   * `image` may be a RawImage from makeSyntheticReceipt or decoded fixture bytes.
   * Set `constrain: true` to apply JSON-schema constrained decoding.
   */
  async attempt({
    prompt,
    image,
    maxNewTokens = 64,
    generateKwargs = {},
    inspectTemplate = false,
    constrain = false,
    schemaName = 'full',
  }) {
    const { processor, model } = state;
    if (!processor) throw new Error('call load() first');

    const messages = [{ role: 'user', content: [{ type: 'image' }, { type: 'text', text: prompt }] }];
    const text = processor.apply_chat_template(messages, { add_generation_prompt: true });
    const inputs = await processor(text, image ? [image] : null);

    let logits_processor;
    let constraintMs = null;
    if (constrain) {
      const tc0 = performance.now();
      logits_processor = new StructuredOutputProcessor(processor.tokenizer, {
        type: 'json_schema',
        json_schema: SCHEMAS[schemaName] ?? SCHEMAS.full,
      });
      constraintMs = performance.now() - tc0;
    }

    const t0 = performance.now();
    const output = await model.generate({
      ...inputs,
      max_new_tokens: maxNewTokens,
      do_sample: false,
      // See vlm-worker.js: the penalty conflicts with a grammar that may
      // legitimately require repeated tokens.
      repetition_penalty: constrain ? 1.0 : 1.1,
      ...(logits_processor ? { logits_processor } : {}),
      ...generateKwargs,
    });
    const ms = performance.now() - t0;

    const ids = output.slice(null, [inputs.input_ids.dims[1], null]);
    const decoded = processor.tokenizer.batch_decode(ids, { skip_special_tokens: true })[0];

    return {
      template: inspectTemplate ? text : undefined,
      templateLength: text.length,
      imageTokenCount: (text.match(/<image>/g) ?? []).length
        + (text.match(/<img>/g) ?? []).length
        + (text.match(/<row_\d+_col_\d+>/g) ?? []).length,
      inputTokens: inputs.input_ids.dims[1],
      pixelValuesDims: inputs.pixel_values?.dims ?? null,
      outputDims: output.dims,
      ms: Math.round(ms),
      constrained: constrain,
      constraintMs,
      decoded: (decoded ?? '').slice(0, 800),
    };
  },
};

window.__VLM_DEBUG_READY__ = true;
