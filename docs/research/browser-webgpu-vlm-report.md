# Running small VLMs in the browser via WebGPU for an on-device receipt parser

**Research date: 2026-09-17.** All versions and model lists below were verified against live sources on that date, not from memory. Every non-obvious claim carries an inline link.

**TL;DR for the decision:** WebLLM (0.2.85) does support image input, but its *prebuilt* model list contains **exactly two VLM entries and both are Phi-3.5-vision (4.2B, ~2.8 GB)** — there is no sub-1B VLM path in WebLLM today. The sub-1B story lives entirely in **transformers.js v4.3.0**, where the best options are `onnx-community/LFM2.5-VL-450M-ONNX` (~392 MB at q4f16, best OCR per MB), `HuggingFaceTB/SmolVLM-256M-Instruct` (~189 MB at q4f16, Apache-2.0) and `HuggingFaceTB/SmolVLM-500M-Instruct` (~358 MB at q4f16, Apache-2.0). Also new and directly relevant: **transformers.js 4.3.0 shipped built-in JSON-schema constrained generation yesterday** (`@huggingface/transformers-structured-output`).

**How to read this:** the sections are numbered 1–5 in the order asked, but **§3 (Practical browser constraints) sits at the end of the document** because it was the deepest research pass and was revised last. An unabridged companion file with the full browser/WebGPU research — per-platform limit measurement tables, the mobile defect list, and a nine-item uncertainty list — is at `webgpu-vlm-sept-2026.md` in this workspace.

**Contents (reading order):**
1. [WebLLM `@mlc-ai/web-llm`](#1-webllm-mlc-aiweb-llm) — line 11
2. [transformers.js `@huggingface/transformers`](#2-transformersjs-huggingfacetransformers) — line 72
3. [Recommended candidate shortlist](#4-recommended-candidate-shortlist) — line 201
4. [Gotchas that will bite an implementer](#5-gotchas-that-will-bite-an-implementer) — line 226
5. [Confidence & gaps](#confidence--gaps) — line 267
6. [Practical browser constraints (§3)](#3-practical-browser-constraints) — line 295

---

## 1. WebLLM (`@mlc-ai/web-llm`)

### Version

- Current npm version: **`0.2.85`**, published from tarball `web-llm-0.2.85.tgz` ([npm registry](https://registry.npmjs.org/@mlc-ai/web-llm/latest)). The package ships a single 6.6 MB `lib/index.js` plus `.d.ts` files, not per-module JS ([jsDelivr package file listing](https://data.jsdelivr.com/v1/packages/npm/@mlc-ai/web-llm@0.2.85)).

### Does it support vision/multimodal image input? — **Yes, genuinely supported, but only for one model family**

Vision support landed in [PR #563 "[Vision] Support Phi-3.5-vision, the first VLM in WebLLM"](https://github.com/mlc-ai/web-llm/pull/563), merged **2024-09-23**. It generalised `Conversation.messages` to `Array<ChatCompletionContentPart>`, added `getChunkedPrefillInputData()` (a single image cannot be chunked while embedding), replaced `forward()` with `embedAndForward()`, and added `getImageEmbeddings()` / `getTokensEmbeddings()`.

The API is OpenAI-compatible content parts, exactly as in the PR description:

```js
const messages = [{
  role: "user",
  content: [
    { type: "text", text: "List the items in the image concisely." },
    { type: "image_url", image_url: { url: "https://.../sunset.jpg" } },
  ],
}];
const reply = await engine.chat.completions.create({ stream: false, messages });
```

The `ModelType` enum in the shipped v0.2.85 source contains a first-class `VLM` member ([`src/config.ts` @ tag v0.2.85](https://raw.githubusercontent.com/mlc-ai/web-llm/v0.2.85/src/config.ts)).

### Exact VLM entries in `prebuiltAppConfig.model_list`

I grepped `src/config.ts` at the published tag and found **exactly two** `model_type: ModelType.VLM` entries. Both are Phi-3.5-vision-instruct (4.2B params):

| `model_id` | `vram_required_MB` | context | `low_resource_required` | model lib |
|---|---|---|---|---|
| `Phi-3.5-vision-instruct-q4f16_1-MLC` | **3952.18** | 4096 | true | `Phi-3.5-vision-instruct-q4f16_1_cs2k-webgpu.wasm` |
| `Phi-3.5-vision-instruct-q4f32_1-MLC` | **5879.84** | 4096 | true | `Phi-3.5-vision-instruct-q4f32_1_cs2k-webgpu.wasm` |

Source: [`src/config.ts` @ v0.2.85, Phi-3.5-vision block](https://raw.githubusercontent.com/mlc-ai/web-llm/v0.2.85/src/config.ts). (`modelVersion` is `v0_2_84/base`.)

**Download size**: the q4f16 weight repo `mlc-ai/Phi-3.5-vision-instruct-q4f16_1-MLC` contains 107 `params_shard_*.bin` files plus `tokenizer.json` (1.85 MB) ([HF tree API](https://huggingface.co/api/models/mlc-ai/Phi-3.5-vision-instruct-q4f16_1-MLC/tree/main)). Summing the shard sizes gives **≈ 2.77 GB** of weights. So even the smallest WebLLM VLM is a ~2.8 GB download needing ~3.95 GB VRAM — it will not fit comfortably on a phone.

### Nothing else in the list is a VLM

Every other `model_list` entry is text-only. Notably `gemma3-1b-it-q4f16_1-MLC` (vram 711.07 MB) is present but is **text-only** (no `model_type`, no vision fields), so it is *not* a vision option — 1B Gemma 3 checkpoints are text-only. Confirmed by grepping the v0.2.85 config: 43 matches for "gemma", zero of them VLMs ([config.ts](https://raw.githubusercontent.com/mlc-ai/web-llm/v0.2.85/src/config.ts)).

### In-flight VLM work (as of 2026-09-17)

- [PR #804 "[VLM] Dynamic image embed size and generic vision framework"](https://github.com/mlc-ai/web-llm/pull/804) — **merged 2026-05-04**. Replaces the hardcoded `IMAGE_EMBED_SIZE` (1921, Phi-3.5-V specific) with per-model computation, adds BOI/EOI token wrapping for models that need it (both `boi_token_index`/`eoi_token_index` and `vision_start_token_id`/`vision_end_token_id`), and exposes `model_type` / `model_config` from `mlc-chat-config.json` through `ChatConfig`. This is the generic framework the other VLMs need.
- [PR #805 "[VLM] Add Gemma 3 Vision support"](https://github.com/mlc-ai/web-llm/pull/805) — **still open** (opened 2026-04-01, last updated 2026-04-15). Adds `gemma3_v` handling, fixed square resize to `image_size`, single tile (no tiling).
- [PR #806 "[VLM] Add Qwen3.5 Vision support"](https://github.com/mlc-ai/web-llm/pull/806) — **still open** (opened 2026-04-01, last updated 2026-04-15). `computeImageEmbedSize` = `(image_size/patch_size/spatial_merge_size)^2 = 196`.
- The shipped `ChatConfig` doc comment already cites `"gemma3_v"` as an example `model_type` ([config.ts](https://raw.githubusercontent.com/mlc-ai/web-llm/v0.2.85/src/config.ts)), so the plumbing exists ahead of a prebuilt model entry.

### Rough edges an implementer will hit

- **Vision was broken for months.** [Issue #727 "Getting vision models to work — Phi 3.5, Gemma 3"](https://github.com/mlc-ai/web-llm/issues/727) (opened 2025-10-16, closed 2026-02-12), [issue #657 "Issue loading Phi 3.5 vision models"](https://github.com/mlc-ai/web-llm/issues/657) (closed 2026-02-20) and [issue #770](https://github.com/mlc-ai/web-llm/issues/770) (closed 2026-02-18) all report the same fatal error: `ValueError: Cannot find parameter in cache: vision_embed_tokens.img_processor.vision_model.embeddings.position_embedding.q_weight`. A fix PR ([#765](https://github.com/mlc-ai/web-llm/pull/765)) was merged 2026-01-30. Anything written against WebLLM vision before ~Feb 2026 is stale.
- **Accuracy complaints are unresolved.** [Issue #586 "Accuracy of Phi-3.5 Vision Models in Web-LLM is way off"](https://github.com/mlc-ai/web-llm/issues/586) has been **open since 2024-09-30**, with side-by-side examples where WebLLM's Phi-3.5-vision fails to read simple handwritten maths that NVIDIA NIM handles. For dense receipt text this is a real risk.
- **No small VLM has ever been requested successfully.** [Issue #482 "Model request: moondream (tiny vision model)"](https://github.com/mlc-ai/web-llm/issues/482) is **still open since 2024-06-18**; [issue #625 (Llama-3.2-vision)](https://github.com/mlc-ai/web-llm/issues/625) is open since 2024-11-07.

### WebLLM's compensating strength: structured JSON

WebLLM's headline feature for a receipt parser is built-in, WASM-implemented grammar-constrained JSON: "WebLLM supports state-of-the-art JSON mode structured generation, implemented in the WebAssembly portion of the model library for optimal performance", exposed through OpenAI's `response_format` ([WebLLM README](https://raw.githubusercontent.com/mlc-ai/web-llm/main/README.md), and `GenerationConfig.response_format` in [config.ts](https://raw.githubusercontent.com/mlc-ai/web-llm/v0.2.85/src/config.ts)). It also supports `logit_bias`, `seed`, SRI integrity verification of downloaded artifacts, and four cache backends (`cache` default, `indexeddb`, `opfs`, `cross-origin`) ([WebLLM README](https://raw.githubusercontent.com/mlc-ai/web-llm/main/README.md)). If a 4 GB model were acceptable, this would be the strongest JSON guarantee available in-browser.

---

## 2. transformers.js (`@huggingface/transformers`)

### Version — and a documentation trap

- Current npm version: **`4.3.0`**, released **2026-09-16** ([npm registry](https://registry.npmjs.org/@huggingface/transformers/latest), [GitHub release 4.3.0](https://github.com/huggingface/transformers.js/releases/tag/4.3.0)).
- Version history: `3.8.1` (2025-12-02) → **`4.0.0` (2026-03-30)** → `4.1.0` / `4.2.0` (2026-04-23) → **`4.3.0` (2026-09-16)** ([releases API](https://github.com/huggingface/transformers.js/releases)).
- **Trap:** the HF docs sidebar still advertises **`v3.8.1` as "the latest stable version"** and warns that `main` "requires installation from source" ([WebGPU guide, main](https://huggingface.co/docs/transformers.js/guides/webgpu)). The `v3.8.1` guide pages I cite below are therefore one major version behind npm. The docs are stale relative to the registry — trust the release notes and the model cards over the versioned guide pages.
- **v4 changed the WebGPU story substantially:** a new WebGPU runtime "completely rewritten in C++" (ONNX Runtime WebGPU execution provider), support for models over 8B params, and — relevant to us — **WebGPU enabled for Safari 26+** in 4.3.0 ([v4 blog post](https://huggingface.co/blog/transformersjs-v4), [4.3.0 release notes](https://github.com/huggingface/transformers.js/releases/tag/4.3.0)).
- Runtime dependency is `onnxruntime-web` **`1.31.0-dev.20260914-8d85527a0`** and `onnxruntime-node 1.30.0` ([npm registry](https://registry.npmjs.org/@huggingface/transformers/latest)).

### Which image-text tasks are supported

`image-to-text` / image-text-to-text is supported; **`visual-question-answering` is still listed as ❌ unsupported** in the README task table ([transformers.js README](https://raw.githubusercontent.com/huggingface/transformers.js/main/README.md)). In practice VQA is done through the `image-text-to-text` / `AutoModelForImageTextToText` API with a prompt.

### Architectures available on the WebGPU device

From the transformers.js README model list and the v4 release notes, the vision-language architectures implemented are:

| Architecture | Supported | Evidence |
|---|---|---|
| SmolVLM / Idefics3 | ✅ | [PR #1059 "Add support for idefics3 (SmolVLM)"](https://github.com/huggingface/transformers.js/pull/1059); "Re-enable SmolVLM" in 4.1.0 ([PR #1648](https://github.com/huggingface/transformers.js/releases/tag/4.1.0)) |
| Florence-2 | ✅ | README model list; [Florence-2 ONNX card](https://huggingface.co/onnx-community/Florence-2-base-ft) |
| Moondream (moondream1) | ✅ | [README model list](https://raw.githubusercontent.com/huggingface/transformers.js/main/README.md) |
| Qwen2-VL | ✅ | README model list |
| Qwen2.5-VL / Qwen3-VL / Qwen3.5 / Qwen3.5-MoE | ✅ (architecture) | [v4.0.0 PR #1551](https://github.com/huggingface/transformers.js/releases/tag/4.0.0) |
| LLaVA, LLaVA-OneVision, LLaVA-Qwen2 (FastVLM) | ✅ | README model list; [FastVLM card](https://huggingface.co/onnx-community/FastVLM-0.5B-ONNX) |
| Gemma 3 VLM, Gemma3n, Gemma 4 | ✅ | [v4.0.0 PR #1601](https://github.com/huggingface/transformers.js/releases/tag/4.0.0); 4.3.0 pins the WebGPU KV cache for Gemma3n/Gemma4 |
| PaliGemma / PaliGemma2 | ✅ | README model list |
| LFM2-VL / LFM2.5-VL | ✅ | [v4.0.0 PR #1569](https://github.com/huggingface/transformers.js/releases/tag/4.0.0) |
| Phi3V | ✅ | README model list |
| OCR-specific: GLM-OCR, LightOnOCR, Dolphin, TrOCR, Donut, Pix2Struct | ✅ | [v4.0.0 PR #1582](https://github.com/huggingface/transformers.js/releases/tag/4.0.0) |

WebGPU is selected with `device: "webgpu"`. Per the README, the **default dtype is `"fp32"` for WebGPU** and `"q8"` for WASM, with `"fp16"`, `"q8"`, `"int8"`, `"uint8"`, `"q4"`, `"bnb4"`, `"q4f16"` as alternatives, plus **per-module dtype maps** ([README](https://raw.githubusercontent.com/huggingface/transformers.js/main/README.md), [dtypes guide](https://huggingface.co/docs/transformers.js/v3.8.1/guides/dtypes)).

### Model-by-model verification with exact ONNX paths and sizes

All sizes below are the raw byte sizes from the Hub tree APIs (graph + `_data` sidecar where present), summed over the components the runtime actually loads.

#### `HuggingFaceTB/SmolVLM-256M-Instruct` — genuinely 256M params ✅

Params: SmolLM2-135M-Instruct text decoder + `google/siglip-base-patch16-512` 93M vision encoder ([model card](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct/raw/main/README.md)). ONNX weights live **in the base repo's `onnx/` subfolder** (there is no `onnx-community/SmolVLM-256M-Instruct` repo — that ID 404s). Sources: [HF tree API](https://huggingface.co/api/models/HuggingFaceTB/SmolVLM-256M-Instruct/tree/main/onnx), [arch tag `idefics3`](https://huggingface.co/api/models?search=SmolVLM&library=transformers.js&limit=30).

| dtype | components (MB) | total |
|---|---|---|
| q4f16 | decoder 77.03 + embed 56.77 + vision 55.04 | **≈ 188.8 MB** |
| q4 | decoder 86.56 + embed 113.54 + vision 63.78 | **≈ 263.9 MB** |
| q8/int8 | decoder 137.22 + embed 28.39 + vision 94.25 | **≈ 259.9 MB** |
| fp16 | decoder 270.41 + embed 56.77 + vision 187.29 | **≈ 514.5 MB** |
| fp32 | decoder 540.64 + embed 113.54 + vision 374.32 | **≈ 1028.5 MB** |

The card states it "can run inference on one image with under 1GB of GPU RAM" ([model card](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct/raw/main/README.md)). There is an official WebGPU demo Space ([`HuggingFaceTB/SmolVLM-256M-Instruct-WebGPU`](https://huggingface.co/spaces/HuggingFaceTB/SmolVLM-256M-Instruct-WebGPU)) whose source is [`transformers.js-examples/smolvlm-webgpu/src/worker.js`](https://raw.githubusercontent.com/huggingface/transformers.js-examples/main/smolvlm-webgpu/src/worker.js) — **note it uses `dtype: "fp32"`, not a quantized dtype**, i.e. the official demo downloads ≈ 1 GB.

Quality for receipts: OCRBench **52.6**, DocVQA **58.3**, MMStar 34.6 ([model card eval table](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct/raw/main/README.md)). Document understanding was 25% of its training mixture ([same card](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct/raw/main/README.md)).

#### `HuggingFaceTB/SmolVLM-500M-Instruct` — genuinely ~500M params ✅

SmolLM2-360M-Instruct + the same 93M SigLIP encoder. [Tree API](https://huggingface.co/api/models/HuggingFaceTB/SmolVLM-500M-Instruct/tree/main/onnx).

| dtype | components (MB) | total |
|---|---|---|
| q4f16 | decoder 205.33 + embed 94.62 + vision 57.69 | **≈ 357.6 MB** |
| q4 | decoder 229.12 + embed 189.24 + vision 66.73 | **≈ 485.1 MB** |
| q8/int8 | decoder 365.04 + embed 47.31 + vision 98.97 | **≈ 511.3 MB** |
| fp16 | decoder 725.49 + embed 94.62 + vision 196.73 | **≈ 1016.8 MB** |
| fp32 | decoder 1450.43 + embed 189.24 + vision 393.19 | **≈ 2032.9 MB** |

Quality: OCRBench **61.0**, DocVQA **70.5**, MMStar 38.3. The SmolVLM blog explicitly frames these two as browser candidates: "This release comes with four checkpoints… loadable directly to transformers, MLX and ONNX, and we have demos for transformers and WebGPU (with ONNX)" ([SmolVLM Grows Smaller blog, 2025-01-23](https://huggingface.co/blog/smolervlm)). Official WebGPU Space: [`HuggingFaceTB/SmolVLM-500M-Instruct-WebGPU`](https://huggingface.co/spaces/HuggingFaceTB/SmolVLM-500M-Instruct-WebGPU).

Preprocessing detail that matters for receipts: SmolVLM-256/500 "use **64 visual tokens to encode image patches of size 512×512**. Larger images are divided into patches, each encoded separately" and encode "at a rate of **4096 pixels per token**, compared to 1820 pixels per token in the 2B model" ([model card](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct/raw/main/README.md), [blog](https://huggingface.co/blog/smolervlm)). The processor's default `size={"longest_edge": N*512}` with `N=4` yields a 2048×2048 input, i.e. up to 16 patches.

#### `onnx-community/LFM2.5-VL-450M-ONNX` — genuinely 450M params ✅ (best sub-1B OCR)

405M-ish: LFM2.5-350M text backbone + SigLIP2 NaFlex shape-optimized 86M vision encoder, 32,768-token context, 65,536 vocab, multilingual ([model card](https://huggingface.co/onnx-community/LFM2.5-VL-450M-ONNX/raw/main/README.md)). [Tree API](https://huggingface.co/api/models/onnx-community/LFM2.5-VL-450M-ONNX/tree/main/onnx).

| dtype | components (MB) | total |
|---|---|---|
| q4f16 | decoder 221.41 + embed 115.61 + vision 54.75 | **≈ 391.8 MB** |
| q4 | decoder 260.08 + embed 231.21 + vision 63.50 | **≈ 554.8 MB** |
| q8/quantized | decoder 442.82 + embed 57.80 + vision 109.87 | **≈ 610.5 MB** |
| fp16 | decoder 725.35 + embed 134.22 + vision 188.86 | **≈ 1048.4 MB** |

Quality: **OCRBench 684/1000, InfoVQA 43.02, MMStar 43.00, RealWorldQA 58.43**, clearly ahead of SmolVLM2-500M (OCRBench 609, InfoVQA 24.64) on the same VLMEvalKit harness ([benchmark table in the card](https://huggingface.co/onnx-community/LFM2.5-VL-450M-ONNX/raw/main/README.md)). It also does bounding-box prediction and (text-only) function calling. The card recommends `transformers.js v4.2.1 or newer` and an official WebGPU demo exists ([`LiquidAI/LFM2.5-VL-450M-WebGPU`](https://huggingface.co/spaces/LiquidAI/LFM2.5-VL-450M-WebGPU)). Caveats: license is **LFM1.0, not Apache/MIT**, and the card says it is "not well-suited for knowledge-intensive tasks or **fine-grained OCR**".

#### `onnx-community/granite-docling-258M-ONNX` — genuinely 258M params ✅ (document specialist)

Idefics3 architecture with `siglip2-base-patch16-512` vision encoder + Granite 165M LLM ([IBM model card](https://huggingface.co/ibm-granite/granite-docling-258M/raw/main/README.md)). 88.5 MB graph files are negligible; the `_data` sidecars dominate. [Tree API](https://huggingface.co/api/models/onnx-community/granite-docling-258M-ONNX/tree/main/onnx).

| dtype | components (MB) | total |
|---|---|---|
| q4f16 | decoder 93.35 + embed 115.61 + vision 54.75 | **≈ 263.7 MB** |
| q4 | decoder 104.72 + embed 231.21 + vision 63.50 | **≈ 399.4 MB** |
| q8/quantized | decoder 166.21 + embed 57.80 + vision 93.88 | **≈ 317.9 MB** |
| fp16 | decoder 329.06 + embed 115.61 + vision 187.02 | **≈ 631.7 MB** |
| fp32 | decoder 658.12 + embed 231.21 + vision 374.04 | **≈ 1263.4 MB** |

It is a *document conversion* model: page → DocTags markup (not JSON), needs `do_image_splitting: true` and up to 4096 new tokens ([usage snippet in the ONNX card](https://huggingface.co/onnx-community/granite-docling-258M-ONNX/raw/main/README.md)). Table structure TEDS 0.97, code recognition F1 0.988, OCRBench 500, MMStar 0.30 ([IBM card](https://huggingface.co/ibm-granite/granite-docling-258M/raw/main/README.md)). Explicit limitation from IBM: "**not intended for general image understanding**".

#### `onnx-community/Florence-2-base-ft` — 0.23B, sub-1B ✅ but not a chat model

Florence-2-base is 0.23B params. Files: `encoder_model` (DaViT vision tower), `vision_encoder`, `decoder_model_merged`, `embed_tokens` ([tree API](https://huggingface.co/api/models/onnx-community/Florence-2-base-ft/tree/main/onnx)).

| config | components (MB) | total |
|---|---|---|
| all-q4f16 | embed 78.78 + vision 62.42 + encoder 25.71 + decoder 56.54 | **≈ 223.5 MB** |
| docs-recommended mix (embed fp16, vision fp16, encoder q4, decoder q4) | 78.78 + 183.93 + 30.06 + 64.39 | **≈ 357.2 MB** |

The docs are explicit that Florence-2 needs per-module dtype: "Some encoder-decoder models, like Whisper or Florence-2, are extremely sensitive to quantization settings: especially of the encoder" ([dtypes guide](https://huggingface.co/docs/transformers.js/v3.8.1/guides/dtypes)). It is a seq2seq model driven by **task tokens** like `<MORE_DETAILED_CAPTION>`, not a chat model, so there is no instruction-following path to JSON ([card](https://huggingface.co/onnx-community/Florence-2-base-ft)).

#### `Xenova/moondream2` — **not sub-1B** (moondream1 arch, ~1.86B), and old

ONNX exists (`moondream1` arch, `Moondream` auto-model, base model `vikhyatk/moondream2`) — [HF API](https://huggingface.co/api/models/Xenova/moondream2?blobs=true). Sizes: `decoder_model_merged_q4f16.onnx` 740.85 MB, `embed_tokens_q4.onnx` 419.43 MB, `vision_encoder_q4.onnx` 279.54 MB → **≈ 1.44 GB** at the cheapest usable mix. Last modified 2025-07-30. There is an experimental Space `Xenova/experimental-moondream-webgpu`. Treat as too heavy for phones.

#### `onnx-community/Qwen2-VL-2B-Instruct` — **too heavy, do not use**

[Tree API](https://huggingface.co/api/models/onnx-community/Qwen2-VL-2B-Instruct/tree/main/onnx). At q4f16: decoder 869.38 MB + embed 466.75 MB + **vision_encoder 1332.20 MB** → **≈ 2.67 GB**. The full-precision decoder graph alone needs a **6.17 GB** `decoder_model_merged.onnx_data`. Not viable in a browser tab, on any device.

#### Qwen2.5-VL / Qwen3-VL small — architecture yes, **ONNX weights no**

transformers.js v4 supports the Qwen2.5-VL / Qwen3-VL / Qwen3.5 architectures ([v4.0.0 PR #1551](https://github.com/huggingface/transformers.js/releases/tag/v4.0.0)), but a Hub search filtered to `library=transformers.js` + `pipeline_tag=image-text-to-text` returns **no first-party Qwen2.5-VL ONNX repo** ([HF models API](https://huggingface.co/api/models?filter=transformers.js&pipeline_tag=image-text-to-text&limit=100&sort=downloads&direction=-1)); a search for `Qwen2.5-VL` returns only safetensors, GGUF and AWQ repos ([HF search API](https://huggingface.co/api/models?search=Qwen2.5-VL&limit=40&sort=downloads&direction=-1)). A few unvetted third-party conversions exist (e.g. `huggingworld/Qwen3.5-0.8B-ONNX`, `huggingworld/Qwen3-VL-2B-Instruct-ONNX`, both with near-zero downloads). **Conclusion: Qwen2.5-VL-3B is not runnable out of the box today, and at 3B it is the wrong size class for this app anyway.**

#### Other sub-1B-ish options found, with caveats

- `llava-hf/llava-onevision-qwen2-0.5b-ov-hf` — 0.5B but the Qwen2 vocab makes the embed table dominate: q4f16 = decoder 287.35 + embed 272.38 + vision 228.39 = **≈ 788.1 MB** for half a billion params ([tree API](https://huggingface.co/api/models/llava-hf/llava-onevision-qwen2-0.5b-ov-hf/tree/main/onnx)). Poor size-to-quality ratio.
- `onnx-community/nanoLLaVA-1.5` — q4f16 = 270.23 + 311.17 + 228.69 = **≈ 810.1 MB**, and it is ~1.1B (Qwen2-0.5B + SigLIP-SO400M), so **not sub-1B** ([tree API](https://huggingface.co/api/models/onnx-community/nanoLLaVA-1.5/tree/main/onnx)).
- `onnx-community/FastVLM-0.5B-ONNX` (Apple) — 0.5B, `llava_qwen2` arch. Card-recommended mix (embed fp16 + vision q4 + decoder q4) = 271.81 + 505.21 + 317.45 = **≈ 1094.5 MB**; all-q4f16 = 282.25 + 271.81 + 252.70 = **≈ 806.8 MB** ([tree API](https://huggingface.co/api/models/onnx-community/FastVLM-0.5B-ONNX/tree/main/onnx)). Official Apple WebGPU demo exists ([`apple/fastvlm-webgpu`](https://huggingface.co/spaces/apple/fastvlm-webgpu)), but the license is **`apple-amlr`** (Apple ML Research license), not OSI-permissive ([card](https://huggingface.co/onnx-community/FastVLM-0.5B-ONNX)).
- `onnx-community/gemma-3n-E2B-it-ONNX` (E2B ≈ 2B effective) and `onnx-community/paligemma2-3b-*` — too big.

---

## 4. Recommended candidate shortlist

Assumptions: sub-1B params, must run in one browser tab on a mid-range phone **or** a laptop, input = one receipt photo, output = JSON.

| Rank | Model ID | Runtime | Params | Best download | OCR/receipt fitness | License | Key limitation |
|---|---|---|---|---|---|---|---|
| **1** | `onnx-community/LFM2.5-VL-450M-ONNX` | transformers.js ≥ 4.2.1, `device:"webgpu"` | 450M (real) | **≈ 392 MB** q4f16 (embed fp16 / decoder q4f16 / vision fp16 per card) | Best measured OCR of the sub-500M field: OCRBench 684/1000, InfoVQA 43.0; native-resolution tiling preserves receipt aspect ratios; supports bbox output | LFM1.0 (not OSI) | Card admits weakness on "fine-grained OCR"; non-permissive license |
| **2** | `HuggingFaceTB/SmolVLM-500M-Instruct` | transformers.js, `device:"webgpu"` | ~500M | **≈ 358 MB** q4f16 | OCRBench 61.0 / DocVQA 70.5; 25–41% document-heavy training mix; official HF WebGPU demo | Apache-2.0 | Weaker than LFM on InfoVQA; 512×512 tiling inflates token count on tall receipts |
| **3** | `HuggingFaceTB/SmolVLM-256M-Instruct` | transformers.js, `device:"webgpu"` | 256M (**smallest real VLM**) | **≈ 189 MB** q4f16 | Fits comfortably even on iOS; OCRBench 52.6 / DocVQA 58.3 — readable totals, unreliable line-item digits | Apache-2.0 | Lowest accuracy of the shortlist; expect to need a repair/retry loop |
| **4** | `onnx-community/granite-docling-258M-ONNX` | transformers.js, `device:"webgpu"` | 258M | **≈ 264 MB** q4f16 | Purpose-built page/document parsing; table TEDS 0.97; strong code/equation recognition | Apache-2.0 | **Emits DocTags markup, not JSON**; explicitly "not intended for general image understanding"; needs 4096-token budget |
| **5** | `onnx-community/Florence-2-base-ft` | transformers.js, per-module dtype | 0.23B | **≈ 224 MB** all-q4f16 (≈ 357 MB with the fp16 vision encoder the docs recommend) | `<OCR_WITH_REGION>` / `<MORE_DETAILED_CAPTION>` task tokens are a good fit for a two-stage receipt pipeline | MIT | Seq2seq task-token API, **no chat template and no instruction following** — you cannot prompt it for JSON |
| **6** | `onnx-community/FastVLM-0.5B-ONNX` | transformers.js, `device:"webgpu"` | 0.5B | ≈ 807 MB all-q4f16 / ≈ 1.09 GB card-recommended | Apple-designed for fast vision encoding; official WebGPU demo | `apple-amlr` | ~2× the bytes of SmolVLM-500M for comparable class; non-OSI license → probably not worth it |

**Flagged as too heavy or not actually runnable:**
- `Phi-3.5-vision-instruct-q4f16_1-MLC` (WebLLM) — 4.2B, **≈ 2.77 GB download, 3952 MB VRAM**; the *only* WebLLM VLM. Fails the sub-1B and phone constraints outright.
- `onnx-community/Qwen2-VL-2B-Instruct` — **≈ 2.67 GB** at q4f16 because of a 1.33 GB vision encoder.
- `Xenova/moondream2` — ~1.86B, **≈ 1.44 GB**.
- `onnx-community/nanoLLaVA-1.5` / `llava-onevision-qwen2-0.5b-ov-hf` — **≈ 810 MB / ≈ 788 MB**; nanoLLaVA is not sub-1B.
- **Qwen2.5-VL-3B / Qwen3-VL small** — architecture supported, **no first-party ONNX weights**, 3B is out of scope anyway.
- **Gemma 3 Vision on WebLLM** — [PR #805](https://github.com/mlc-ai/web-llm/pull/805) still open; no prebuilt entry.

**Suggested default:** build against `HuggingFaceTB/SmolVLM-500M-Instruct` (Apache-2.0, official HF WebGPU demo, ~358 MB) as the shipping default, with `LFM2.5-VL-450M-ONNX` as the quality tier if the LFM1.0 license is acceptable, and `SmolVLM-256M-Instruct` as the low-end/phone tier. Keep `granite-docling-258M` in reserve as a second-stage document reader rather than the JSON emitter.

---

## 5. Gotchas that will bite an implementer

1. **`AutoProcessor` + chat templates are mandatory, and they differ per family.** SmolVLM/LFM take OpenAI-ish nested content parts (`{role:"user", content:[{type:"image"},{type:"text",text}]}`) passed through `processor.apply_chat_template(messages, {add_generation_prompt:true})`, then `processor(image, prompt, {add_special_tokens:false})` ([SmolVLM worker.js](https://raw.githubusercontent.com/huggingface/transformers.js-examples/main/smolvlm-webgpu/src/worker.js), [LFM2.5-VL card](https://huggingface.co/onnx-community/LFM2.5-VL-450M-ONNX/raw/main/README.md)). FastVLM uses a flat string `"<image>Describe this image in detail."` instead ([FastVLM card](https://huggingface.co/onnx-community/FastVLM-0.5B-ONNX)). Florence-2 has **no chat template at all** — you call `processor.construct_prompts(task)` with a literal task token ([Florence-2 card](https://huggingface.co/onnx-community/Florence-2-base-ft)). Hard-coding one family's prompt format will silently produce garbage on another.

2. **SmolVLM image tiling is a memory/precision dial you must set explicitly.** The official demo carries the comment: *"Set `do_image_splitting: true` to split images into multiple patches. NOTE: This uses more memory, but can provide more accurate results"* — and it is **commented out** in the demo, while `granite-docling-258M`'s card sets `do_image_splitting: true` ([worker.js](https://raw.githubusercontent.com/huggingface/transformers.js-examples/main/smolvlm-webgpu/src/worker.js), [granite-docling card](https://huggingface.co/onnx-community/granite-docling-258M-ONNX/raw/main/README.md)). For a tall, text-dense receipt, splitting is where your digit accuracy comes from — but it multiplies the visual token count (64 tokens per 512×512 patch at a 2048×2048 default input) and therefore prefill time and KV-cache size. You can also shrink the input with the processor's `size={"longest_edge": N*512}` ([SmolVLM card](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct/raw/main/README.md)). LFM2.5-VL exposes tunable `min_image_tokens` / `max_image_tokens` plus `do_image_splitting` and a thumbnail pass ([LFM card](https://huggingface.co/onnx-community/LFM2.5-VL-450M-ONNX/raw/main/README.md)).

3. **KV-cache reuse across repeated questions on the same image is *not* reliable yet.** The official SmolVLM WebGPU demo caches `past_key_values` but ships with the reuse **disabled**: `// TODO: Add back when fixed` / `// past_key_values: past_key_values_cache` ([worker.js](https://raw.githubusercontent.com/huggingface/transformers.js-examples/main/smolvlm-webgpu/src/worker.js)). transformers.js 4.1.0 did land "Cached generation improvements (+ `past_key_values` via pipeline function)" ([4.1.0 release notes](https://github.com/huggingface/transformers.js/releases/tag/4.1.0)). **Design for a one-shot extraction prompt**, and treat prefill reuse as a stretch optimization to validate per model.

4. **`use_external_data_format` / `_data` sidecars: good for the buffer limit, expensive in round-trips.** All three top candidates ship the big tensors as `<name>.onnx_data` sidecars with a tiny graph file, e.g. LFM2.5-VL's `decoder_model_merged_q4f16.onnx` is 187 KB with a 221 MB `_data` partner, and `granite-docling`'s `decoder_model_merged_q4f16.onnx` is 284 KB with a 93 MB `_data` partner ([LFM tree](https://huggingface.co/api/models/onnx-community/LFM2.5-VL-450M-ONNX/tree/main/onnx), [granite tree](https://huggingface.co/api/models/onnx-community/granite-docling-258M-ONNX/tree/main/onnx)). That keeps each individual allocation under GPU limits, but it means more files to fetch and to cache. Contrast `onnx-community/Qwen2-VL-2B-Instruct`, where the *single* `decoder_model_merged.onnx_data` is **6.17 GB** ([tree](https://huggingface.co/api/models/onnx-community/Qwen2-VL-2B-Instruct/tree/main/onnx)) — a single buffer of that size cannot work in a browser. Use `ModelRegistry.get_pipeline_files` + `get_file_metadata` to compute exact download size before committing to a model ([v4 blog](https://huggingface.co/blog/transformersjs-v4)).

5. **Quantize per module, not uniformly.** Uniform 4-bit destroys Florence-2 ("extremely sensitive to quantization settings: especially of the encoder" — [dtypes guide](https://huggingface.co/docs/transformers.js/v3.8.1/guides/dtypes)), and the recommended FastVLM config keeps `embed_tokens: "fp16"` + `vision_encoder: "q4"` while the decoder is `q4` ([FastVLM card](https://huggingface.co/onnx-community/FastVLM-0.5B-ONNX)); LFM2.5-VL's card recommends `vision_encoder: "fp16"` with a `q4f16` decoder ([LFM card](https://huggingface.co/onnx-community/LFM2.5-VL-450M-ONNX/raw/main/README.md)). Note also that in these repos `q4` and `bnb4` are frequently the *same* bytes as fp32 for `embed_tokens` (e.g. FastVLM `embed_tokens_q4` = 543.6 MB = `embed_tokens` size), so "q4" can cost you more than `q4f16`.

6. **WASM fallback changes your default dtype and your headers.** If `device` is not set you get WASM with **`q8` as the default dtype**, versus **`fp32` for WebGPU** ([README](https://raw.githubusercontent.com/huggingface/transformers.js/main/README.md)). Multithreaded WASM needs `SharedArrayBuffer`, which needs cross-origin isolation (`Cross-Origin-Opener-Policy: same-origin` + `Cross-Origin-Embedder-Policy: require-corp`). See §3 for the verified header/WASM-build details. Also set `env.useWasmCache = true` if you want offline reuse of the runtime files ([v4 blog](https://huggingface.co/blog/transformersjs-v4)).

7. **Feature-detect `shader-f16` before choosing dtypes.** The canonical pattern from the official demo:

```js
const adapter = await navigator.gpu.requestAdapter();
if (!adapter) throw new Error("WebGPU is not supported (no adapter found)");
const fp16_supported = adapter.features.has("shader-f16");
```
([worker.js](https://raw.githubusercontent.com/huggingface/transformers.js-examples/main/smolvlm-webgpu/src/worker.js)). Any `q4f16`/`fp16` plan needs this gate plus a `q4`/`q8` fallback. See §3 for which platforms actually expose `shader-f16`.

8. **Forcing JSON: use the new first-party constrained decoder, not prompt engineering.** transformers.js **4.3.0 (2026-09-16)** ships `@huggingface/transformers-structured-output` (version **4.3.0**, peer dep `@huggingface/transformers@^4.3.0`, described as "Dependency-free constrained generation for Transformers.js", keywords `constrained-generation`/`structured-output`/`json-schema`) ([npm registry](https://registry.npmjs.org/@huggingface/transformers-structured-output/latest)). Release notes: "Constrain generation to a JSON schema, JSON object, or regular expression with the experimental, dependency-free `@huggingface/transformers-structured-output` package" ([PR #1758 / release 4.3.0](https://github.com/huggingface/transformers.js/releases/tag/4.3.0)). Usage is a `logits_processor`:

```js
const processor = new StructuredOutputProcessor(generator.tokenizer, {
  type: "json_schema",
  json_schema: { type: "object", properties: { /* ... */ }, required: [...], additionalProperties: false },
});
await generator(messages, { max_new_tokens: 512, do_sample: false, logits_processor: [processor] });
```

Two documented caveats: it is **experimental**, currently supports **one generated sequence at a time**, and you must "set a sufficient token budget so the output can finish" ([release notes](https://github.com/huggingface/transformers.js/releases/tag/4.3.0)). It was published *yesterday* relative to this report, so expect churn. Alternatives: the third-party [`transformers-llguidance`](https://github.com/dsh0416/transformers-llguidance) (JSON Schema / regex / Lark CFG via llguidance WASM) whose README lists "Currently requires the WASM module to be built from source" as a limitation; or WebLLM's built-in JSON mode, which is mature but only reachable with a 4 GB model.

9. **Budget token count *and* CPU-side image processing — these are two different bottlenecks.** On the decode side, `MAX_NEW_TOKENS` is 1024 in the official SmolVLM demo and 4096 for granite-docling ([worker.js](https://raw.githubusercontent.com/huggingface/transformers.js-examples/main/smolvlm-webgpu/src/worker.js), [granite card](https://huggingface.co/onnx-community/granite-docling-258M-ONNX/raw/main/README.md)), and a tiny VLM asked for a full JSON receipt will happily emit a paragraph per line item — so pair the schema constraint (item 8) with a **short, flat schema**, `do_sample: false`, and `InterruptableStoppingCriteria`. On the prefill side, the strongest published finding for this size class is that **CPU-side image decode/resize/normalize and tokenization dominate time-to-first-token**, not GPU compute ([Sony AI arXiv 2603.16987](https://arxiv.org/html/2603.16987v1)); in a browser that work happens in JS/WASM. Downscale the receipt *before* handing it to the processor, and instrument preprocessing separately from inference.

10. **Model-naming churn.** `AutoModelForVision2Seq` (SmolVLM demo, granite-docling card) and `AutoModelForImageTextToText` (FastVLM, LFM2.5-VL cards) are both current in the same version; pick per model card rather than assuming one is the modern spelling. Similarly the "official" SmolVLM ONNX weights live in the base `HuggingFaceTB/*` repos while most others live in `onnx-community/*` — there is no single naming convention, and guessing `onnx-community/SmolVLM-256M-Instruct` 404s.

---

## Confidence & gaps

**High confidence (verified directly against primary sources on 2026-09-17):**
- WebLLM npm `0.2.85`; exactly two `ModelType.VLM` entries in the published `prebuiltAppConfig`, both Phi-3.5-vision; their `vram_required_MB` and `cs2k` model libs; ~2.77 GB of weight shards.
- transformers.js npm `4.3.0` released 2026-09-16; the `@huggingface/transformers-structured-output` package and its API shape; v4.0.0/4.1.0/4.2.0 content.
- All per-file ONNX byte sizes, summed from the Hub tree APIs. Totals are arithmetic on those byte counts, so they are exact for the file set I named; they exclude tokenizer/config JSON (typically 1–10 MB).
- Parameter counts for SmolVLM-256M/500M, granite-docling-258M, LFM2.5-VL-450M, Florence-2-base (0.23B) and moondream2 (~1.86B) come from the vendors' own model cards.
- The §3 browser support matrix, spec-default limits, `shader-f16`/COOP-COEP mechanics and storage quotas were verified against primary sources: the raw [caniuse dataset](https://raw.githubusercontent.com/Fyrd/caniuse/main/features-json/webgpu.json), [MDN](https://developer.mozilla.org/en-US/docs/Web/API/GPUSupportedLimits), the [W3C Candidate Recommendation Draft](https://www.w3.org/TR/webgpu/), Chrome release notes, and onnxruntime-web source.
- The "~2 GB" question is resolved as a **binding-range** limit ([gpuweb #6338](https://github.com/gpuweb/gpuweb/issues/6338), still open, with in-thread hardware measurements) and the decisive ORT-web behaviour — always requesting `adapter.limits` maxima, conditionally enabling `shader-f16` — was read from [`backend-webgpu.ts`](https://raw.githubusercontent.com/microsoft/onnxruntime/main/js/web/lib/wasm/jsep/backend-webgpu.ts).

**Medium confidence / explicitly uncertain:**
- **Florence-2 component accounting.** The repo contains *both* `encoder_model*` and `vision_encoder*` files and I could not fully verify from primary sources which set the current transformers.js v4 runtime resolves for `Florence2ForConditionalGeneration`. The dtypes guide's per-module example names all four keys (`embed_tokens`, `vision_encoder`, `encoder_model`, `decoder_model_merged`), which is why I list both totals. Verify with `ModelRegistry.get_pipeline_files('image-text-to-text', 'onnx-community/Florence-2-base-ft', opts)` before quoting a number.
- **"Sub-1B" is fuzzy for encoder-decoder VLMs.** Florence-2-base's 0.23B is the full model (DaViT + BART); SmolVLM-500M's 500M includes its 93M encoder. LFM2.5-VL-450M's 450M = 350M LM + 86M encoder, so the numbers are comparable, but "params" alone does not predict browser cost — the embedding table does. FastVLM is nominally 0.5B yet ≈ 807 MB at q4f16 because of a 271 MB embedding table.
- **Reported benchmark numbers are not apples-to-apples.** SmolVLM publishes OCRBench on a 0–100 scale (52.6, 61.0) while LFM2.5-VL publishes 0–1000 (684). I converted LFM's to ~68.4% for comparison; the harnesses (VLMEvalKit) match but the exact prompt/settings may not.
- **Which dtype transformers.js actually picks when you omit `dtype` for these specific VLMs.** The README says fp32 is the WebGPU default, and the official SmolVLM demo does use fp32 — but I did not confirm whether a quantized default is auto-selected for models that lack fp32 files.
- **Whether the in-repo `HuggingFaceTB/SmolVLM-256M-Instruct/onnx` folder is fully v4-runtime-compatible.** It was converted in early 2025 and "re-enabled" in 4.1.0; I did not find a per-model compatibility matrix, and the official demo targets it, which is strong but indirect evidence.

**Could not verify / open questions:**
- **End-to-end latency numbers on real hardware.** There is **no published TTFT / prefill / decode measurement for a small VLM in a browser on a named dGPU, iGPU or phone.** The closest data points are LLM decode throughput on an M3 Max ([WebLLM paper](https://arxiv.org/abs/2412.15803)), a browser *text*-LLM figure on an M4 Pro Max ([v4 notes](https://github.com/huggingface/transformers.js/releases/tag/4.0.0)), and SmolVLM-256M TTFT measured with **vLLM on an H100** ([Sony AI](https://arxiv.org/html/2603.16987v1)). **Treat any per-device latency claim for this workload as unmeasured until you benchmark it yourself** — §3.5 lists the documented pieces and the derived cost model.
- **Phone-specific VLM size budget.** No first-party source states a tested size budget for Android Chrome or iOS Safari, and the mobile defects in §3.7 (iOS texture leak, iOS 26 command-buffer throttling, Adreno device-lost, moto g54 device-creation failure) mean "the weights fit" is not the same as "it works". §3.7 is the checklist to test against.
- **Whether Chromium's Android `shader-f16` enablement is polyfilled or native.** Survey coverage on Adreno 7xx/8xx (96–99%) implies a polyfill shipped, but no release note, bug or commit was found announcing it (§3.3). Coverage is measured; the mechanism is not.
- **The spec's normative *maximum* (as opposed to default) values** for `maxBufferSize` / `maxStorageBufferBindingSize`. I have observed platform tiers from an issue thread and a survey, not the spec table.
- **caniuse vs BCD/W3C disagreement** on Firefox (141+ "disabled by default" in caniuse) and Chrome-for-Android (caniuse says 152). §3.1 uses BCD/wiki for versions; which caniuse table is stale remains unresolved.
- **Whether `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` (the `smolvlm` arch, not `idefics3`) works via transformers.js** — it has an `onnx` tag but I did not inspect its ONNX folder or confirm runtime support.
- **Exact behaviour of the structured-output processor with a multimodal `generate()` call.** The 4.3.0 example uses `text-generation`; I did not verify that `logits_processor` composes correctly with the `image-text-to-text` path and its `inputs_embeds` handling. Test this early — it is the single highest-value unknown for this project.

---

## 3. Practical browser constraints

### 3.1 Browser support matrix (September 2026)

From the caniuse `webgpu` feature dataset ([raw caniuse data](https://raw.githubusercontent.com/Fyrd/caniuse/main/features-json/webgpu.json); `usage_perc_y` = **83.99%**, `usage_perc_a` = 2.95%) plus the Chrome release note for Android.

| Engine / platform | Enabled by default from | Notes |
|---|---|---|
| **Chrome desktop** Win/macOS/ChromeOS | **113** | — |
| **Chrome desktop Linux** | **144** (Intel Gen12+); NVIDIA on Wayland **147** | other Linux configs behind `--enable-unsafe-webgpu` + Vulkan flags ([Implementation Status](https://github.com/gpuweb/gpuweb/wiki/Implementation-Status)); caniuse note #5 "Not enabled on Linux by default" covers 113–143 |
| **Edge** | **113** | Chromium mirror |
| **Opera** | **99** | Chromium mirror |
| **Chrome for Android** | **121** | "enabled by default in Chrome 121 on Android 12 and later devices with Qualcomm and ARM GPUs" ([Chrome 121](https://developer.chrome.com/blog/new-in-webgpu-121)); **139** adds Imagination (Android 16+), Samsung Xclipse "probably 154" ([Implementation Status](https://github.com/gpuweb/gpuweb/wiki/Implementation-Status)) |
| **Samsung Internet** | **24** | |
| **Safari macOS / iOS / iPadOS / visionOS** | **26** | pre-26 behind the `WebGPU` feature flag (17.4–18.7) ([Implementation Status](https://github.com/gpuweb/gpuweb/wiki/Implementation-Status)) |
| **Firefox Windows** | **141** | all contexts **except service workers** ([bug 1942431](https://bugzil.la/1942431)) |
| **Firefox macOS Apple Silicon** | **145** (macOS 26 Tahoe), **147** (all macOS on Apple Silicon) | Intel Macs unsupported |
| **Firefox Linux / macOS Intel** | Nightly only | not in Stable |
| **Firefox Android** | not supported | flag `gfx.webgpu.ignore-blocklist` in Beta/Nightly |
| **UC / QQ / Baidu browsers** | not supported | |

**Source conflict you should know about.** caniuse's live table prints Firefox 141–159 as "Disabled by default" and lists Chrome-for-Android support only from **152** — both contradict the W3C [Implementation Status wiki](https://github.com/gpuweb/gpuweb/wiki/Implementation-Status) and [MDN browser-compat-data](https://raw.githubusercontent.com/mdn/browser-compat-data/main/api/GPU.json) (`chrome_android: 121`). Use the wiki/BCD for **versions** and caniuse for **usage share**. On share they also disagree slightly: the raw caniuse JSON I pulled gives `usage_perc_y` **83.99%** + `usage_perc_a` **2.95%**, while caniuse.com/webgpu displayed 85.72% + 1.63% on the same day.

Spec status: **W3C Candidate Recommendation Draft, 15 September 2026** ([W3C TR](https://www.w3.org/TR/webgpu/)) — not yet a final Recommendation.

Also relevant for reach: Chrome shipped a **compatibility mode** (OpenGL ES 3.1-class, requested via `requestAdapter({ featureLevel: "compatibility" })`) in **Chrome 146**, and the **`core-features-and-limits`** feature in **Chrome 139** ([Chrome 146](https://developer.chrome.com/blog/new-in-webgpu-146), [Chrome 139](https://developer.chrome.com/blog/new-in-webgpu-139)). Compatibility mode **reduces the available limits**, so a weights-heavy VLM runner should detect `core-features-and-limits` and fall back rather than assume desktop-class limits.

Runtime-side corroboration: transformers.js 4.3.0 shipped "**Enable WebGPU for Safari 26 and above**" ([release notes](https://github.com/huggingface/transformers.js/releases/tag/4.3.0), [PR #1700](https://github.com/huggingface/transformers.js/pull/1700)). The v3.8.1 WebGPU guide still describes Firefox and Safari as purely flag-gated and cites "around 70% global WebGPU support as of October 2024" ([guide](https://huggingface.co/docs/transformers.js/guides/webgpu)) — stale.

**Practical read for a PWA:** target Chrome/Edge desktop, Chrome for Android 121+ (and ≥139 for Imagination GPUs, ~154 for Xclipse), and Safari 26 on iOS/iPadOS/macOS. Firefox desktop is Windows-first and has **no WebGPU inside service workers**; Firefox Android has none. Those users need an explicit "not supported" path, not a degraded one. §3.7 lists the phone-specific failure modes, which are the real deployment risk.

### 3.2 Hardware limits: the "~2 GB single buffer" question

Spec defaults, from MDN's `GPUSupportedLimits` table ([MDN](https://developer.mozilla.org/en-US/docs/Web/API/GPUSupportedLimits)):

| Limit | Spec default |
|---|---|
| `maxBufferSize` | **268,435,456 bytes (256 MB)** |
| `maxStorageBufferBindingSize` | **134,217,728 bytes (128 MB)** |
| `maxUniformBufferBindingSize` | 65,536 bytes |
| `maxStorageBuffersPerShaderStage` | 8 |
| `maxComputeWorkgroupStorageSize` | 16,384 bytes |
| `maxComputeInvocationsPerWorkgroup` | 256 |
| `maxTextureDimension2D` | 8192 |
| `maxTextureDimension3D` | 2048 |
| `maxTextureArrayLayers` | 256 |

**The "~2 GB limit" is real — and it is a *binding-range* limit, not an allocation limit.** This is the most important correction in this section:

- gpuweb issue [#6338 "Support binding larger buffer ranges >2GB to shaders"](https://github.com/gpuweb/gpuweb/issues/6338) is **still open** (created 2026-07-15). The spec editor's summary: "it's larger buffer **binding ranges** that is a spec issue. There's no spec issue with larger `GPUBuffer` allocations, but **you can't use it all in one shader invocation**."
- Measured platform behaviour from that thread: **Apple M2 Max / Chrome 151** reports `maxBufferSize` and `maxStorageBufferBindingSize` both **4294967292** (4 GiB − 4, Dawn's top `Limits.cpp` tier), and requesting one byte more fails validation instead of clamping; **NVIDIA Tesla T4 / Linux / Vulkan** deliberately clamps `maxStorageBufferBindingSize` to **2147483644** (2 GiB − 4) ([crbug 435684920](https://issues.chromium.org/issues/435684920)); **Windows/D3D12** sets `maxBufferSize` to **exactly 2 GiB unconditionally on every device**; **Qualcomm below Adreno 8xx on D3D12** gets only **256 MiB** binding size — "Raw Buffers can only address 2^28 bytes instead of the guaranteed 2^31."
- Survey coverage ([Web3D Survey `maxBufferSize`](https://web3dsurvey.com/webgpu/limits/maxBufferSize), [`maxStorageBufferBindingSize`](https://web3dsurvey.com/webgpu/limits/maxStorageBufferBindingSize)): `maxBufferSize` ≥2 GiB on **79% overall** (100% Windows, 77% Android, 67% macOS; iOS only 80% ≥1 GiB); `maxStorageBufferBindingSize` ≥2 GiB on just **16% overall** (99% Windows, 63% macOS, 35% Android, **no ≥2 GiB tier on iOS**; Samsung Internet 68% ≥256 MiB).
- Provenance caveat: those figures are named engineers' issue comments and live survey reports, not vendor documentation. Trust the *shape* — a 256 MiB binding floor on older Adreno, a hard 2 GiB cap on D3D12, 4 GiB on Metal — and treat exact digits as approximate. I also could not extract the spec's normative *maximum* column for these limits (my spec fetch truncated before the limits table), so the maxima here are observed tiers, not quoted statute.

**The actionable fact: onnxruntime-web — the engine under transformers.js — does not inherit the spec defaults.** `WebGpuBackend.initialize()` in [`backend-webgpu.ts`](https://raw.githubusercontent.com/microsoft/onnxruntime/main/js/web/lib/wasm/jsep/backend-webgpu.ts) *always* requests the adapter's own maxima:

```ts
const deviceDescriptor: GPUDeviceDescriptor = {
  requiredLimits: {
    maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
    maxComputeWorkgroupsPerDimension: adapter.limits.maxComputeWorkgroupsPerDimension,
    maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
    maxBufferSize: adapter.limits.maxBufferSize,
    maxComputeInvocationsPerWorkgroup: adapter.limits.maxComputeInvocationsPerWorkgroup,
    maxComputeWorkgroupSizeX/Y/Z: adapter.limits.maxComputeWorkgroupSize[X/Y/Z],
  },
  requiredFeatures,
};
const requireFeatureIfAvailable = (feature: GPUFeatureName) =>
  adapter.features.has(feature) && requiredFeatures.push(feature) && true;
requireFeatureIfAvailable('shader-f16');
requireFeatureIfAvailable('subgroups');
```

So **128 MiB / 256 MiB is what a *bare* `requestDevice()` gives you, not what your VLM will run with.** The risk therefore moves from "do my weights fit in 256 MiB" to "**does any single *binding* fit**" — which is exactly where Adreno <8xx (256 MiB) and D3D12 (2 GiB) bite. (Source: [onnxruntime-web `backend-webgpu.ts`](https://raw.githubusercontent.com/microsoft/onnxruntime/main/js/web/lib/wasm/jsep/backend-webgpu.ts).)

Related hard ceilings from the same runtime's large-model documentation: Chrome's max `ArrayBuffer` is ~2 GB (`0x7fe00000`), the ONNX protobuf format has a **2 GB file limit**, and **WebAssembly has a 4 GB memory limit — "Currently, there is no way for ONNX Runtime Web to run models larger than 4GB."** Weights beyond that are streamed via `externalData: [{ path, data }]`, where `path` must match the `location` string inside the protobuf, and both `.onnx` and `.data` can be supplied as Blobs from IndexedDB ([ORT large models](https://onnxruntime.ai/docs/tutorials/web/large-models.html)).

**This is also why ONNX conversions ship external-data sidecars.** The `.onnx` + `.onnx_data` split keeps each file, and therefore each allocation, under those ceilings. Compare `onnx-community/Qwen2-VL-2B-Instruct`, whose single `decoder_model_merged.onnx_data` is **6.17 GB** ([tree](https://huggingface.co/api/models/onnx-community/Qwen2-VL-2B-Instruct/tree/main/onnx)) — unusable under every ceiling above — versus `onnx-community/LFM2.5-VL-450M-ONNX`, whose q4f16 `_data` files are 221 / 116 / 55 MB ([tree](https://huggingface.co/api/models/onnx-community/LFM2.5-VL-450M-ONNX/tree/main/onnx)). Note `use_external_data_format` is an **export/conversion-time** flag (optimum / onnxruntime Python tooling); I found no transformers.js *runtime* option by that name — the runtime side consumes the split through `externalData`.

**Practical heuristic:** every q4f16 weight file in shortlist ranks 1–5 is ≤ 222 MB, comfortably under even the 256 MiB bare-request floor and the Adreno-6xx 256 MiB binding ceiling. Two instructive exceptions: `onnx-community/FastVLM-0.5B-ONNX` q4f16 files are 282 / 272 / 253 MB ([tree](https://huggingface.co/api/models/onnx-community/FastVLM-0.5B-ONNX/tree/main/onnx)), and Qwen2-VL-2B is an order of magnitude over. (File size proxies buffer pressure rather than equalling it — ORT can stream one large external-data file into several smaller buffers.)

### 3.3 `shader-f16`

`shader-f16` is an **optional** WebGPU feature, not core.

- Requesting it when unavailable fails device creation, so branch on it. The canonical check, as used by the official SmolVLM WebGPU demo:

```js
const adapter = await navigator.gpu.requestAdapter();
if (!adapter) throw new Error("WebGPU is not supported (no adapter found)");
const fp16_supported = adapter.features.has("shader-f16");
```
([worker.js](https://raw.githubusercontent.com/huggingface/transformers.js-examples/main/smolvlm-webgpu/src/worker.js)).

- **Qualcomm was the original problem — and the spec was deliberately not changed.** gpuweb [#5006](https://github.com/gpuweb/gpuweb/issues/5006) reported Android coverage at the time of `shaderFloat16` 76.5%, `storageBuffer16BitAccess` 64% and `uniformAndStorageBuffer16BitAccess` 42%, and that "the list of devices that support the feature contains exactly 0 Qualcomm devices" (Adreno has `storageBuffer16BitAccess` but not `uniformAndStorageBuffer16BitAccess`). The WGSL group **decided not to change the spec** and instead let implementations polyfill uniform-buffer f16 loads; the issue **closed 2025-10-09** ([#5006 thread](https://github.com/gpuweb/gpuweb/issues/5006)). So the folklore is out of date: it is no longer "all Qualcomm is excluded".
- **Current measured coverage** ([Web3D Survey `shader-f16`](https://web3dsurvey.com/webgpu/features/shader-f16)): **93.39% overall** — iOS 100%, macOS 99.86%, Windows 90.43%, Android 86.63%, **Linux 70.23%**. By GPU architecture: Adreno **8xx 99.0%**, **7xx 96.7%**, **6xx 29.2%**, **5xx 0%**; ARM Mali/Bifrost/Valhall 100%; Apple 100%; NVIDIA Ampere/Lovelace/Blackwell ~98.5% but **Maxwell and Pascal 0%**; Intel Gen-8 28%. **The devices to gate on are therefore Adreno 6xx and older, NVIDIA Maxwell/Pascal, and Linux generally** — and a Linux + RTX 3060 Ti machine has been observed with `adapter.features.has("shader-f16") === false` ([HF Space discussion #1](https://huggingface.co/spaces/webml-community/Qwen3.5-WebGPU/discussions/1), where the app broke because it hardcoded `vision_encoder: "fp16"`).
- **Coverage is measured; the mechanism is unverified.** Survey coverage on Adreno 7xx/8xx at 96–99% is impossible if Chromium still excluded Qualcomm, so a polyfill evidently shipped — but no Chrome release note or Dawn commit announcing it was found. Rely on the runtime check, not on an assumption about the platform.
- WebLLM encodes this per model via `required_features`: `Llama-2-7b-chat-hf-q4f16_1-MLC`, `TinyLlama-1.1B-Chat-v1.0-q4f16_1-MLC`, `phi-2-q4f16_1-MLC` and `gemma-2b-it-q4f16_1-MLC` all require `shader-f16`, while their `q4f32_1` siblings do not ([config.ts @ v0.2.85](https://raw.githubusercontent.com/mlc-ai/web-llm/v0.2.85/src/config.ts)). That is the shape of the fallback you need. onnxruntime-web takes the opportunistic route instead — `requireFeatureIfAvailable('shader-f16')` — so device creation never fails on that account, though that does **not** guarantee an fp16-quantized graph runs *well* without it.
- **Implication for this project:** choose the dtype at **runtime** from `adapter.features.has('shader-f16')`; never hardcode `fp16`/`q4f16`. Ship a `q4`/`q8` fallback for every `q4f16`/`fp16` recommendation in §2 and §4 — on many Android phones and on Linux that fallback is the primary path, at roughly 1.3–2× the bytes (SmolVLM-500M: 358 MB q4f16 → 485 MB q4).
- **What the fallback costs you when f16 *is* available elsewhere:** +28% prefill and +41% decode for Llama-2-7B f16 vs f32 on an Apple M1 Pro ([Chrome 120 blog](https://developer.chrome.com/blog/new-in-webgpu-120)). That is the size of the win you forgo by defaulting down — and of the breakage you avoid by not defaulting up.

### 3.4 COOP/COEP and the WASM fallback

`SharedArrayBuffer` — which multithreaded WebAssembly needs — requires cross-origin isolation. MDN states the exact requirements; the document must be returned with:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp      # or: credentialless
```

and the `cross-origin-isolated` directive of `Permissions-Policy` must not block access. You then read `window.crossOriginIsolated` to branch ([MDN `crossOriginIsolated`](https://developer.mozilla.org/en-US/docs/Web/API/Window/crossOriginIsolated), [MDN `SharedArrayBuffer`](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/SharedArrayBuffer)). Without those headers, `postMessage()` throws for `SharedArrayBuffer` and multithreaded WASM is unavailable.

Three points to internalise:

- **WebGPU inference does not need `SharedArrayBuffer`.** Cross-origin isolation is required for `SharedArrayBuffer`, high-resolution timers and `measureUserAgentSpecificMemory`; COOP/COEP are orthogonal to WebGPU ([web.dev cross-origin isolation guide](https://web.dev/articles/cross-origin-isolation-guide)). SAB is needed only by the multithreaded **WASM** backend — the CPU fallback — and a compatibility matrix states it plainly: "SharedArrayBuffer — Enables multi-threaded WASM inference… Requires Cross-Origin Isolation headers (COOP/COEP). **Not required for basic functionality**" ([LocalMode](https://localmode.dev/blog/compatibility/webgpu-support)). **If you ship WebGPU-only inference you can skip COOP/COEP entirely** and thereby avoid the CDN/CORP and popup breakage below.
- **COEP `require-corp` breaks third-party embeds; COOP `same-origin` breaks popups.** Every cross-origin subresource — your model weights on a CDN, WASM binaries, images — must then send `Cross-Origin-Resource-Policy: cross-origin` (or be loaded with CORS / the `crossorigin` attribute). `COEP: credentialless` exists in Chrome 96+ but "isn't supported by any other browsers yet", and `COOP: same-origin` severs the relationship with cross-origin openers, which breaks OAuth and payment popups. `Report-Only` variants of both headers exist for staging ([web.dev](https://web.dev/articles/cross-origin-isolation-guide)). Budget for this before enabling it.
- **Two runtime builds and an explicit thread knob.** ORT-web's `ort.env.wasm.numThreads` **defaults to `0`**, meaning "half of `navigator.hardwareConcurrency` or `4`, whichever is smaller", and "only when the browser supports WebAssembly multi-threading **and** `crossOriginIsolated` mode is enabled, multi-threading will be enabled". `numThreads = 1` forces single-threading. Threaded and non-threaded artifacts are separate files (`ort-wasm-simd.jsep.wasm` vs `ort-wasm-simd-threaded.jsep.wasm`), and the JS bundle and the `.wasm` must come from the same build or init fails on mismatched names ([ORT env flags](https://onnxruntime.ai/docs/tutorials/web/env-flags-and-session-options.html)). `q8` is the WASM default dtype while WebGPU defaults to `fp32` ([transformers.js README](https://raw.githubusercontent.com/huggingface/transformers.js/main/README.md)); set `env.useWasmCache = true` for offline reuse ([v4 blog](https://huggingface.co/blog/transformersjs-v4)).
- **One hard incompatibility:** ORT-web's `env.wasm.proxy` worker **cannot** be combined with the WebGPU execution provider, because "a GPU buffer is not transferable" ([ORT env flags](https://onnxruntime.ai/docs/tutorials/web/env-flags-and-session-options.html)). If you want the model off the main thread, use a dedicated Web Worker around the WebGPU path — as the official [SmolVLM demo](https://raw.githubusercontent.com/huggingface/transformers.js-examples/main/smolvlm-webgpu/src/worker.js) does — rather than the WASM proxy.

### 3.5 Cold-start vs warm latency — what is actually documented

**I could not find a single vendor-published, hardware-named latency figure for a 250M–1B VLM doing one image plus a short answer in a browser.** That is a genuine gap, not a search failure. What *is* documented, and why it does not transfer:

| Documented number | Source / hardware | Why it does not transfer |
|---|---|---|
| Llama-3.1-8B 4-bit **41.1 tok/s** in-browser vs 57.7 native MLC-LLM (71.2% retained); Phi-3.5-mini 71.1 vs 89.3 (79.6%) | [WebLLM paper arXiv 2412.15803v2](https://arxiv.org/abs/2412.15803) — M3 Max, Chrome Canary 133 | LLM decode only; no TTFT, no VLM, no phone |
| GPT-OSS 20B (q4f16) **~60 tokens/sec on an M4 Pro Max** | [transformers.js v4.0.0 release notes](https://github.com/huggingface/transformers.js/releases/tag/4.0.0) | 20B text-only MoE; a sanity check on the v4 runtime, nothing more |
| `f16` vs `f32` = **+28% prefill, +41% decode** (Llama-2-7B) | [Chrome 120 blog](https://developer.chrome.com/blog/new-in-webgpu-120) — Apple M1 Pro | A ratio, not an absolute |
| **SmolVLM-256M TTFT 344.7 ms → 22.8 ms** optimized, 533 tok/s, E2E 427.6 → 85.1 ms; InternVL3-2B 124 → 57.7 ms | [Sony AI arXiv 2603.16987](https://arxiv.org/html/2603.16987v1) — **vLLM on an H100, batch 1** | Server GPU, not a browser — but the *finding* transfers (below) |
| SmolVLM-256M "can run inference on **one image with under 1GB of GPU RAM**" | [SmolVLM-256M card](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct/raw/main/README.md) | Memory, not time |
| SmolVLM (2B) needs **min 5.02 GB GPU RAM**; 81 tokens per 384×384 patch; prompt + 1 image ≈ **1.2k tokens** | [SmolVLM blog, Nov 2024](https://huggingface.co/blog/smolvlm) | 2B model, not browser |
| SmolVLM 256M/500M throughput vs batch size | [SmolVLM blog](https://huggingface.co/blog/smolervlm) | Explicitly "throughput benchmarks ran on **A100**" |

The Sony AI result is the most useful thing in that table even though it is server-side, because its central finding is about *where the time goes* in exactly our size class: "previously overlooked **CPU-side operations, such as image processing and text tokenization, often dominate latency**." Their biggest single win on SmolVLM-256M was replacing PIL decoding (344.7 → 241.6 ms) and collapsing redundant image transforms (→ 71.9 ms). **A browser does the equivalent work in JS/WASM on the main thread or in a worker, and it is the same class of cost** — so decode throughput is not the metric to optimize first; image preprocessing and tokenization are. ([Sony AI arXiv 2603.16987](https://arxiv.org/html/2603.16987v1).)

So the honest answer is: **measure it, and design for a cold-start-dominated first run.** The cost model you *can* reason about from verified inputs:

- **Cold start is dominated by the download:** 189 MB (SmolVLM-256M q4f16) → 392 MB (LFM2.5-VL-450M q4f16) → 486 MB (SmolVLM-500M q4). One-time, then browser-cached. Use `ModelRegistry.is_pipeline_cached()` and `get_file_metadata()` to show real size and progress before the user commits ([v4 blog](https://huggingface.co/blog/transformersjs-v4)). WebLLM's default Cache API backend plays the same role, with `opfs`/`indexeddb` alternatives ([WebLLM README](https://raw.githubusercontent.com/mlc-ai/web-llm/main/README.md)).
- **Prefill is dominated by visual tokens, and receipt photos hurt.** Derived arithmetic (my calculation from cited constants, not a published benchmark): SmolVLM-256/500M encode at **4096 pixels per token** ([card](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct/raw/main/README.md)) and the processor default is a 2048×2048 input (`size={"longest_edge": N*512}`, N=4) → 4,194,304 px ÷ 4096 ≈ **1024 visual tokens for one image**, i.e. 16 patches at 64 visual tokens per 512×512 patch ([card](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct/raw/main/README.md)). A portrait receipt upscaled to fill that square pays the same prefill cost as a landscape photo, so **tight pre-cropping is a first-order latency win.** LFM2.5-VL lets you cap this explicitly (`min_image_tokens=32`, `max_image_tokens=256`, `do_image_splitting=True` — [LFM card](https://huggingface.co/onnx-community/LFM2.5-VL-450M-ONNX/raw/main/README.md)), i.e. 4–8× fewer visual tokens than SmolVLM's default; that is probably the single largest lever on phone latency.
- **Decode scales with your JSON size.** The official SmolVLM demo allocates 1024 new tokens and granite-docling 4096 ([worker.js](https://raw.githubusercontent.com/huggingface/transformers.js-examples/main/smolvlm-webgpu/src/worker.js), [granite card](https://huggingface.co/onnx-community/granite-docling-258M-ONNX/raw/main/README.md)); a tiny VLM left unconstrained will use them. Constrain the schema and bound `max_new_tokens`.
- **Later runs on the same image** should be much cheaper if KV-cache reuse works — but §5.3 shows the official demo currently ships that path disabled. Do not promise it.

### 3.6 Storage and caching

How the runtime caches: transformers.js uses the **Cache API by default** (`env.useBrowserCache = true`, cache name `env.cacheKey = 'transformers-cache'`), with `useFSCache`/`cacheDir` in Node, `useCustomCache` for custom backends, and `useWasmCache` for the runtime binaries. Monitor with `navigator.storage.estimate()` → `{usage, quota}` ([transformers.js caching reference](https://raw.githubusercontent.com/huggingface/skills/main/skills/transformers-js/references/CACHE.md), [configuration reference](https://raw.githubusercontent.com/huggingface/skills/main/skills/transformers-js/references/CONFIGURATION.md)). WebLLM documents four backends (`cache` default, `indexeddb`, `opfs`, `cross-origin`) and warns that "the Cache API is the most well-tested in WebLLM as of now" ([WebLLM README](https://raw.githubusercontent.com/mlc-ai/web-llm/main/README.md)). transformers.js v4 adds `ModelRegistry.is_pipeline_cached` / `clear_pipeline_cache` and a `progress_total` progress event ([v4 blog](https://huggingface.co/blog/transformersjs-v4)).

Quotas and eviction — this is where a PWA plan can quietly fail ([web.dev "Storage for the web"](https://web.dev/articles/storage-for-the-web), [MDN storage quotas and eviction](https://developer.mozilla.org/en-US/docs/Web/API/Storage_API/Storage_quotas_and_eviction_criteria)):

| Browser | Documented quota | Eviction behaviour |
|---|---|---|
| **Chrome / Chromium** | Browser up to **80% of total disk**; a single origin up to **60%**. Incognito ~**5%**. With "clear cookies and site data when you close all windows" enabled, quota collapses to **~300 MB** | Best-effort; under pressure Chromium evicts **all data of the least-recently-used origin first**, then the next, until under limit |
| **Firefox** | Browser up to **50% of free disk**; best-effort origin limit = smaller of **10% of disk or a 10 GiB per-site group**; with persistent storage granted, up to 50% of disk capped at 8 TiB | Best-effort LRU-per-origin; `navigator.storage.persist()` triggers a **user prompt** |
| **Safari (desktop + iOS)** | **~1 GB per origin**, after which Safari prompts and raises the limit in **~200 MB increments** (web.dev's author notes "I could not find any official documentation on this") | Since **iOS/iPadOS 13.4 and Safari 13.1**, script-writable storage — IndexedDB, service worker registrations, **and the Cache API** — is **evicted after 7 days of Safari use without user interaction**. **Installed PWAs added to the home screen are exempt** ([WebKit blog](https://webkit.org/blog/10218/full-third-party-cookie-blocking-and-more/)) |

**Design consequences.** Everything above is best-effort unless you call `navigator.storage.persist()` — and Safari and most Chromium browsers decide that request silently based on the user's interaction history rather than prompting ([MDN](https://developer.mozilla.org/en-US/docs/Web/API/Storage_API/Storage_quotas_and_eviction_criteria)). Concretely: on Chrome/Firefox desktop a 189–486 MB model cache is comfortable; **on Safari plan for ≤1 GB without a prompt**, which is fine for every shortlist candidate but rules out a 2 GB+ fp16 fallback; and **on iOS the 7-day eviction cap is the strongest argument for shipping as an installable PWA**, since that is the documented exemption. Always request persistence and handle `QuotaExceededError` from both IndexedDB transactions and `cache.put()`.

### 3.7 Phone-specific failure modes — the real deployment risk

Mobile WebGPU in 2026 works but is not yet boring. These are documented, currently-open issues rather than hypotheticals:

- **iOS: texture churn leaks Metal memory and forces a page reload.** WebKit bug [312563](https://bugs.webkit.org/show_bug.cgi?id=312563) (filed 2026-04-17, still NEW/P2) — destroying and recreating WebGPU textures does not release Metal memory; a Unity app force-reloads after 3–5 render-scale changes (iPhone 12 and iPhone 15 Pro both reproduce). It reproduces "in both Safari **and Chrome on iOS** (identical behaviour, consistent with a shared WebKit/Metal layer)" and **not** on Android. **Implication: do not allocate and destroy textures per inference.** Allocate once, reuse, and keep resolution changes out of the hot loop.
- **iOS 26 needs throttled command submission.** llama.cpp PR [21533](https://github.com/ggml-org/llama.cpp/pull/21533) (merged 2026-04-07): "on iOS 26, the WebGPU backend tends to crash unless the number of operations + submitted command buffers is pretty severely throttled." Their fix detects iOS from the **User-Agent**, because `adapter.info` returns only `"apple"` — so you cannot feature-detect your way out of this one. A WebKit bug was filed ([311598](https://bugs.webkit.org/show_bug.cgi?id=311598)). This is a *runtime-level* concern for whoever maintains your inference backend; if you depend on transformers.js/ORT-web, verify iOS 26 specifically rather than assuming.
- **Other live crashes/devices:** GPU-process leak on iPadOS/macOS 26 ([WebKit 303203](https://bugs.webkit.org/show_bug.cgi?id=303203)); device lost on Safari 26 with Emscripten builds ([imgui #9103](https://github.com/ocornut/imgui/issues/9103)); Qualcomm Adreno `VK_ERROR_DEVICE_LOST` during WebLLM engine init ([web-llm #836](https://github.com/mlc-ai/web-llm/issues/836)); outright device-creation failure on at least one common handset, the Motorola moto g54 5G ([Chromium 559589664](https://issues.chromium.org/issues/559589664)).
- **False-positive capability detection.** `navigator.gpu` plus a working adapter can still fail at model init on Android ([LocalMode](https://localmode.dev/blog/compatibility/webgpu-support)). Wrap model loading in try/catch with a real fallback path — a server call, or "try again on Wi-Fi/desktop" — rather than trusting feature detection.
- **iOS chip requirement: none documented.** I found no Apple documentation of a minimum chip; the practical floor is the **iOS 26** version requirement, and `shader-f16` is at 100% on surveyed iOS reports ([Implementation Status](https://github.com/gpuweb/gpuweb/wiki/Implementation-Status), [Web3D Survey](https://web3dsurvey.com/webgpu/features/shader-f16)).
