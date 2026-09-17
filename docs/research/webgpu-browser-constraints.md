# WebGPU for on-device VLM inference — state of play, 2026-09-17

All claims carry a source link. Numbers labelled **[measured]** come from a named benchmark/hardware; **[estimate]** means I derived or inferred it; **[unverified]** means I could not confirm it.

---

## 1. Browser support matrix (Sept 2026)

**Spec status:** WebGPU is a **W3C Candidate Recommendation Draft, 15 September 2026** ([w3.org/TR/webgpu](https://www.w3.org/TR/webgpu/)). Global usage per caniuse: **85.72% + 1.63% = 87.35%** ([caniuse.com/webgpu](https://caniuse.com/webgpu)).

| Browser / platform | Shipped (enabled by default) | Flag / partial | Source |
|---|---|---|---|
| **Chrome desktop** (Win/macOS/ChromeOS) | **113** | — | [gpuweb Implementation Status](https://github.com/gpuweb/gpuweb/wiki/Implementation-Status), [caniuse](https://caniuse.com/webgpu) |
| **Chrome desktop, Linux** | **144** — Intel Gen12+ only; NVIDIA (driver 535.183.01+) on Wayland **147**; others behind `--enable-unsafe-webgpu` + Vulkan flags | 144/147 | [Implementation Status](https://github.com/gpuweb/gpuweb/wiki/Implementation-Status), [Chrome 144](https://developer.chrome.com/blog/new-in-webgpu-144), [Chrome 147-148](https://developer.chrome.com/blog/new-in-webgpu-147-148) |
| **Chrome current stable range** | caniuse lists Chrome 113–156 all "supported" | — | [caniuse](https://caniuse.com/webgpu) |
| **Edge** | **113** (mirror of Chrome), caniuse shows 113–152 | — | [caniuse](https://caniuse.com/webgpu), [MDN BCD `edge: mirror`](https://raw.githubusercontent.com/mdn/browser-compat-data/main/api/GPU.json) |
| **Chrome for Android** | **121**, Android 12+ with Qualcomm/ARM GPUs; **139** adds Imagination (Android 16+); Samsung Xclipse "probably 154"; others TBD | 121/139 | [Implementation Status](https://github.com/gpuweb/gpuweb/wiki/Implementation-Status), [Chrome 121 blog](https://developer.chrome.com/blog/new-in-webgpu-121), MDN BCD `chrome_android: 121` |
| **Firefox Windows** | **141** (enabled by default) | all contexts *except service workers* | [Implementation Status](https://github.com/gpuweb/gpuweb/wiki/Implementation-Status), [MDN Firefox experimental features](https://raw.githubusercontent.com/mdn/content/main/files/en-us/mozilla/firefox/experimental_features/index.md) |
| **Firefox macOS (Apple Silicon)** | **145** on macOS 26+/Tahoe, then **147** on all macOS versions on Apple Silicon | Intel Macs: **not supported**; Linux: Nightly only | [Implementation Status](https://github.com/gpuweb/gpuweb/wiki/Implementation-Status), MDN BCD notes ([bug 2004105](https://bugzil.la/2004105), [bug 2006676](https://bugzil.la/2006676)) |
| **Firefox Linux / macOS Intel** | **Nightly only** | enabled by default in Nightly, not Stable/Beta | [Implementation Status](https://github.com/gpuweb/gpuweb/wiki/Implementation-Status) |
| **Firefox Android** | **No** | behind flag: Beta/Nightly + `about:config` → `gfx.webgpu.ignore-blocklist` | MDN BCD `firefox_android: false`; [Implementation Status](https://github.com/gpuweb/gpuweb/wiki/Implementation-Status) |
| **Safari macOS / iOS / iPadOS / visionOS** | **26** — enabled by default | 17.4–18.7 / 15–17.3 behind flag (17.4–18.7 "disabled by default" in caniuse) | [Implementation Status](https://github.com/gpuweb/gpuweb/wiki/Implementation-Status), MDN BCD `safari: 26` |
| **Opera** | **99** (Chromium mirror) | — | [caniuse](https://caniuse.com/webgpu) |
| **Samsung Internet** | **24** | — | [caniuse](https://caniuse.com/webgpu) |

**Flag names to quote for pre-ship builds:** `dom.webgpu.enabled` (Firefox), `chrome://flags/#enable-unsafe-webgpu` (Chromium), Safari's WebGPU feature flag pre-26. ([transformers.js WebGPU guide](https://raw.githubusercontent.com/huggingface/transformers.js/main/packages/transformers/docs/source/guides/webgpu.md), [Chrome troubleshooting](https://developer.chrome.com/docs/web-platform/webgpu/troubleshooting-tips))

**Conflict to be aware of:** caniuse still prints Firefox 141–159 as **"Disabled by default"**, and lists Chrome for Android support only from **152**. Both disagree with the W3C Implementation Status wiki and MDN BCD (Firefox enabled-by-default on Windows since 141; Chrome Android since 121). caniuse's Firefox row also carries a 156 entry with no version range, i.e. its table is mid-edit. Treat caniuse as the usage-share source and the wiki/BCD as the version source.

**Compatibility mode** (relevant to old Android): Chrome shipped a WebGPU **compatibility mode** for OpenGL ES 3.1 in **Chrome 146** (Feb 2026), requested via `requestAdapter({ featureLevel: "compatibility" })`, plus the `"core-features-and-limits"` feature in **Chrome 139**. ([Chrome 146](https://developer.chrome.com/blog/new-in-webgpu-146), [Chrome 139](https://developer.chrome.com/blog/new-in-webgpu-139)). Compatibility mode reduces the available limits, so a limits-hungry VLM runner should check for `core-features-and-limits`.

Firefox notes: "Supports all contexts except service workers" ([bug 1942431](https://bugzil.la/1942431)); on AC power a discrete adapter is returned by default only on dual-GPU macOS in Chrome ([MDN BCD](https://raw.githubusercontent.com/mdn/browser-compat-data/main/api/GPU.json)); on Windows Chrome always uses the same adapter as the rest of Chrome (usually the iGPU) and `powerPreference` is ignored — force with `chrome://flags/#force-high-performance-gpu` ([Chrome troubleshooting](https://developer.chrome.com/docs/web-platform/webgpu/troubleshooting-tips)).

---

## 2. Mobile

**Android Chrome:** yes, **enabled by default from Chrome 121** (Jan 2024), on **Android 12+** devices with **Qualcomm and ARM GPUs**; rolled out progressively, with Android 11 and other vendors "depending on further testing" ([Chrome 121 blog](https://developer.chrome.com/blog/new-in-webgpu-121), [chromium:1497815](https://bugs.chromium.org/p/chromium/issues/detail?id=1497815)). By 2026 the Implementation Status wiki shows **Imagination** added at Chrome 139 (Android 16+) and **Samsung Xclipse** "probably 154". MDN BCD independently records `chrome_android: 121` ([BCD](https://raw.githubusercontent.com/mdn/browser-compat-data/main/api/GPU.json)).

**iOS Safari:** yes, **from iOS 26 / Safari 26, enabled by default** on iOS, iPadOS, visionOS ([Implementation Status](https://github.com/gpuweb/gpuweb/wiki/Implementation-Status)). iOS 17.4+ had it behind the WebKit feature flag. I found **no Apple documentation of a chip/device requirement**; a compatibility guide states "No specific chip requirement has been documented by Apple" ([LocalMode, May 2026](https://localmode.dev/blog/compatibility/webgpu-support)). Given Apple ships Metal 3 + f16 on 100% of surveyed iOS reports (below), the practical floor is more likely the iOS version than the chip.

**Known phone memory limits / crashes — this is the weakest part of the ecosystem:**

- **iOS texture-churn leak → forced page reload.** WebKit bug **312563** (filed 2026-04-17, still NEW, P2/Major, Safari 26/iOS 26): destroying and recreating WebGPU textures does not release Metal memory; a Unity app force-reloads after 3–5 render-scale changes. **iPhone 12**: `{1.0, 0.9, 1.0}` crashes; **iPhone 15 Pro**: `{1.4, 1.3, 1.4}` crashes. Reproduces "in both Safari and Chrome on iOS (identical behavior, consistent with a shared WebKit/Metal layer)"; does **not** reproduce on Android WebGPU or native iOS/Android. Engineer comment: "During resize we do preserve the prior render target contents for one frame though they should be discarded… there is some accumulation occurring which is unexpected." ([bugs.webkit.org 312563](https://bugs.webkit.org/show_bug.cgi?id=312563))
- **iOS 26 command-buffer throttling.** llama.cpp PR **21533** ("ggml-webgpu: parameterize submission size and add iOS specific limits", merged 2026-04-07): "on iOS 26, the WebGPU backend tends to crash unless the number of operations + submitted command buffers is pretty severely throttled"; the fix detects iOS from the User-Agent because `adapter.info` only returns `"apple"`. A WebKit bug was filed: [311598](https://bugs.webkit.org/show_bug.cgi?id=311598). ([PR 21533](https://github.com/ggml-org/llama.cpp/pull/21533))
- **GPU-process leak on iPadOS/macOS 26** when running a simple WebGPU app: WebKit bug [303203](https://bugs.webkit.org/show_bug.cgi?id=303203).
- **Device lost on Safari 26 (macOS + iOS)** when building with Emscripten: [imgui issue 9103](https://github.com/ocornut/imgui/issues/9103).
- **Qualcomm Adreno `VK_ERROR_DEVICE_LOST`** on WebLLM engine init: [web-llm issue 836](https://github.com/mlc-ai/web-llm/issues/836) (collaborator: repro attempts inconclusive; suspected 0.2.80 GPU-kernel move).
- **Device creation outright fails** on at least one common device: Chromium issue [559589664](https://issues.chromium.org/issues/559589664) (Motorola moto g54 5G).
- **False positives in detection:** `navigator.gpu` + a working adapter can still fail at model init on Android; wrap loading in try/catch and fall back ([LocalMode](https://localmode.dev/blog/compatibility/webgpu-support)).

**Android WebView:** MDN BCD records `webview_android: mirror` of `chrome_android`, i.e. **121** ([BCD](https://raw.githubusercontent.com/mdn/browser-compat-data/main/api/GPU.json)). I found **no WebView-specific release note**; Android WebView version numbers no longer track Chrome versions, so the practical rule is "WebView supports it if its Chromium is ≥121 and the device GPU vendor is supported" **[unverified]**.

**Samsung Internet:** shipped **24** per caniuse; the W3C wiki implies Xclipse support around Chrome 154. Survey data for Samsung Internet specifically: `shader-f16` **94.05%**, but `maxStorageBufferBindingSize` only **68% ≥256 MiB, 42% ≥512 MiB** and `maxBufferSize` 84% ≥2 GiB ([Web3D Survey](https://web3dsurvey.com/webgpu/limits/maxStorageBufferBindingSize)) — i.e. binding size is the mobile constraint, not total buffer size.

---

## 3. Hardware limits

### Spec defaults (what you get with a bare `requestDevice()`)

Authoritative table from MDN (which mirrors the spec's limits table):

| Limit | Spec default |
|---|---|
| `maxBufferSize` | **268435456 bytes (256 MiB)** |
| `maxStorageBufferBindingSize` | **134217728 bytes (128 MiB)** |
| `maxUniformBufferBindingSize` | 65536 bytes |
| `maxComputeWorkgroupStorageSize` | **16384 bytes** |
| `maxComputeInvocationsPerWorkgroup` | 256 |
| `maxComputeWorkgroupSizeX/Y/Z` | 256 / 256 / 64 |
| `maxComputeWorkgroupsPerDimension` | 65535 |
| `maxTextureDimension1D / 2D / 3D` | **8192 / 8192** / 2048 |
| `maxTextureArrayLayers` | 256 |
| `maxStorageBuffersPerShaderStage` | 8 |
| `maxSampledTexturesPerShaderStage` | 16 |

Source: [MDN `GPUSupportedLimits` (full table with defaults)](https://raw.githubusercontent.com/mdn/content/main/files/en-us/web/api/gpusupportedlimits/index.md).

### Raising limits

- `requestDevice({ requiredLimits })` **can only raise up to the adapter's own `adapter.limits` values**. If a requested limit exceeds the adapter's value (or is not a valid limit), the promise **rejects with `OperationError`**; if a feature in `requiredFeatures` is unsupported, it rejects with **`TypeError`** ([MDN `GPUAdapter.requestDevice()`](https://raw.githubusercontent.com/mdn/content/main/files/en-us/web/api/gpuadapter/requestdevice/index.md)).
- Requesting **unknown limit names does not error** — they come back `undefined` (MDN, same page).
- Adapters usually **report quantized limit tiers**, not exact hardware values, to limit fingerprinting: "If your GPU's actual limit is 16384, the browser will still report 8192" ([MDN GPUSupportedLimits](https://raw.githubusercontent.com/mdn/content/main/files/en-us/web/api/gpusupportedlimits/index.md)).

### Real-world adapter maxes (survey, Sept 2026)

Web3D Survey aggregates live `webgpureport.org`-style reports. "≥ X" = share of reports whose value is at least X.

**`maxBufferSize`** ([web3dsurvey.com/webgpu/limits/maxBufferSize](https://web3dsurvey.com/webgpu/limits/maxBufferSize)):

| Scope | ≥256 MiB | ≥1 GiB | ≥2147483647 (~2 GiB) |
|---|---|---|---|
| Overall | 100% | 97% | **79%** |
| Windows | 100% | — | **100%** (≥2147483644) |
| macOS | 100% | 94% | 67% |
| Linux | 100% | 100% | 79% (≥2147483644: 87%) |
| Android | 100% | 100% | **77%** |
| iOS | 100% | **80%** | (no ≥2 GiB row) |
| Chrome/Chromium | 100% | 99% | 94% |
| Safari | 100% | 81% | (none) |
| Samsung Internet | 100% | 100% | 84% |

**`maxStorageBufferBindingSize`** ([web3dsurvey.com/.../maxStorageBufferBindingSize](https://web3dsurvey.com/webgpu/limits/maxStorageBufferBindingSize)):

| Scope | ≥128 MiB | ≥1 GiB | ≥1717986916 (~1.6 GiB) | ≥2147483645 (~2 GiB) |
|---|---|---|---|---|
| Overall | 100% | 91% | 78% | **16%** |
| Windows | 100% | 100% | — | **99%** (≥2147483643) |
| macOS | 100% | 94% | 82% | 63% |
| Linux | 100% | 76% | — | 71% (≥2147483644) |
| Android | 100% | **36%** | — | 35% (≥2147483639) |
| iOS | 100% | **81%** | — | (no ≥2 GiB row) |
| Safari | 100% | 81% | 24% | (none) |
| Samsung Internet | 100% | — | — | (only 68% ≥256 MiB, 42% ≥512 MiB) |

**`maxComputeWorkgroupStorageSize`** ([survey](https://web3dsurvey.com/webgpu/limits/maxComputeWorkgroupStorageSize)): 100% ≥16384, **99% ≥32765**, only **5–6% ≥65536**. Vendor: every surveyed vendor ≥32768 except Apple (99% ≥32767). So the practical target is **16 KiB guaranteed / 32 KiB nearly universal**.

**`maxTextureDimension2D`** ([survey](https://web3dsurvey.com/webgpu/limits/maxTextureDimension2D)): **100% ≥8192**, **99% ≥16382/16384** (Mac 99% ≥16384, Android 98%). Practical target: 8192 safe, 16384 nearly universal.

**No per-vendor deep tables** for NVIDIA/AMD/Adreno/Mali `maxBufferSize` were retrievable from the survey pages I fetched (the vendor tables are truncated on the public pages). The Android-specific vendor signal below is from the gpuweb issue thread and Dawn source references, which is stronger evidence anyway.

### Is there still a ~2 GB / 4 GB single-buffer limit?

**Yes, and it is a binding-range limit, not (only) an allocation limit.** gpuweb issue **#6338 "Support binding larger buffer ranges >2GB to shaders"** is still **open** (created 2026-07-15, last updated 2026-08-14). Key facts from that thread:

- Specification editor (kainino0x): "it's larger buffer *binding ranges* that is a spec issue. There's no spec issue with larger `GPUBuffer` allocations, but you can't use it all in one shader invocation." ([#6338 comment](https://github.com/gpuweb/gpuweb/issues/6338#issuecomment-4982743971))
- **M2 Max, Chrome 151 [measured]**: `maxBufferSize` **4294967292** and `maxStorageBufferBindingSize` **4294967292** (= 4 GiB − 4, the top `Limits.cpp` tier); requesting one byte more fails with `GPUValidationError` rather than clamping. Dawn's Metal backend computes `UINT32_MAX` first because "we pass storage (and vertex) buffer sizes to MSL as u32."
- **NVIDIA Tesla T4, Linux, headless Chrome 148, Vulkan [measured]**: `maxBufferSize` 4294967292, `maxStorageBufferBindingSize` **2147483644** (= 2 GiB − 4). Dawn's Vulkan backend clamps NV binding size to 2 GiB − 4 citing [crbug.com/435684920](https://issues.chromium.org/issues/435684920).
- **Windows/D3D12**: Dawn sets `maxBufferSize` to `kAssumedMaxBufferSize` = **exactly 2 GiB, unconditionally on every device** ("D3D12 has no documented limit on the buffer size… Use 2 GB when the limit is unavailable"). The same constant is used by the Vulkan, D3D11 and OpenGL backends.
- **Qualcomm below Adreno 8xx, D3D12**: Dawn sets `maxStorageBufferBindingSize` to **256 MiB**, commented "Due to hardware limitation, Raw Buffers can only address 2^28 bytes instead of the guaranteed 2^31 bytes."
- 64-bit indices: jimblandy — "hardware support for 64-bit indices is very sparse, so although we could add support to the spec, the devices that could use it would be rare."
- A production workload at 15.7 GB of resident GPU buffers exists (Qwen3.6-35B-A3B, 3-bit MoE) where the constraint is **buffer *count*, not binding range**.

Caveat: these specific numbers come from issue comments by named engineers/users, not from vendor documentation. The Dawn source statements are quoted in-thread.

**Practical consequence for VLMs:** a 1–2 B-parameter VLM in q4 weighs ~0.6–1.5 GB and will *fit* in one or two `GPUBuffer`s on most desktops, but you must chunk weight storage so no single **binding** exceeds ~128 MiB–2 GiB depending on platform, and you must request the adapter max explicitly (below). ORT-web already does exactly that.

### f16 textures/buffers

- `rgba16float` etc. are **core texture formats** and are usable for storage/attachment per the spec's texture-format capability tables ([WebGPU spec, §26.1 Texture Format Capabilities](https://www.w3.org/TR/webgpu/#texture-format-caps)).
- Using **`f16` in WGSL** (including 16-bit values in storage/uniform buffers) requires the **`shader-f16`** feature plus `enable f16;` — see §4. Without it, `createShaderModule()` produces a validation error ([Chrome 120 blog](https://developer.chrome.com/blog/new-in-webgpu-120)).

---

## 4. `shader-f16`

### Where it exists

Spec-defined optional feature ([WebGPU spec §25.11 `"shader-f16"`](https://www.w3.org/TR/webgpu/#shader-f16)). On Vulkan, both Chrome and Firefox require **`VK_KHR_shader_float16_int8`** with `shaderFloat16` **plus** `VK_KHR_16bit_storage` (or Vulkan 1.1) with `storageBuffer16BitAccess` **and** `uniformAndStorageBuffer16BitAccess` ([gpuweb #5006](https://github.com/gpuweb/gpuweb/issues/5006)).

**History:** issue #5006 ("`shader-f16` requirements exclude all Qualcomm devices", Dec 2024) reported Android coverage at the time of `shaderFloat16` 76.5%, `storageBuffer16BitAccess` 64%, `uniformAndStorageBuffer16BitAccess` 42% — and **"the list of devices that support the feature contains exactly 0 Qualcomm devices."** Root cause: Adreno has `storageBuffer16BitAccess` but not `uniformAndStorageBuffer16BitAccess`, and the spec forbids emulating 2-byte access when only 4-byte writes exist. WGSL group **decided not to change the spec** and to let implementations polyfill uniform-buffer f16 loads (`load<f16>(i) => unpack2x16float(load<u32>(i/2))[i % 2]`). Issue closed 2025-10-09. ([#5006 comments](https://github.com/gpuweb/gpuweb/issues/5006#issuecomment-3386845834), [minutes](https://github.com/gpuweb/gpuweb/issues/5006#issuecomment-2768449973))

### Current real-world coverage (survey, Sept 2026) — measured

| Scope | `shader-f16` support |
|---|---|
| **Overall** | **93.39%** |
| Android | **86.63%** (Android page: 85.89%) |
| iOS | **100%** |
| macOS | 99.86% |
| Windows | 90.43% |
| Linux | **70.23%** |
| Chrome/Chromium | 92.53% · Edge 93.37% · Safari 100% · Firefox 92.82% · Samsung Internet 94.05% |

By vendor: arm **100%**, apple 99.99%, img-tec 100%, samsung 100%, intel 95.18%, amd 90.91%, nvidia 83.79%, **qualcomm 71.13%**, google **0%**, microsoft 66.67%.

By architecture: qualcomm **adreno-8xx 99.05–99.22%**, **adreno-7xx 96.74–96.82%**, **adreno-6xx 29.16–29.39%**, **adreno-5xx 0%**; arm bifrost/valhall/gen-5 100%; samsung rdna-2/rdna-3 100%; apple 100%; nvidia ampere/blackwell/lovelace ~98.5%, **kepler 3.1%, maxwell 0%, pascal 0%**; amd gcn-5 99.67%, gcn-4 25.31%, rdna-1/2/3 ~100%; intel gen-12lp 99.85%, gen-11 100%, gen-9 86.63%, gen-8 28.22%.

Sources: [Web3D Survey shader-f16](https://web3dsurvey.com/webgpu/features/shader-f16), [shader-f16 on Android](https://web3dsurvey.com/webgpu/features/shader-f16/platform/Android).

**Does Chromium now expose `shader-f16` on Android?** The survey implies **yes on most modern Android** — Adreno 7xx/8xx at ~97–99% and Mali/Valhall/Bifrost at 100% would be impossible if Chromium still excluded all Qualcomm and if the f16 polyfill had not landed. **But I could not find a Chrome release note or Chromium bug explicitly announcing an Android `shader-f16` polyfill/enablement**, so treat the mechanism as **[unverified]** and the *coverage* as measured fact. Older Adreno 5xx/6xx devices remain the main gap.

### Correct code pattern

```js
const adapter = await navigator.gpu.requestAdapter();
if (!adapter) throw new Error("No WebGPU adapter");

const hasF16 = adapter.features.has("shader-f16");
const device = await adapter.requestDevice({
  requiredFeatures: hasF16 ? ["shader-f16"] : [],
  requiredLimits: {
    maxBufferSize: adapter.limits.maxBufferSize,
    maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
    maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
  },
});
```

Then branch in WGSL via a type alias:

```js
const header = hasF16 ? `enable f16;\nalias min16float = f16;` : `alias min16float = f32;`;
```

Both snippets are the pattern shown in the [Chrome 120 "New in WebGPU" post](https://developer.chrome.com/blog/new-in-webgpu-120).

### What happens if you request it and it's unavailable

Per MDN, `requestDevice({ requiredFeatures: [...] })` rejects with a **`TypeError`** when the adapter lacks a requested feature ([MDN `GPUAdapter.requestDevice()`](https://raw.githubusercontent.com/mdn/content/main/files/en-us/web/api/gpuadapter/requestdevice/index.md)). MDN's own example therefore gates on `adapter.features.has(...)` before pushing into `requiredFeatures`. (The spec text has historically used `OperationError` for this case in some revisions; MDN documents `TypeError`. Either way the result is a rejected promise, so the guarded pattern above is the only safe one.)

### How libraries handle it

**onnxruntime-web (the engine under transformers.js) — verified in source.** `WebGpuBackend.initialize()` builds the device descriptor like this ([`js/web/lib/wasm/jsep/backend-webgpu.ts`, main](https://raw.githubusercontent.com/microsoft/onnxruntime/main/js/web/lib/wasm/jsep/backend-webgpu.ts)):

```ts
const requiredFeatures: GPUFeatureName[] = [];
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
...
requireFeatureIfAvailable('shader-f16');
requireFeatureIfAvailable('subgroups');
```

Two takeaways: (a) ORT-web **always requests the adapter's maximum limits**, so it never inherits the 256 MiB / 128 MiB spec defaults — this is the single most important line for your question 3; (b) it **opportunistically** enables `shader-f16` (and `subgroups`, `timestamp-query`) rather than requiring them, so device creation never fails on that account. Note this does not by itself guarantee an `fp16`-quantized graph runs well on a device without `shader-f16`.

**transformers.js** exposes dtype choice rather than feature negotiation: `{ dtype: "fp16" | "q4" | "q4f16" | "q8" | "fp32" | "bnb4" | ... }`, with **per-module dtypes** for encoders that are quantization-sensitive ([transformers.js dtypes guide](https://huggingface.co/docs/transformers.js/v3.8.1/guides/dtypes)). Its own VLM example hardcodes `vision_encoder: "fp16"` for Florence-2 — which is exactly the pattern that breaks on devices without `shader-f16`.

**Real-world failure of the naive pattern:** HF Space `webml-community/Qwen3.5-WebGPU`, discussion #1 (Apr 2026): "Linux + Chrome 144 + RTX 3060 Ti + Vulkan… `navigator.gpu.requestAdapter()` succeeds but `adapter.features.has("shader-f16") === false`. The Space hardcodes `vision_encoder` to fp16. Suggest adding q8/fp32 fallback logic." ([HF discussion](https://huggingface.co/spaces/webml-community/Qwen3.5-WebGPU/discussions/1)). This matches the survey's Linux **70.23%** figure. **Implication for your app: choose the dtype at runtime from `adapter.features.has('shader-f16')`, don't hardcode fp16.**

**Measured payoff when it *is* available:** on an Apple **M1 Pro**, the `f16` implementation of Llama-2-7B in the WebLLM chat demo was **+28% prefill and +41% decode** versus `f32` [measured] ([Chrome 120 blog](https://developer.chrome.com/blog/new-in-webgpu-120)).

---

## 5. COOP/COEP and SharedArrayBuffer

**Exact headers required to enable cross-origin isolation** ([web.dev "A guide to enable cross-origin isolation"](https://web.dev/articles/cross-origin-isolation-guide)):

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

Verify with `self.crossOriginIsolated === true` in the console (same page). COEP `require-corp` means every cross-origin subresource (model weights on a CDN, WASM binaries, images) must send `Cross-Origin-Resource-Policy: cross-origin` (or be loaded with CORS / `crossorigin` attribute); `COEP: credentialless` exists in Chrome 96+ but "it's not supported by any other browsers yet"; `COOP: same-origin` breaks OAuth/payment popups (same page). Assessment (`Report-Only` variants) exist for both headers.

**How ORT-web picks multithreaded vs single-threaded** ([onnxruntime.ai "The 'env' Flags and Session Options"](https://onnxruntime.ai/docs/tutorials/web/env-flags-and-session-options.html)):

- `ort.env.wasm.numThreads` default is **`0`** = "determined by ONNX Runtime Web based on the environment. In browsers, it will be set to **half of `navigator.hardwareConcurrency` or `4`, whichever is smaller**."
- "Only when the browser supports WebAssembly multi-threading **and** `crossOriginIsolated` mode is enabled, multi-threading will be enabled. See Cross Origin Isolation Guide."
- `ort.env.wasm.numThreads = 1` **forces single-threading**.
- The threaded and non-threaded artifacts are separately named — the docs' own `wasmPaths` example lists both `ort-wasm-simd.jsep.wasm` and `ort-wasm-simd-threaded.jsep.wasm` (same page). Both the JS bundle and the `.wasm` must come from the same build or init fails on mismatched minified names.

**Does WebGPU inference need SharedArrayBuffer?** **No.** Cross-origin isolation is required for `SharedArrayBuffer` (and high-resolution timers / `measureUserAgentSpecificMemory`), and COOP/COEP are orthogonal to WebGPU ([web.dev COI guide](https://web.dev/articles/cross-origin-isolation-guide)). SAB/cross-origin isolation is needed only by the **multi-threaded WASM** backend, i.e. the CPU fallback path — a third-party compatibility matrix states it plainly: "SharedArrayBuffer — Enables multi-threaded WASM inference… Requires Cross-Origin Isolation headers (COOP/COEP). **Not required for basic functionality**" ([LocalMode](https://localmode.dev/blog/compatibility/webgpu-support)). If you ship WebGPU-only inference you can skip COOP/COEP entirely and thereby avoid the CDN/CORP and popup breakage that COEP causes. The one caveat: ORT-web's `env.wasm.proxy` worker **cannot work with the WebGPU EP** ("a GPU buffer is not transferable") and cannot run under a restrictive CSP ([ORT env docs](https://onnxruntime.ai/docs/tutorials/web/env-flags-and-session-options.html)).

---

## 6. Cold-start vs warm latency for small VLMs (250M–1B) in a browser

### What I could find that is documented and hardware-named

**A. WebLLM paper (arXiv 2412.15803v2, revised 13 Apr 2026) — LLMs, not VLMs; decode throughput only.** [measured]

| Model (4-bit) | WebLLM (tok/s) | MLC-LLM native (tok/s) | Retained |
|---|---|---|---|
| Llama-3.1-8B | 41.1 | 57.7 | 71.2% |
| Phi-3.5-mini (3.8B) | 71.1 | 89.3 | 79.6% |

Hardware: **Apple MacBook Pro M3 Max**, WebLLM v0.2.75 in **Chrome Canary 133.0.6870.0 (arm64)**, vs MLC-LLM commit `d23d6f5` natively. The paper also cites "a 4-bit-quantized 3B-parameter model generates ~90 tokens/s on an Apple M3 laptop." ([arXiv abs](https://arxiv.org/abs/2412.15803), [HTML](https://arxiv.org/html/2412.15803v2)). No TTFT, no cold-start/download time, no VLM, no phone.

**B. `f16` vs `f32` speedup, Llama-2-7B, Apple M1 Pro (WebLLM demo).** [measured] +28% prefill, +41% decode with `shader-f16` ([Chrome 120 blog](https://developer.chrome.com/blog/new-in-webgpu-120)).

**C. Sony AI "Empirical Recipes for Efficient and Compact VLMs" (arXiv 2603.16987, 17 Mar 2026) — real VLMs including SmolVLM-256M, but server-side.** [measured, **NOT browser**]

| Model | TTFT baseline | TTFT optimized | Throughput | E2E baseline → optimized |
|---|---|---|---|---|
| SmolVLM-256M | **344.7 ms** | **22.8 ms** (−93%) | 460.7 → **533.3 tok/s** | 427.6 → **85.1 ms** (−80%) |
| InternVL3-2B | 124.0 ms | 57.7 ms (−53%) | 270.9 → 292.7 tok/s | 180.9 → 101.3 ms (−44%) |

Setup: **vLLM on an H100, batch size 1**, LMMs-Eval COCO2017 captioning, 5,000 multimodal requests. The paper's central finding is directly relevant to browsers: in the compact regime **CPU-side operations (image decode/resize/normalize, text tokenization) dominate TTFT**, not GPU compute — "previously overlooked CPU-side operations, such as image processing and text tokenization, often dominate latency." Their single biggest win on SmolVLM-256M was replacing PIL decoding (344.7 → 241.6 ms) and collapsing redundant image transforms (→ 71.9 ms). Note that a browser does image preprocessing in JS/WASM on the main thread or a worker, which is the same class of cost. ([HTML](https://arxiv.org/html/2603.16987v1))

**D. SmolVLM / SmolVLM2 model facts (Hugging Face blogs).** SmolVLM (2B) requires a **minimum 5.02 GB GPU RAM** in transformers; it encodes each 384×384 patch to **81 tokens**, and encodes "our test prompt and a single image in **1.2k tokens**" (vs 16k for Qwen2-VL); prefill throughput is **3.3–4.5×** and generation throughput **7.5–16×** faster than Qwen2-VL, measured with one or two input images ([SmolVLM blog, Nov 2024](https://huggingface.co/blog/smolvlm)). SmolVLM2 ships **256M / 500M / 2.2B**, with an iPhone app running the 500M model fully locally ([SmolVLM2 blog, Feb 2025](https://huggingface.co/blog/smolvlm2)). **No wall-clock browser numbers.**

**E. Library-level claims.** transformers.js v3 announcement headlines "WebGPU support (**up to 100x faster than WASM**)" — no hardware named, no per-model figures ([HF blog](https://huggingface.co/blog/transformersjs-v3)). A compatibility guide claims WebLLM "30–90 tokens/second" on GPU vs wllama WASM "5–20 tokens/second" — again no model/hardware named ([LocalMode](https://localmode.dev/blog/compatibility/webgpu-support)). A VLM-focused blog giving "50–150 ms local latency" and "up to 5× vs WASM" **explicitly labels its own numbers as "Indicative figures, not our measurements"** ([Vucense](https://vucense.com/dev-corner/webgpu-acceleration-local-vision-language-models-lumina/)) — do not cite it as measured.

### What I could NOT find — be explicit

- **No published TTFT, prefill time, or decode tokens/s for a small VLM running in a browser, on any named desktop dGPU / laptop iGPU / phone.** The only numbers I found that satisfy "VLM + named hardware" are the Sony AI H100/vLLM figures (C), which are not browser. The only browser numbers with named hardware are WebLLM's *LLM* decode throughput on an M3 Max (A) and the M1 Pro f16 delta (B).
- No documented **cold-start / first-load** timing (model download + ORT init + first inference) for a VLM in a browser. I found no source that measures it. A Sitpoint article titled "WebGPU vs WebASM: Browser Inference Benchmarks" appeared in search results but returned **HTTP 403** on fetch, so I could not read its numbers.

### Best available estimate, clearly labelled [estimate]

Compose from sourced pieces rather than guess a single number:

| Component | Basis |
|---|---|
| Download | SmolVLM2-256M ≈ 256M params; q4 → ~0.15 GB, q8/fp16 → ~0.3–0.5 GB. **I did not verify exact ONNX file byte sizes** (HF `/api/models/{id}?blobs=true` returned 401 in this environment). |
| First-load init | Includes cache write + ORT session create + **WebGPU shader compilation**; Chrome reports DXC improved compute-shader compile speed **~20%** vs FXC on Windows ([Chrome 121](https://developer.chrome.com/blog/new-in-webgpu-121)), but no absolute number is published. |
| Warm decode (desktop dGPU/iGPU) | WebLLM's 41–71 tok/s for 4-bit 3–8B LLMs on an M3 Max is the closest anchor; a 256M VLM should be **faster** but is usually decode-bound by memory bandwidth and by per-token JS overhead. **[estimate]** |
| Warm decode (phone) | **No sourced number.** In the same class as the vendor-agnostic "30–90 tok/s" claim, i.e. not trustworthy per-device. **[estimate / unverified]** |
| Prefill / image encode | SmolVLM encodes prompt+1 image to ~1.2k tokens ([SmolVLM blog](https://huggingface.co/blog/smolvlm)); on H100+vLLM that costs ~345 ms baseline. On a phone, budget **hundreds of ms to >1 s** for prefill alone, plus CPU-side image preprocessing. **[estimate]** |

**Recommendation for measurement:** instrument in-page with `performance.now()` around (a) fetch/cache read, (b) `InferenceSession.create`, (c) first `generate`, (d) steady-state tokens/s, and report per-device-class; the literature does not have this data.

---

## 7. Storage and caching

### How transformers.js caches

- **Browser: Cache API by default.** `env.useBrowserCache = true` (default), cache name `env.cacheKey` = `'transformers-cache'`. "Models are cached after the first download… Cache persists across page reloads and browser sessions." Disable with `env.useBrowserCache = false`. ([transformers.js Caching Reference, `skills/transformers-js/references/CACHE.md`](https://raw.githubusercontent.com/huggingface/skills/main/skills/transformers-js/references/CACHE.md))
- **Node:** `env.useFSCache` (default true), `env.cacheDir` (default `./.cache`).
- **Custom backends:** `env.useCustomCache` + `env.customCache` implementing `{ match(url), put(url, response) }` (same doc).
- **WASM binaries:** `env.useWasmCache` (default true) ([CONFIGURATION.md](https://raw.githubusercontent.com/huggingface/skills/main/skills/transformers-js/references/CONFIGURATION.md)).
- Monitor with `navigator.storage.estimate()` → `{usage, quota}` (same docs).

### Quotas and eviction

| Browser | Documented quota | Eviction |
|---|---|---|
| **Chrome / Chromium** | Browser may use **80% of total disk**; **a single origin up to 60% of total disk**. Incognito ~**5%**. If "clear cookies and site data when you close all windows" is on, quota drops to **~300 MB**. | Best-effort; on pressure Chromium evicts **all data of the least-recently-used origin first**, then the next, until under limit. ([web.dev "Storage for the web"](https://web.dev/articles/storage-for-the-web)) |
| **Firefox** | Browser may use **50% of free disk**. MDN: best-effort origin limit = **smaller of 10% of disk or a 10 GiB group limit per site**; with persistent storage granted, up to **50% of disk capped at 8 TiB** and exempt from the group limit. (web.dev's older text says an eTLD+1 group "may use up to 2 GB".) | Best-effort LRU-per-origin eviction; `navigator.storage.persist()` triggers a **user prompt** in Firefox. ([MDN Storage quotas and eviction criteria](https://developer.mozilla.org/en-US/docs/Web/API/Storage_API/Storage_quotas_and_eviction_criteria), [web.dev](https://web.dev/articles/storage-for-the-web)) |
| **Safari (desktop + iOS)** | **~1 GB per origin**, then Safari prompts and raises the limit in **~200 MB increments**; web.dev author states "I could not find any official documentation on this." Installed PWAs get a **separate** storage container. | Since **iOS/iPadOS 13.4 and Safari 13.1 on macOS**, script-writable storage (IndexedDB, service worker registrations, **Cache API**) is **evicted after 7 days of Safari use without user interaction**. **Installed PWAs added to the home screen are exempt.** ([web.dev](https://web.dev/articles/storage-for-the-web), [WebKit blog on the 7-day cap](https://webkit.org/blog/10218/full-third-party-cookie-blocking-and-more/)) |

Best-effort vs persistent: everything above is **best-effort by default**; `navigator.storage.persist()` opts into persistent storage. "Safari and most Chromium-based browsers automatically approve or deny the request based on the user's history of interaction with the site and do not show any prompts" ([MDN](https://developer.mozilla.org/en-US/docs/Web/API/Storage_API/Storage_quotas_and_eviction_criteria)). Storage Buckets allow per-bucket eviction prioritization ([web.dev](https://web.dev/articles/storage-for-the-web)).

**Practical realistic VLM cache size:** desktop Chromium/Firefox — hundreds of MB to several GB is fine. Safari — plan for **≤1 GB without a prompt**, and note that a 1B-parameter fp16 VLM (~2 GB) will hit the prompt or fail. Always request persistence and handle `QuotaExceededError` from both IndexedDB transactions and `cache.put()` ([web.dev](https://web.dev/articles/storage-for-the-web)).

### `use_external_data_format` and large-model handling

- **Why it exists (ORT-web, "Working with Large Models"):** max `ArrayBuffer` in Chrome is **0x7fe00000 bytes (~2 GB)**, so `response.arrayBuffer()` on a large file can fail; ORT-web works around this by allocating via `new WebAssembly.Memory()`, but such an ArrayBuffer **is not transferable**, so it cannot be used with the proxy-worker feature. **ONNX protobuf has a 2 GB file limit** — "If an ONNX model is larger than 2GB, it's usually generated with external data." **WebAssembly has a 4 GB memory limit** — "Currently, there is no way for ONNX Runtime Web to run models larger than 4GB." ([onnxruntime.ai/docs/tutorials/web/large-models.html](https://onnxruntime.ai/docs/tutorials/web/large-models.html))
- **Browser-side API:** you must pass the external-data mapping yourself, because JS cannot read the filesystem:

```js
const mySession = await ort.InferenceSession.create(modelUrl, {
  externalData: [{ path: './model_a.data', data: externalDataUrl }], // data: URL | Blob | Uint8Array
});
```

  `path` must match the weight's `location` string inside the protobuf. Same doc shows loading both `.onnx` and `.data` from **IndexedDB** as Blobs. `onnxruntime-web`'s `env.wasm.wasmPaths` accepts an object of `{ "ort-wasm-simd-threaded.jsep.wasm": url }` for self-hosting ([ORT env docs](https://onnxruntime.ai/docs/tutorials/web/env-flags-and-session-options.html)).
- **Caveat / [unverified]:** I read the transformers.js Caching, Configuration and WebGPU guides and ORT-web's GPU/large-model/env pages; **none of them documents a `use_external_data_format` option by that name**. `use_external_data_format` is an **export/conversion-time** flag (optimum / onnxruntime Python tooling) that produces the `.onnx` + `.onnx_data` split; the browser side consumes it via `externalData`. If your pipeline relies on that flag, verify it against the current optimum/transformers.js converter docs rather than the runtime docs — I did not find a runtime-side page naming it.
- Practical guidance: keep total weights under **2 GB per protobuf/ArrayBuffer**, shard as needed, prefer **q4/q8** for phones, request `navigator.storage.persist()`, and remember Safari's 7-day eviction can wipe a warm cache for casual users.

---

## Uncertain / could not verify

1. **caniuse vs MDN/W3C conflict.** caniuse lists Firefox 141–159 as "Disabled by default" and Chrome for Android only from **152**; [MDN BCD](https://raw.githubusercontent.com/mdn/browser-compat-data/main/api/GPU.json) and the [W3C wiki](https://github.com/gpuweb/gpuweb/wiki/Implementation-Status) say Firefox enabled-by-default on Windows since **141** and Chrome Android since **121**. I trust BCD/wiki for versions and caniuse for usage share, but I could not resolve which caniuse table is stale.
2. **Whether/how Chromium enabled `shader-f16` on Android.** Survey coverage (Adreno 7xx/8xx ~97–99%) implies it is enabled now, but I found **no Chrome release note, Chromium bug, or Dawn commit** announcing the Android f16 polyfill. The mechanism is unverified; the coverage is measured.
3. **Spec "maximum" column values** for `maxBufferSize` and `maxStorageBufferBindingSize` (as distinct from defaults). I got the full **default** table from MDN but could not isolate the spec's per-limit maximums; search snippets of the W3C CRD were ambiguous.
4. **Exact ONNX weight file sizes / download times** for SmolVLM2-256M/500M or Qwen2-VL-2B ONNX. HF `?blobs=true` returned 401 here; I only have parameter counts from the blogs.
5. **Any browser VLM latency number with named hardware.** No source gives TTFT/prefill/decode for a VLM *in a browser* on a named dGPU, iGPU, or phone. Sony AI's SmolVLM numbers are H100/vLLM. The Sitpoint "WebGPU vs WebASM" benchmark article returned HTTP 403 and could not be read.
6. **Android WebView specifics.** BCD mirrors `chrome_android` (121) but I found no WebView-specific release note, and WebView version numbers no longer map to Chrome versions.
7. **iOS-specific throttling values** in llama.cpp PR 21533 (how many ops/command buffers are allowed). I read the PR description, not the diff.
8. **`use_external_data_format` in transformers.js runtime docs** — not found by that name; see §7.
9. **Whether `reportAdapterInfo`/`info.architecture` is usable for Adreno-vs-other branching.** On iOS `adapter.info` reportedly returns only `"apple"` ([llama.cpp PR 21533](https://github.com/ggml-org/llama.cpp/pull/21533)), so UA sniffing is being used in practice; I did not verify Android behavior.
