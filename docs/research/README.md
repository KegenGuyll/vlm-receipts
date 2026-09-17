# Research notes

Background research that informed this benchmark's design. These are **inputs, not results** —
they were produced by literature/Hub surveys before any measurement, and where the benchmark later
contradicted them, the benchmark wins and the discrepancy is noted in `../../README.md` and
`results/REPORT.md`.

| Document | What it covers |
|---|---|
| `browser-webgpu-vlm-report.md` | State of in-browser VLM inference: WebLLM's vision support, transformers.js architecture coverage, ranked shortlist of sub-1B candidates, and implementation gotchas. |
| `webgpu-browser-constraints.md` | Browser/WebGPU constraints in depth: support matrix, buffer limits, mobile `shader-f16` availability, storage eviction, KV-cache reuse status. |
| `hf-receipt-datasets.md` | Survey of HuggingFace receipt datasets: schema coverage, licences, download mechanics, and why `albertobarnabo/synthetic-receipts-ocr` was chosen. |

## How much to trust these

They were accurate enough to steer the build correctly on several points that mattered — that
WebLLM has no sub-1B vision path, that `onnx-community/SmolVLM-*` repos do not exist (the weights
live in the base `HuggingFaceTB/*` repos), and that the datasets-server REST API works anonymously.

Two claims did **not** survive contact with measurements:

- The shortlist rated `LFM2.5-VL-450M` highly, which held up — it won. But an early note treated it
  as difficult to run in a worker; that turned out to be a bug in this harness, not the model.
- Constrained decoding via `@huggingface/transformers-structured-output` was flagged as the
  highest-value untested item. It was tested, and it does not apply in this stack — see
  `results/REPORT.md` §4.

Prefer the benchmark results over these documents wherever the two disagree.
