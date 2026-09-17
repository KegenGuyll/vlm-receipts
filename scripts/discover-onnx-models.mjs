// Discover which candidate small-VLM repos actually publish ONNX weights usable
// by transformers.js, and how big the quantized assets are.
const CANDIDATES = [
  'HuggingFaceTB/SmolVLM-256M-Instruct',
  'HuggingFaceTB/SmolVLM-500M-Instruct',
  'HuggingFaceTB/SmolVLM2-500M-Video-Instruct',
  'HuggingFaceTB/SmolVLM2-256M-Video-Instruct',
  'HuggingFaceTB/SmolVLM-Instruct',
  'Xenova/moondream2',
  'onnx-community/moondream2-20250414',
  'onnx-community/moondream2-text-model-ONNX',
  'onnx-community/Florence-2-base-ft',
  'onnx-community/Florence-2-base',
  'onnx-community/Qwen2.5-VL-3B-Instruct',
  'onnx-community/Qwen2-VL-2B-Instruct',
  'onnx-community/gemma-3-4b-it-ONNX',
  'onnx-community/GLM-4.1V-9B-Thinking-ONNX',
  'onnx-community/glm-ocr',
  'Xenova/vit-gpt2-image-captioning',
];

const QUANTS = ['q4f16', 'q4', 'fp16', 'int8', 'uint8', 'quantized'];

function fmt(bytes) {
  if (bytes == null) return '?';
  return (bytes / 1048576).toFixed(1) + 'MB';
}

async function getJson(url) {
  const res = await fetch(url, { headers: { 'user-agent': 'vlm-receipts-bench/0.1' } });
  if (!res.ok) return { error: `${res.status} ${res.statusText}` };
  return res.json();
}

const rows = [];
for (const id of CANDIDATES) {
  const info = await getJson(`https://huggingface.co/api/models/${id}?blobs=true`);
  if (info.error) {
    rows.push({ id, status: `ERROR ${info.error}`, onnx: 0 });
    continue;
  }
  const files = (info.siblings ?? []).filter((s) => /\.onnx$/i.test(s.rfilename));
  const config = (info.siblings ?? []).some((s) => /preprocessor_config\.json$/i.test(s.rfilename));
  // Prefer weights under an `onnx/` subfolder (what transformers.js resolves).
  const inOnnxDir = files.filter((f) => f.rfilename.startsWith('onnx/') || f.rfilename.includes('/onnx/'));
  const pool = inOnnxDir.length ? inOnnxDir : files;
  const found = {};
  for (const q of QUANTS) {
    const hit = pool.filter((f) => f.rfilename.includes(q));
    if (hit.length) {
      found[q] = {
        files: hit.length,
        bytes: hit.reduce((a, f) => a + (f.size ?? 0), 0),
      };
    }
  }
  rows.push({
    id,
    status: 'OK',
    downloads: info.downloads ?? 0,
    license: info.cardData?.license ?? '?',
    pipeline: info.pipeline_tag ?? '?',
    onnxFiles: files.length,
    onnxInSubdir: inOnnxDir.length > 0,
    hasPreprocessor: config,
    quants: found,
    baseFiles: pool.filter((f) => !QUANTS.some((q) => f.rfilename.includes(q))).map((f) => f.rfilename),
  });
  await new Promise((r) => setTimeout(r, 250));
}

for (const r of rows) {
  console.log('='.repeat(100));
  if (r.status !== 'OK') {
    console.log(`${r.id}  -> ${r.status}`);
    continue;
  }
  console.log(`${r.id}`);
  console.log(`  pipeline=${r.pipeline} license=${r.license} downloads=${r.downloads}`);
  console.log(`  onnxFiles=${r.onnxFiles} inSubdir=${r.onnxInSubdir} hasPreprocessorConfig=${r.hasPreprocessor}`);
  for (const [q, v] of Object.entries(r.quants)) {
    console.log(`    ${q.padEnd(10)} ${String(v.files).padStart(2)} file(s)  ${fmt(v.bytes)}`);
  }
  if (r.baseFiles.length) console.log(`    base onnx: ${r.baseFiles.join(', ')}`);
}
