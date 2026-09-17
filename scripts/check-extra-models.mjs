/**
 * Verify the additional sub-1B candidates surfaced by research: confirm each
 * repo exists, publishes ONNX weights transformers.js can resolve, and report
 * real quantized sizes before adding any of them to the benchmark lineup.
 *
 *   node scripts/check-extra-models.mjs
 */
const CANDIDATES = [
  'onnx-community/LFM2.5-VL-450M-ONNX',
  'onnx-community/LFM2-VL-450M-ONNX',
  'onnx-community/granite-docling-258M-ONNX',
  'onnx-community/FastVLM-0.5B-ONNX',
  'onnx-community/SmolVLM-256M-Instruct',
  'onnx-community/SmolVLM2-256M-Video-Instruct',
  'onnx-community/Qwen3-VL-2B-Instruct-ONNX',
  'LiquidAI/LFM2.5-VL-450M',
  'ibm-granite/granite-docling-258M',
];

const QUANTS = ['q4f16', 'q4', 'fp16', 'int8', 'uint8', 'q8', 'bnb4'];

async function getJson(url) {
  const res = await fetch(url, { headers: { 'user-agent': 'vlm-receipts-bench/0.1' } });
  return { status: res.status, body: res.ok ? await res.json() : null };
}

for (const id of CANDIDATES) {
  const { status, body } = await getJson(`https://huggingface.co/api/models/${id}?blobs=true`);
  if (!body) {
    console.log(`${id.padEnd(48)} HTTP ${status} — DOES NOT EXIST / GATED`);
    continue;
  }
  const files = (body.siblings ?? []).filter((s) => /\.onnx$/i.test(s.rfilename));
  const onnxDir = files.filter((f) => f.rfilename.startsWith('onnx/'));
  const pool = onnxDir.length ? onnxDir : files;
  const hasConfig = (body.siblings ?? []).some((s) => s.rfilename === 'config.json');
  const hasPreproc = (body.siblings ?? []).some((s) => s.rfilename === 'preprocessor_config.json');

  console.log(`\n${id}`);
  console.log(`  pipeline=${body.pipeline_tag ?? '?'}  license=${body.cardData?.license ?? '?'}  downloads=${body.downloads ?? 0}`);
  console.log(`  onnxFiles=${files.length} inOnnxDir=${onnxDir.length > 0} config=${hasConfig} preprocessor=${hasPreproc}`);
  for (const q of QUANTS) {
    const hit = pool.filter((f) => f.rfilename.includes(q));
    if (hit.length) {
      const bytes = hit.reduce((a, f) => a + (f.size ?? 0), 0);
      console.log(`    ${q.padEnd(8)} ${String(hit.length).padStart(2)} file(s)  ${(bytes / 1048576).toFixed(1)}MB`);
    }
  }
  if (!pool.length) console.log('    (no onnx weights)');
  await new Promise((r) => setTimeout(r, 200));
}
