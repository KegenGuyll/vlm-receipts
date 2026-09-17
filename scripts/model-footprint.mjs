/**
 * Resolve the TRUE download size of an ONNX model repo.
 *
 * `scripts/check-extra-models.mjs` under-reports badly: quantized ONNX exports
 * keep weights in external `.onnx_data` sidecar files, so summing only `*.onnx`
 * can report 0.7 MB for a 400 MB model. This sums every file transformers.js
 * would actually fetch under `onnx/`, and reports the architecture so we can
 * check it against what the installed transformers.js supports.
 *
 *   node scripts/model-footprint.mjs [repo ...]
 */
const REPOS = process.argv.slice(2).filter((a) => !a.startsWith('--'));
const TARGETS = REPOS.length
  ? REPOS
  : [
      'onnx-community/LFM2.5-VL-450M-ONNX',
      'onnx-community/LFM2-VL-450M-ONNX',
      'onnx-community/granite-docling-258M-ONNX',
      'HuggingFaceTB/SmolVLM-500M-Instruct',
    ];

const QUANTS = ['q4f16', 'q4', 'fp16', 'int8', 'uint8', 'quantized'];

for (const id of TARGETS) {
  const res = await fetch(`https://huggingface.co/api/models/${id}?blobs=true`);
  if (!res.ok) {
    console.log(`${id}  -> HTTP ${res.status}`);
    continue;
  }
  const info = await res.json();
  const files = info.siblings ?? [];

  // Config tells us the architecture transformers.js must recognize.
  let arch = '?';
  let modelType = '?';
  try {
    const cfgRes = await fetch(`https://huggingface.co/${id}/raw/main/config.json`);
    if (cfgRes.ok) {
      const cfg = await cfgRes.json();
      arch = Array.isArray(cfg.architectures) ? cfg.architectures.join(',') : '?';
      modelType = cfg.model_type ?? '?';
    }
  } catch { /* leave unknown */ }

  console.log(`\n${id}`);
  console.log(`  pipeline=${info.pipeline_tag ?? '?'} license=${info.cardData?.license ?? '?'} downloads=${info.downloads ?? 0}`);
  console.log(`  architectures=${arch}  model_type=${modelType}`);

  const onnxFiles = files.filter((f) => f.rfilename.startsWith('onnx/'));
  for (const q of QUANTS) {
    const hit = onnxFiles.filter((f) => f.rfilename.includes(q));
    if (!hit.length) continue;
    const onnxBytes = hit.reduce((a, f) => a + (f.size ?? 0), 0);
    // External weight sidecars are named after the .onnx file, e.g.
    // onnx/decoder_model_merged_q4.onnx_data
    const sidecars = files.filter(
      (f) => f.rfilename.endsWith('.onnx_data') && hit.some((h) => f.rfilename.startsWith(h.rfilename)),
    );
    const sideBytes = sidecars.reduce((a, f) => a + (f.size ?? 0), 0);
    console.log(
      `    ${q.padEnd(9)} ${String(hit.length).padStart(2)} onnx (${(onnxBytes / 1048576).toFixed(1)}MB)` +
      ` + ${String(sidecars.length).padStart(2)} data (${(sideBytes / 1048576).toFixed(1)}MB)` +
      ` = ${((onnxBytes + sideBytes) / 1048576).toFixed(0)}MB`,
    );
  }

  const totalOnnxDir = onnxFiles.reduce((a, f) => a + (f.size ?? 0), 0);
  const allData = files.filter((f) => f.rfilename.endsWith('.onnx_data')).reduce((a, f) => a + (f.size ?? 0), 0);
  console.log(`  onnx/ total: ${((totalOnnxDir + allData) / 1048576).toFixed(0)}MB across ${onnxFiles.length} files`);
  await new Promise((r) => setTimeout(r, 200));
}
