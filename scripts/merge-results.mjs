/**
 * Merge several benchmark result files into one.
 *
 * Benchmarks are often run a model at a time (a full run of five models takes a
 * while, and a crash halfway should not cost the earlier models). The report and
 * dashboard readers expect a single result set, so this combines them into one
 * ranking while keeping the provenance explicit — the merged file records which
 * source files fed it, so a partially-merged result is never mistaken for a
 * clean single run.
 *
 * Configuration must match across inputs (prompt, decode settings, fixture
 * count, split); a mismatch would make the comparison meaningless, so it fails
 * loudly instead of silently producing a bad table.
 *
 *   node scripts/merge-results.mjs --out results/bench-all.json a.json b.json
 *                                 [--force]
 */
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import path from 'node:path';

function arg(name, fallback) {
  const i = process.argv.indexOf(`--${name}`);
  return i !== -1 && process.argv[i + 1] ? process.argv[i + 1] : fallback;
}
const has = (name) => process.argv.includes(`--${name}`);

const outPath = arg('out', null);
const inputs = process.argv
  .slice(2)
  .filter((a) => !a.startsWith('--') && a.endsWith('.json') && a !== outPath);

if (!inputs.length || !outPath) {
  console.error('usage: node scripts/merge-results.mjs --out results/bench-all.json <file.json> [...]');
  process.exit(1);
}

const loaded = [];
for (const f of inputs) {
  const d = JSON.parse(await readFile(f, 'utf8'));
  loaded.push({ file: f, data: d });
}

/** Fields that must agree, or the merged ranking would compare unlike things. */
const IDENTITY = [
  ['dataset', (d) => d.dataset],
  ['split', (d) => d.split],
  ['fixtureCount', (d) => d.fixtureCount],
  ['maxNewTokens', (d) => d.maxNewTokens],
  ['prompt', (d) => d.prompt?.id],
  ['constrain', (d) => d.constrain],
  ['schemaName', (d) => d.schemaName ?? null],
];

const mismatches = [];
for (const [label, get] of IDENTITY) {
  const values = [...new Set(loaded.map((l) => JSON.stringify(get(l.data))))];
  if (values.length > 1) mismatches.push(`${label}: ${values.join(' vs ')}`);
}

if (mismatches.length) {
  console.error('[merge] refusing to merge incompatible result files:');
  for (const m of mismatches) console.error(`  ${m}`);
  if (!has('force')) {
    console.error('Re-run with --force only if you are certain the comparison is still meaningful.');
    process.exit(1);
  }
  console.warn('[merge] --force given; proceeding despite mismatches');
}

const base = loaded[0].data;
const runs = [];
const seen = new Map();
for (const { file, data } of loaded) {
  for (const r of data.runs ?? []) {
    if (seen.has(r.model)) {
      console.warn(`[merge] duplicate model ${r.model} (keeping the one from ${seen.get(r.model)})`);
      continue;
    }
    seen.set(r.model, file);
    runs.push({ ...r, _sources: [file] });
  }
}

const merged = {
  ...base,
  generatedAt: new Date().toISOString(),
  mergedFrom: inputs,
  mergedNote: `Merged from ${inputs.length} result file(s). Each model ran as its own benchmark process.`,
  runs,
};

await mkdir(path.dirname(outPath), { recursive: true });
await writeFile(outPath, JSON.stringify(merged, null, 2));

console.log(`[merge] ${inputs.length} file(s) -> ${outPath}`);
for (const r of runs) {
  const acc = r.summary ? `${(r.summary.accuracy * 100).toFixed(1)}%` : 'FAILED';
  console.log(`  ${String(r.model).padEnd(46)} ${acc.padStart(7)}  (${seen.get(r.model)})`);
}
console.log(`[merge] ${runs.length} models, ${base.fixtureCount} receipts each`);
