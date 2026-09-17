// Repair mojibake in any source file, and report what was fixed.
import { readFile, writeFile } from 'node:fs/promises';

const FIXES = [
  ['\u00e2\u20ac\u201d', '\u2014'], // em dash
  ['\u00e2\u20ac\u201c', '\u2013'], // en dash
  ['\u00e2\u20ac\u00a6', '\u2026'], // ellipsis
  ['\u00e2\u20ac\u2122', '\u2019'], // right single quote
  ['\u00e2\u20ac\u02dc', '\u201c'], // left double quote
  ['\u00e2\u20ac\u009d', '\u201d'], // right double quote
  ['\u00c2\u00a7', '\u00a7'], // section sign
  ['\u00c2\u00b7', '\u00b7'], // middle dot
  ['\u00c3\u00a9', '\u00e9'], // e-acute
];

const files = process.argv.slice(2);
if (!files.length) {
  console.error('usage: node scripts/repair-encoding.mjs <file ...>');
  process.exit(1);
}

let totalFixed = 0;
for (const f of files) {
  let text = await readFile(f, 'utf8');
  const before = text;
  let n = 0;
  for (const [bad, good] of FIXES) {
    const parts = text.split(bad);
    if (parts.length > 1) {
      n += parts.length - 1;
      text = parts.join(good);
    }
  }
  if (text !== before) {
    await writeFile(f, text, 'utf8');
    console.log(`[repair] ${f}: fixed ${n} sequence(s)`);
    totalFixed += n;
  } else {
    console.log(`[repair] ${f}: clean`);
  }
}
console.log(`[repair] total ${totalFixed}`);
