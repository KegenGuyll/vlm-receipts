/**
 * Screenshot a tall HTML page in vertical slices.
 *
 * A full-page capture of the dashboard now exceeds the 8192px image limit, so
 * verification screenshots are taken in halves. Kept as a script because
 * "the page got too long to screenshot" recurs as the report grows.
 *
 *   node scripts/screenshot.mjs results/dashboard.html [--slices 2] [--out results/shot]
 */
import { chromium } from 'playwright-core';
import path from 'node:path';
import { resolve } from 'node:path';

function arg(name, fallback) {
  const i = process.argv.indexOf(`--${name}`);
  return i !== -1 && process.argv[i + 1] ? process.argv[i + 1] : fallback;
}

const page_path = process.argv.slice(2).find((a) => !a.startsWith('--'));
if (!page_path) {
  console.error('usage: node scripts/screenshot.mjs <page.html> [--slices 2] [--out results/shot]');
  process.exit(1);
}
const slices = Number(arg('slices', '2'));
const outBase = arg('out', path.join('results', 'shot'));
const width = Number(arg('width', '1200'));

const CHROME = 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';
const browser = await chromium.launch({
  executablePath: CHROME,
  headless: true,
  args: ['--no-sandbox', '--disable-gpu-sandbox', '--no-first-run', '--disable-crash-reporter'],
});
const page = await browser.newPage({ viewport: { width, height: 1000 }, deviceScaleFactor: 1 });
await page.goto(`file:///${resolve(page_path).replace(/\\/g, '/')}`, { waitUntil: 'load' });

const total = await page.evaluate(() => document.body.scrollHeight);
const sliceH = Math.ceil(total / slices);
console.log(`[shot] page is ${total}px tall -> ${slices} slice(s) of ${sliceH}px`);

for (let i = 0; i < slices; i++) {
  const y = i * sliceH;
  const height = Math.min(sliceH, total - y);
  if (height <= 0) break;
  const out = `${outBase}-${i + 1}.png`;
  // Resize the viewport to the slice and scroll to its top, then screenshot the
  // viewport. A `clip` on a full-page capture fails once the page is taller than
  // the maximum surface Chromium will rasterize.
  await page.setViewportSize({ width, height });
  await page.evaluate((scrollY) => window.scrollTo(0, scrollY), y);
  await page.waitForTimeout(120);
  await page.screenshot({ path: out });
  console.log(`  ${out} (y=${y}..${y + height})`);
}

await browser.close();
