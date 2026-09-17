// @ts-check
/**
 * Step 0 diagnostic: prove we can drive system Chrome headless and obtain a
 * real WebGPU adapter/device with the features a VLM runtime needs.
 *
 * Serves the page over http://127.0.0.1 so the origin is a secure context
 * (WebGPU is gated on secure contexts; file:// is not reliable).
 */
import { createServer } from 'node:http';
import { chromium } from 'playwright-core';

const CHROME_PATHS = [
  'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
  'C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe',
  'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',
  'C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe',
];

const PAGE = `<!doctype html>
<html><head><meta charset="utf-8"><title>webgpu-probe</title></head>
<body><div id="app">probing</div></body></html>`;

const server = createServer((_req, res) => {
  res.writeHead(200, {
    'content-type': 'text/html; charset=utf-8',
    'cross-origin-opener-policy': 'same-origin',
    'cross-origin-embedder-policy': 'require-corp',
  });
  res.end(PAGE);
});

await new Promise((r) => server.listen(0, '127.0.0.1', r));
const { port } = /** @type {import('node:net').AddressInfo} */ (server.address());
const url = `http://127.0.0.1:${port}/`;
console.log(`[probe] serving ${url}`);

const browser = await chromium.launch({
  executablePath: CHROME_PATHS[0],
  headless: process.env.HEADED !== '1',
  args: [
    '--no-sandbox',
    '--disable-gpu-sandbox',
    '--enable-unsafe-webgpu',
    '--no-first-run',
    '--no-default-browser-check',
    '--disable-crash-reporter',
    '--disable-breakpad',
  ],
});

const page = await browser.newPage();
await page.goto(url, { waitUntil: 'load' });

const result = await page.evaluate(async () => {
  const out = {};
  out.secureContext = window.isSecureContext;
  out.userAgent = navigator.userAgent;
  out.hardwareConcurrency = navigator.hardwareConcurrency;
  out.deviceMemory = navigator.deviceMemory ?? null;
  out.hasNavigatorGpu = !!navigator.gpu;
  if (!navigator.gpu) return { ...out, verdict: 'NO_WEBGPU' };
  try {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) return { ...out, verdict: 'NO_ADAPTER' };
    const info = adapter.info ?? (adapter.requestAdapterInfo ? await adapter.requestAdapterInfo() : {});
    out.adapter = { vendor: info.vendor, architecture: info.architecture, device: info.device, description: info.description };
    out.features = [...adapter.features].sort();
    out.hasShaderF16 = adapter.features.has('shader-f16');
    const device = await adapter.requestDevice();
    out.limits = {
      maxBufferSize: device.limits.maxBufferSize,
      maxStorageBufferBindingSize: device.limits.maxStorageBufferBindingSize,
      maxComputeWorkgroupStorageSize: device.limits.maxComputeWorkgroupStorageSize,
      maxComputeInvocationsPerWorkgroup: device.limits.maxComputeInvocationsPerWorkgroup,
    };
    // Prove we can actually run compute, not just request a device.
    const gpuBuffer = device.createBuffer({ size: 64, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    out.computeBufferAllocated = gpuBuffer.size === 64;
    gpuBuffer.destroy();
    return { ...out, verdict: 'WEBGPU_OK' };
  } catch (err) {
    return { ...out, verdict: 'ERROR', error: String(err && err.message ? err.message : err) };
  }
});

console.log('\n=== WebGPU probe result ===');
console.log(JSON.stringify(result, null, 2));

await browser.close();
server.close();
