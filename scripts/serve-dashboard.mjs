/**
 * Serve the generated dashboard for viewing in a browser.
 *
 * A file:// page works too, but serving it means the dashboard can be opened at
 * a stable localhost URL, shared, and reloaded after a rebuild without hunting
 * for the path. Zero dependencies, and it deliberately serves ONLY the
 * dashboard, so nothing else in the repo is exposed.
 *
 *   node scripts/serve-dashboard.mjs [--port 5180] [--host 127.0.0.1]
 *                                    [--file results/dashboard.html]
 */
import { createServer } from 'node:http';
import { readFile, stat } from 'node:fs/promises';
import path from 'node:path';

function arg(name, fallback) {
  const i = process.argv.indexOf(`--${name}`);
  return i !== -1 && process.argv[i + 1] ? process.argv[i + 1] : fallback;
}

const port = Number(arg('port', '5180'));
const host = arg('host', '127.0.0.1');
const file = path.resolve(arg('file', path.join('results', 'dashboard.html')));

try {
  await stat(file);
} catch {
  console.error(`[serve-dashboard] no dashboard at ${file}`);
  console.error('Build it first:  node scripts/dashboard.mjs results/bench-final.rescored.json');
  process.exit(1);
}

const server = createServer(async (req, res) => {
  const url = new URL(req.url ?? '/', `http://${host}:${port}`);

  if (url.pathname === '/health') {
    res.writeHead(200, { 'content-type': 'application/json' });
    res.end(JSON.stringify({ ok: true, file }));
    return;
  }

  // Serve the dashboard for any other path so the root URL just works.
  try {
    const body = await readFile(file);
    res.writeHead(200, {
      'content-type': 'text/html; charset=utf-8',
      'cache-control': 'no-store',
      'content-length': body.length,
    });
    res.end(body);
  } catch (err) {
    res.writeHead(500, { 'content-type': 'text/plain' });
    res.end(`failed to read dashboard: ${err.message}`);
  }
});

server.listen(port, host, () => {
  const url = `http://${host}:${port}/`;
  console.log(`[serve-dashboard] serving ${file}`);
  console.log(`[serve-dashboard] ${url}`);
});
