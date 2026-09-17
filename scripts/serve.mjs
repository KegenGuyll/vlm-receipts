/**
 * Start the harness dev server for benchmarking.
 *
 * The watcher is disabled via BENCH_NO_WATCH so that editing a file mid-run
 * cannot trigger an HMR reload. A reload destroys the page's execution context
 * and silently kills a multi-minute model benchmark, and on Windows the watcher
 * can also crash on a locked scratch file (EBUSY).
 *
 * Cross-platform without a `cross-env` dependency.
 */
import { spawn } from 'node:child_process';

const child = spawn(process.execPath, ['node_modules/vite/bin/vite.js'], {
  stdio: 'inherit',
  env: { ...process.env, BENCH_NO_WATCH: '1' },
});

child.on('exit', (code) => process.exit(code ?? 0));
for (const sig of ['SIGINT', 'SIGTERM']) {
  process.on(sig, () => child.kill(sig));
}
