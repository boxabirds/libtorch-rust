/**
 * Build script for WebGPU demo
 * Bundles the app and copies static assets
 */

import { copyFileSync, mkdirSync, existsSync, rmSync } from 'fs';
import { join } from 'path';

// Clean dist folder
if (existsSync('./dist')) {
  rmSync('./dist', { recursive: true });
}
mkdirSync('./dist', { recursive: true });

console.log('📦 Building WebGPU demo...\n');

// Bundle the app
const result = await Bun.build({
  entrypoints: ['./src/main.tsx'],
  outdir: './dist',
  target: 'browser',
  format: 'esm',
  splitting: true,
  minify: false,
  sourcemap: 'external',
});

if (!result.success) {
  console.error('❌ Build failed:');
  for (const log of result.logs) {
    console.error(log);
  }
  process.exit(1);
}

console.log('✅ Bundle created');

// Copy index.html
copyFileSync('./index.html', './dist/index.html');
console.log('✅ Copied index.html');

// Copy public folder
if (existsSync('./public')) {
  const copyDir = (src: string, dest: string) => {
    mkdirSync(dest, { recursive: true });
    const entries = require('fs').readdirSync(src, { withFileTypes: true });

    for (const entry of entries) {
      const srcPath = join(src, entry.name);
      const destPath = join(dest, entry.name);

      if (entry.isDirectory()) {
        copyDir(srcPath, destPath);
      } else {
        copyFileSync(srcPath, destPath);
      }
    }
  };

  copyDir('./public', './dist/public');
  console.log('✅ Copied public folder');
}

console.log('\n🎉 Build complete! Output in ./dist');
console.log('   Run: bun run serve-dist');
