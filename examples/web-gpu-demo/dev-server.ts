/**
 * Development server for WebGPU demo
 * Builds the app and serves the dist folder with live reload
 */

import { watch } from 'fs';

const PORT = 3000;
let building = false;

// Initial build
await buildApp();

// Watch for changes and rebuild
console.log('👀 Watching for changes...\n');
watch('./src', { recursive: true }, async (event, filename) => {
  if (building) return;
  console.log(`📝 File changed: ${filename}`);
  await buildApp();
});

// Start server
Bun.serve({
  port: PORT,
  async fetch(req) {
    const url = new URL(req.url);
    let filePath = url.pathname;

    // Serve index.html for root
    if (filePath === '/') {
      filePath = '/index.html';
    }

    // Map /models/ to dist/public/models/
    if (filePath.startsWith('/models/')) {
      const distPath = `./dist/public${filePath}`;
      const file = Bun.file(distPath);
      if (await file.exists()) {
        return new Response(file);
      }
    }

    // Serve from dist/public for /public paths
    if (filePath.startsWith('/public/')) {
      const distPath = `./dist${filePath}`;
      const file = Bun.file(distPath);
      if (await file.exists()) {
        return new Response(file);
      }
    }

    // Serve from dist folder
    const file = Bun.file(`./dist${filePath}`);
    if (await file.exists()) {
      return new Response(file);
    }

    // 404
    return new Response('Not Found', { status: 404 });
  },
});

console.log(`🚀 Dev server running at http://localhost:${PORT}`);
console.log(`📄 Open http://localhost:${PORT} in your browser\n`);

// Build function
async function buildApp() {
  if (building) return;
  building = true;

  try {
    const result = await Bun.build({
      entrypoints: ['./src/main.tsx'],
      outdir: './dist',
      target: 'browser',
      format: 'esm',
      splitting: true,
      minify: false,
      sourcemap: 'inline',
      publicPath: '/',
    });

    if (!result.success) {
      console.error('❌ Build failed:');
      for (const log of result.logs) {
        console.error(log);
      }
      building = false;
      return;
    }

    // Copy index.html
    await Bun.write('./dist/index.html', await Bun.file('./index.html').text());

    // Copy public folder to dist/public
    await Bun.$`mkdir -p ./dist/public`.quiet();
    await Bun.$`cp -r ./public/* ./dist/public/`.quiet().catch(() => {
      // Ignore if public is empty
    });

    console.log('✅ Build complete');
  } catch (err) {
    console.error('❌ Build error:', err);
  } finally {
    building = false;
  }
}
