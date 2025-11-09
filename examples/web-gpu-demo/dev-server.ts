/**
 * Development server for WebGPU demo
 * Serves static files from public/ at root path
 */

const PORT = 3000;

const server = Bun.serve({
  port: PORT,
  async fetch(req) {
    const url = new URL(req.url);
    let filePath = url.pathname;

    // Serve index.html for root
    if (filePath === '/') {
      filePath = '/index.html';
    }

    // Try to serve from public folder first
    if (filePath.startsWith('/models/')) {
      const publicPath = `./public${filePath}`;
      const file = Bun.file(publicPath);
      if (await file.exists()) {
        return new Response(file);
      }
    }

    // Serve other files from project root
    const file = Bun.file(`.${filePath}`);
    if (await file.exists()) {
      return new Response(file);
    }

    // 404
    return new Response('Not Found', { status: 404 });
  },
});

console.log(`🚀 Dev server running at http://localhost:${PORT}`);
console.log(`📁 Serving public/ at root path`);
console.log(`📄 Open http://localhost:${PORT} in your browser`);
