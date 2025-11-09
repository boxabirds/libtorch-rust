#!/bin/bash
set -e

echo "🔨 Building WASM module..."

# Navigate to wasm-trainer directory
cd ../../libtorch-wasm-trainer

# Build WASM
echo "📦 Running wasm-pack build..."
RUSTFLAGS='--cfg wasm_js' wasm-pack build --target web --out-dir pkg

# Copy to web-gpu-demo
echo "📋 Copying WASM files to web-gpu-demo..."
rm -rf ../examples/web-gpu-demo/public/wasm
cp -r pkg ../examples/web-gpu-demo/public/wasm

echo "✅ WASM build complete!"
echo "   WASM module available at: public/wasm/"
echo ""
echo "🚀 Start the dev server:"
echo "   bun run dev"
