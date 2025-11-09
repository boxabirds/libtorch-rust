# LibTorch-Rust WebGPU Browser Demo

This is a browser-based demonstration of WebGPU compute operations that mirrors the Rust implementation in `examples/gpu_demo.rs`. It showcases that the same WGSL (WebGPU Shading Language) shaders work in both:

- **Native applications** (via Rust + wgpu)
- **Web browsers** (via JavaScript + WebGPU API)

## Features

- ✅ **Same WGSL Shaders** - Identical compute shaders as the Rust implementation
- ✅ **Cross-Platform GPU** - Works on Chrome, Edge, Safari with WebGPU support
- ✅ **Large-Scale Operations** - Realistic data sizes (millions of elements)
- ✅ **Performance Metrics** - Throughput and GFLOPS measurements
- ✅ **Beautiful UI** - Modern React interface with real-time status
- ✅ **Works Out of the Box** - Benchmarks run immediately without setup

## Demos Included

### GPU Benchmarks (Works Immediately)

| Operation | Size | Memory | Metric |
|-----------|------|--------|--------|
| Element-wise Add | 10M elements | 40 MB/tensor | M ops/sec |
| Element-wise Mul | 5M elements | 20 MB/tensor | M ops/sec |
| Matrix Multiply | 512×512 | 1 MB/matrix | GFLOPS |
| ReLU Activation | 8M elements | 32 MB | M ops/sec |
| Sigmoid Activation | 6M elements | 24 MB | M ops/sec |

### MNIST Digit Recognition (Optional, Requires Training)

Interactive digit drawing with neural network inference. Requires model training (see below).

## Prerequisites

### Browser Requirements

- **Chrome** 113+ or **Edge** 113+
- **Safari** 18+ (macOS)
- **Firefox** Nightly with WebGPU enabled

### Development Requirements

- [Bun](https://bun.sh) 1.0+ (JavaScript runtime)

## Installation

```bash
cd examples/web-gpu-demo

# Install dependencies
bun install
```

## Running the Demo

### Development Mode (Recommended)

```bash
bun run dev
```

This will start the dev server at http://localhost:3000.

**The GPU benchmarks work immediately!** Just click "Initialize WebGPU" and then "Run GPU Benchmarks".

### Alternative: Simple HTTP Server

```bash
# Using Python
python3 -m http.server 8000

# Or use the npm script
bun run serve
```

Then navigate to `http://localhost:8000`

## Training MNIST Model (Optional)

The MNIST demo is **optional** and requires trained model weights. The GPU benchmarks work without this step.

To enable the MNIST demo:

```bash
# Install PyTorch
pip install torch torchvision

# Train the model (~2-3 minutes)
python scripts/train_mnist.py
```

This will:
- Train a 784→128→10 MLP on MNIST
- Export weights to `public/models/mnist-mlp.json`
- Achieve ~97-98% test accuracy

### Weight File Format

```json
{
  "weights": {
    "fc1_weight": [...],  // 128×784 flat array
    "fc1_bias": [...],    // 128 array
    "fc2_weight": [...],  // 10×128 flat array
    "fc2_bias": [...]     // 10 array
  }
}
```

## Browser Compatibility

### ✅ Fully Supported

- **Chrome/Edge 113+** (Windows, macOS, Linux)
- **Safari 18+** (macOS with Apple Silicon or Intel)

### 🚧 Experimental

- **Firefox Nightly** with `dom.webgpu.enabled` flag

### ❌ Not Supported

- Internet Explorer
- Older browser versions
- Browsers without GPU hardware access

## Architecture

### File Structure

```
web-gpu-demo/
├── src/
│   ├── webgpu/
│   │   ├── device.ts        # GPU initialization
│   │   ├── shaders.ts       # WGSL compute shaders
│   │   └── operations.ts    # High-level GPU operations
│   ├── App.tsx              # Main React component
│   └── main.tsx             # React entry point
├── index.html               # HTML entry point
├── package.json             # Bun configuration
└── README.md               # This file
```

### Code Comparison: Rust vs Browser

The implementation demonstrates perfect API parity:

**Rust (via wgpu):**
```rust
let device = GpuDevice::new().await?;
let a = GpuTensor::from_slice(&data, &[size], device)?;
let result = a.add(&b).await?;
```

**Browser (via WebGPU API):**
```typescript
const device = await initializeGPU();
const result = await elementwiseAdd(device, a, b);
```

Both use the **exact same WGSL shaders**!

## Performance Notes

### Expected Performance

On modern GPUs (Apple M1/M2, NVIDIA RTX, AMD RX):

- **Element-wise ops:** 100-500+ million ops/sec
- **Matrix multiply (512×512):** 50-200+ GFLOPS
- **Activations:** Similar to element-wise operations

### Performance Factors

1. **GPU Hardware** - Dedicated GPUs > Integrated GPUs
2. **Browser** - Chrome/Edge often faster than Safari
3. **Power Settings** - "High Performance" mode recommended
4. **Background Apps** - Close other GPU-intensive applications

## Troubleshooting

### "WebGPU Not Supported"

- Update your browser to the latest version
- Check if your GPU supports WebGPU
- On macOS: Ensure Safari 18+ or Chrome 113+
- On Windows/Linux: Use Chrome 113+ or Edge 113+

### Poor Performance

- Close other tabs/applications using GPU
- Check if running on integrated vs dedicated GPU
- Try different browser (Chrome vs Safari)
- Monitor GPU usage in system tools

### Shader Compilation Errors

- Check browser console for detailed error messages
- Verify WebGPU is enabled (chrome://gpu in Chrome)
- Update graphics drivers

## Comparison with Rust Implementation

This browser demo validates the WebGPU strategy by showing:

1. **Same Shaders Work** - WGSL is truly cross-platform
2. **Similar Performance** - Browser WebGPU is competitive with native
3. **Easy Development** - TypeScript provides great developer experience
4. **Real ML Workloads** - Can handle millions of elements efficiently

This proves that Phase 3 (WASM compilation) of the libtorch-rust roadmap is viable.

## Next Steps

After Phase 3 (WASM Compilation):

1. Compile Rust libtorch-rust to WASM
2. Replace TypeScript ops with WASM imports
3. Provide Python-like API in browser
4. Enable full ML training in browser

## Related Files

- **Rust GPU Demo:** `../gpu_demo.rs`
- **Rust GPU Implementation:** `../../libtorch-rust-sys/src/gpu/`
- **Strategy Document:** `../../docs/20251109-0845-webgpu-strategy.md`
- **GPU Implementation Doc:** `../../docs/20251109-GPU_IMPLEMENTATION.md`

## License

Same as libtorch-rust project.

## Contributing

This demo is part of the libtorch-rust project. See the main README for contribution guidelines.
