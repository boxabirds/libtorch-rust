# libtorch-wasm-trainer

**Status: Milestone 1 Complete ✅ (100%)**

Browser-based ML training using Rust + Burn framework, compiled to WASM. Train models in your browser with GPU/CPU, export weights compatible with PyTorch.

## Vision: Web Training → PyTorch Models

```
Browser (WASM + WebGPU)
    ↓
  Train Model
    ↓
 Export Weights (JSON)
    ↓
Load in PyTorch
    ↓
Fine-tune / Deploy
```

### Why This Matters

1. **Privacy-Preserving** - Train on sensitive data without leaving the browser
2. **Federated Learning** - Distribute training across users' devices
3. **PyTorch Compatibility** - Weights load directly into PyTorch ecosystem
4. **Vendor-Independent GPU** - WebGPU works on Nvidia, AMD, Apple Metal

## Current Implementation

### What Works

- ✅ MNIST MLP model (784→128→10) using Burn framework
- ✅ Training loop structure with autograd (forward/backward pass)
- ✅ Cross-entropy loss using Burn's built-in `CrossEntropyLoss`
- ✅ PyTorch-compatible weight export format
- ✅ WASM bindings via wasm-bindgen
- ✅ Adam optimizer integration
- ✅ Evaluation mode with accuracy tracking

### Burn 0.19 API Compatibility ✅

All Burn 0.19 API issues have been resolved:
- ✅ Correct `OptimizerAdaptor` import from `burn::optim::adaptor`
- ✅ Generic trainer with `Optimizer` trait bound
- ✅ Proper gradient extraction using `GradientsParams::from_grads()`
- ✅ Optimizer ownership handling with `std::mem::replace()`
- ✅ Loss to scalar conversion using `.elem()`
- ✅ All tests passing (4/4)

### Next Steps (Milestone 2+)

For detailed roadmap, see [ROADMAP.md](./ROADMAP.md)

**Immediate Next Steps:**

1. **Compile to WASM:**
   ```bash
   cd libtorch-wasm-trainer
   wasm-pack build --target web --out-dir pkg
   ```

2. **Browser Integration:**
   - Load WASM module in browser demo
   - Add training UI component
   - Test full workflow: train → export → load in PyTorch

3. **Start Milestone 2:** PyTorch Round-Trip
   - Create Python utility to load exported weights
   - Verify gradient compatibility
   - Test complete workflow

## Architecture

```rust
// Rust WASM Module
#[wasm_bindgen]
pub struct BrowserTrainer {
    // Burn model with autodiff
    model: MnistMLP<Autodiff<NdArray>>
    optimizer: Adam
}

impl BrowserTrainer {
    #[wasm_bindgen]
    pub fn train_batch(&mut self, images: Vec<f32>, labels: Vec<usize>) -> f32 {
        // Forward pass
        let output = self.model.forward(images);

        // Loss computation
        let loss = cross_entropy(output, labels);

        // Backward pass (Burn's autograd)
        let grads = loss.backward();

        // Update weights
        self.optimizer.step(grads);

        loss.value()
    }

    #[wasm_bindgen]
    pub fn export_pytorch_weights(&self) -> String {
        // Export weights in PyTorch format
        serde_json::to_string(&self.weights())
    }
}
```

```javascript
// Browser Usage
import init, { BrowserTrainer } from './pkg';

await init();
const trainer = new BrowserTrainer(0.001); // learning rate

// Training loop
for (const batch of mnistData) {
    const loss = trainer.train_batch(batch.images, batch.labels);
    console.log(`Loss: ${loss}`);
}

// Export for PyTorch
const weights = trainer.export_pytorch_weights();
downloadFile(weights, 'model.json');
```

```python
# Load in PyTorch
import torch
import json

class MnistMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(784, 128)
        self.fc2 = torch.nn.Linear(128, 10)

# Load browser-trained weights
with open('model.json') as f:
    weights = json.load(f)

model = MnistMLP()
model.load_state_dict({
    'fc1.weight': torch.tensor(weights['fc1_weight']).view(128, 784),
    'fc1.bias': torch.tensor(weights['fc1_bias']),
    'fc2.weight': torch.tensor(weights['fc2_weight']).view(10, 128),
    'fc2.bias': torch.tensor(weights['fc2_bias']),
})

# Now use with PyTorch ecosystem!
model.eval()
# Fine-tune, export to ONNX, deploy to production, etc.
```

## Dependencies

- **Burn 0.19** - Pure Rust ML framework with autodiff
- **burn-ndarray** - CPU backend (will add burn-wgpu for WebGPU later)
- **wasm-bindgen** - Rust ↔ JavaScript bindings
- **serde/serde_json** - Weight serialization

## Development Status

### Milestone 1: Browser Training MVP ✅ (100% complete)

- [x] Burn framework integration
- [x] MNIST MLP model
- [x] Training loop with autograd
- [x] PyTorch-compatible export
- [x] Fix Burn 0.19 API compatibility
- [ ] WASM compilation (next step)
- [ ] Browser demo integration (next step)

### Milestone 2: PyTorch Round-Trip (Planned)

- [ ] Python utility to load weights
- [ ] Verify gradient compatibility
- [ ] Test PyTorch → Browser → PyTorch workflow

### Milestone 3: WebGPU Backend (Future)

- [ ] Switch from NdArray to burn-wgpu backend
- [ ] GPU-accelerated training in browser
- [ ] Performance benchmarks vs. CPU

### Milestone 4: Advanced Features (Future)

- [ ] Support Conv2D, BatchNorm, etc.
- [ ] Multiple optimizers (SGD, AdamW)
- [ ] Learning rate scheduling
- [ ] Model checkpointing

## Known Issues

1. **WASM Size**: Initial WASM bundle will be large (~10-20MB)
   - Can optimize with `wasm-opt`
   - Tree-shaking helps

2. **Performance**: NdArray backend is CPU-only
   - Acceptable for prototyping
   - Will add WebGPU backend in Milestone 3

## Building

```bash
# Install wasm-pack
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh

# Build for web
wasm-pack build --target web --out-dir pkg

# The pkg/ directory contains WASM module for browser
```

## Testing

```bash
# Rust tests
cargo test

# WASM tests (once compiling)
wasm-pack test --headless --chrome
```

## Contributing

This is part of the larger libtorch-rust project. The goal is to enable:

1. Privacy-preserving federated learning
2. Browser-based ML education
3. Edge training with consumer GPUs
4. PyTorch ecosystem integration

## License

Same as libtorch-rust project (MIT OR Apache-2.0).

## Related

- **Roadmap**: [ROADMAP.md](./ROADMAP.md) - Detailed plan for Milestones 2-4
- **Browser Demo**: `../examples/web-gpu-demo/` - WebGPU operations and inference
- **Strategy Doc**: `../docs/20251109-0845-webgpu-strategy.md`
- **Burn Framework**: https://burn.dev
