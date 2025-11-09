# WebGPU-First Strategy for LibTorch-Rust

**Document**: Strategic Plan for WebGPU-Based Web Training
**Date**: 2025-11-09 08:45
**Vision**: Pure Rust ML framework with GPU acceleration via WebGPU, running in browser and native

---

## 🚀 Strategic Insight: This Changes Everything

### Why WebGPU + Web Training is a Game Changer

**Previous positioning**: "Slower pure-Rust alternative to tch-rs"
**New positioning**: "Browser-based ML training framework with GPU acceleration"

**Comparison Matrix**:

| Feature | tch-rs | PyTorch.js | **libtorch-rust + WebGPU** |
|---------|--------|------------|----------------------------|
| **Language** | Rust + C++ | JavaScript | **Pure Rust** |
| **Training** | ✅ Native only | ⚠️ Limited | **✅ Browser + Native** |
| **GPU** | CUDA/Metal only | WebGL/WebGPU | **✅ WebGPU (universal)** |
| **Browser** | ❌ No | ✅ Yes | **✅ Yes** |
| **Native** | ✅ Yes | ❌ No | **✅ Yes** |
| **C++ Deps** | Required | None | **None** |
| **WASM** | ❌ No | ✅ Yes | **✅ Yes** |
| **Performance** | Best (native) | Medium | **Good (GPU)** |

**Killer Feature**: Train ML models in the browser with GPU acceleration, then deploy the same code natively.

---

## 🎯 Unique Value Proposition

### What No One Else Offers

1. **Browser-Based Training with GPU** 🔥
   - Run training in browser with WebGPU acceleration
   - No server needed for small models
   - Privacy-preserving (data never leaves device)
   - Works on any OS (Windows, Mac, Linux, ChromeOS)

2. **Write Once, Run Anywhere**
   - Same Rust code for browser and native
   - Same WebGPU backend for all platforms
   - No CUDA lock-in, no Metal lock-in

3. **Pure Rust End-to-End**
   - No C++ dependencies
   - Easy to compile to WASM
   - Memory-safe throughout

4. **Educational & Research**
   - Interactive ML notebooks in browser
   - Live training visualization
   - Shareable via URL

5. **Edge Computing**
   - Train on device
   - Privacy-first ML
   - Offline training capability

---

## 🏗️ Architecture: WebGPU-First Design

### Core Principle

**CPU operations are NOT the performance target. GPU compute shaders are.**

```
┌─────────────────────────────────────────────┐
│         LibTorch-Rust User API              │
│  (Tensor, nn::Module, Optimizer, etc.)      │
└─────────────────────────────────────────────┘
                    ▼
┌─────────────────────────────────────────────┐
│         Autograd/Computational Graph        │
│     (Gradient tracking, backward pass)      │
└─────────────────────────────────────────────┘
                    ▼
┌─────────────────────────────────────────────┐
│            Operation Dispatch               │
│  ┌────────────────┬────────────────────┐   │
│  │   GPU Backend  │   CPU Fallback     │   │
│  │   (WebGPU)     │   (Current impl)   │   │
│  └────────────────┴────────────────────┘   │
└─────────────────────────────────────────────┘
                    ▼
┌─────────────────────────────────────────────┐
│              WebGPU Runtime                 │
│  ┌──────────────────────────────────────┐  │
│  │  Browser: wgpu via web-sys           │  │
│  │  Native:  wgpu via Vulkan/Metal/DX12 │  │
│  └──────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
```

---

## 📋 Revised MECE Implementation Plan

### New Categories (WebGPU-Centric)

1. **GPU Compute Backend** (Replaces CPU optimization)
2. **Training Infrastructure** (Autograd, backward)
3. **WebAssembly Support** (Browser deployment)
4. **Browser Integration** (Web APIs, visualization)
5. **Performance & Optimization** (GPU-specific)

---

## 🎯 Phase 1: GPU Backend Foundation (4-6 weeks)

### Goal: Get basic GPU operations working

#### Week 1-2: WebGPU Infrastructure

**Setup**:
```toml
[dependencies]
wgpu = "0.19"
bytemuck = "1.14"  # For GPU buffer casting
pollster = "0.3"   # For async in examples

[target.'cfg(target_arch = "wasm32")'.dependencies]
wasm-bindgen = "0.2"
wasm-bindgen-futures = "0.4"
web-sys = { version = "0.3", features = ["Window", "Navigator", "Gpu"] }
console_error_panic_hook = "0.1"
```

**Tasks**:
- [ ] Initialize wgpu device and queue (browser + native)
- [ ] Create GPU buffer management abstraction
- [ ] Implement buffer upload/download
- [ ] Add GPU/CPU memory synchronization
- [ ] Write basic GPU compute shader (add two arrays)

**Deliverable**: Can run simple GPU compute shader on browser and native

---

#### Week 3-4: Core GPU Operations

**Implement WebGPU compute shaders for**:

1. **Element-wise operations** (easy):
   ```wgsl
   @group(0) @binding(0) var<storage, read> input_a: array<f32>;
   @group(0) @binding(1) var<storage, read> input_b: array<f32>;
   @group(0) @binding(2) var<storage, read_write> output: array<f32>;

   @compute @workgroup_size(256)
   fn add(@builtin(global_invocation_id) global_id: vec3<u32>) {
       let idx = global_id.x;
       output[idx] = input_a[idx] + input_b[idx];
   }
   ```

2. **Matrix multiplication** (critical):
   - Use tiled/blocked matmul algorithm
   - Workgroup shared memory
   - Expected: 50-100x faster than CPU naive

3. **Activation functions**:
   - ReLU, Sigmoid, Tanh (parallel over elements)

4. **Reductions** (harder):
   - Sum, mean via parallel reduction
   - Multiple shader passes

**Deliverable**: Core operations run on GPU, faster than CPU

---

#### Week 5-6: Tensor GPU Backend

**Abstraction Layer**:
```rust
pub enum TensorBackend {
    Cpu(CpuTensor),
    Gpu(GpuTensor),
}

pub struct GpuTensor {
    buffer: wgpu::Buffer,
    shape: Vec<usize>,
    strides: Vec<usize>,
    dtype: DType,
    device: GpuDevice,
}

impl GpuTensor {
    pub async fn add(&self, other: &GpuTensor) -> Result<GpuTensor> {
        // Dispatch GPU compute shader
        self.device.execute_shader("add", &[self.buffer, other.buffer])
    }

    pub async fn to_cpu(&self) -> Result<CpuTensor> {
        // Download from GPU
    }

    pub fn from_cpu(cpu: &CpuTensor, device: &GpuDevice) -> Result<GpuTensor> {
        // Upload to GPU
    }
}
```

**Tasks**:
- [ ] Implement GpuTensor with WebGPU buffers
- [ ] Add CPU↔GPU data transfer
- [ ] Implement lazy execution (build command buffer)
- [ ] Add automatic GPU kernel selection
- [ ] GPU memory pooling

**Deliverable**: Tensor operations automatically use GPU when available

**Benchmark Target**:
- Matrix 500×500: <2ms (currently 156ms CPU)
- Add 10K elements: <100µs (currently 10.4µs CPU)

---

## 🎯 Phase 2: Training Infrastructure (4-6 weeks)

### Goal: Enable training with GPU-accelerated backward pass

#### Week 7-9: Autograd System

**Computational Graph**:
```rust
pub struct Tensor {
    data: TensorBackend,  // CPU or GPU
    grad: Option<Box<Tensor>>,
    grad_fn: Option<Arc<dyn BackwardFn>>,
    requires_grad: bool,
}

pub trait BackwardFn: Send + Sync {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor>;
}

// Example: Addition backward
struct AddBackward {
    input_a: Weak<Tensor>,
    input_b: Weak<Tensor>,
}

impl BackwardFn for AddBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        // Gradient of a + b is 1 for both inputs
        vec![grad_output.clone(), grad_output.clone()]
    }
}
```

**Tasks**:
- [ ] Add requires_grad flag to Tensor
- [ ] Implement computational graph tracking
- [ ] Add backward() method
- [ ] Implement backward functions for all operations
- [ ] Add gradient accumulation
- [ ] Implement no_grad() context

**Deliverable**: Can compute gradients (on GPU!)

---

#### Week 10-11: Optimizers and Loss Functions

**GPU-Accelerated Optimizers**:
```rust
pub struct GpuSgd {
    params: Vec<GpuTensor>,
    gradients: Vec<GpuTensor>,
    lr: f32,
    momentum: f32,
    velocity: Vec<GpuTensor>,  // Stored on GPU
}

impl GpuSgd {
    pub async fn step(&mut self) {
        // Single GPU kernel for all param updates
        // params = params - lr * grads (on GPU)
        self.device.execute_shader("sgd_step", &self.params_and_grads)
    }
}
```

**Loss Functions** (GPU kernels):
- MSE loss (parallel reduction)
- CrossEntropy loss (parallel softmax + log)
- Binary CrossEntropy

**Tasks**:
- [ ] Implement SGD with GPU updates
- [ ] Implement Adam with GPU updates
- [ ] Implement MSE loss (forward + backward)
- [ ] Implement CrossEntropy loss (forward + backward)
- [ ] Add learning rate scheduling

**Deliverable**: Can train models end-to-end on GPU

---

#### Week 12: Training Loop and Validation

**Example Training**:
```rust
async fn train_mnist() -> Result<()> {
    let device = GpuDevice::new().await?;
    let mut model = MnistNet::new(&device);
    let mut optimizer = Adam::new(model.parameters(), 0.001)?;

    for epoch in 0..10 {
        for (images, labels) in train_loader {
            // Forward pass (GPU)
            let output = model.forward(&images).await?;
            let loss = cross_entropy(&output, &labels).await?;

            // Backward pass (GPU)
            loss.backward().await?;

            // Optimizer step (GPU)
            optimizer.step().await?;
            optimizer.zero_grad();
        }

        println!("Epoch {}: loss = {}", epoch, loss.item());
    }

    Ok(())
}
```

**Tasks**:
- [ ] Port all examples to GPU backend
- [ ] Add MNIST training example
- [ ] Validate gradient correctness (vs numerical gradients)
- [ ] Add training progress logging
- [ ] Benchmark training throughput

**Deliverable**: Can train MNIST in browser with GPU acceleration

---

## 🎯 Phase 3: WebAssembly & Browser (2-3 weeks)

### Goal: Deploy to browser with full functionality

#### Week 13-14: WASM Compilation

**Build Setup**:
```toml
[lib]
crate-type = ["cdylib", "rlib"]  # For WASM

[profile.release]
opt-level = "z"      # Optimize for size
lto = true           # Link-time optimization
codegen-units = 1    # Better optimization
```

**Tasks**:
- [ ] Add wasm-pack build configuration
- [ ] Implement async operations for browser
- [ ] Handle browser GPU initialization
- [ ] Add WASM bindings for Tensor API
- [ ] Optimize WASM bundle size (<2MB goal)

**Build Command**:
```bash
wasm-pack build --target web --release
```

---

#### Week 15: Browser Demo Application

**Create Interactive Demo**:
```html
<!DOCTYPE html>
<html>
<head>
    <title>LibTorch-Rust Browser Training</title>
    <script type="module">
        import init, { train_mnist, Tensor } from './pkg/libtorch_rust.js';

        async function runTraining() {
            await init();  // Initialize WASM

            // Train MNIST in browser!
            const model = await train_mnist({
                epochs: 5,
                batch_size: 32,
                learning_rate: 0.001,
                onProgress: (epoch, loss) => {
                    document.getElementById('loss').textContent =
                        `Epoch ${epoch}: Loss = ${loss.toFixed(4)}`;
                }
            });

            console.log("Training complete!");
        }
    </script>
</head>
<body>
    <button onclick="runTraining()">Train MNIST in Browser</button>
    <div id="loss">Not started</div>
    <canvas id="visualization"></canvas>
</body>
</html>
```

**Features**:
- [ ] Live training visualization
- [ ] GPU usage monitoring
- [ ] Interactive hyperparameter tuning
- [ ] Model export/import
- [ ] Inference playground

**Deliverable**: Full browser-based training demo

---

## 🎯 Phase 4: Advanced Features (4+ weeks)

### Week 16-17: More Layers (GPU Kernels)

**Convolution** (critical for vision):
```wgsl
// 2D Convolution compute shader
@compute @workgroup_size(16, 16)
fn conv2d(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let out_y = global_id.y;
    let out_x = global_id.x;

    var sum = 0.0;
    for (var ky = 0u; ky < kernel_h; ky++) {
        for (var kx = 0u; kx < kernel_w; kx++) {
            let in_y = out_y * stride + ky;
            let in_x = out_x * stride + kx;
            sum += input[in_y][in_x] * kernel[ky][kx];
        }
    }
    output[out_y][out_x] = sum + bias;
}
```

**Layers to implement**:
- [ ] Conv2d (GPU compute shader)
- [ ] BatchNorm2d (GPU parallel ops)
- [ ] Dropout (GPU random numbers)
- [ ] MaxPool2d (GPU parallel reduction)
- [ ] LayerNorm (GPU normalization)

---

### Week 18-19: Data Loading & Augmentation

**Browser-Friendly Data Loading**:
```rust
// Load MNIST from URL
pub async fn load_mnist_from_url(url: &str) -> Result<Dataset> {
    #[cfg(target_arch = "wasm32")]
    {
        let response = web_sys::window()
            .unwrap()
            .fetch_with_str(url)
            .await?;
        let bytes = response.array_buffer().await?;
        parse_mnist(&bytes)
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        let bytes = std::fs::read(url)?;
        parse_mnist(&bytes)
    }
}
```

**GPU Data Augmentation**:
- Random crop (GPU shader)
- Random flip (GPU shader)
- Color jitter (GPU shader)
- Normalization (GPU parallel)

---

### Week 20+: Production Features

**Model Serialization**:
```rust
// Save model weights
model.save_safetensors("model.safetensors")?;

// Load in browser
let model = MnistNet::load_from_url(
    "https://example.com/model.safetensors"
).await?;
```

**Mixed Precision Training** (fp16 on GPU):
- Automatic mixed precision
- Gradient scaling
- fp16 GPU kernels

**Distributed Training** (multi-GPU in browser?):
- WebRTC for peer-to-peer training
- Federated learning in browser

---

## 📊 Performance Targets (GPU)

### Expected Performance (WebGPU)

| Operation | CPU (Current) | CPU (Optimized) | **GPU (WebGPU)** |
|-----------|--------------|-----------------|------------------|
| Matrix 500×500 | 156ms | 5ms | **<2ms** ✨ |
| Matrix 100×100 | 1.0ms | 50µs | **<100µs** ✨ |
| Add 10K | 10.4µs | 2µs | **<50µs** ✨ |
| Conv2d 224×224 | N/A | Very slow | **<5ms** ✨ |
| MNIST Forward (32) | 6ms est | 1.5ms | **<500µs** ✨ |
| MNIST Training Epoch | ~30s est | ~10s | **<2s** ✨ |

**Target**: GPU performance competitive with native PyTorch on CPU, faster than PyTorch.js

---

## 🎯 Success Metrics

### Technical Milestones

**Phase 1 (GPU Backend)**:
- ✅ Matrix multiply 100x faster than CPU naive
- ✅ Can run all operations on GPU
- ✅ Works in browser and native

**Phase 2 (Training)**:
- ✅ Can compute gradients correctly
- ✅ Can train MNIST to >95% accuracy
- ✅ Training runs entirely on GPU

**Phase 3 (Browser)**:
- ✅ WASM bundle <2MB
- ✅ Can train in browser with WebGPU
- ✅ Interactive demo works on Chrome/Edge/Safari

**Phase 4 (Production)**:
- ✅ Can train ResNet-18 on CIFAR-10
- ✅ Supports common vision architectures
- ✅ Model serialization works

### User-Facing Goals

1. **"Train MNIST in your browser in under 10 seconds"**
2. **"No Python, no CUDA, no server required"**
3. **"Privacy-first ML: your data never leaves your device"**
4. **"Write once, run in browser and native"**

---

## 🔍 Competitive Differentiation

### vs PyTorch (tch-rs)

| Feature | PyTorch/tch-rs | libtorch-rust + WebGPU |
|---------|----------------|------------------------|
| Browser training | ❌ No | ✅ Yes |
| WASM support | ❌ No | ✅ Yes |
| Universal GPU | ❌ CUDA/ROCm only | ✅ WebGPU (all) |
| C++ dependencies | ❌ Required | ✅ None |
| Setup complexity | ❌ High | ✅ Low (just browser) |

### vs TensorFlow.js

| Feature | TensorFlow.js | libtorch-rust + WebGPU |
|---------|---------------|------------------------|
| Language | JavaScript | Rust (type-safe) |
| GPU backend | WebGL/WebGPU | WebGPU only (modern) |
| Native support | ⚠️ Via Node.js | ✅ First-class |
| Performance | Good | Better (Rust + WASM) |
| Memory safety | ❌ JS runtime | ✅ Rust guarantees |

### vs ONNX Runtime Web

| Feature | ONNX Runtime Web | libtorch-rust + WebGPU |
|---------|------------------|------------------------|
| Training | ❌ Inference only | ✅ Full training |
| Custom models | ⚠️ Limited | ✅ Full flexibility |
| GPU | ✅ WebGPU | ✅ WebGPU |

**Unique Position**: Only pure-Rust framework with full training + WebGPU + browser support

---

## 🚧 Challenges & Mitigations

### Challenge 1: WebGPU Async in Rust

**Problem**: WebGPU is async, Rust async in WASM is tricky

**Mitigation**:
- Use wasm-bindgen-futures
- Provide sync API with internal async (for browser)
- Use pollster for native blocking

### Challenge 2: Shader Development

**Problem**: Writing WGSL shaders is complex

**Mitigation**:
- Start with simple operations
- Use shader templates
- Test on native first (better debugging)
- Consider shader library (e.g., from burn)

### Challenge 3: WASM Bundle Size

**Problem**: ML frameworks can be large

**Mitigation**:
- Feature flags for operations
- Tree-shaking via wasm-pack
- LTO and optimization
- Target: <2MB compressed

### Challenge 4: Browser Compatibility

**Problem**: WebGPU not universally supported yet

**Mitigation**:
- Provide CPU fallback
- Check feature support at runtime
- Clear documentation of requirements
- Target Chrome/Edge/Safari (WebGPU available)

### Challenge 5: GPU Memory Limits

**Problem**: Browsers limit GPU memory

**Mitigation**:
- Implement memory pooling
- Add out-of-core training for large models
- Stream data efficiently
- Clear error messages on OOM

---

## 📚 Technology Stack

### Core Dependencies

```toml
[dependencies]
# GPU backend
wgpu = "0.19"                    # WebGPU API
bytemuck = { version = "1.14", features = ["derive"] }
pollster = "0.3"                 # Async blocking for native

# WASM support
[target.'cfg(target_arch = "wasm32")'.dependencies]
wasm-bindgen = "0.2"
wasm-bindgen-futures = "0.4"
web-sys = { version = "0.3", features = [
    "Window", "Navigator", "Gpu", "GpuDevice",
    "GpuBuffer", "GpuQueue", "console"
]}
js-sys = "0.3"
console_error_panic_hook = "0.1"

# Core functionality (existing)
rand = "0.8"
safetensors = "0.4"

[dev-dependencies]
criterion = "0.5"
approx = "0.5"
```

### Build Tools

```toml
[package.metadata.wasm-pack.profile.release]
wasm-opt = ["-O4", "-Oz"]  # Aggressive optimization
```

---

## 🎓 Learning Resources

**For Team**:
1. WebGPU Fundamentals: https://webgpufundamentals.org/
2. wgpu Tutorial: https://sotrh.github.io/learn-wgpu/
3. WGSL Spec: https://www.w3.org/TR/WGSL/
4. Burn (Rust DL framework with GPU): https://github.com/tracel-ai/burn

**Reference Implementations**:
- Burn: Pure Rust with wgpu backend
- Candle: Pure Rust with CUDA/Metal
- TensorFlow.js: Browser ML reference

---

## 🎯 Decision: Revised Strategy

### **RECOMMENDATION: WebGPU-First Approach**

**Phase 1 (Now - Month 2)**: GPU Backend Foundation
- Build WebGPU compute infrastructure
- Implement core GPU kernels
- Get 10-100x speedup over CPU

**Phase 2 (Month 2-3)**: Training on GPU
- Add autograd with GPU backward pass
- GPU-accelerated optimizers
- Train MNIST in browser

**Phase 3 (Month 3-4)**: Browser Polish
- WASM optimization
- Interactive demos
- Documentation

**Phase 4 (Month 4+)**: Advanced Features
- More layers (Conv2d, etc.)
- Production features
- Ecosystem building

### Why This Path?

1. **Differentiation**: No one else has this combo (Rust + Training + WebGPU + Browser)
2. **Performance**: GPU solves the speed problem elegantly
3. **Practicality**: WebGPU works on all platforms
4. **Future-proof**: WebGPU is the future of web graphics/compute
5. **Excitement**: Training in browser is genuinely novel

### Skip the CPU Optimization Plan

The MECE CPU optimization plan is **obsolete** if going WebGPU-first:
- GPU compute shaders are inherently 10-100x faster
- WebGPU handles cross-platform (Vulkan/Metal/DX12)
- Browser + Native with same code
- Focus effort on GPU kernels, not CPU SIMD

**Exception**: Keep simple CPU fallback for debugging and compatibility

---

## 📋 Next Actions

**Immediate (This Week)**:
1. [ ] Prototype basic wgpu setup (browser + native)
2. [ ] Write simple compute shader (add two arrays)
3. [ ] Verify WebGPU works in browser
4. [ ] Create project structure for GPU backend

**Sprint 1 (Next 2 Weeks)**:
1. [ ] Implement GPU buffer management
2. [ ] Write compute shaders for: add, mul, matmul
3. [ ] Add CPU↔GPU transfer
4. [ ] Benchmark GPU vs CPU

**Sprint 2 (Weeks 3-4)**:
1. [ ] Complete all element-wise operations on GPU
2. [ ] Implement GPU matrix multiplication
3. [ ] Add GPU activation functions
4. [ ] Create GpuTensor abstraction

---

## 🎬 Conclusion

**WebGPU changes the entire value proposition**:

❌ **Old story**: "Pure Rust PyTorch that's slower"
✅ **New story**: "Train ML models in your browser with GPU acceleration"

This is:
- **Technically feasible**: wgpu is mature, WebGPU is shipping
- **Strategically sound**: Unique position in market
- **Practically useful**: Privacy-first ML, edge training, education
- **Future-proof**: WebGPU is the future

**Recommendation**: Abandon CPU optimization path, go all-in on WebGPU backend.

---

**Document Version**: 1.0
**Status**: Proposed Strategic Direction
**Next Review**: After Phase 1 prototype (2 weeks)
