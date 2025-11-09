# libtorch-wasm-trainer: Milestones 2-4 Roadmap

**Status**: Milestone 1 Complete ✅ (100%)
**Created**: 2025-11-09
**Vision**: Privacy-preserving browser training with PyTorch ecosystem integration

---

## Overview

This roadmap covers the path from "Milestone 1: Browser Training MVP" (complete) to a production-ready browser-based training system with full PyTorch compatibility and GPU acceleration.

### Key Differentiators
1. **PyTorch Compatibility** - Weights export/import seamlessly
2. **Privacy-First** - All training happens client-side
3. **Vendor-Independent GPU** - WebGPU works across Nvidia, AMD, Apple Metal
4. **Production-Ready** - Not just a toy demo

---

## Milestone 2: PyTorch Round-Trip

**Timeline**: 1-2 weeks
**Status**: Not Started
**Goal**: Verify complete interoperability between browser training and PyTorch

### Why This Matters

Milestone 1 exports weights in PyTorch-compatible format, but we need to **prove** the round-trip works:
- Browser → PyTorch: Load weights and continue training
- PyTorch → Browser: Initialize browser training with pre-trained weights
- Gradient Compatibility: Ensure both frameworks compute identical gradients

This unlocks **real-world use cases**:
- Fine-tune PyTorch models in browser
- Federated learning with PyTorch aggregation server
- Transfer learning workflows

### Tasks

#### 2.1 Python Weight Loader (3 days)

**Files to Create**:
- `python/load_browser_weights.py` - Utility to load JSON weights into PyTorch
- `python/export_to_browser.py` - Export PyTorch model to browser-compatible JSON
- `python/test_compatibility.py` - Automated compatibility tests

**Implementation**:

```python
# load_browser_weights.py
import torch
import json
from typing import Dict, Any

class BrowserWeightLoader:
    """Load weights trained in browser into PyTorch models"""

    @staticmethod
    def load_mnist_mlp(weights_path: str) -> torch.nn.Module:
        """
        Load browser-trained MNIST MLP weights

        Args:
            weights_path: Path to exported JSON weights

        Returns:
            PyTorch model with loaded weights
        """
        with open(weights_path) as f:
            data = json.load(f)

        # Extract weights
        weights = data.get('weights', data)

        # Create PyTorch model
        model = torch.nn.Sequential(
            torch.nn.Linear(784, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10)
        )

        # Load weights (handle shape conversion)
        state_dict = {
            '0.weight': torch.tensor(weights['fc1_weight']).view(128, 784),
            '0.bias': torch.tensor(weights['fc1_bias']),
            '2.weight': torch.tensor(weights['fc2_weight']).view(10, 128),
            '2.bias': torch.tensor(weights['fc2_bias']),
        }

        model.load_state_dict(state_dict)
        return model

    @staticmethod
    def export_to_browser(model: torch.nn.Module, output_path: str):
        """Export PyTorch model to browser-compatible format"""
        state_dict = model.state_dict()

        weights = {
            'fc1_weight': state_dict['0.weight'].flatten().tolist(),
            'fc1_bias': state_dict['0.bias'].tolist(),
            'fc2_weight': state_dict['2.weight'].flatten().tolist(),
            'fc2_bias': state_dict['2.bias'].tolist(),
        }

        output = {
            'model_type': 'mnist-mlp',
            'framework': 'pytorch',
            'weights': weights,
            'metadata': {
                'exported_from': 'pytorch',
                'version': torch.__version__
            }
        }

        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
```

**Testing Strategy**:
```python
# test_compatibility.py
def test_weight_loading():
    """Test that browser weights load correctly"""
    # 1. Train simple model in browser (synthetic data)
    # 2. Export weights
    # 3. Load into PyTorch
    # 4. Compare forward pass outputs
    pass

def test_gradient_compatibility():
    """Test that gradients match between frameworks"""
    # 1. Initialize both frameworks with same weights
    # 2. Run forward pass on identical input
    # 3. Compute loss
    # 4. Backward pass
    # 5. Compare gradients (should be identical within floating point tolerance)
    pass

def test_roundtrip():
    """Test PyTorch → Browser → PyTorch"""
    # 1. Create PyTorch model
    # 2. Export to browser format
    # 3. "Train" in browser (1 batch)
    # 4. Load back into PyTorch
    # 5. Verify weights changed correctly
    pass
```

**Success Criteria**:
- ✅ Load browser weights into PyTorch without errors
- ✅ Forward pass outputs match within 1e-5 tolerance
- ✅ Gradients match within 1e-4 tolerance
- ✅ Round-trip maintains model performance

#### 2.2 Gradient Verification (2 days)

**Goal**: Prove Burn and PyTorch compute identical gradients

**Approach**:
1. Initialize both frameworks with **identical** random seed weights
2. Feed **identical** input batch
3. Compute loss with **identical** loss function
4. Run backward pass
5. Extract gradients and compare numerically

**Expected Challenge**: Floating point differences in:
- Matrix multiplication order
- Activation functions
- Loss computation

**Tolerance**: Accept differences < 1e-4 (standard for neural networks)

**Files**:
- `tests/gradient_compatibility.rs` - Rust test harness
- `python/test_gradients.py` - Python comparison script

#### 2.3 Fine-Tuning Workflow (2 days)

**Use Case**: Pre-train in PyTorch, fine-tune in browser

**Implementation**:
```javascript
// Browser: Load pre-trained PyTorch model
import init, { BrowserTrainer } from './pkg';

// Load pre-trained weights from PyTorch
const response = await fetch('/models/pretrained-mnist.json');
const pretrainedWeights = await response.json();

// Create trainer with pre-trained initialization
const trainer = BrowserTrainer.from_pytorch_weights(pretrainedWeights);

// Fine-tune on user's local data
for (const batch of userLocalData) {
    const loss = trainer.train_batch(batch.images, batch.labels);
}

// Export fine-tuned weights
const finetuned = trainer.export_pytorch_weights();
```

**Rust Changes Required**:
```rust
// Add to lib.rs
#[wasm_bindgen]
impl BrowserTrainer {
    /// Create trainer from PyTorch pre-trained weights
    #[wasm_bindgen]
    pub fn from_pytorch_weights(weights_json: &str) -> Result<BrowserTrainer, JsValue> {
        let weights: PyTorchWeights = serde_json::from_str(weights_json)
            .map_err(|e| JsValue::from_str(&format!("Parse error: {}", e)))?;

        let device = NdArrayDevice::Cpu;
        let config = TrainingConfig::default();
        let mut model = MnistMLP::new(&device);

        // Load weights into model
        // TODO: Need to implement model.load_weights(weights)

        let optim = AdamConfig::new().init();
        let trainer = MnistTrainer { model, optim, config, device };

        Ok(Self { trainer })
    }
}
```

**Success Criteria**:
- ✅ Load PyTorch weights into browser trainer
- ✅ Fine-tune for N batches
- ✅ Export and verify weights changed correctly
- ✅ Document workflow with examples

#### 2.4 Documentation & Examples (2 days)

**Deliverables**:
1. **Tutorial**: "Train in Browser, Deploy with PyTorch"
2. **Example Notebook**: `examples/pytorch_roundtrip.ipynb`
3. **API Documentation**: Weight format specification
4. **Blog Post**: "Privacy-Preserving ML with Browser Training"

**Example Notebook Outline**:
```markdown
# Browser Training → PyTorch Deployment

## Part 1: Train in Browser
- Load MNIST data
- Initialize BrowserTrainer
- Train for 5 epochs
- Export weights

## Part 2: Load into PyTorch
- Load exported weights
- Evaluate on test set
- Verify accuracy matches

## Part 3: Fine-Tune in PyTorch
- Add regularization
- Train for 5 more epochs
- Compare performance

## Part 4: Deploy
- Export to ONNX
- Deploy to production
```

### Milestone 2 Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Gradient mismatch due to numerical precision | High | Accept 1e-4 tolerance, document differences |
| Weight layout differences (row-major vs column-major) | High | Add explicit transpose/reshape logic |
| Burn API changes in weight access | Medium | Pin Burn version, add CI tests |
| Browser RNG differs from PyTorch | Low | Use fixed seeds for testing |

### Milestone 2 Success Criteria

- [ ] Python utility loads browser weights with zero manual conversion
- [ ] Gradients match within 1e-4 tolerance on 100 random test cases
- [ ] Round-trip (PyTorch → Browser → PyTorch) preserves model quality
- [ ] Complete Jupyter notebook demonstrating workflow
- [ ] Documentation published (tutorial + API reference)

**Estimated Effort**: 1-2 weeks (1 developer)

---

## Milestone 3: WebGPU Backend

**Timeline**: 2-3 weeks
**Status**: Not Started
**Goal**: GPU-accelerated training in browser using WebGPU

### Why This Matters

Current implementation uses **NdArray (CPU backend)**:
- ✅ Works immediately
- ✅ Good for prototyping
- ❌ **Slow** for real training (10-100x slower than GPU)

WebGPU enables:
- **Real-time training** on realistic datasets
- **Larger models** (CNNs, ResNets)
- **Competitive performance** vs native PyTorch on GPU
- **Cross-platform** GPU support (Vulkan/Metal/DX12)

### Tasks

#### 3.1 Burn WGPU Backend Integration (1 week)

**Current State**:
```rust
// lib.rs - Current CPU backend
type WasmBackend = Autodiff<NdArray>;
```

**Target State**:
```rust
// lib.rs - WebGPU backend
use burn::backend::wgpu::{Wgpu, WgpuDevice};

type WasmBackend = Autodiff<Wgpu>;

#[wasm_bindgen]
impl BrowserTrainer {
    #[wasm_bindgen(constructor)]
    pub async fn new(learning_rate: f64) -> Result<BrowserTrainer, JsValue> {
        // Initialize WebGPU device
        let device = WgpuDevice::default();

        let config = TrainingConfig::new(learning_rate, 32);
        let model = MnistMLP::new(&device);
        let optim = AdamConfig::new().init();

        let trainer = MnistTrainer { model, optim, config, device };
        Ok(Self { trainer })
    }
}
```

**Cargo.toml Changes**:
```toml
[dependencies]
burn = { version = "0.19", default-features = false, features = ["wgpu", "train"] }
burn-wgpu = { version = "0.19" }

[target.'cfg(target_arch = "wasm32")'.dependencies]
wgpu = { version = "22", features = ["webgpu"] }
```

**Challenges**:
1. **Async Initialization**: WebGPU device creation is async
   - Solution: Make `BrowserTrainer::new()` async
   - JavaScript: `const trainer = await new BrowserTrainer(0.001);`

2. **Feature Detection**: Not all browsers support WebGPU
   - Solution: Add fallback to NdArray backend
   - Check `navigator.gpu` before initialization

3. **Memory Management**: GPU memory limits
   - Solution: Batch size tuning, memory profiling

**Implementation Plan**:

**Phase 1: Basic WebGPU Support (3 days)**
- Update Cargo.toml with wgpu features
- Replace NdArray with Wgpu backend
- Add async initialization
- Verify compilation

**Phase 2: Browser Testing (2 days)**
- Test on Chrome Canary (WebGPU stable)
- Test on Firefox Nightly (experimental)
- Test on Safari Technology Preview
- Document browser requirements

**Phase 3: Fallback Strategy (2 days)**
```rust
#[wasm_bindgen]
pub enum BackendType {
    WebGPU,
    CPU,
}

#[wasm_bindgen]
impl BrowserTrainer {
    /// Create trainer with automatic backend selection
    #[wasm_bindgen]
    pub async fn new_auto(learning_rate: f64) -> BrowserTrainer {
        // Try WebGPU first
        if webgpu_available() {
            Self::new_webgpu(learning_rate).await
        } else {
            // Fallback to CPU
            Self::new_cpu(learning_rate)
        }
    }
}
```

#### 3.2 Performance Benchmarking (3 days)

**Benchmark Suite**:

```rust
// benches/training_speed.rs
#[wasm_bindgen_test]
async fn bench_train_batch_cpu_vs_gpu() {
    let batch_sizes = vec![16, 32, 64, 128];

    for batch_size in batch_sizes {
        // CPU backend
        let mut trainer_cpu = BrowserTrainer::new_cpu(0.001);
        let start = performance.now();
        for _ in 0..100 {
            trainer_cpu.train_batch(images, labels);
        }
        let cpu_time = performance.now() - start;

        // GPU backend
        let mut trainer_gpu = BrowserTrainer::new_webgpu(0.001).await;
        let start = performance.now();
        for _ in 0..100 {
            trainer_gpu.train_batch(images, labels);
        }
        let gpu_time = performance.now() - start;

        console.log(&format!("Batch {}: CPU={}ms, GPU={}ms, Speedup={:.2}x",
            batch_size, cpu_time, gpu_time, cpu_time / gpu_time));
    }
}
```

**Metrics to Track**:
- Training throughput (samples/sec)
- Memory usage (MB)
- Latency (ms per batch)
- Model size limits

**Target Performance**:
- **10x faster** than CPU backend for batch_size=32
- **50x faster** than CPU backend for batch_size=128
- **Memory efficient**: Train on 10MB GPU memory budget

#### 3.3 Browser Demo Integration (2 days)

**Update** `examples/web-gpu-demo/`:

```javascript
// Add new tab: "Training Demo"
import init, { BrowserTrainer } from '../../libtorch-wasm-trainer/pkg';

async function initTrainingDemo() {
    await init();

    // Check WebGPU support
    if (!navigator.gpu) {
        showWarning("WebGPU not supported. Using CPU backend (slower).");
        trainer = new BrowserTrainer.new_cpu(0.001);
    } else {
        showSuccess("WebGPU detected! Training will use GPU acceleration.");
        trainer = await BrowserTrainer.new_webgpu(0.001);
    }

    // Load MNIST subset (1000 samples for demo)
    const mnistData = await fetch('/data/mnist_subset.json').then(r => r.json());

    // Training UI
    const trainButton = document.getElementById('train-button');
    trainButton.onclick = async () => {
        for (let epoch = 0; epoch < 5; epoch++) {
            for (const batch of mnistData) {
                const loss = trainer.train_batch(batch.images, batch.labels);
                updateLossChart(loss);
            }

            // Evaluate
            const [loss, accuracy] = trainer.eval_batch(testImages, testLabels);
            updateAccuracyChart(accuracy);
        }

        // Export weights
        const weights = trainer.export_pytorch_weights();
        downloadJSON(weights, 'trained_model.json');
    };
}
```

**New Demo Features**:
- Real-time loss/accuracy charts
- GPU vs CPU performance comparison
- Weight visualization
- Export trained model button

#### 3.4 Memory Optimization (2 days)

**Challenge**: Browser GPU memory limits (~256MB typical)

**Strategies**:
1. **Gradient Checkpointing**: Recompute activations during backward pass
2. **Mixed Precision**: Use FP16 for forward pass, FP32 for weights
3. **Batch Size Tuning**: Automatically detect optimal batch size
4. **Model Sharding**: Split large models across CPU/GPU

**Implementation**:
```rust
#[wasm_bindgen]
pub struct TrainingConfig {
    learning_rate: f64,
    batch_size: usize,
    use_mixed_precision: bool,
    gradient_checkpointing: bool,
}

impl BrowserTrainer {
    /// Auto-tune batch size for available GPU memory
    pub async fn auto_tune_batch_size(&mut self) -> usize {
        let mut batch_size = 128;
        loop {
            match self.try_train_batch(batch_size).await {
                Ok(_) => return batch_size,
                Err(OutOfMemory) => {
                    batch_size /= 2;
                    if batch_size < 8 {
                        panic!("Insufficient GPU memory");
                    }
                }
            }
        }
    }
}
```

### Milestone 3 Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| WebGPU not widely supported yet | High | Provide CPU fallback, document browser requirements |
| Burn's WGPU backend has bugs | High | Test extensively, report issues upstream, have workarounds |
| GPU memory constraints | Medium | Implement auto-tuning, provide memory usage warnings |
| Performance doesn't meet expectations | Medium | Profile, optimize hot paths, consider alternative backends |
| WASM binary size too large | Low | Use wasm-opt, tree-shaking, code splitting |

### Milestone 3 Success Criteria

- [ ] WebGPU backend compiles and runs in Chrome/Firefox/Safari
- [ ] Training is 10x faster than CPU backend
- [ ] Automatic fallback to CPU when WebGPU unavailable
- [ ] Browser demo shows real-time training with visualizations
- [ ] Memory usage stays under 256MB
- [ ] Benchmark results published (CPU vs GPU comparison)
- [ ] Documentation updated with browser compatibility matrix

**Estimated Effort**: 2-3 weeks (1 developer)

---

## Milestone 4: Advanced Features

**Timeline**: Ongoing (3-6 months)
**Status**: Not Started
**Goal**: Production-ready features for real-world applications

### Why This Matters

Milestones 1-3 provide a **working prototype**. Milestone 4 adds features needed for **production use**:
- Support diverse model architectures (CNNs, Transformers)
- Multiple training strategies (optimizers, schedulers)
- Robustness (checkpointing, error handling)
- Usability (better APIs, debugging tools)

### 4.1 Advanced Model Architectures (3-4 weeks)

#### Conv2D Support
**Use Case**: Image classification beyond MLP

```rust
// model.rs - Add CNN support
#[derive(Module, Debug)]
pub struct MnistCNN<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    pool: AdaptiveAvgPool2d,
    fc1: Linear<B>,
    fc2: Linear<B>,
}

impl<B: Backend> MnistCNN<B> {
    pub fn new(device: &B::Device) -> Self {
        let conv1 = Conv2dConfig::new([1, 32], [3, 3]).init(device);
        let conv2 = Conv2dConfig::new([32, 64], [3, 3]).init(device);
        let pool = AdaptiveAvgPool2dConfig::new([7, 7]).init();
        let fc1 = LinearConfig::new(64 * 7 * 7, 128).init(device);
        let fc2 = LinearConfig::new(128, 10).init(device);

        Self { conv1, conv2, pool, fc1, fc2 }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 2> {
        // input: [batch, 1, 28, 28]
        let x = self.conv1.forward(input);
        let x = activation::relu(x);
        let x = self.conv2.forward(x);
        let x = activation::relu(x);
        let x = self.pool.forward(x);
        let x = x.flatten(1, 3); // [batch, 64*7*7]
        let x = self.fc1.forward(x);
        let x = activation::relu(x);
        self.fc2.forward(x)
    }
}
```

**Tasks**:
- [ ] Implement Conv2D model
- [ ] Add BatchNorm support
- [ ] Add Dropout for regularization
- [ ] Test on CIFAR-10
- [ ] Export/import weights to PyTorch

#### Transformer Support (Advanced)
**Use Case**: Text classification, sequence modeling

```rust
#[derive(Module, Debug)]
pub struct TransformerClassifier<B: Backend> {
    embedding: Embedding<B>,
    transformer: TransformerEncoder<B>,
    fc: Linear<B>,
}
```

**Challenges**:
- Attention mechanism implementation
- Positional encodings
- Memory requirements (quadratic in sequence length)

**Timeline**: 2-3 weeks

### 4.2 Multiple Optimizers (1 week)

**Current**: Only Adam optimizer

**Target**: Support all major optimizers

```rust
#[wasm_bindgen]
pub enum OptimizerType {
    Adam,
    AdamW,
    SGD,
    RMSprop,
}

#[wasm_bindgen]
impl BrowserTrainer {
    #[wasm_bindgen(constructor)]
    pub fn new_with_optimizer(
        learning_rate: f64,
        optimizer: OptimizerType
    ) -> Self {
        let optim = match optimizer {
            OptimizerType::Adam => AdamConfig::new().init(),
            OptimizerType::AdamW => AdamWConfig::new().init(),
            OptimizerType::SGD => SgdConfig::new().init(),
            OptimizerType::RMSprop => RmspropConfig::new().init(),
        };
        // ... rest of initialization
    }
}
```

**Tasks**:
- [ ] Integrate SGD, AdamW, RMSprop
- [ ] Add momentum, weight decay options
- [ ] Benchmark convergence speed
- [ ] Document optimizer selection guide

### 4.3 Learning Rate Scheduling (1 week)

**Importance**: Critical for training convergence

```rust
pub trait LRScheduler {
    fn step(&mut self, epoch: usize) -> f64;
}

pub struct CosineAnnealingLR {
    initial_lr: f64,
    min_lr: f64,
    total_epochs: usize,
}

impl LRScheduler for CosineAnnealingLR {
    fn step(&mut self, epoch: usize) -> f64 {
        let progress = epoch as f64 / self.total_epochs as f64;
        self.min_lr + (self.initial_lr - self.min_lr) *
            (1.0 + (progress * std::f64::consts::PI).cos()) / 2.0
    }
}

#[wasm_bindgen]
impl BrowserTrainer {
    #[wasm_bindgen]
    pub fn train_batch_with_scheduler(&mut self,
        images: Vec<f32>,
        labels: Vec<usize>,
        epoch: usize
    ) -> f32 {
        let lr = self.scheduler.step(epoch);
        self.trainer.config.learning_rate = lr;
        self.trainer.train_batch(&images, &labels)
    }
}
```

**Schedulers to Implement**:
- [x] Constant (current)
- [ ] Step decay
- [ ] Exponential decay
- [ ] Cosine annealing
- [ ] One-cycle

### 4.4 Model Checkpointing (1 week)

**Use Case**: Resume training after browser refresh

```rust
#[wasm_bindgen]
impl BrowserTrainer {
    /// Save checkpoint to browser storage
    #[wasm_bindgen]
    pub fn save_checkpoint(&self, checkpoint_name: &str) -> Result<(), JsValue> {
        let checkpoint = Checkpoint {
            model_weights: self.trainer.export_weights(),
            optimizer_state: self.trainer.optim.state(),
            epoch: self.current_epoch,
            loss_history: self.loss_history.clone(),
        };

        let json = serde_json::to_string(&checkpoint)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        // Save to localStorage
        let window = web_sys::window().unwrap();
        let storage = window.local_storage().unwrap().unwrap();
        storage.set_item(checkpoint_name, &json)
            .map_err(|e| JsValue::from_str("Storage error"))?;

        Ok(())
    }

    /// Load checkpoint from browser storage
    #[wasm_bindgen]
    pub fn load_checkpoint(checkpoint_name: &str) -> Result<BrowserTrainer, JsValue> {
        let window = web_sys::window().unwrap();
        let storage = window.local_storage().unwrap().unwrap();
        let json = storage.get_item(checkpoint_name)
            .map_err(|_| JsValue::from_str("Storage error"))?
            .ok_or_else(|| JsValue::from_str("Checkpoint not found"))?;

        let checkpoint: Checkpoint = serde_json::from_str(&json)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        // Restore trainer state
        let mut trainer = Self::from_pytorch_weights(&checkpoint.model_weights)?;
        trainer.current_epoch = checkpoint.epoch;
        trainer.loss_history = checkpoint.loss_history;
        // TODO: Restore optimizer state

        Ok(trainer)
    }
}
```

**Features**:
- Save/load from localStorage
- Export/import as files
- Automatic checkpointing every N epochs
- Resume training from checkpoint

### 4.5 Data Augmentation (1 week)

**Use Case**: Improve model generalization

```rust
pub trait Augmentation {
    fn apply(&self, image: &[f32]) -> Vec<f32>;
}

pub struct RandomRotation {
    max_degrees: f32,
}

impl Augmentation for RandomRotation {
    fn apply(&self, image: &[f32]) -> Vec<f32> {
        // Rotate image by random angle in [-max_degrees, max_degrees]
        // TODO: Implement efficient rotation
    }
}

#[wasm_bindgen]
impl BrowserTrainer {
    #[wasm_bindgen]
    pub fn train_batch_with_augmentation(
        &mut self,
        images: Vec<f32>,
        labels: Vec<usize>,
    ) -> f32 {
        // Apply augmentations
        let augmented_images = self.augmentations
            .iter()
            .fold(images, |imgs, aug| aug.apply(&imgs));

        self.trainer.train_batch(&augmented_images, &labels)
    }
}
```

**Augmentations to Implement**:
- [ ] Random horizontal flip
- [ ] Random rotation
- [ ] Random crop
- [ ] Color jittering
- [ ] Mixup / CutMix (advanced)

### 4.6 Mixed Precision Training (1 week)

**Benefit**: 2x faster training, 2x less memory

```rust
#[wasm_bindgen]
pub struct TrainingConfig {
    learning_rate: f64,
    batch_size: usize,
    use_fp16: bool, // NEW
}

impl<B: AutodiffBackend> MnistTrainer<B, O> {
    pub fn train_batch_fp16(&mut self, images: &[f32], labels: &[usize]) -> f32 {
        // Convert inputs to FP16
        let images_fp16 = images.iter().map(|&x| half::f16::from_f32(x)).collect();

        // Forward pass in FP16
        let output_fp16 = self.model.forward_fp16(images_fp16);

        // Loss computation in FP32 (stability)
        let output_fp32 = output_fp16.cast::<f32>();
        let loss = self.loss_fn.forward(output_fp32, labels);

        // Backward pass
        let grads = loss.backward();

        // Update in FP32
        // ...
    }
}
```

**Challenges**:
- Numerical stability (loss scaling)
- Not all operations support FP16
- Browser WebGPU FP16 support varies

### 4.7 Better Error Handling & Debugging (1 week)

**Current State**: Panics crash the browser tab

**Target State**: Graceful error handling with debugging tools

```rust
#[wasm_bindgen]
pub enum TrainingError {
    OutOfMemory,
    InvalidInput,
    NumericalInstability,
    BackendError,
}

#[wasm_bindgen]
impl BrowserTrainer {
    #[wasm_bindgen]
    pub fn train_batch_safe(
        &mut self,
        images: Vec<f32>,
        labels: Vec<usize>
    ) -> Result<f32, TrainingError> {
        // Validate inputs
        if images.len() != labels.len() * 784 {
            return Err(TrainingError::InvalidInput);
        }

        // Try training with error recovery
        match self.trainer.train_batch(&images, &labels) {
            Ok(loss) if loss.is_nan() => {
                // Numerical instability detected
                self.reduce_learning_rate();
                Err(TrainingError::NumericalInstability)
            }
            Ok(loss) => Ok(loss),
            Err(e) => Err(TrainingError::BackendError),
        }
    }
}
```

**Debugging Tools**:
- Gradient visualization
- Activation distribution plots
- Weight histogram tracking
- Loss spike detection

### 4.8 Production Deployment Features (2 weeks)

**Real-world requirements**:

1. **Federated Learning Support**
```rust
#[wasm_bindgen]
impl BrowserTrainer {
    /// Compute model delta (for federated aggregation)
    #[wasm_bindgen]
    pub fn compute_delta(&self, initial_weights: &str) -> String {
        let initial: PyTorchWeights = serde_json::from_str(initial_weights).unwrap();
        let current = self.trainer.export_weights();

        let delta = PyTorchWeights {
            fc1_weight: current.fc1_weight.iter().zip(&initial.fc1_weight)
                .map(|(c, i)| c - i).collect(),
            // ... compute delta for all params
        };

        serde_json::to_string(&delta).unwrap()
    }
}
```

2. **Differential Privacy**
```rust
pub struct DPConfig {
    epsilon: f64,  // Privacy budget
    delta: f64,
    clip_norm: f64,
}

impl<B, O> MnistTrainer<B, O> {
    pub fn train_batch_dp(&mut self, images: &[f32], labels: &[usize], dp_config: &DPConfig) -> f32 {
        // Clip gradients
        let grads = self.compute_gradients(images, labels);
        let clipped_grads = clip_gradients(grads, dp_config.clip_norm);

        // Add noise
        let noisy_grads = add_gaussian_noise(clipped_grads, dp_config);

        // Update
        self.model = self.optim.step(self.config.learning_rate, self.model, noisy_grads);
    }
}
```

3. **Model Compression**
```rust
#[wasm_bindgen]
impl BrowserTrainer {
    /// Export quantized model (INT8)
    #[wasm_bindgen]
    pub fn export_quantized(&self) -> String {
        let weights = self.trainer.export_weights();

        // Quantize to INT8
        let quantized = quantize_weights(weights, 8);

        serde_json::to_string(&quantized).unwrap()
    }
}
```

### Milestone 4 Success Criteria

**Core Features**:
- [ ] CNN architecture support (Conv2D, BatchNorm, Dropout)
- [ ] 4+ optimizers (Adam, AdamW, SGD, RMSprop)
- [ ] 3+ LR schedulers (Step, Exponential, Cosine)
- [ ] Checkpointing with localStorage
- [ ] 5+ data augmentations
- [ ] Mixed precision training (FP16)

**Production Features**:
- [ ] Graceful error handling (no crashes)
- [ ] Debugging tools (grad viz, activation plots)
- [ ] Federated learning support (delta computation)
- [ ] Differential privacy (DP-SGD)
- [ ] Model compression (quantization)

**Documentation**:
- [ ] API reference for all features
- [ ] Tutorial for each major feature
- [ ] Production deployment guide
- [ ] Benchmark suite results

**Estimated Effort**: 3-6 months (1-2 developers)

---

## Cross-Cutting Concerns

### Testing Strategy

**Unit Tests** (Rust):
```bash
cargo test
```

**Integration Tests** (WASM):
```bash
wasm-pack test --headless --chrome
```

**End-to-End Tests** (Browser):
```javascript
// Playwright tests
test('Complete training workflow', async ({ page }) => {
    await page.goto('/demo');
    await page.click('#train-button');
    await page.waitForSelector('.training-complete');
    const accuracy = await page.textContent('.accuracy');
    expect(parseFloat(accuracy)).toBeGreaterThan(0.95);
});
```

**Gradient Tests** (Comparison with PyTorch):
```bash
python tests/test_gradients.py
```

### CI/CD Pipeline

```yaml
# .github/workflows/wasm-trainer.yml
name: WASM Trainer CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Install Rust
        uses: actions-rs/toolchain@v1
      - name: Install wasm-pack
        run: curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
      - name: Run tests
        run: cd libtorch-wasm-trainer && cargo test
      - name: Build WASM
        run: cd libtorch-wasm-trainer && wasm-pack build --target web
      - name: Run WASM tests
        run: wasm-pack test --headless --chrome

  python-roundtrip:
    runs-on: ubuntu-latest
    steps:
      - name: Test PyTorch compatibility
        run: python tests/test_compatibility.py

  benchmark:
    runs-on: ubuntu-latest
    steps:
      - name: Run benchmarks
        run: cargo bench --features benchmark
```

### Documentation Plan

**User Documentation**:
- Getting Started Guide
- API Reference (Rust + JavaScript)
- Tutorial Series:
  - "Your First Browser Training"
  - "PyTorch Round-Trip"
  - "GPU-Accelerated Training"
  - "Production Deployment"

**Developer Documentation**:
- Architecture Overview
- Contributing Guide
- Burn Framework Integration Guide
- WebGPU Backend Details

**Research Documentation**:
- Gradient Compatibility Analysis
- Performance Benchmarks
- Privacy Analysis (Differential Privacy)
- Federated Learning Case Studies

### Community & Ecosystem

**Blog Posts**:
1. "Introducing libtorch-wasm-trainer" (launch announcement)
2. "Privacy-Preserving ML in Your Browser" (use cases)
3. "WebGPU Performance: CPU vs GPU Training" (benchmarks)
4. "Building Federated Learning with Rust + WASM" (advanced)

**Conference Talks** (potential):
- RustConf: "High-Performance ML in the Browser"
- PyTorch Conference: "Browser Training with PyTorch Compatibility"
- WebGPU Summit: "Training Neural Networks with WebGPU"

**Integration Examples**:
- React + WASM Trainer starter template
- Vue.js demo application
- Svelte federated learning example
- Node.js aggregation server (for federated learning)

---

## Timeline Summary

| Milestone | Duration | Cumulative | Key Deliverable |
|-----------|----------|-----------|-----------------|
| ✅ Milestone 1 | 2 weeks | 2 weeks | Browser training MVP with PyTorch export |
| Milestone 2 | 1-2 weeks | 4 weeks | Complete PyTorch round-trip verification |
| Milestone 3 | 2-3 weeks | 7 weeks | GPU-accelerated training with WebGPU |
| Milestone 4 | 3-6 months | 7 months | Production-ready with advanced features |

**Total Time to Production**: ~7 months (1 developer) or ~3.5 months (2 developers)

---

## Resource Requirements

**Development**:
- 1-2 Rust developers (intermediate to advanced)
- 1 Python developer (for PyTorch integration)
- 1 Frontend developer (for browser demo)

**Hardware**:
- Development machines with GPU (for testing WebGPU)
- Browser compatibility testing lab (Chrome, Firefox, Safari)
- CI/CD infrastructure

**External Dependencies**:
- Burn framework (active development, API may change)
- WebGPU spec (still evolving)
- Browser support (Chrome stable, Firefox/Safari experimental)

---

## Success Metrics

**Technical Metrics**:
- ✅ Gradient accuracy: < 1e-4 difference vs PyTorch
- ✅ GPU speedup: > 10x vs CPU
- ✅ Model accuracy: Match PyTorch baseline
- ✅ Memory efficiency: Train on 256MB GPU budget
- ✅ WASM size: < 5MB compressed

**Adoption Metrics**:
- GitHub stars: 500+ (indicates interest)
- npm downloads: 1000+/month (indicates usage)
- Tutorial completions: 100+ (indicates learning)
- Production deployments: 10+ companies

**Community Metrics**:
- Contributors: 10+ active
- Issues resolved: 90%+ within 1 week
- Documentation coverage: 100% of public APIs
- Blog post views: 10,000+

---

## Risk Assessment

### High-Risk Items

1. **Burn API Stability**
   - **Risk**: Breaking changes in Burn 0.20+
   - **Mitigation**: Pin versions, maintain compatibility layer
   - **Probability**: High (Burn is pre-1.0)

2. **WebGPU Browser Support**
   - **Risk**: Limited browser support delays adoption
   - **Mitigation**: CPU fallback, clear documentation
   - **Probability**: Medium (Chrome stable, others experimental)

3. **Gradient Compatibility**
   - **Risk**: Numerical differences break PyTorch round-trip
   - **Mitigation**: Extensive testing, tolerance thresholds
   - **Probability**: Medium

### Medium-Risk Items

4. **Performance Expectations**
   - **Risk**: WebGPU slower than expected
   - **Mitigation**: Profiling, optimization, set realistic expectations
   - **Probability**: Low

5. **WASM Bundle Size**
   - **Risk**: Binary too large for practical use (>10MB)
   - **Mitigation**: Lazy loading, code splitting, wasm-opt
   - **Probability**: Medium

### Low-Risk Items

6. **Memory Constraints**
   - **Risk**: Browser OOM on large models
   - **Mitigation**: Batch size tuning, streaming
   - **Probability**: Low (we control model size)

---

## Conclusion

This roadmap transforms libtorch-wasm-trainer from a **working prototype** (Milestone 1 ✅) to a **production-ready system** for privacy-preserving machine learning.

**Key Differentiators**:
1. **PyTorch Compatibility** - Unique in browser-based training
2. **WebGPU Performance** - Competitive with native training
3. **Privacy-First** - All training client-side
4. **Production-Ready** - Not just a toy demo

**Next Immediate Steps**:
1. ✅ Compile Milestone 1 to WASM
2. ✅ Integrate with browser demo
3. → Start Milestone 2: Python utility for weight loading

**Long-Term Vision**:
Enable a new paradigm of **privacy-preserving federated learning** where:
- Users train on their data without uploading it
- Models aggregate improvements across millions of devices
- PyTorch ecosystem provides production deployment
- Cross-platform GPU acceleration makes it practical

---

**Questions? Feedback?**

Open an issue on GitHub or reach out to the maintainers.

**Let's build the future of privacy-preserving ML! 🚀**
