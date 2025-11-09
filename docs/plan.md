# WebGPU Backpropagation Implementation Plan

## Executive Summary

This plan details the implementation of GPU-accelerated training for libtorch-rust by building a complete autograd system with WebGPU compute shaders. The goal is to enable practical browser-based training, not just CPU toy demos.

**Current State:**
- ✅ WebGPU forward pass (inference only)
- ✅ CPU tensor operations
- ❌ No autograd/gradient tracking
- ❌ No backward pass
- ❌ Optimizer operations are stubs (see `nn/optimizer.rs:51-119`)

**Target State:**
- ✅ Full autograd tape system
- ✅ WebGPU backward pass shaders
- ✅ GPU-accelerated optimizer operations
- ✅ End-to-end training in browser with WASM

**Scope:** Port all PyTorch training tests and implement corresponding functionality

---

## Phase 1: Autograd Foundation (CPU)

**Goal:** Build gradient tracking and computational graph infrastructure

### 1.1 Gradient Tensor Storage

**Tests to Port:**
- `test/cpp/api/autograd.cpp::AutogradTest.SimpleGrad`
- `test/cpp/api/autograd.cpp::AutogradTest.GradientStorage`

**Implementation:**
- [ ] `libtorch-rust-sys/src/autograd/tensor_grad.rs`
  - Add `grad` field to TensorImpl
  - Add `requires_grad` flag
  - Implement `set_requires_grad()` and `requires_grad()`
  - Add gradient accumulation logic

**Validation:**
- Tensor can store gradient
- `requires_grad` flag propagates correctly
- Gradients accumulate on repeated backward passes

### 1.2 Computational Graph (Tape-based)

**Tests to Port:**
- `test/cpp/api/autograd.cpp::AutogradTest.SimpleTape`
- `test/cpp/api/autograd.cpp::AutogradTest.EdgeCases`

**Implementation:**
- [ ] `libtorch-rust-sys/src/autograd/tape.rs`
  - Define `GradNode` trait for operations
  - Define `Edge` struct connecting tensors to nodes
  - Implement topological sort for backward traversal
  - Thread-local tape storage (for gradient context)

- [ ] `libtorch-rust-sys/src/autograd/context.rs`
  - `GradMode::is_enabled()` - check if recording
  - `NoGradGuard` - disable gradient tracking
  - `set_grad_enabled(bool)` - global control

**Validation:**
- Operations record edges when `requires_grad=true`
- Topological sort produces correct backward order
- `no_grad` context prevents recording

### 1.3 Backward Pass Infrastructure

**Tests to Port:**
- `test/cpp/api/autograd.cpp::AutogradTest.Backward`
- `test/cpp/api/autograd.cpp::AutogradTest.BackwardWithGradient`

**Implementation:**
- [ ] `libtorch-rust-sys/src/autograd/backward.rs`
  - `Tensor::backward()` - trigger gradient computation
  - `Tensor::backward_with_gradient(grad)` - custom output gradient
  - Implement reverse-mode traversal
  - Call `GradNode::apply()` for each operation

**Validation:**
- Simple chain rule: `z = x * y`, compute `dz/dx` and `dz/dy`
- Multi-path gradients accumulate correctly
- Error on backward without `requires_grad`

---

## Phase 2: Backward Operations (CPU)

**Goal:** Implement gradient formulas for all operations (CPU first, GPU later)

### 2.1 Basic Arithmetic Gradients

**Tests to Port:**
- `test/cpp/api/autograd.cpp::AutogradTest.AddBackward`
- `test/cpp/api/autograd.cpp::AutogradTest.SubBackward`
- `test/cpp/api/autograd.cpp::AutogradTest.MulBackward`
- `test/cpp/api/autograd.cpp::AutogradTest.DivBackward`

**Implementation:**
- [ ] `libtorch-rust-sys/src/autograd/ops/arithmetic.rs`
  - `AddBackward` node: broadcasts gradient to input shapes
  - `SubBackward` node: negates gradient for subtrahend
  - `MulBackward` node: `grad * other_input`
  - `DivBackward` node: implements quotient rule

**Gradient Formulas:**
```
Add: d(x+y)/dx = 1,  d(x+y)/dy = 1
Sub: d(x-y)/dx = 1,  d(x-y)/dy = -1
Mul: d(x*y)/dx = y,  d(x*y)/dy = x
Div: d(x/y)/dx = 1/y, d(x/y)/dy = -x/y²
```

**Validation:**
- Numerical gradient check (finite differences)
- Broadcasting: `[3,1] + [3,5]` gradients sum correctly
- Second-order gradients (optional for Phase 2)

### 2.2 Matrix Operations Gradients

**Tests to Port:**
- `test/cpp/api/autograd.cpp::AutogradTest.MatmulBackward`
- `test/cpp/api/autograd.cpp::AutogradTest.TransposeBackward`
- `test/cpp/api/autograd.cpp::AutogradTest.ReshapeBackward`

**Implementation:**
- [ ] `libtorch-rust-sys/src/autograd/ops/linalg.rs`
  - `MatmulBackward`: matrix multiply with transposed inputs
  - `TransposeBackward`: transpose gradient
  - `ReshapeBackward`: reshape gradient to input shape

**Gradient Formulas:**
```
Matmul C = A @ B:
  dL/dA = dL/dC @ B^T
  dL/dB = A^T @ dL/dC

Transpose: dL/dX = (dL/dY)^T
Reshape: dL/dX = reshape(dL/dY, X.shape)
```

**Validation:**
- Matrix chain rule: `z = (A @ B) @ C`
- Batch matmul: `[B, M, K] @ [B, K, N]`
- In-place transpose gradients

### 2.3 Activation Function Gradients

**Tests to Port:**
- `test/cpp/api/autograd.cpp::AutogradTest.ReluBackward`
- `test/cpp/api/autograd.cpp::AutogradTest.SigmoidBackward`
- `test/cpp/api/autograd.cpp::AutogradTest.TanhBackward`
- `test/cpp/api/autograd.cpp::AutogradTest.SoftmaxBackward`

**Implementation:**
- [ ] `libtorch-rust-sys/src/autograd/ops/activation.rs`
  - `ReluBackward`: gradient = input > 0 ? grad : 0
  - `SigmoidBackward`: gradient = grad * sigmoid(x) * (1 - sigmoid(x))
  - `TanhBackward`: gradient = grad * (1 - tanh²(x))
  - `SoftmaxBackward`: Jacobian-vector product

**Gradient Formulas:**
```
ReLU: d/dx = 1 if x > 0 else 0
Sigmoid: d/dx = σ(x) * (1 - σ(x))
Tanh: d/dx = 1 - tanh²(x)
Softmax: Jacobian J[i,j] = softmax[i] * (δ[i,j] - softmax[j])
```

**Validation:**
- ReLU gradient is zero for negative inputs
- Sigmoid gradient matches numerical
- Softmax: sum of output gradients = 0 (probability constraint)

### 2.4 Reduction Operations Gradients

**Tests to Port:**
- `test/cpp/api/autograd.cpp::AutogradTest.SumBackward`
- `test/cpp/api/autograd.cpp::AutogradTest.MeanBackward`
- `test/cpp/api/autograd.cpp::AutogradTest.MaxBackward`

**Implementation:**
- [ ] `libtorch-rust-sys/src/autograd/ops/reduction.rs`
  - `SumBackward`: broadcast gradient to input shape
  - `MeanBackward`: broadcast gradient / num_elements
  - `MaxBackward`: gradient to max element only (argmax mask)

**Validation:**
- Sum over all dims: gradient broadcasts to original shape
- Mean: gradient scaled by 1/N
- Max: only one element receives gradient

---

## Phase 3: WebGPU Backward Shaders

**Goal:** Implement GPU-accelerated gradient computation

### 3.1 GPU Kernel Infrastructure

**Implementation:**
- [ ] `libtorch-rust-sys/src/gpu/kernels/backward/mod.rs`
  - Shader template system for gradient ops
  - Workgroup size optimization (256 threads)
  - Shared memory for reductions

### 3.2 Matrix Multiplication Backward

**Tests to Port:**
- `test/cpp/api/autograd.cpp::AutogradTest.MatmulBackwardGPU`

**Implementation:**
- [ ] `libtorch-rust-sys/src/gpu/kernels/backward/matmul.wgsl`
  ```wgsl
  // Compute dL/dA = dL/dC @ B^T
  @group(0) @binding(0) var<storage, read> grad_output: array<f32>;
  @group(0) @binding(1) var<storage, read> B: array<f32>;
  @group(0) @binding(2) var<storage, read_write> grad_A: array<f32>;

  @compute @workgroup_size(16, 16)
  fn matmul_backward_A(@builtin(global_invocation_id) global_id: vec3<u32>) {
      // Implement: grad_A[i,k] = sum_j(grad_output[i,j] * B[j,k])
  }
  ```

**Validation:**
- Matches CPU implementation exactly
- Performance: >100 GFLOPS on test matrix
- Batched operations work

### 3.3 Activation Function Backward Shaders

**Implementation:**
- [ ] `libtorch-rust-sys/src/gpu/kernels/backward/relu.wgsl`
- [ ] `libtorch-rust-sys/src/gpu/kernels/backward/sigmoid.wgsl`
- [ ] `libtorch-rust-sys/src/gpu/kernels/backward/softmax.wgsl`

**Validation:**
- Bit-exact match with CPU (within floating point tolerance)
- Benchmark: <1ms for 1M element tensor

### 3.4 Element-wise Operation Shaders

**Implementation:**
- [ ] `libtorch-rust-sys/src/gpu/kernels/backward/elementwise.wgsl`
  - Single shader with operation mode parameter
  - Fused multiply-add for efficiency
  - Vectorized loads (vec4<f32>)

**Validation:**
- Broadcasting handled correctly
- Performance: memory bandwidth bound

---

## Phase 4: Loss Functions

**Goal:** Implement differentiable loss functions

### 4.1 Classification Losses

**Tests to Port:**
- `test/cpp/api/loss.cpp::LossTest.CrossEntropy`
- `test/cpp/api/loss.cpp::LossTest.NLLLoss`

**Implementation:**
- [ ] `libtorch-rust/src/nn/loss.rs`
  ```rust
  pub fn cross_entropy(logits: &Tensor, targets: &Tensor) -> Tensor {
      // Numerically stable: -log(exp(x[target]) / sum(exp(x)))
      // = log(sum(exp(x))) - x[target]
  }
  ```

- [ ] `libtorch-rust-sys/src/gpu/kernels/loss/cross_entropy.wgsl`

**Validation:**
- Numerically stable for large logits
- Gradient sums to 1.0 (probability simplex)
- Reduction modes: mean, sum, none

### 4.2 Regression Losses

**Tests to Port:**
- `test/cpp/api/loss.cpp::LossTest.MSE`
- `test/cpp/api/loss.cpp::LossTest.L1Loss`

**Implementation:**
- [ ] MSE: `(pred - target)²`
- [ ] L1: `|pred - target|`
- [ ] Huber: smooth L1 loss

**Validation:**
- Gradient magnitude tests
- Reduction consistency

---

## Phase 5: Optimizers

**Goal:** Implement GPU-accelerated weight update algorithms

### 5.1 SGD Optimizer

**Tests to Port:**
- `test/cpp/api/optim.cpp::OptimTest.SGD`
- `test/cpp/api/optim.cpp::OptimTest.SGDMomentum`

**Implementation:**
- [ ] `libtorch-rust/src/nn/optim/sgd.rs`
  ```rust
  impl Optimizer for SGD {
      fn step(&mut self) {
          for param in &self.params {
              // param -= lr * grad
              // With momentum: velocity = momentum * velocity + grad
              //                param -= lr * velocity
          }
      }
  }
  ```

- [ ] `libtorch-rust-sys/src/gpu/kernels/optim/sgd.wgsl`
  ```wgsl
  @compute @workgroup_size(256)
  fn sgd_step(
      @builtin(global_invocation_id) global_id: vec3<u32>,
      @group(0) @binding(0) var<storage, read_write> params: array<f32>,
      @group(0) @binding(1) var<storage, read> grads: array<f32>,
      @group(0) @binding(2) var<uniform> lr: f32,
  ) {
      let idx = global_id.x;
      params[idx] -= lr * grads[idx];
  }
  ```

**Validation:**
- Converges on simple quadratic
- Momentum accumulates correctly
- Weight decay applied properly

### 5.2 Adam Optimizer

**Tests to Port:**
- `test/cpp/api/optim.cpp::OptimTest.Adam`
- `test/cpp/api/optim.cpp::OptimTest.AdamW`

**Implementation:**
- [ ] `libtorch-rust/src/nn/optim/adam.rs`
  ```rust
  // m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
  // v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
  // m̂_t = m_t / (1 - β₁^t)
  // v̂_t = v_t / (1 - β₂^t)
  // θ_t = θ_{t-1} - α * m̂_t / (√v̂_t + ε)
  ```

- [ ] `libtorch-rust-sys/src/gpu/kernels/optim/adam.wgsl`

**Validation:**
- Bias correction at early steps
- Converges faster than SGD on noisy gradients
- AdamW decoupled weight decay

### 5.3 Learning Rate Schedulers

**Tests to Port:**
- `test/cpp/api/optim.cpp::OptimTest.LRScheduler`

**Implementation:**
- [ ] Step decay
- [ ] Exponential decay
- [ ] Cosine annealing
- [ ] One-cycle policy

---

## Phase 6: Integration Tests

**Goal:** End-to-end training validation

### 6.1 Simple Network Training

**Tests to Port:**
- `test/cpp/api/integration.cpp::IntegrationTest.SimpleNet`

**Implementation:**
- [ ] `libtorch-rust/tests/training_tests.rs::test_simple_linear_regression`
  ```rust
  // y = 2x + 3 with noise
  // Train linear model to recover weights
  #[test]
  fn test_simple_linear_regression() {
      let vs = VarStore::new(Device::Cpu);
      let linear = nn::linear(&vs.root(), 1, 1);
      let opt = nn::Adam::new(0.01).build(&vs);

      for epoch in 0..100 {
          let x = Tensor::randn(&[32, 1], ...);
          let y = &x * 2.0 + 3.0; // true function

          let pred = linear.forward(&x);
          let loss = (&pred - &y).pow(2.0).mean(Kind::Float);

          opt.zero_grad();
          loss.backward();
          opt.step();
      }

      // Check learned weight ≈ 2.0, bias ≈ 3.0
  }
  ```

**Validation:**
- Loss decreases monotonically
- Learned parameters match true values
- GPU and CPU results match

### 6.2 MNIST MLP Training

**Tests to Port:**
- `test/cpp/api/integration.cpp::IntegrationTest.MNIST`

**Implementation:**
- [ ] `libtorch-rust/tests/training_tests.rs::test_mnist_mlp`
  - 784 → 128 → 10 network
  - ReLU activations
  - Cross-entropy loss
  - Achieves >95% test accuracy

**Validation:**
- Training loss decreases
- Test accuracy improves
- Gradients don't vanish/explode

### 6.3 GPU vs CPU Parity

**Implementation:**
- [ ] `libtorch-rust/tests/training_tests.rs::test_gpu_cpu_parity`
  - Run identical training on CPU and GPU
  - Compare loss curves
  - Assert numerical equivalence (within tolerance)

**Validation:**
- Loss difference <1e-5 per step
- Final weights match within 1e-4

---

## Phase 7: WASM Integration

**Goal:** Enable browser training with WebGPU

### 7.1 WASM-Compatible Autograd

**Implementation:**
- [ ] Ensure no threading (WASM single-threaded)
- [ ] Replace `std::sync` with WASM-safe alternatives
- [ ] Test in browser console

### 7.2 Rewrite libtorch-wasm-trainer

**Implementation:**
- [ ] `libtorch-wasm-trainer/src/lib.rs`
  - **Remove all Burn dependencies**
  - Use `libtorch-rust` with WebGPU backend
  - Expose same WASM API

**Validation:**
- Browser training demo works
- Performance: >100 samples/sec on M1/M2 MacBook
- Weights are PyTorch-compatible

### 7.3 Browser Training Demo

**Implementation:**
- [ ] Update `TrainingDemo.tsx` to show real GPU usage
- [ ] Display GPU utilization metrics
- [ ] Compare CPU vs GPU training speed

---

## Phase 8: Performance Optimization

**Goal:** Achieve production-grade performance

### 8.1 Kernel Fusion

**Implementation:**
- [ ] Fuse activation + bias add
- [ ] Fuse matmul + activation
- [ ] Fuse gradient computation chains

**Target:** 2x speedup on common patterns

### 8.2 Memory Optimization

**Implementation:**
- [ ] In-place operations where safe
- [ ] Gradient checkpointing
- [ ] Memory pool for temporary buffers

**Target:** 50% memory reduction

### 8.3 Async Execution

**Implementation:**
- [ ] Pipeline GPU kernel submissions
- [ ] Overlap compute and memory transfers
- [ ] Batch multiple operations

**Target:** 80% GPU utilization

---

## Testing Strategy

### Numerical Gradient Checking

All backward operations must pass:
```rust
fn numerical_gradient_check<F>(f: F, x: &Tensor, eps: f64) -> bool
where F: Fn(&Tensor) -> Tensor
{
    let analytical = {
        let y = f(x);
        y.backward();
        x.grad().clone()
    };

    let numerical = {
        let mut grad = Tensor::zeros_like(x);
        for i in 0..x.numel() {
            x[i] += eps;
            let y_plus = f(x);
            x[i] -= 2.0 * eps;
            let y_minus = f(x);
            x[i] += eps;

            grad[i] = (y_plus - y_minus) / (2.0 * eps);
        }
        grad
    };

    (&analytical - &numerical).abs().max() < 1e-4
}
```

### Performance Benchmarks

Every operation must have:
- CPU baseline timing
- GPU timing
- Memory usage tracking
- Regression tests (no slowdowns)

### Integration Test Coverage

- [ ] 100% of PyTorch training tests ported
- [ ] All tests pass on CPU
- [ ] All tests pass on WebGPU
- [ ] All tests pass in WASM

---

## Success Criteria

### Phase 1-2 Complete:
- ✅ Simple autograd works: `z = x * y; z.backward()` computes gradients
- ✅ All arithmetic/matmul gradients pass numerical checks

### Phase 3 Complete:
- ✅ GPU backward pass 10x faster than CPU
- ✅ MNIST training runs on WebGPU

### Phase 4-5 Complete:
- ✅ Adam optimizer converges faster than SGD
- ✅ Cross-entropy loss is numerically stable

### Phase 6 Complete:
- ✅ MNIST MLP achieves >95% accuracy in <60 seconds
- ✅ GPU/CPU results match within 1e-4

### Phase 7 Complete:
- ✅ Browser training works without Burn
- ✅ 1000-sample MNIST training in <10 seconds on laptop GPU

### Phase 8 Complete:
- ✅ 80%+ GPU utilization during training
- ✅ Memory usage <2GB for MNIST training

---

## Development Workflow

### For Each Operation:

1. **Port Test** from PyTorch (`test/cpp/api/autograd.cpp`)
2. **Implement CPU version** in `libtorch-rust-sys/src/autograd/ops/`
3. **Verify with numerical gradients**
4. **Implement GPU shader** in `libtorch-rust-sys/src/gpu/kernels/backward/`
5. **Benchmark** CPU vs GPU
6. **Document** gradient formula and edge cases

### Test-Driven Development:

```bash
# 1. Write failing test
cargo test test_relu_backward -- --nocapture
# FAIL: ReLU gradient not implemented

# 2. Implement operation
vim libtorch-rust-sys/src/autograd/ops/activation.rs

# 3. Test passes
cargo test test_relu_backward
# PASS

# 4. Add GPU version
vim libtorch-rust-sys/src/gpu/kernels/backward/relu.wgsl

# 5. Benchmark
cargo bench bench_relu_backward
# CPU: 1.2ms, GPU: 0.08ms (15x speedup)
```

---

## Risk Mitigation

### Risk: WebGPU shader bugs hard to debug

**Mitigation:**
- Implement CPU version first (easier to debug)
- Use `wgpu-validation-layers` for shader errors
- Print intermediate values to storage buffers

### Risk: Gradient numerical instability

**Mitigation:**
- Use PyTorch's numerically stable formulas
- Add epsilon to denominators
- Clamp gradients to prevent overflow

### Risk: WASM threading limitations

**Mitigation:**
- Design single-threaded from start
- Use WebWorkers for parallelism if needed
- Profile early in browser

---

## Milestones & Timeline

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| Phase 1 | 2 weeks | Autograd tape system working |
| Phase 2 | 3 weeks | All CPU gradients implemented |
| Phase 3 | 4 weeks | GPU backward shaders complete |
| Phase 4 | 1 week | Loss functions working |
| Phase 5 | 2 weeks | SGD and Adam optimizers |
| Phase 6 | 2 weeks | MNIST training end-to-end |
| Phase 7 | 1 week | WASM integration |
| Phase 8 | 2 weeks | Performance optimization |
| **Total** | **17 weeks** | **Production-ready GPU training** |

---

## References

### PyTorch Test Files:
- `pytorch/test/cpp/api/autograd.cpp` - Core autograd tests
- `pytorch/test/cpp/api/modules.cpp` - Neural network module tests
- `pytorch/test/cpp/api/optim.cpp` - Optimizer tests
- `pytorch/test/cpp/api/integration.cpp` - End-to-end training tests

### PyTorch Source:
- `pytorch/torch/csrc/autograd/` - Autograd C++ implementation
- `pytorch/aten/src/ATen/native/cuda/` - CUDA kernels
- `pytorch/c10/core/TensorImpl.h` - Tensor internals

### Academic Papers:
- "Automatic Differentiation in Machine Learning: a Survey" (Baydin et al., 2018)
- "Adam: A Method for Stochastic Optimization" (Kingma & Ba, 2014)

---

## Conclusion

This plan provides a complete roadmap to transform libtorch-rust from inference-only to a full training framework with GPU acceleration. By systematically porting PyTorch's tests and implementing each component with numerical validation, we ensure correctness and compatibility.

The end result: **Real GPU-accelerated machine learning training in the browser**, not CPU toy demos.
