# Performance Optimization Plan for LibTorch-Rust Inference

**Document**: MECE Performance Improvement Plan
**Date**: 2025-11-09 08:39
**Scope**: Inference-only performance optimization
**Target**: Achieve competitive performance with tch-rs (within 2x for CPU operations)
**Current Status**: 2-200x slower than baseline (see ASSESSMENT_REPORT.md)

---

## Executive Summary

This document provides a **Mutually Exclusive, Collectively Exhaustive (MECE)** plan to improve libtorch-rust inference performance from current 0.005x-0.6x speed to target ≥0.5x speed (within 2x of tch-rs).

**Key Insight**: The performance gap is caused by five independent categories of issues. Addressing all five is necessary and sufficient to achieve competitive performance.

**Expected Outcome**: 10-100x speedup across operations, making inference viable for production use.

---

## MECE Framework

Performance improvements are categorized into **5 mutually exclusive dimensions**:

```
Performance = f(Algorithm, Computation, Memory, Parallelism, Integration)
```

1. **Algorithm Efficiency** - What algorithms we use
2. **Computational Efficiency** - How we execute computations
3. **Memory Efficiency** - How we manage data
4. **Parallelism** - How we use multiple resources
5. **System Integration** - How we interface with hardware/libraries

Each dimension is independent and collectively they cover all performance optimization opportunities.

---

## Category 1: Algorithm Efficiency

**Definition**: Choice of algorithms and their complexity
**Current Impact**: 20-200x slowdown on matrix operations
**Target Impact**: 50-100x speedup

### 1.1 Matrix Multiplication (CRITICAL - P0)

**Current State**:
```rust
// Naive O(n³) triple loop
for i in 0..m {
    for j in 0..n {
        for k in 0..k {
            result[i][j] += a[i][k] * b[k][j];
        }
    }
}
```
**Impact**: 195x slower at 500×500

**Improvement Options** (Choose one, mutually exclusive):

#### Option A: BLAS Integration (RECOMMENDED)
- **Approach**: Use existing BLAS library via FFI
- **Library**: `blas-src` + `cblas-sys` crates
- **Expected Speedup**: 50-100x
- **Effort**: Medium (2-3 days)
- **Implementation**:
  ```rust
  // Use cblas_sgemm for f32, cblas_dgemm for f64
  unsafe {
      cblas::sgemm(
          Layout::RowMajor,
          Transpose::None,
          Transpose::None,
          m, n, k,
          1.0,
          a.as_ptr(), k,
          b.as_ptr(), n,
          0.0,
          result.as_mut_ptr(), n,
      );
  }
  ```
- **Pros**: Battle-tested, optimal performance
- **Cons**: Requires C library (but lighter than full libtorch)

#### Option B: Pure Rust Optimized (ALTERNATIVE)
- **Approach**: Cache-blocked matrix multiply with SIMD
- **Expected Speedup**: 20-30x
- **Effort**: High (1-2 weeks)
- **Implementation**:
  - Block size: 32×32 or 64×64 (cache-friendly)
  - SIMD vectorization (4-8 elements at once)
  - Loop unrolling
- **Pros**: Pure Rust, no dependencies
- **Cons**: Still slower than BLAS, high maintenance

#### Option C: Use `matrixmultiply` Crate (COMPROMISE)
- **Approach**: Pure Rust, pre-optimized library
- **Library**: `matrixmultiply = "0.3"`
- **Expected Speedup**: 30-50x
- **Effort**: Low (1 day)
- **Pros**: Pure Rust, well-maintained, faster than naive
- **Cons**: Still slower than BLAS

**Recommendation**: **Option C for Phase 1** (quick win), **Option A for Phase 2** (optimal)

**Action Items**:
- [ ] Integrate `matrixmultiply` crate for gemm operations
- [ ] Add feature flag for BLAS backend
- [ ] Implement BLAS-based matmul behind feature flag
- [ ] Benchmark all three approaches

**Success Metrics**:
- 500×500 matmul: <5ms (currently 156ms)
- 100×100 matmul: <100µs (currently 1ms)

---

### 1.2 Transpose via Stride Manipulation (HIGH - P0)

**Current State**:
```rust
// Copies entire tensor
let mut transposed = Vec::new();
for j in 0..cols {
    for i in 0..rows {
        transposed.push(data[i * cols + j]);  // Full copy!
    }
}
```
**Impact**: 10-100x slower than necessary

**Improvement**:
```rust
// Zero-copy transpose (stride manipulation)
pub fn transpose(&self) -> Result<Self> {
    if self.ndim() != 2 {
        return Err(TchError::ShapeError("2D only".into()));
    }

    Ok(TensorImpl {
        storage: self.storage.clone(),  // Arc clone (cheap!)
        shape: vec![self.shape[1], self.shape[0]],  // Swap
        strides: vec![self.strides[1], self.strides[0]],  // Swap
        offset: self.offset,
        dtype: self.dtype,
        device: self.device,
    })
}
```

**Expected Speedup**: 10-100x (O(1) vs O(n))
**Effort**: Low (1 day)

**Caveat**: Requires implementing stride-aware element access throughout codebase

**Action Items**:
- [ ] Implement stride-aware indexing in all operations
- [ ] Add `is_transposed()` check
- [ ] Replace copying transpose with stride manipulation
- [ ] Add `make_contiguous()` when needed for performance

**Success Metrics**:
- Transpose time: O(1) regardless of size
- No memory copies for transpose operation

---

### 1.3 Reshape via Stride Manipulation (MEDIUM - P1)

**Current State**: Always copies data
**Improvement**: Zero-copy reshape when contiguous

```rust
pub fn reshape(&self, new_shape: &[usize]) -> Result<Self> {
    if new_numel != self.numel() { return Err(...); }

    // Zero-copy if contiguous
    if self.is_contiguous() {
        return Ok(TensorImpl {
            storage: self.storage.clone(),
            shape: new_shape.to_vec(),
            strides: Self::compute_strides(new_shape),
            offset: self.offset,
            dtype: self.dtype,
            device: self.device,
        });
    }

    // Copy only if non-contiguous
    self.contiguous()?.reshape(new_shape)
}
```

**Expected Speedup**: 5-10x for reshape operations
**Effort**: Low (1 day)

---

## Category 2: Computational Efficiency

**Definition**: How efficiently we execute individual operations
**Current Impact**: 2-5x slowdown on element-wise operations
**Target Impact**: 2-4x speedup

### 2.1 Eliminate f64 Conversion Overhead (CRITICAL - P0)

**Current State**: ALL operations convert through f64
```rust
// Current (SLOW):
let data = self.to_vec_f64();  // f32 → f64 copy
let result: Vec<f64> = data.iter().map(|&x| x.max(0.0)).collect();
let f32_data: Vec<f32> = result.iter().map(|&x| x as f32).collect();  // f64 → f32 copy
```

**Impact**:
- 2x memory usage during operations
- 2x type conversion overhead
- Cache pollution

**Improvement**: Generic operations in native dtype
```rust
// Improved (FAST):
impl TensorImpl {
    pub fn relu(&self) -> Result<Self> {
        match self.dtype {
            DType::Float => self.unary_op_f32(|x| x.max(0.0)),
            DType::Double => self.unary_op_f64(|x| x.max(0.0)),
            // ...
        }
    }

    fn unary_op_f32<F>(&self, f: F) -> Result<Self>
    where F: Fn(f32) -> f32
    {
        let data = self.storage.as_f32_slice()?;  // No conversion!
        let result: Vec<f32> = data.iter().map(|&x| f(x)).collect();
        Ok(Self::from_slice_f32(&result, &self.shape)?)
    }
}
```

**Expected Speedup**: 2-4x for all element-wise operations
**Effort**: Medium (3-5 days to refactor all operations)

**Action Items**:
- [ ] Create generic operation traits
- [ ] Implement f32-native operations
- [ ] Implement f64-native operations
- [ ] Remove all `to_vec_f64()` calls
- [ ] Add dtype-specific fast paths

**Success Metrics**:
- Zero f32↔f64 conversions in hot paths
- Element-wise ops: <1µs for 1000 elements (currently ~1.1µs)

---

### 2.2 SIMD Vectorization (HIGH - P0)

**Current State**: Scalar operations, relies on auto-vectorization
**Impact**: Missing 4-8x potential speedup

**Improvement Options**:

#### Option A: Portable SIMD (RECOMMENDED for stability)
```rust
use std::simd::{f32x8, SimdFloat};

fn add_simd_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    const LANES: usize = 8;

    // Vectorized portion
    for i in (0..a.len()).step_by(LANES) {
        if i + LANES <= a.len() {
            let va = f32x8::from_slice(&a[i..]);
            let vb = f32x8::from_slice(&b[i..]);
            let result = va + vb;
            result.copy_to_slice(&mut out[i..]);
        }
    }

    // Scalar remainder
    let remainder = a.len() % LANES;
    for i in (a.len() - remainder)..a.len() {
        out[i] = a[i] + b[i];
    }
}
```

**Expected Speedup**: 4-8x for element-wise operations
**Effort**: Medium (1 week for common operations)
**Requires**: Nightly Rust (currently) or wait for stable

#### Option B: Platform-Specific Intrinsics (ALTERNATIVE)
```rust
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
unsafe fn add_avx(a: &[f32], b: &[f32], out: &mut [f32]) {
    for i in (0..a.len()).step_by(8) {
        if i + 8 <= a.len() {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i));
            let result = _mm256_add_ps(va, vb);
            _mm256_storeu_ps(out.as_mut_ptr().add(i), result);
        }
    }
}
```

**Expected Speedup**: 4-8x
**Effort**: High (need platform-specific code)
**Pros**: Works on stable Rust
**Cons**: Maintenance burden, platform-specific

**Recommendation**: Option A (portable_simd) - worth nightly requirement

**Action Items**:
- [ ] Enable `portable_simd` feature
- [ ] Implement SIMD for: add, sub, mul, div
- [ ] Implement SIMD for: relu, sigmoid (using SIMD math)
- [ ] Benchmark vs scalar
- [ ] Add feature flag for SIMD/scalar selection

**Success Metrics**:
- Element-wise add (10K): <2µs (currently 10.4µs)
- ReLU (10K): <1.5µs (currently 5.8µs)

---

### 2.3 Loop Optimizations (MEDIUM - P1)

**Techniques**:
1. **Loop unrolling** (4-8 iterations at a time)
2. **Loop fusion** (combine multiple passes)
3. **Strength reduction** (cheaper operations)

**Example - Loop Fusion**:
```rust
// Before (2 passes):
let tmp = tensor.relu();
let result = tmp.mul_scalar(2.0);

// After (1 fused pass):
let result = tensor.map(|x| x.max(0.0) * 2.0);
```

**Expected Speedup**: 1.5-2x for complex operations
**Effort**: Medium (ongoing optimization)

---

## Category 3: Memory Efficiency

**Definition**: How we allocate, copy, and access memory
**Current Impact**: Dominates small tensor operations
**Target Impact**: 2-3x speedup on small ops

### 3.1 Eliminate Unnecessary Allocations (CRITICAL - P0)

**Current State**: Every operation allocates 2-4 new vectors
```rust
// Current:
fn binary_op(&self, other: &TensorImpl, f: F) -> Result<TensorImpl> {
    let self_data = self.to_vec_f64();      // Allocation #1
    let other_data = other.to_vec_f64();    // Allocation #2
    let result_data: Vec<f64> = self_data   // Allocation #3
        .iter()
        .zip(other_data.iter())
        .map(|(&a, &b)| f(a, b))
        .collect();
    self.create_result_tensor(result_data, ...) // Allocation #4
}
```

**Improvement**: Operate on slices directly
```rust
// Improved:
fn binary_op_f32<F>(&self, other: &TensorImpl, f: F) -> Result<TensorImpl>
where F: Fn(f32, f32) -> f32
{
    let a = self.storage.as_f32_slice()?;    // No allocation!
    let b = other.storage.as_f32_slice()?;   // No allocation!

    let mut result = Vec::with_capacity(a.len());  // Single allocation
    for i in 0..a.len() {
        result.push(f(a[i], b[i]));
    }

    Ok(Self::from_vec_f32(result, &self.shape)?)
}
```

**Expected Speedup**: 2-3x for small tensors (<1000 elements)
**Effort**: Medium (3-4 days)

---

### 3.2 Memory Pooling (MEDIUM - P1)

**Current State**: Every operation allocates from heap
**Improvement**: Reuse allocations via memory pool

```rust
use std::sync::Mutex;

thread_local! {
    static POOL: Mutex<Vec<Vec<f32>>> = Mutex::new(Vec::new());
}

fn get_buffer(size: usize) -> Vec<f32> {
    POOL.with(|pool| {
        let mut p = pool.lock().unwrap();
        p.pop().map(|mut v| { v.clear(); v.reserve(size); v })
            .unwrap_or_else(|| Vec::with_capacity(size))
    })
}

fn return_buffer(buf: Vec<f32>) {
    if buf.capacity() < 1_000_000 {  // Don't pool huge buffers
        POOL.with(|pool| pool.lock().unwrap().push(buf));
    }
}
```

**Expected Speedup**: 1.5-2x for operations on small/medium tensors
**Effort**: Medium (2-3 days)

---

### 3.3 In-Place Operations (MEDIUM - P1)

**Current State**: All operations create new tensors
**Improvement**: Add in-place variants when reference count = 1

```rust
impl TensorImpl {
    pub fn add_(&mut self, other: &TensorImpl) -> Result<()> {
        // In-place addition (no allocation)
        let data = self.storage.as_mut_f32_slice()?;
        let other_data = other.storage.as_f32_slice()?;

        for i in 0..data.len() {
            data[i] += other_data[i];
        }
        Ok(())
    }
}
```

**Expected Speedup**: 2x for mutation chains
**Effort**: Medium (1 week for all operations)

---

### 3.4 Cache Optimization (MEDIUM - P2)

**Techniques**:
1. **Data layout**: Row-major vs column-major awareness
2. **Blocking**: Tile operations to fit in cache
3. **Prefetching**: Software prefetch for large operations

**For Matrix Multiply**:
```rust
const BLOCK_SIZE: usize = 64;  // Fits in L1 cache

for i in (0..m).step_by(BLOCK_SIZE) {
    for j in (0..n).step_by(BLOCK_SIZE) {
        for k in (0..k_dim).step_by(BLOCK_SIZE) {
            // Multiply micro-blocks
            matmul_block(a, b, c, i, j, k, BLOCK_SIZE);
        }
    }
}
```

**Expected Speedup**: 2-3x for large operations
**Effort**: High (only if not using BLAS)

---

## Category 4: Parallelism

**Definition**: Using multiple CPU cores or execution units
**Current Impact**: Missing N-core speedup
**Target Impact**: 4-8x speedup on multi-core CPUs

### 4.1 Thread-Level Parallelism with Rayon (HIGH - P1)

**Current State**: All operations single-threaded
**Improvement**: Parallelize large tensor operations

```rust
use rayon::prelude::*;

impl TensorImpl {
    pub fn add_parallel(&self, other: &TensorImpl) -> Result<Self> {
        const PARALLEL_THRESHOLD: usize = 10_000;

        if self.numel() < PARALLEL_THRESHOLD {
            return self.add(other);  // Use serial for small
        }

        let a = self.storage.as_f32_slice()?;
        let b = other.storage.as_f32_slice()?;

        let result: Vec<f32> = a.par_iter()
            .zip(b.par_iter())
            .map(|(&x, &y)| x + y)
            .collect();

        Ok(Self::from_vec_f32(result, &self.shape)?)
    }
}
```

**Expected Speedup**: 4-8x on 8-core CPU for large tensors
**Effort**: Medium (1 week)

**Heuristics for when to parallelize**:
- Element-wise ops: ≥10,000 elements
- Reductions: ≥100,000 elements
- Matrix multiply: ≥100×100 (if not using BLAS)

**Action Items**:
- [ ] Add `rayon` dependency
- [ ] Parallelize: add, sub, mul, div (for large tensors)
- [ ] Parallelize: relu, sigmoid (for large tensors)
- [ ] Parallelize: reductions (sum, mean)
- [ ] Add auto-tuning for parallel threshold

**Success Metrics**:
- Large add (100K): 4x faster on 8-core CPU
- Matrix multiply: Let BLAS handle parallelism

---

### 4.2 Instruction-Level Parallelism (MEDIUM - P2)

**Techniques**:
1. **Pipelining**: Arrange code to maximize CPU pipeline usage
2. **Avoid dependencies**: Allow multiple operations in flight
3. **Unroll loops**: Expose more ILP

**Example**:
```rust
// Poor ILP (dependency chain):
for i in 0..n {
    sum += data[i];  // Each iteration depends on previous
}

// Better ILP (parallel reduction):
let mut sum0 = 0.0;
let mut sum1 = 0.0;
let mut sum2 = 0.0;
let mut sum3 = 0.0;

for i in (0..n).step_by(4) {
    sum0 += data[i];
    sum1 += data[i+1];
    sum2 += data[i+2];
    sum3 += data[i+3];
}

let sum = sum0 + sum1 + sum2 + sum3;
```

**Expected Speedup**: 1.5-2x for reductions
**Effort**: Low (part of loop optimization)

---

## Category 5: System Integration

**Definition**: How we interact with OS, hardware, and libraries
**Current Impact**: N/A (no integration currently)
**Target Impact**: Enable GPU acceleration (future)

### 5.1 BLAS Library Integration (CRITICAL - P0)

**See Category 1.1** - This is the algorithm choice, included here for completeness

**Libraries**:
- Intel MKL (fastest, proprietary)
- OpenBLAS (fast, open source)
- BLIS (modern, open source)
- Apple Accelerate (macOS)

**Integration Strategy**:
```toml
[dependencies]
blas-src = { version = "0.10", optional = true, features = ["openblas"] }
cblas-sys = { version = "0.1", optional = true }

[features]
default = ["blas"]
blas = ["blas-src", "cblas-sys"]
pure-rust = []  # Fallback to pure Rust implementation
```

---

### 5.2 Platform-Specific Optimizations (LOW - P2)

**Platform Detection**:
```rust
#[cfg(target_arch = "x86_64")]
fn detect_features() -> CpuFeatures {
    CpuFeatures {
        avx2: is_x86_feature_detected!("avx2"),
        avx512: is_x86_feature_detected!("avx512f"),
        fma: is_x86_feature_detected!("fma"),
    }
}

// Dispatch to best implementation
pub fn add_optimized(a: &[f32], b: &[f32], out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { add_avx2(a, b, out) };
        }
    }

    add_scalar(a, b, out);  // Fallback
}
```

**Effort**: High (ongoing)
**Expected Speedup**: 10-20% on top of SIMD

---

### 5.3 GPU Acceleration (FUTURE - NOT INFERENCE OPTIMIZATION)

**Status**: Out of scope for this plan (requires separate architecture)
**Future Consideration**: WebGPU, CUDA, Metal support

---

## Implementation Roadmap

### Phase 1: Critical Path (Target: 10-20x speedup) - 2-3 weeks

**Week 1: Algorithmic Improvements**
- [ ] **Day 1-2**: Integrate `matrixmultiply` crate (30-50x matmul speedup)
- [ ] **Day 3**: Zero-copy transpose via strides (10-100x transpose speedup)
- [ ] **Day 4**: Zero-copy reshape via strides (5-10x reshape speedup)
- [ ] **Day 5**: Benchmark and validate

**Expected Result**: Matrix operations 30-50x faster

**Week 2: Computational Efficiency**
- [ ] **Day 1-3**: Eliminate f64 conversions (2-4x element-wise speedup)
- [ ] **Day 4-5**: Enable portable_simd (4-8x element-wise speedup)

**Expected Result**: Element-wise operations 8-32x faster overall

**Week 3: Memory Optimization**
- [ ] **Day 1-2**: Eliminate unnecessary allocations (2-3x small tensor speedup)
- [ ] **Day 3**: Implement memory pooling (1.5-2x speedup)
- [ ] **Day 4-5**: Integration testing and benchmarking

**Expected Result**: Small tensor operations 3-6x faster

**Phase 1 Target**: Achieve 0.3-0.5x speed (2-3x slower than tch-rs) ✅

---

### Phase 2: Parallelization (Target: Additional 2-4x) - 1-2 weeks

**Week 4: Thread-Level Parallelism**
- [ ] **Day 1-3**: Integrate Rayon for large operations
- [ ] **Day 4-5**: Auto-tuning parallel thresholds

**Expected Result**: Large tensor operations 4-8x faster on multi-core

**Phase 2 Target**: Achieve 0.5-0.8x speed (1.25-2x slower than tch-rs) ✅

---

### Phase 3: Advanced Optimizations (Target: Additional 1.5-2x) - 2-3 weeks

**Week 5-6: Polish**
- [ ] Loop optimizations and fusion
- [ ] In-place operation variants
- [ ] Cache optimization
- [ ] Platform-specific tuning

**Phase 3 Target**: Achieve 0.7-1.0x speed (competitive with tch-rs) 🎯

---

### Phase 4: BLAS Integration (Target: Optimal performance) - 1 week

**Week 7: Production BLAS**
- [ ] Implement BLAS backend behind feature flag
- [ ] Performance comparison with matrixmultiply
- [ ] Documentation and examples

**Phase 4 Target**: Achieve 0.9-1.2x speed (match or exceed tch-rs for some ops) 🚀

---

## Success Metrics

### Benchmark Targets (Post-Optimization)

| Operation | Current | Phase 1 Target | Phase 2 Target | Final Target |
|-----------|---------|----------------|----------------|--------------|
| **Matrix 500×500** | 156ms | 5ms | 2ms | 1ms |
| **Matrix 100×100** | 1.0ms | 50µs | 30µs | 20µs |
| **Add (10K elements)** | 10.4µs | 2µs | 1µs | 0.8µs |
| **ReLU (10K)** | 5.8µs | 1.5µs | 1µs | 0.8µs |
| **Sigmoid (10K)** | 53.4µs | 15µs | 10µs | 8µs |
| **MNIST Forward (batch=32)** | 6ms (est) | 1.5ms | 750µs | 500µs |

**Overall Target**: Within 2x of tch-rs performance across all operations

---

## Risk Management

### High-Risk Items

1. **Portable SIMD on Nightly**
   - **Risk**: API instability, requires nightly Rust
   - **Mitigation**: Also implement scalar fallback, feature flag
   - **Alternative**: Use platform-specific intrinsics on stable

2. **BLAS Dependency**
   - **Risk**: Adds C dependency (defeats "pure Rust" goal)
   - **Mitigation**: Make optional via feature flag, pure-Rust default
   - **Alternative**: Use `matrixmultiply` crate (pure Rust)

3. **Stride Manipulation Complexity**
   - **Risk**: Bugs in stride calculations cause silent errors
   - **Mitigation**: Extensive testing, property-based tests
   - **Validation**: Compare output with copying version

### Medium-Risk Items

1. **Memory Pool Contention**
   - **Risk**: Thread contention on global pool
   - **Mitigation**: Use thread-local pools

2. **Rayon Overhead**
   - **Risk**: Parallel overhead exceeds gains for some sizes
   - **Mitigation**: Careful threshold tuning, benchmarking

---

## Measurement and Validation

### Continuous Benchmarking

```bash
# Run benchmarks after each change
cargo bench --bench tensor_benchmarks
cargo bench --bench nn_benchmarks

# Compare with baseline
critcmp baseline current
```

### Regression Testing

```rust
#[test]
fn test_performance_regression() {
    let start = Instant::now();
    let result = large_matmul_benchmark();
    let duration = start.elapsed();

    assert!(duration < Duration::from_millis(5),
        "Performance regression: matmul took {:?}", duration);
}
```

### Correctness Validation

```rust
#[test]
fn test_optimized_vs_naive() {
    let a = random_tensor([100, 100]);
    let b = random_tensor([100, 100]);

    let result_naive = matmul_naive(&a, &b);
    let result_optimized = matmul_optimized(&a, &b);

    assert_approx_eq!(result_naive, result_optimized, 1e-4);
}
```

---

## Dependencies and Prerequisites

### Required Crates

```toml
[dependencies]
# Phase 1
matrixmultiply = "0.3"          # Pure Rust optimized matmul

# Phase 2
portable-simd = "0.1"           # Requires nightly
rayon = "1.10"                  # Parallel iterators

# Phase 3 (optional)
blas-src = { version = "0.10", features = ["openblas"], optional = true }
cblas-sys = { version = "0.1", optional = true }

[features]
default = ["matrixmultiply"]
blas = ["blas-src", "cblas-sys"]
simd = ["portable-simd"]        # Requires nightly
```

### Environment Requirements

- Rust nightly (for portable_simd)
- OR Rust stable 1.75+ (if using platform intrinsics)
- Multi-core CPU (for parallel benchmarks)
- 16GB RAM (for large benchmarks)

---

## Appendix A: MECE Validation

**Proof that categories are Mutually Exclusive**:

1. **Algorithm** ≠ Computation: Algorithm is *which* method, computation is *how* we execute
2. **Computation** ≠ Memory: Computation is processing, memory is data movement
3. **Memory** ≠ Parallelism: Memory is allocation/access, parallelism is concurrent execution
4. **Parallelism** ≠ Integration: Parallelism is internal, integration is external
5. **Integration** ≠ Algorithm: Integration is using external libs, algorithm is our logic

**Proof that categories are Collectively Exhaustive**:

All performance factors fall into one category:
- Slow due to bad algorithm? → Category 1
- Slow due to inefficient computation? → Category 2
- Slow due to memory overhead? → Category 3
- Slow due to single-threading? → Category 4
- Slow due to missing hardware/lib support? → Category 5

∴ MECE framework is valid ∎

---

## Appendix B: Quick Wins Summary

**Top 5 Highest Impact/Effort Ratio**:

1. **Integrate matrixmultiply** (1 day, 30x speedup for matmul) 🏆
2. **Zero-copy transpose** (1 day, 10-100x speedup for transpose) 🏆
3. **Remove f64 conversions** (3 days, 2-4x speedup for everything) 🏆
4. **Add Rayon parallelism** (3 days, 4-8x speedup for large ops) 🏆
5. **Enable SIMD** (5 days, 4-8x speedup for element-wise) 🏆

**Expected Total Improvement**: 10-50x speedup in 2-3 weeks

---

## Document Control

**Version**: 1.0
**Author**: Performance Analysis Team
**Status**: Approved for Implementation
**Next Review**: After Phase 1 completion

**Change Log**:
- 2025-11-09 08:39: Initial version created
