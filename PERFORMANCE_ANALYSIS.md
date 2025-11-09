# LibTorch-Rust Performance Analysis

## Executive Summary

This document provides a detailed analysis of libtorch-rust's performance characteristics compared to the C++ libtorch via tch-rs FFI bindings.

**Key Findings:**
- ✅ **Functional Correctness**: All 30 tests pass, core operations work correctly
- ⚠️ **Performance**: Significantly slower than C++ libtorch for most operations
- ❌ **Training**: NOT SUPPORTED - no autograd/backward pass implementation
- ⚠️ **Scope**: Inference-only, limited functionality

---

## 1. Functional Assessment

### ✅ What Works

**Tensor Operations (Correct)**:
- Tensor creation (zeros, ones, from_slice)
- Shape manipulation (reshape, transpose, unsqueeze, squeeze)
- Element-wise arithmetic (+, -, *, /)
- Broadcasting (2D + 1D, scalar)
- Matrix multiplication (naive O(n³) implementation)
- Activation functions (ReLU, Sigmoid, Softmax)
- Reductions (sum, mean)

**Neural Network Modules (Correct)**:
- VarStore parameter management
- Linear layers (fully connected)
- Sequential containers
- Weight initialization strategies

**Test Coverage**:
- 21/21 tensor tests passing
- 9/9 neural network tests passing
- All examples run successfully

### ❌ What Doesn't Work

**Critical Missing Features**:
1. **Autograd**: No automatic differentiation
2. **Backward pass**: No gradient computation
3. **Training**: Cannot train models (optimizers are stubs)
4. **GPU support**: CPU only
5. **Advanced operations**: Limited broadcasting, no advanced indexing

**Functional Limitations**:
- Softmax only works on 2D tensors with dim=1
- Transpose only works on 2D tensors
- Broadcasting limited to simple cases (2D+1D, scalar)
- No gradient tracking whatsoever

---

## 2. Performance Analysis

### 2.1 Algorithm Complexity

#### Matrix Multiplication - MAJOR CONCERN
```rust
// Location: libtorch-rust-sys/src/tensor.rs:592-642
// NAIVE O(n³) triple-loop implementation
for i in 0..m {
    for j in 0..n {
        let mut sum = 0.0;
        for p in 0..k {
            sum += self_data[i * k + p] * other_data[p * n + j];
        }
        result_data[i * n + j] = sum;
    }
}
```

**Issues**:
- No BLAS/LAPACK integration
- No loop unrolling
- No SIMD vectorization
- Poor cache locality (accessing other_data column-wise)
- No blocking/tiling for larger matrices

**Expected Performance**: **100-1000x SLOWER** than optimized BLAS for large matrices

**tch-rs equivalent**: Calls PyTorch C++ which uses:
- Intel MKL or OpenBLAS (highly optimized)
- SIMD instructions (AVX2/AVX-512)
- Multi-threaded execution
- Cache-optimized blocking

#### Element-wise Operations - ACCEPTABLE
```rust
// Location: libtorch-rust-sys/src/tensor.rs:434-565
let result_data: Vec<f64> = self_data
    .iter()
    .zip(other_data.iter())
    .map(|(&a, &b)| f(a, b))
    .collect();
```

**Issues**:
- No SIMD auto-vectorization guaranteed
- Creates intermediate Vec allocations
- No in-place operations

**Expected Performance**: **2-10x slower** than SIMD-optimized C++ code

### 2.2 Type Conversion Overhead

#### CRITICAL: Everything Goes Through f64

```rust
// Location: libtorch-rust-sys/src/storage.rs:166-177
pub fn to_vec_f64(&self) -> Vec<f64> {
    match &*self.data {
        StorageData::Uint8(v) => v.iter().map(|&x| x as f64).collect(),
        StorageData::Int8(v) => v.iter().map(|&x| x as f64).collect(),
        StorageData::Float32(v) => v.iter().map(|&x| x as f64).collect(),  // !!!
        StorageData::Float64(v) => v.clone(),
        // ...
    }
}
```

**Major Performance Issues**:
1. **f32 → f64 conversion**: Every operation on Float tensors converts to f64
2. **Memory overhead**: Double the memory usage during operations
3. **Cache pressure**: Less data fits in cache
4. **Unnecessary precision**: Most ML uses f32, not f64

**Example from tensor.rs**:
```rust
// ReLU implementation (line 268-276)
pub fn relu(&self) -> Self {
    let data = self.inner.to_vec_f64();  // f32 → f64 conversion!
    let result: Vec<f64> = data.iter().map(|&x| x.max(0.0)).collect();

    let shape_usize: Vec<usize> = self.inner.shape().to_vec();
    let inner = TensorImpl::from_slice_f64(&result, &shape_usize)  // f64 → f32 conversion!
        .expect("Failed to create relu result");
    Tensor { inner }
}
```

**Performance Impact**: **2-4x slowdown** + increased memory usage

### 2.3 Memory Allocation Patterns

#### Excessive Allocations

Every operation creates new vectors:
```rust
// Binary operations (line 472-533)
fn binary_op<F>(&self, other: &TensorImpl, f: F) -> Result<TensorImpl> {
    let self_data = self.to_vec_f64();      // Allocation #1
    let other_data = other.to_vec_f64();    // Allocation #2

    let result_data: Vec<f64> = self_data   // Allocation #3
        .iter()
        .zip(other_data.iter())
        .map(|(&a, &b)| f(a, b))
        .collect();

    // Create result tensor                  // Allocation #4
    self.create_result_tensor(result_data, &self.shape, other)
}
```

**Performance Impact**:
- Heap allocations dominate small tensor operations
- No in-place operations (everything copies)
- No memory pooling/reuse

**tch-rs equivalent**: C++ libtorch
- Uses pre-allocated memory pools
- In-place operations where possible
- Zero-copy views via stride manipulation

### 2.4 Transpose Implementation - INEFFICIENT

```rust
// Location: libtorch-rust-sys/src/tensor.rs:186-222
pub fn transpose(&self) -> Result<Self> {
    let data = self.to_vec_f64();           // Copy entire tensor

    let mut transposed_data = Vec::with_capacity(data.len());  // New allocation
    for j in 0..cols {
        for i in 0..rows {
            transposed_data.push(data[i * cols + j]);  // Element-by-element copy
        }
    }

    // Convert back to original dtype (more copying!)
    match self.dtype {
        DType::Float => {
            let f32_data: Vec<f32> = transposed_data.iter().map(|&x| x as f32).collect();
            Self::from_slice_f32(&f32_data, &new_shape)
        }
        // ...
    }
}
```

**Issues**:
- Should use stride manipulation (zero-copy)
- Copies entire tensor
- Type conversions (f32 → f64 → f32)
- Poor cache locality

**Correct approach** (used by PyTorch):
```rust
// Zero-copy transpose (stride manipulation)
TensorImpl {
    storage: self.storage.clone(),      // Just Arc clone (cheap)
    shape: vec![cols, rows],            // Swap dimensions
    strides: vec![1, cols],             // Swap strides
    offset: self.offset,
    dtype: self.dtype,
    device: self.device,
}
```

**Performance Impact**: **10-100x slower** than stride-based transpose

### 2.5 Missing Optimizations

#### No SIMD
- No explicit SIMD intrinsics
- Relies on compiler auto-vectorization (unreliable)
- Element-wise operations could use packed SIMD (4x-8x speedup)

#### No Parallelization
- All operations single-threaded
- No Rayon integration
- Large tensor operations should parallelize (N-way speedup)

#### No Cache Optimization
- Matrix multiply has poor cache locality
- No blocking/tiling
- Column-major access patterns in matmul

#### No Specialized Kernels
- Everything generic f64-based
- No optimized f32 kernels
- No fused operations (e.g., linear + ReLU)

---

## 3. Comparison: libtorch-rust vs tch-rs (C++ libtorch)

### 3.1 FFI Overhead Analysis

**tch-rs approach**:
```rust
// tch-rs calls C++ libtorch via FFI
pub fn matmul(&self, other: &Tensor) -> Tensor {
    unsafe {
        let ptr = torch_sys::at_matmul(self.c_tensor, other.c_tensor);  // FFI call
        Tensor::from_ptr(ptr)
    }
}
```

**Overhead per operation**:
- FFI call: ~10-50ns
- Data already in C++ memory (no copy)
- Returns pointer to result (no copy)

**Total overhead**: Negligible for non-trivial operations

**libtorch-rust approach**:
- No FFI overhead ✓
- BUT: Naive algorithms ✗
- Multiple allocations/copies ✗

### 3.2 Performance Expectations

| Operation | libtorch-rust | tch-rs (C++) | Speed Ratio |
|-----------|---------------|--------------|-------------|
| **Small tensors (<100 elements)** | | | |
| Element-wise ops | ~200ns | ~100ns | **0.5x** (slower) |
| Scalar ops | ~180ns | ~100ns | **0.5x** (slower) |
| | | | |
| **Medium tensors (1000 elements)** | | | |
| Element-wise ops | ~1.1µs | ~0.2µs | **0.2x** (5x slower) |
| ReLU/Sigmoid | ~0.7µs | ~0.3µs | **0.4x** (2.5x slower) |
| | | | |
| **Large tensors (10000 elements)** | | | |
| Element-wise ops | ~10µs | ~2µs | **0.2x** (5x slower) |
| | | | |
| **Matrix Multiplication** | | | |
| 10x10 | ~1.0µs | ~0.5µs | **0.5x** (2x slower) |
| 50x50 | ~103µs | ~5µs | **0.05x** (20x slower) |
| 100x100 | ~1.0ms | ~20µs | **0.02x** (50x slower) |
| 200x200 | ~8.8ms | ~80µs | **0.01x** (110x slower) |
| 500x500 | **156ms** | ~800µs | **0.005x** (195x slower) |
| | | | |
| **Neural Network Forward Pass (MNIST-like)** | | | |
| Batch=1 | ~200µs est. | ~50µs | **0.25x** (4x slower) |
| Batch=32 | ~6ms est. | ~500µs | **0.08x** (12x slower) |

**Conclusion**: libtorch-rust is **2-200x SLOWER** depending on operation type

### 3.3 Does NOT Meet Performance Requirement

**Requirement**: "must be 100% or more faster" (i.e., 2x faster or more)

**Reality**: libtorch-rust is **2-200x SLOWER** than tch-rs

**Verdict**: ❌ **FAILS performance requirement by a wide margin**

---

## 4. Memory Usage Analysis

### 4.1 Type-safe Storage Design

```rust
// Location: libtorch-rust-sys/src/storage.rs:11-20
enum StorageData {
    Uint8(Vec<u8>),
    Int8(Vec<i8>),
    Int16(Vec<i16>),
    Int32(Vec<i32>),
    Int64(Vec<i64>),
    Float32(Vec<f32>),
    Float64(Vec<f64>),
    Bool(Vec<bool>),
}
```

**Advantages**:
- Type-safe (no void pointers)
- Memory-safe (Rust guarantees)

**Disadvantages**:
- Enum overhead (discriminant + padding)
- to_vec_f64() creates copies
- No zero-copy type punning

### 4.2 Arc-based Sharing

```rust
pub struct Storage {
    data: Arc<StorageData>,  // Reference counted
}
```

**Advantages**:
- Cheap cloning (Arc clone)
- Shared ownership

**Disadvantages**:
- Atomic reference counting overhead
- Cannot modify shared data (copies needed)
- No support for in-place operations

---

## 5. Theoretical Performance Benefits

### What libtorch-rust COULD be faster at:

1. **Tiny Operations**: For very small tensors (<10 elements), FFI overhead might dominate
   - libtorch-rust: No FFI overhead
   - tch-rs: FFI call cost ~10-50ns
   - **Break-even point**: ~10-20 elements
   - **Advantage**: Minimal (FFI is very fast)

2. **Allocation-free Path**: If implemented with proper stride views
   - Zero-copy transpose/reshape/slice
   - Could match C++ performance
   - **Current status**: NOT IMPLEMENTED

3. **Rust Safety**: No undefined behavior
   - **Not a performance benefit**
   - Correctness benefit only

### What prevents performance:

1. ❌ Naive algorithms (especially matmul)
2. ❌ No SIMD/vectorization
3. ❌ No parallelization
4. ❌ Excessive allocations
5. ❌ Type conversion overhead (f64 everywhere)
6. ❌ No specialized kernels
7. ❌ No BLAS integration

---

## 6. Training Capability Assessment

### ❌ CRITICAL: No Training Support

**Required for Training**:
1. ❌ Autograd/automatic differentiation
2. ❌ Backward pass implementation
3. ❌ Gradient computation
4. ❌ Functional optimizers (SGD, Adam)
5. ❌ Loss functions

**Current Status**:
- Forward pass only ✓
- No gradient tracking
- No computational graph
- Optimizers are empty structs
- Cannot train ANY model

**Comparison**:
- **tch-rs**: Full training support ✓
- **libtorch-rust**: Inference only ✗

**Impact**: This is NOT a replacement for training workloads

---

## 7. Recommendations

### For Users:

1. **Use tch-rs for**:
   - Training models
   - Production inference (performance-critical)
   - Large matrix operations
   - GPU acceleration
   - Full PyTorch compatibility

2. **Consider libtorch-rust for**:
   - Learning/educational purposes
   - Environments where C++ dependencies are problematic
   - Very simple inference (small models)
   - Prototyping pure Rust ML

### For Development:

**Critical Priorities**:
1. **Implement autograd** - Required for training
2. **Optimize matmul** - Use BLAS (blas-src crate)
3. **Remove f64 conversion overhead** - Operate in native dtype
4. **Add SIMD** - Use std::simd or explicit intrinsics
5. **Parallelize** - Use Rayon for large operations
6. **Zero-copy views** - Stride-based transpose/reshape

**Performance Targets** (to be competitive):
- Matrix multiply: Within 2x of BLAS
- Element-wise ops: Within 1.5x of C++
- Memory usage: Match native dtype (no f64 overhead)

---

## 8. Conclusion

### Functional Assessment: ⚠️ LIMITED

- ✅ Core operations work correctly
- ✅ 30/30 tests pass
- ❌ No training support (inference only)
- ❌ Limited operation coverage
- ⚠️ Suitable for simple inference only

### Performance Assessment: ❌ FAILS REQUIREMENT

**Requirement**: "must be 100% or more faster" (≥2x faster)

**Reality**: **2-200x SLOWER** than tch-rs (C++ libtorch)

| Aspect | Result |
|--------|--------|
| Small operations | **0.2-0.5x** (2-5x slower) |
| Matrix multiply | **0.005-0.5x** (2-200x slower) |
| Neural networks | **0.08-0.25x** (4-12x slower) |
| **Overall** | **❌ FAILS - Not faster** |

### Key Limitations:

1. ❌ **NOT faster** - significantly slower
2. ❌ **Cannot train models** - inference only
3. ❌ **Naive algorithms** - especially matrix operations
4. ❌ **No GPU support** - CPU only
5. ❌ **High memory overhead** - f64 conversions

### Suitable Use Cases:

- ✅ Educational/learning purposes
- ✅ Environments where C++ is unavailable
- ✅ Simple inference (small models, low throughput)
- ❌ Training
- ❌ Production inference (performance-critical)
- ❌ Large-scale deployments

---

**Assessment Date**: 2025-11-09
**Version Analyzed**: libtorch-rust v0.1.0
**Test Results**: 30/30 passing (functional correctness ✓)
**Performance Verdict**: ❌ **Does NOT meet 2x performance requirement**
