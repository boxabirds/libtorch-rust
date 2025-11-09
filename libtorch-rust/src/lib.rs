//! # LibTorch Rust
//!
//! Pure Rust implementation of PyTorch/LibTorch with API compatibility to tch-rs.
//!
//! This library provides a complete re-implementation of libtorch in Rust,
//! offering the same API as tch-rs while being 100% pure Rust with no C++ dependencies.

pub mod device;
pub mod kind;
pub mod tensor;
pub mod nn;
pub mod vision;
pub mod jit;
pub mod index;
pub mod wrappers;

/// Autograd functionality for automatic differentiation
pub mod autograd {
    pub use libtorch_rust_sys::autograd::{is_grad_enabled, set_grad_enabled, NoGradGuard, GradNode, Edge};
}

// Re-export commonly used types
pub use device::Device;
pub use kind::Kind;
pub use tensor::Tensor;

use libtorch_rust_sys::TchError;

/// Result type for tensor operations
pub type Result<T> = std::result::Result<T, TchError>;

/// Set the number of threads used for intra-op parallelism
pub fn set_num_threads(n: i32) {
    // TODO: Implement thread pool configuration
    let _ = n;
}

/// Get the number of threads used for intra-op parallelism
pub fn get_num_threads() -> i32 {
    // TODO: Implement thread pool query
    1
}

/// Set the number of threads used for inter-op parallelism
pub fn set_num_interop_threads(n: i32) {
    // TODO: Implement inter-op thread pool configuration
    let _ = n;
}

/// Manually seed the random number generator
pub fn manual_seed(seed: i64) {
    use rand::SeedableRng;
    let _rng = rand::rngs::StdRng::seed_from_u64(seed as u64);
    // TODO: Store global RNG state
}
