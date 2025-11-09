use crate::Tensor;

/// Trait for neural network modules
pub trait Module {
    /// Forward pass
    fn forward(&self, input: &Tensor) -> Tensor;
}
