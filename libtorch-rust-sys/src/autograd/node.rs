use crate::TensorImpl;
use std::sync::Arc;

/// Node in the computational graph representing an operation
///
/// Each backward operation (e.g., AddBackward, MulBackward) implements this trait
/// to define how gradients flow backward through that operation.
pub trait GradNode: Send + Sync {
    /// Apply gradient computation (backward pass)
    ///
    /// Takes gradients flowing backward from the output and computes
    /// gradients with respect to the inputs.
    ///
    /// # Arguments
    /// * `grad_outputs` - Gradients flowing backward from the outputs
    ///
    /// # Returns
    /// Gradients with respect to each input of this operation
    fn apply(&self, grad_outputs: &[&TensorImpl]) -> Vec<TensorImpl>;

    /// Get the edges connecting to input tensors
    fn next_edges(&self) -> &[super::Edge];
}
