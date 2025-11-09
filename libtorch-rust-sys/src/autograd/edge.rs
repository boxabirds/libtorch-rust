use std::sync::Arc;
use super::GradNode;

/// Edge connecting a tensor to the operation that created it
///
/// Represents a connection in the computational graph from a tensor
/// back to the GradNode that produced it.
#[derive(Clone)]
pub struct Edge {
    /// The gradient function (operation) that produced the tensor
    pub function: Arc<dyn GradNode>,

    /// Which input number this edge corresponds to (for operations with multiple inputs)
    pub input_nr: usize,
}

impl Edge {
    pub fn new(function: Arc<dyn GradNode>, input_nr: usize) -> Self {
        Edge { function, input_nr }
    }
}
