use crate::nn::Module;
use crate::Tensor;

/// Sequential container for composing modules
pub struct Sequential {
    layers: Vec<Box<dyn Fn(&Tensor) -> Tensor>>,
}

impl Sequential {
    /// Create a new sequential container
    pub fn new() -> Self {
        Sequential { layers: Vec::new() }
    }

    /// Add a function to the sequence
    pub fn add_fn<F>(mut self, f: F) -> Self
    where
        F: Fn(&Tensor) -> Tensor + 'static,
    {
        self.layers.push(Box::new(f));
        self
    }
}

impl Default for Sequential {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for Sequential {
    fn forward(&self, mut input: &Tensor) -> Tensor {
        let mut current = input.shallow_clone();
        for layer in &self.layers {
            current = layer(&current);
        }
        current
    }
}

/// Convenient function to create a sequential container
pub fn seq() -> Sequential {
    Sequential::new()
}
