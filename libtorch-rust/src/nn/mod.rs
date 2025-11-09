pub mod var_store;
pub mod module;
pub mod linear;
pub mod optimizer;
pub mod init;
pub mod sequential;

pub use var_store::{Path, VarStore};
pub use module::Module;
pub use linear::{Linear, linear};
pub use optimizer::{Optimizer, OptimizerConfig, Sgd, Adam};
pub use init::Init;
pub use sequential::Sequential;

/// Functional API for neural network operations
pub mod func {
    use crate::Tensor;

    /// Cross entropy loss for logits
    pub fn cross_entropy_for_logits(logits: &Tensor, targets: &Tensor) -> Tensor {
        // TODO: Implement proper cross entropy
        // For now, return a placeholder
        let _ = (logits, targets);
        Tensor::zeros(&[1], crate::Kind::Float, crate::Device::Cpu)
    }

    /// Mean squared error loss
    pub fn mse_loss(pred: &Tensor, target: &Tensor) -> Tensor {
        let diff = &(pred - target);
        let squared = &(diff * diff);
        let mean = squared.mean(crate::Kind::Float);
        mean
    }
}
