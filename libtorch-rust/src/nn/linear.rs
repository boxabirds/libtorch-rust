use crate::nn::{Init, Module, Path};
use crate::Tensor;

/// Linear (fully connected) layer
pub struct Linear {
    pub ws: Tensor,
    pub bs: Option<Tensor>,
}

impl Linear {
    /// Create a new linear layer
    pub fn new(vs: &Path, in_dim: i64, out_dim: i64, bias: bool) -> Self {
        let ws = vs.var("weight", &[out_dim, in_dim], Init::KaimingUniform);
        let bs = if bias {
            Some(vs.var("bias", &[out_dim], Init::Const(0.0)))
        } else {
            None
        };

        Linear { ws, bs }
    }
}

impl Module for Linear {
    fn forward(&self, input: &Tensor) -> Tensor {
        let output = input.matmul(&self.ws.tr());

        if let Some(ref bias) = self.bs {
            // Broadcast bias across batch dimension
            &output + bias
        } else {
            output
        }
    }
}

/// Convenient function to create a linear layer
pub fn linear(vs: &Path, in_dim: i64, out_dim: i64) -> Linear {
    Linear::new(vs, in_dim, out_dim, true)
}
