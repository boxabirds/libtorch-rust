/// Backward operations for basic arithmetic
///
/// Each operation (Add, Sub, Mul, Div) has a corresponding backward node
/// that computes gradients during the backward pass.

use crate::autograd::{Edge, GradNode};
use crate::TensorImpl;
use crate::error::Result;
use std::sync::Arc;

/// Backward operation for multiplication: z = x * y
///
/// Gradient formulas:
/// - dL/dx = dL/dz * y
/// - dL/dy = dL/dz * x
pub struct MulBackward {
    /// Value of the other operand (for computing gradient)
    /// For z = x * y, when computing dL/dx, we need y
    x_value: TensorImpl,
    y_value: TensorImpl,

    /// Pointers to original input tensors (for gradient accumulation)
    /// These are raw pointers because we can't store Arc/Rc references
    /// Safety: These pointers are only used during backward(), which is called
    /// while the original tensors are still alive
    x_ptr: *mut TensorImpl,
    y_ptr: *mut TensorImpl,

    /// Edges to the inputs' grad_fns
    next_edges: Vec<Edge>,
}

// Safety: The raw pointers are only dereferenced during backward(),
// which is called while the original tensors are still alive.
// The backward pass is synchronous and completes before the tensors are dropped.
unsafe impl Send for MulBackward {}
unsafe impl Sync for MulBackward {}

impl MulBackward {
    pub fn new(x: &TensorImpl, y: &TensorImpl) -> Arc<Self> {
        let mut edges = Vec::new();

        // Add edge for x if it has a grad_fn
        if let Some(x_grad_fn) = x.grad_fn() {
            edges.push(Edge::new(x_grad_fn.clone(), 0));
        }

        // Add edge for y if it has a grad_fn
        if let Some(y_grad_fn) = y.grad_fn() {
            edges.push(Edge::new(y_grad_fn.clone(), edges.len()));
        }

        Arc::new(MulBackward {
            x_value: x.clone(),
            y_value: y.clone(),
            x_ptr: x as *const TensorImpl as *mut TensorImpl,
            y_ptr: y as *const TensorImpl as *mut TensorImpl,
            next_edges: edges,
        })
    }
}

impl GradNode for MulBackward {
    fn apply(&self, grad_outputs: &[&TensorImpl]) -> Vec<TensorImpl> {
        assert_eq!(grad_outputs.len(), 1, "MulBackward expects 1 grad_output");
        let grad_output = grad_outputs[0];

        let mut grad_inputs = Vec::new();

        // Compute dL/dx = dL/dz * y
        if self.x_value.requires_grad() {
            let grad_x = grad_output.mul(&self.y_value)
                .expect("Failed to compute gradient for x in MulBackward");
            grad_inputs.push(grad_x);
        }

        // Compute dL/dy = dL/dz * x
        if self.y_value.requires_grad() {
            let grad_y = grad_output.mul(&self.x_value)
                .expect("Failed to compute gradient for y in MulBackward");
            grad_inputs.push(grad_y);
        }

        grad_inputs
    }

    fn next_edges(&self) -> &[Edge] {
        &self.next_edges
    }

    fn accumulate_grads(&self, grad_outputs: &[&TensorImpl]) -> Result<()> {
        assert_eq!(grad_outputs.len(), 1, "MulBackward expects 1 grad_output");
        let grad_output = grad_outputs[0];

        // Safety: These pointers are valid during the backward pass
        // because the original tensors outlive the backward call
        unsafe {
            // Only accumulate at leaf tensors (those without grad_fn)
            // Intermediate tensors will have their gradients computed by
            // traversing through their own grad_fn

            // Accumulate gradient for x: dL/dx = dL/dz * y
            if self.x_value.requires_grad() && self.x_value.grad_fn().is_none() && !self.x_ptr.is_null() {
                let grad_x = grad_output.mul(&self.y_value)?;
                (*self.x_ptr).accumulate_grad(grad_x)?;
            }

            // Accumulate gradient for y: dL/dy = dL/dz * x
            if self.y_value.requires_grad() && self.y_value.grad_fn().is_none() && !self.y_ptr.is_null() {
                let grad_y = grad_output.mul(&self.x_value)?;
                (*self.y_ptr).accumulate_grad(grad_y)?;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DType, Device};

    #[test]
    fn test_mul_backward_gradients() {
        // z = x * y, where x=2, y=3
        let x = TensorImpl::from_slice_f32(&[2.0], &[1]).unwrap();
        let y = TensorImpl::from_slice_f32(&[3.0], &[1]).unwrap();

        let mul_backward = MulBackward::new(&x, &y);

        // Gradient flowing backward: dL/dz = 1.0
        let grad_output = TensorImpl::from_slice_f32(&[1.0], &[1]).unwrap();

        let grad_inputs = mul_backward.apply(&[&grad_output]);

        // Since neither x nor y have requires_grad set, grad_inputs should be empty
        assert_eq!(grad_inputs.len(), 0);
    }

    #[test]
    fn test_mul_backward_with_requires_grad() {
        // z = x * y, where x=2, y=3, both require grad
        let mut x = TensorImpl::from_slice_f32(&[2.0], &[1]).unwrap();
        let mut y = TensorImpl::from_slice_f32(&[3.0], &[1]).unwrap();

        x.set_requires_grad(true);
        y.set_requires_grad(true);

        let mul_backward = MulBackward::new(&x, &y);

        // Gradient flowing backward: dL/dz = 1.0
        let grad_output = TensorImpl::from_slice_f32(&[1.0], &[1]).unwrap();

        let grad_inputs = mul_backward.apply(&[&grad_output]);

        // Should have 2 gradients (one for x, one for y)
        assert_eq!(grad_inputs.len(), 2);

        // dL/dx = dL/dz * y = 1.0 * 3.0 = 3.0
        let grad_x_data = grad_inputs[0].to_vec_f64();
        assert_eq!(grad_x_data[0], 3.0);

        // dL/dy = dL/dz * x = 1.0 * 2.0 = 2.0
        let grad_y_data = grad_inputs[1].to_vec_f64();
        assert_eq!(grad_y_data[0], 2.0);
    }
}
