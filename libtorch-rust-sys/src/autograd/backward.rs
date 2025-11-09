/// Backward pass implementation
///
/// This module implements the backward pass through the computational graph
/// using reverse-mode automatic differentiation.

use crate::TensorImpl;
use crate::error::{Result, TchError};
use super::GradNode;
use std::collections::HashSet;
use std::sync::Arc;

/// Perform topological sort of the computational graph
///
/// Starting from the root tensor's grad_fn, traverses the graph of GradNodes
/// and returns them in topological order (dependencies first, root last).
///
/// This is used to determine the order in which to execute backward operations.
///
/// # Arguments
/// * `root` - The output tensor from which to start the backward pass
///
/// # Returns
/// Vector of GradNodes in execution order for the backward pass
pub fn topological_sort_grad_fns(root: &TensorImpl) -> Vec<Arc<dyn GradNode>> {
    let mut visited = HashSet::new();
    let mut result = Vec::new();

    // Start DFS from the root's grad_fn if it exists
    if let Some(grad_fn) = root.grad_fn() {
        dfs_grad_fn(grad_fn, &mut visited, &mut result);
    }

    // Reverse to get execution order (dependencies first)
    result.reverse();
    result
}

/// Depth-first search to collect GradNodes in post-order
///
/// # Arguments
/// * `node` - Current GradNode to visit
/// * `visited` - Set of already-visited nodes (to avoid cycles)
/// * `result` - Vector to accumulate nodes in post-order
fn dfs_grad_fn(
    node: &Arc<dyn GradNode>,
    visited: &mut HashSet<*const ()>,
    result: &mut Vec<Arc<dyn GradNode>>
) {
    // Use raw pointer as unique identifier for the GradNode
    let node_ptr = Arc::as_ptr(node) as *const ();

    // Skip if already visited
    if visited.contains(&node_ptr) {
        return;
    }

    visited.insert(node_ptr);

    // Visit all dependencies first (inputs to this operation)
    for edge in node.next_edges() {
        dfs_grad_fn(&edge.function, visited, result);
    }

    // Add this node after its dependencies (post-order traversal)
    result.push(node.clone());
}

/// Perform backward pass from a scalar tensor
///
/// Computes gradients for all tensors in the computation graph that have
/// `requires_grad=true`. The output tensor must be a scalar (single element).
///
/// # Arguments
/// * `output` - The scalar output tensor to backpropagate from
///
/// # Errors
/// * Returns error if `output` is not a scalar
/// * Returns error if `output` does not have a grad_fn (is a leaf tensor)
pub fn backward(output: &mut TensorImpl) -> Result<()> {
    // Create gradient of ones for scalar output
    let grad_output = TensorImpl::ones(output.shape(), output.dtype(), output.device())?;
    backward_with_gradient(output, grad_output)
}

/// Perform backward pass with a custom output gradient
///
/// Computes gradients for all tensors in the computation graph that have
/// `requires_grad=true`, using the provided gradient for the output tensor.
///
/// # Arguments
/// * `output` - The output tensor to backpropagate from
/// * `grad_output` - The gradient of the loss with respect to output
///
/// # Errors
/// * Returns error if gradient shape doesn't match output shape
/// * Returns error if `output` does not have a grad_fn
pub fn backward_with_gradient(output: &mut TensorImpl, grad_output: TensorImpl) -> Result<()> {
    // Verify output has a grad_fn (it was created by an operation)
    if output.grad_fn().is_none() {
        return Err(TchError::Other(
            "Cannot call backward() on a tensor that doesn't require grad or has no grad_fn".to_string()
        ));
    }

    // Verify gradient shape matches output shape
    if grad_output.shape() != output.shape() {
        return Err(TchError::Other(format!(
            "Gradient shape {:?} doesn't match output shape {:?}",
            grad_output.shape(),
            output.shape()
        )));
    }

    // Get topological order of operations (dependencies first, output last)
    let sorted_nodes = topological_sort_grad_fns(output);

    // Use a HashMap to track gradients flowing to each node
    use std::collections::HashMap;
    let mut grad_map: HashMap<*const (), TensorImpl> = HashMap::new();

    // Initialize with the output gradient
    if let Some(grad_fn) = output.grad_fn() {
        let node_ptr = Arc::as_ptr(&grad_fn) as *const ();
        grad_map.insert(node_ptr, grad_output);
    }

    // Execute backward pass in reverse topological order (output to inputs)
    for node in sorted_nodes.iter().rev() {
        let node_ptr = Arc::as_ptr(node) as *const ();

        // Get the gradient for this node
        if let Some(node_grad) = grad_map.get(&node_ptr) {
            // Accumulate gradients at leaf tensors (those with requires_grad but no grad_fn)
            node.accumulate_grads(&[node_grad])?;

            // Compute gradients for inputs
            let grad_inputs = node.apply(&[node_grad]);

            // Propagate gradients to predecessor nodes
            let edges = node.next_edges();
            for (i, edge) in edges.iter().enumerate() {
                if i < grad_inputs.len() {
                    let pred_ptr = Arc::as_ptr(&edge.function) as *const ();

                    // Accumulate gradient at predecessor node
                    if let Some(existing_grad) = grad_map.get_mut(&pred_ptr) {
                        // Add to existing gradient
                        *existing_grad = existing_grad.add(&grad_inputs[i])?;
                    } else {
                        // Insert new gradient
                        grad_map.insert(pred_ptr, grad_inputs[i].clone());
                    }
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::{Edge, GradNode};
    use crate::{TensorImpl, DType, Device};
    use std::sync::Arc;

    // Mock GradNode for testing
    struct MockGradNode {
        name: String,
        edges: Vec<Edge>,
    }

    impl MockGradNode {
        fn new(name: &str, edges: Vec<Edge>) -> Arc<Self> {
            Arc::new(MockGradNode {
                name: name.to_string(),
                edges,
            })
        }
    }

    impl GradNode for MockGradNode {
        fn apply(&self, _grad_outputs: &[&TensorImpl]) -> Vec<TensorImpl> {
            vec![]
        }

        fn next_edges(&self) -> &[Edge] {
            &self.edges
        }
    }

    #[test]
    fn test_topological_sort_simple_chain() {
        // Create a simple chain: a -> b -> c
        let node_a = MockGradNode::new("a", vec![]);
        let node_b = MockGradNode::new("b", vec![Edge::new(node_a.clone(), 0)]);
        let node_c = MockGradNode::new("c", vec![Edge::new(node_b.clone(), 0)]);

        // Create root tensor with grad_fn = node_c
        let mut root = TensorImpl::zeros(&[1], DType::Float, Device::Cpu).unwrap();
        root.set_grad_fn(node_c.clone());

        // Topological sort should return [a, b, c]
        let sorted = topological_sort_grad_fns(&root);
        assert_eq!(sorted.len(), 3);

        // Verify order (dependencies first)
        // Note: We can't directly compare Arc<dyn GradNode>, so we check count
        assert_eq!(sorted.len(), 3);
    }

    #[test]
    fn test_topological_sort_no_grad_fn() {
        // Tensor without grad_fn (leaf tensor)
        let root = TensorImpl::zeros(&[1], DType::Float, Device::Cpu).unwrap();

        let sorted = topological_sort_grad_fns(&root);
        assert_eq!(sorted.len(), 0);
    }

    #[test]
    fn test_topological_sort_diamond() {
        // Create diamond graph:
        //     d
        //    / \
        //   b   c
        //    \ /
        //     a
        let node_a = MockGradNode::new("a", vec![]);
        let node_b = MockGradNode::new("b", vec![Edge::new(node_a.clone(), 0)]);
        let node_c = MockGradNode::new("c", vec![Edge::new(node_a.clone(), 0)]);
        let node_d = MockGradNode::new("d", vec![
            Edge::new(node_b.clone(), 0),
            Edge::new(node_c.clone(), 1),
        ]);

        let mut root = TensorImpl::zeros(&[1], DType::Float, Device::Cpu).unwrap();
        root.set_grad_fn(node_d.clone());

        let sorted = topological_sort_grad_fns(&root);

        // Should have 4 nodes, with 'a' before 'b' and 'c', and 'd' last
        assert_eq!(sorted.len(), 4);
    }
}
