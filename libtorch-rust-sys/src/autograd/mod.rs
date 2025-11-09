/// Autograd functionality for automatic differentiation
pub mod node;
pub mod edge;
pub mod context;
pub mod backward;

pub use node::GradNode;
pub use edge::Edge;
pub use context::{is_grad_enabled, set_grad_enabled, NoGradGuard};
