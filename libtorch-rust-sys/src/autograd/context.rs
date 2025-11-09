use std::cell::Cell;

thread_local! {
    /// Global gradient mode - controls whether operations record history
    static GRAD_MODE: Cell<bool> = Cell::new(true);
}

/// Check if gradient recording is currently enabled
pub fn is_grad_enabled() -> bool {
    GRAD_MODE.with(|mode| mode.get())
}

/// Set gradient recording mode globally
pub fn set_grad_enabled(enabled: bool) {
    GRAD_MODE.with(|mode| mode.set(enabled));
}

/// RAII guard that disables gradient tracking
///
/// Gradients are automatically re-enabled when the guard is dropped.
///
/// # Example
/// ```
/// use libtorch_rust_sys::autograd::{is_grad_enabled, NoGradGuard};
///
/// assert_eq!(is_grad_enabled(), true);
///
/// {
///     let _guard = NoGradGuard::new();
///     assert_eq!(is_grad_enabled(), false);
/// }
///
/// assert_eq!(is_grad_enabled(), true);
/// ```
pub struct NoGradGuard {
    prev_mode: bool,
}

impl NoGradGuard {
    pub fn new() -> Self {
        let prev_mode = is_grad_enabled();
        set_grad_enabled(false);
        NoGradGuard { prev_mode }
    }
}

impl Drop for NoGradGuard {
    fn drop(&mut self) {
        set_grad_enabled(self.prev_mode);
    }
}

impl Default for NoGradGuard {
    fn default() -> Self {
        Self::new()
    }
}
