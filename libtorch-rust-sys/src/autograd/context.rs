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
/// let x = Tensor::ones(&[2, 2], Kind::Float, Device::Cpu).set_requires_grad(true);
///
/// {
///     let _guard = NoGradGuard::new();
///     let y = &x + &x;  // This operation won't be recorded
/// }
///
/// let z = &x + &x;  // This operation will be recorded
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
