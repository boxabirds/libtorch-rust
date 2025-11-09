use std::fmt;

/// Device specification for tensor operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Device {
    /// CPU device
    Cpu,
    /// CUDA device with specified index
    Cuda(usize),
}

impl Device {
    /// Create a CPU device
    pub fn cpu() -> Self {
        Device::Cpu
    }

    /// Create a CUDA device with index 0
    pub fn cuda() -> Self {
        Device::Cuda(0)
    }

    /// Create a CUDA device with specified index
    pub fn cuda_with_index(index: usize) -> Self {
        Device::Cuda(index)
    }

    /// Returns CUDA device if available, otherwise CPU
    pub fn cuda_if_available() -> Self {
        if Self::is_cuda_available() {
            Device::Cuda(0)
        } else {
            Device::Cpu
        }
    }

    /// Check if CUDA is available
    pub fn is_cuda_available() -> bool {
        // TODO: Implement actual CUDA availability check
        // For now, return false as we're building a pure Rust implementation
        false
    }

    /// Returns true if this is a CPU device
    pub fn is_cpu(&self) -> bool {
        matches!(self, Device::Cpu)
    }

    /// Returns true if this is a CUDA device
    pub fn is_cuda(&self) -> bool {
        matches!(self, Device::Cuda(_))
    }
}

impl Default for Device {
    fn default() -> Self {
        Device::Cpu
    }
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Device::Cpu => write!(f, "cpu"),
            Device::Cuda(idx) => write!(f, "cuda:{}", idx),
        }
    }
}
