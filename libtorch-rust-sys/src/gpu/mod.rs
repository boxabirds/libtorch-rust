/// GPU backend using WebGPU (wgpu)
///
/// This module provides GPU acceleration for tensor operations using WebGPU,
/// which works across all platforms (Vulkan, Metal, DX12, WebGPU in browser).

pub mod device;
pub mod buffer;
pub mod shaders;
pub mod tensor;

pub use device::{GpuDevice, GpuDeviceError};
pub use buffer::GpuBuffer;
pub use tensor::GpuTensor;
