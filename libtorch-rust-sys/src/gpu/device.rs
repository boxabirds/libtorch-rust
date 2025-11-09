use std::sync::Arc;
use wgpu;

/// Errors that can occur when working with GPU devices
#[derive(Debug, thiserror::Error)]
pub enum GpuDeviceError {
    #[error("Failed to request GPU adapter: {0}")]
    AdapterRequest(String),

    #[error("Failed to request GPU device: {0}")]
    DeviceRequest(String),

    #[error("GPU operation failed: {0}")]
    OperationFailed(String),
}

/// WebGPU device wrapper
///
/// This provides a unified interface for GPU operations that works across:
/// - Native: Vulkan, Metal, DX12
/// - Web: WebGPU in browser
pub struct GpuDevice {
    pub(crate) device: Arc<wgpu::Device>,
    pub(crate) queue: Arc<wgpu::Queue>,
    adapter_info: wgpu::AdapterInfo,
}

impl GpuDevice {
    /// Create a new GPU device
    ///
    /// This will automatically select the best available GPU backend:
    /// - Windows: DX12 or Vulkan
    /// - macOS/iOS: Metal
    /// - Linux: Vulkan
    /// - Web: WebGPU
    pub async fn new() -> Result<Self, GpuDeviceError> {
        Self::new_with_backend(wgpu::Backends::all()).await
    }

    /// Create a GPU device with specific backend
    pub async fn new_with_backend(backends: wgpu::Backends) -> Result<Self, GpuDeviceError> {
        // Create instance
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends,
            ..Default::default()
        });

        // Request adapter (GPU)
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| {
                GpuDeviceError::AdapterRequest("No suitable GPU adapter found".to_string())
            })?;

        let adapter_info = adapter.get_info();
        println!("Using GPU: {} ({:?})", adapter_info.name, adapter_info.backend);

        // Request device and queue
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("LibTorch-Rust GPU Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .map_err(|e| GpuDeviceError::DeviceRequest(e.to_string()))?;

        Ok(GpuDevice {
            device: Arc::new(device),
            queue: Arc::new(queue),
            adapter_info,
        })
    }

    /// Get device info
    pub fn info(&self) -> &wgpu::AdapterInfo {
        &self.adapter_info
    }

    /// Get the underlying wgpu device
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// Get the command queue
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// Create a compute pipeline from WGSL shader source
    pub fn create_compute_pipeline(
        &self,
        shader_source: &str,
        entry_point: &str,
    ) -> wgpu::ComputePipeline {
        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Compute Pipeline Layout"),
                bind_group_layouts: &[],
                push_constant_ranges: &[],
            });

        self.device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Compute Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point,
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_device_creation() {
        // This test requires a GPU, so we'll make it optional
        pollster::block_on(async {
            match GpuDevice::new().await {
                Ok(device) => {
                    println!("GPU device created successfully!");
                    println!("Device info: {:?}", device.info());
                }
                Err(e) => {
                    println!("GPU not available (this is OK): {}", e);
                }
            }
        });
    }
}
