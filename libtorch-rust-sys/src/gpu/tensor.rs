use super::{buffer::GpuBuffer, device::GpuDevice, shaders};
use crate::error::{Result, TchError};
use std::sync::Arc;
use wgpu::{self, util::DeviceExt};

/// GPU-accelerated tensor
///
/// All operations are performed on the GPU using WebGPU compute shaders.
pub struct GpuTensor {
    buffer: GpuBuffer,
    shape: Vec<usize>,
    numel: usize,
    device: Arc<GpuDevice>,
}

impl GpuTensor {
    /// Create a new GPU tensor filled with zeros
    pub fn zeros(shape: &[usize], device: Arc<GpuDevice>) -> Self {
        let numel: usize = shape.iter().product();
        let size_bytes = numel * std::mem::size_of::<f32>();

        let buffer = GpuBuffer::new(
            &device,
            size_bytes,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        );

        // Initialize with zeros
        let zeros = vec![0.0f32; numel];
        buffer.write(device.queue(), &zeros);

        GpuTensor {
            buffer,
            shape: shape.to_vec(),
            numel,
            device,
        }
    }

    /// Create a GPU tensor from CPU data
    pub fn from_slice(data: &[f32], shape: &[usize], device: Arc<GpuDevice>) -> Result<Self> {
        let numel: usize = shape.iter().product();
        if data.len() != numel {
            return Err(TchError::ShapeError(format!(
                "Data length {} doesn't match shape {:?} (expected {})",
                data.len(),
                shape,
                numel
            )));
        }

        let buffer = GpuBuffer::from_slice(&device, data);

        Ok(GpuTensor {
            buffer,
            shape: shape.to_vec(),
            numel,
            device,
        })
    }

    /// Download tensor data from GPU to CPU
    pub async fn to_vec(&self) -> Vec<f32> {
        self.buffer.read().await
    }

    /// Get tensor shape
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get number of elements
    pub fn numel(&self) -> usize {
        self.numel
    }

    /// Element-wise addition on GPU
    pub async fn add(&self, other: &GpuTensor) -> Result<GpuTensor> {
        self.elementwise_binary_op(other, shaders::ELEMENTWISE_ADD, "main")
            .await
    }

    /// Element-wise multiplication on GPU
    pub async fn mul(&self, other: &GpuTensor) -> Result<GpuTensor> {
        self.elementwise_binary_op(other, shaders::ELEMENTWISE_MUL, "main")
            .await
    }

    /// Element-wise subtraction on GPU
    pub async fn sub(&self, other: &GpuTensor) -> Result<GpuTensor> {
        self.elementwise_binary_op(other, shaders::ELEMENTWISE_SUB, "main")
            .await
    }

    /// Element-wise division on GPU
    pub async fn div(&self, other: &GpuTensor) -> Result<GpuTensor> {
        self.elementwise_binary_op(other, shaders::ELEMENTWISE_DIV, "main")
            .await
    }

    /// ReLU activation on GPU
    pub async fn relu(&self) -> Result<GpuTensor> {
        self.elementwise_unary_op(shaders::RELU, "main").await
    }

    /// Sigmoid activation on GPU
    pub async fn sigmoid(&self) -> Result<GpuTensor> {
        self.elementwise_unary_op(shaders::SIGMOID, "main").await
    }

    /// Matrix multiplication on GPU
    pub async fn matmul(&self, other: &GpuTensor) -> Result<GpuTensor> {
        // Verify shapes are compatible for matmul
        if self.shape.len() != 2 || other.shape.len() != 2 {
            return Err(TchError::ShapeError(
                "matmul requires 2D tensors".to_string(),
            ));
        }

        let m = self.shape[0] as u32;
        let k = self.shape[1] as u32;
        let k2 = other.shape[0] as u32;
        let n = other.shape[1] as u32;

        if k != k2 {
            return Err(TchError::ShapeError(format!(
                "Matrix dimension mismatch: ({}, {}) x ({}, {})",
                m, k, k2, n
            )));
        }

        // Create output buffer
        let output_numel = (m * n) as usize;
        let output = GpuTensor::zeros(&[m as usize, n as usize], Arc::clone(&self.device));

        // Create dimensions uniform buffer
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Dimensions {
            m: u32,
            n: u32,
            k: u32,
            _padding: u32,
        }

        let dims = Dimensions {
            m,
            n,
            k,
            _padding: 0,
        };

        let dims_buffer = self.device.device().create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Dimensions Buffer"),
                contents: bytemuck::cast_slice(&[dims]),
                usage: wgpu::BufferUsages::UNIFORM,
            },
        );

        // Create compute pipeline
        let shader = self
            .device
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("MatMul Shader"),
                source: wgpu::ShaderSource::Wgsl(shaders::MATMUL_SIMPLE.into()),
            });

        let bind_group_layout =
            self.device
                .device()
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("MatMul Bind Group Layout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        let pipeline_layout =
            self.device
                .device()
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("MatMul Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

        let pipeline = self
            .device
            .device()
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("MatMul Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "main",
            });

        // Create bind group
        let bind_group = self
            .device
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("MatMul Bind Group"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.buffer.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: other.buffer.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: output.buffer.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: dims_buffer.as_entire_binding(),
                    },
                ],
            });

        // Execute compute shader
        let mut encoder = self
            .device
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("MatMul Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("MatMul Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Dispatch workgroups (16x16 workgroup size)
            let workgroups_x = (n + 15) / 16;
            let workgroups_y = (m + 15) / 16;
            compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        self.device.queue().submit(Some(encoder.finish()));

        Ok(output)
    }

    /// Helper for element-wise binary operations
    async fn elementwise_binary_op(
        &self,
        other: &GpuTensor,
        shader_source: &str,
        entry_point: &str,
    ) -> Result<GpuTensor> {
        if self.shape != other.shape {
            return Err(TchError::ShapeError(format!(
                "Shape mismatch: {:?} vs {:?}",
                self.shape, other.shape
            )));
        }

        // Create output buffer
        let output = GpuTensor::zeros(&self.shape, Arc::clone(&self.device));

        // Create shader and pipeline
        let shader = self
            .device
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Elementwise Binary Op Shader"),
                source: wgpu::ShaderSource::Wgsl(shader_source.into()),
            });

        let bind_group_layout =
            self.device
                .device()
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Elementwise Bind Group Layout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        let pipeline_layout =
            self.device
                .device()
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Elementwise Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

        let pipeline = self
            .device
            .device()
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Elementwise Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point,
            });

        // Create bind group
        let bind_group = self
            .device
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Elementwise Bind Group"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.buffer.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: other.buffer.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: output.buffer.buffer().as_entire_binding(),
                    },
                ],
            });

        // Execute compute shader
        let mut encoder = self
            .device
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Elementwise Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Elementwise Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Dispatch workgroups (256 threads per group)
            let workgroups = (self.numel as u32 + 255) / 256;
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }

        self.device.queue().submit(Some(encoder.finish()));

        Ok(output)
    }

    /// Helper for element-wise unary operations
    async fn elementwise_unary_op(
        &self,
        shader_source: &str,
        entry_point: &str,
    ) -> Result<GpuTensor> {
        // Create output buffer
        let output = GpuTensor::zeros(&self.shape, Arc::clone(&self.device));

        // Create shader and pipeline
        let shader = self
            .device
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Elementwise Unary Op Shader"),
                source: wgpu::ShaderSource::Wgsl(shader_source.into()),
            });

        let bind_group_layout =
            self.device
                .device()
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Unary Bind Group Layout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        let pipeline_layout =
            self.device
                .device()
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Unary Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

        let pipeline = self
            .device
            .device()
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Unary Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point,
            });

        // Create bind group
        let bind_group = self
            .device
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Unary Bind Group"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.buffer.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: output.buffer.buffer().as_entire_binding(),
                    },
                ],
            });

        // Execute compute shader
        let mut encoder = self
            .device
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Unary Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Unary Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Dispatch workgroups (256 threads per group)
            let workgroups = (self.numel as u32 + 255) / 256;
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }

        self.device.queue().submit(Some(encoder.finish()));

        Ok(output)
    }
}
