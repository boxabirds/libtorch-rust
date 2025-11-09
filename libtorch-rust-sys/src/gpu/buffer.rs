use super::device::GpuDevice;
use bytemuck;
use std::sync::Arc;
use wgpu::{self, util::DeviceExt};

/// GPU buffer wrapper for tensor data
pub struct GpuBuffer {
    buffer: wgpu::Buffer,
    size: usize,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
}

impl GpuBuffer {
    /// Create a new GPU buffer with the given size (in bytes)
    pub fn new(device: &GpuDevice, size: usize, usage: wgpu::BufferUsages) -> Self {
        let buffer = device.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tensor GPU Buffer"),
            size: size as u64,
            usage,
            mapped_at_creation: false,
        });

        GpuBuffer {
            buffer,
            size,
            device: Arc::clone(&device.device),
            queue: Arc::clone(&device.queue),
        }
    }

    /// Create a GPU buffer from a slice of data
    pub fn from_slice<T: bytemuck::Pod>(device: &GpuDevice, data: &[T]) -> Self {
        let bytes = bytemuck::cast_slice(data);
        let buffer = device
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Tensor GPU Buffer (Init)"),
                contents: bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            });

        GpuBuffer {
            buffer,
            size: bytes.len(),
            device: Arc::clone(&device.device),
            queue: Arc::clone(&device.queue),
        }
    }

    /// Upload data to GPU buffer
    pub fn write<T: bytemuck::Pod>(&self, queue: &wgpu::Queue, data: &[T]) {
        let bytes = bytemuck::cast_slice(data);
        assert!(
            bytes.len() <= self.size,
            "Data size exceeds buffer capacity"
        );
        queue.write_buffer(&self.buffer, 0, bytes);
    }

    /// Download data from GPU buffer
    pub async fn read<T: bytemuck::Pod>(&self) -> Vec<T> {
        let size = self.size;

        // Create staging buffer for readback
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: size as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create command encoder
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Copy Encoder"),
            });

        // Copy from GPU buffer to staging buffer
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &staging_buffer, 0, size as u64);

        // Submit commands
        self.queue.submit(Some(encoder.finish()));

        // Map buffer for reading
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures::channel::oneshot::channel();

        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });

        // Wait for GPU to finish and map completion
        // wgpu 27+ automatically processes pending operations when awaiting
        receiver.await.unwrap().unwrap();

        // Read data
        let data = buffer_slice.get_mapped_range();
        let result: Vec<T> = bytemuck::cast_slice(&data).to_vec();

        drop(data);
        staging_buffer.unmap();

        result
    }

    /// Get the underlying wgpu buffer
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    /// Get buffer size in bytes
    pub fn size(&self) -> usize {
        self.size
    }
}
