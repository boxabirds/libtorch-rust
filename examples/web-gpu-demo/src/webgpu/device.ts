/**
 * WebGPU Device Initialization
 *
 * Handles GPU device detection and initialization in the browser.
 */

export interface GpuDeviceInfo {
  adapter: GPUAdapter;
  device: GPUDevice;
  info: {
    vendor: string;
    architecture: string;
    device: string;
    description: string;
  };
}

export class WebGPUNotSupportedError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'WebGPUNotSupportedError';
  }
}

export async function initializeGPU(): Promise<GpuDeviceInfo> {
  // Check if WebGPU is supported
  if (!navigator.gpu) {
    throw new WebGPUNotSupportedError(
      'WebGPU is not supported in this browser. Try Chrome 113+, Edge 113+, or Safari 18+'
    );
  }

  // Request adapter with high performance preference
  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: 'high-performance',
  });

  if (!adapter) {
    throw new WebGPUNotSupportedError(
      'Failed to get WebGPU adapter. Your GPU may not support WebGPU.'
    );
  }

  // Get adapter info
  const adapterInfo = await adapter.requestAdapterInfo();

  // Request device
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxBufferSize: adapter.limits.maxBufferSize,
      maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
    },
  });

  // Handle device lost
  device.lost.then((info) => {
    console.error('WebGPU device was lost:', info.message);
    if (info.reason !== 'destroyed') {
      console.error('Device loss reason:', info.reason);
    }
  });

  // Handle uncaptured errors
  device.onuncapturederror = (event) => {
    console.error('Uncaptured WebGPU error:', event.error);
  };

  return {
    adapter,
    device,
    info: {
      vendor: adapterInfo.vendor || 'Unknown',
      architecture: adapterInfo.architecture || 'Unknown',
      device: adapterInfo.device || 'Unknown',
      description: adapterInfo.description || 'Unknown',
    },
  };
}

export function isWebGPUSupported(): boolean {
  return 'gpu' in navigator;
}
