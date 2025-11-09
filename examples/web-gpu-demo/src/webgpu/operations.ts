/**
 * WebGPU Tensor Operations
 *
 * High-level API for GPU-accelerated tensor operations in the browser.
 */

import * as shaders from './shaders';

export interface OperationResult {
  data: Float32Array;
  timeMs: number;
  throughput?: number; // operations per second
  gflops?: number;
}

function createBuffer(
  device: GPUDevice,
  data: Float32Array,
  usage: GPUBufferUsageFlags
): GPUBuffer {
  const buffer = device.createBuffer({
    size: data.byteLength,
    usage,
    mappedAtCreation: true,
  });

  new Float32Array(buffer.getMappedRange()).set(data);
  buffer.unmap();

  return buffer;
}

async function readBuffer(device: GPUDevice, buffer: GPUBuffer): Promise<Float32Array> {
  const size = buffer.size;
  const stagingBuffer = device.createBuffer({
    size,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  const commandEncoder = device.createCommandEncoder();
  commandEncoder.copyBufferToBuffer(buffer, 0, stagingBuffer, 0, size);
  device.queue.submit([commandEncoder.finish()]);

  await stagingBuffer.mapAsync(GPUMapMode.READ);
  const result = new Float32Array(stagingBuffer.getMappedRange().slice(0));
  stagingBuffer.unmap();
  stagingBuffer.destroy();

  return result;
}

function createComputePipeline(
  device: GPUDevice,
  shaderCode: string,
  entryPoint: string = 'main'
): GPUComputePipeline {
  const shaderModule = device.createShaderModule({
    code: shaderCode,
  });

  return device.createComputePipeline({
    layout: 'auto',
    compute: {
      module: shaderModule,
      entryPoint,
    },
  });
}

export async function elementwiseAdd(
  device: GPUDevice,
  a: Float32Array,
  b: Float32Array
): Promise<OperationResult> {
  if (a.length !== b.length) {
    throw new Error('Input arrays must have the same length');
  }

  const startTime = performance.now();

  // Create buffers
  const bufferA = createBuffer(
    device,
    a,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  );
  const bufferB = createBuffer(
    device,
    b,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  );
  const bufferOut = device.createBuffer({
    size: a.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  // Create pipeline
  const pipeline = createComputePipeline(device, shaders.ELEMENTWISE_ADD);

  // Create bind group
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: bufferA } },
      { binding: 1, resource: { buffer: bufferB } },
      { binding: 2, resource: { buffer: bufferOut } },
    ],
  });

  // Dispatch compute shader
  const commandEncoder = device.createCommandEncoder();
  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(pipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatchWorkgroups(Math.ceil(a.length / 256));
  passEncoder.end();
  device.queue.submit([commandEncoder.finish()]);

  // Read result
  const result = await readBuffer(device, bufferOut);

  // Cleanup
  bufferA.destroy();
  bufferB.destroy();
  bufferOut.destroy();

  const endTime = performance.now();
  const timeMs = endTime - startTime;
  const throughput = a.length / (timeMs / 1000);

  return { data: result, timeMs, throughput };
}

export async function elementwiseMul(
  device: GPUDevice,
  a: Float32Array,
  b: Float32Array
): Promise<OperationResult> {
  if (a.length !== b.length) {
    throw new Error('Input arrays must have the same length');
  }

  const startTime = performance.now();

  const bufferA = createBuffer(
    device,
    a,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  );
  const bufferB = createBuffer(
    device,
    b,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  );
  const bufferOut = device.createBuffer({
    size: a.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  const pipeline = createComputePipeline(device, shaders.ELEMENTWISE_MUL);

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: bufferA } },
      { binding: 1, resource: { buffer: bufferB } },
      { binding: 2, resource: { buffer: bufferOut } },
    ],
  });

  const commandEncoder = device.createCommandEncoder();
  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(pipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatchWorkgroups(Math.ceil(a.length / 256));
  passEncoder.end();
  device.queue.submit([commandEncoder.finish()]);

  const result = await readBuffer(device, bufferOut);

  bufferA.destroy();
  bufferB.destroy();
  bufferOut.destroy();

  const endTime = performance.now();
  const timeMs = endTime - startTime;
  const throughput = a.length / (timeMs / 1000);

  return { data: result, timeMs, throughput };
}

export async function relu(
  device: GPUDevice,
  input: Float32Array
): Promise<OperationResult> {
  const startTime = performance.now();

  const bufferIn = createBuffer(
    device,
    input,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  );
  const bufferOut = device.createBuffer({
    size: input.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  const pipeline = createComputePipeline(device, shaders.RELU);

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: bufferIn } },
      { binding: 1, resource: { buffer: bufferOut } },
    ],
  });

  const commandEncoder = device.createCommandEncoder();
  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(pipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatchWorkgroups(Math.ceil(input.length / 256));
  passEncoder.end();
  device.queue.submit([commandEncoder.finish()]);

  const result = await readBuffer(device, bufferOut);

  bufferIn.destroy();
  bufferOut.destroy();

  const endTime = performance.now();
  const timeMs = endTime - startTime;
  const throughput = input.length / (timeMs / 1000);

  return { data: result, timeMs, throughput };
}

export async function sigmoid(
  device: GPUDevice,
  input: Float32Array
): Promise<OperationResult> {
  const startTime = performance.now();

  const bufferIn = createBuffer(
    device,
    input,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  );
  const bufferOut = device.createBuffer({
    size: input.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  const pipeline = createComputePipeline(device, shaders.SIGMOID);

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: bufferIn } },
      { binding: 1, resource: { buffer: bufferOut } },
    ],
  });

  const commandEncoder = device.createCommandEncoder();
  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(pipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatchWorkgroups(Math.ceil(input.length / 256));
  passEncoder.end();
  device.queue.submit([commandEncoder.finish()]);

  const result = await readBuffer(device, bufferOut);

  bufferIn.destroy();
  bufferOut.destroy();

  const endTime = performance.now();
  const timeMs = endTime - startTime;
  const throughput = input.length / (timeMs / 1000);

  return { data: result, timeMs, throughput };
}

export async function matmul(
  device: GPUDevice,
  a: Float32Array,
  b: Float32Array,
  M: number,
  N: number,
  K: number
): Promise<OperationResult> {
  if (a.length !== M * K || b.length !== K * N) {
    throw new Error('Matrix dimensions do not match input array sizes');
  }

  const startTime = performance.now();

  const bufferA = createBuffer(
    device,
    a,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  );
  const bufferB = createBuffer(
    device,
    b,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  );
  const bufferOut = device.createBuffer({
    size: M * N * 4, // Float32 = 4 bytes
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  const dimensions = new Uint32Array([M, N, K]);
  const bufferDims = createBuffer(
    device,
    new Float32Array(dimensions.buffer),
    GPUBufferUsage.STORAGE
  );

  const pipeline = createComputePipeline(device, shaders.MATMUL);

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: bufferA } },
      { binding: 1, resource: { buffer: bufferB } },
      { binding: 2, resource: { buffer: bufferOut } },
      { binding: 3, resource: { buffer: bufferDims } },
    ],
  });

  const commandEncoder = device.createCommandEncoder();
  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(pipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatchWorkgroups(Math.ceil(N / 16), Math.ceil(M / 16));
  passEncoder.end();
  device.queue.submit([commandEncoder.finish()]);

  const result = await readBuffer(device, bufferOut);

  bufferA.destroy();
  bufferB.destroy();
  bufferOut.destroy();
  bufferDims.destroy();

  const endTime = performance.now();
  const timeMs = endTime - startTime;

  // Matrix multiply is 2*M*N*K FLOPs
  const flops = 2 * M * N * K;
  const gflops = flops / (timeMs / 1000) / 1e9;

  return { data: result, timeMs, gflops };
}

export async function softmax(
  device: GPUDevice,
  input: Float32Array
): Promise<OperationResult> {
  const startTime = performance.now();

  const bufferIn = createBuffer(
    device,
    input,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  );
  const bufferOut = device.createBuffer({
    size: input.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  const params = new Uint32Array([input.length]);
  const bufferParams = createBuffer(
    device,
    new Float32Array(params.buffer),
    GPUBufferUsage.STORAGE
  );

  const pipeline = createComputePipeline(device, shaders.SOFTMAX);

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: bufferIn } },
      { binding: 1, resource: { buffer: bufferOut } },
      { binding: 2, resource: { buffer: bufferParams } },
    ],
  });

  const commandEncoder = device.createCommandEncoder();
  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(pipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatchWorkgroups(1); // Single workgroup for softmax
  passEncoder.end();
  device.queue.submit([commandEncoder.finish()]);

  const result = await readBuffer(device, bufferOut);

  bufferIn.destroy();
  bufferOut.destroy();
  bufferParams.destroy();

  const endTime = performance.now();
  const timeMs = endTime - startTime;

  return { data: result, timeMs };
}
