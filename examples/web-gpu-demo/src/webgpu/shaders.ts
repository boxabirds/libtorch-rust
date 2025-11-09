/**
 * WebGPU Compute Shaders (WGSL)
 *
 * These are the same shaders used in the Rust implementation.
 * They demonstrate that the same WGSL code works in both native (via wgpu)
 * and browser (via WebGPU API) environments.
 */

export const ELEMENTWISE_ADD = `
@group(0) @binding(0)
var<storage, read> input_a: array<f32>;

@group(0) @binding(1)
var<storage, read> input_b: array<f32>;

@group(0) @binding(2)
var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < arrayLength(&output)) {
        output[idx] = input_a[idx] + input_b[idx];
    }
}
`;

export const ELEMENTWISE_MUL = `
@group(0) @binding(0)
var<storage, read> input_a: array<f32>;

@group(0) @binding(1)
var<storage, read> input_b: array<f32>;

@group(0) @binding(2)
var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < arrayLength(&output)) {
        output[idx] = input_a[idx] * input_b[idx];
    }
}
`;

export const RELU = `
@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < arrayLength(&output)) {
        output[idx] = max(0.0, input[idx]);
    }
}
`;

export const SIGMOID = `
@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < arrayLength(&output)) {
        output[idx] = 1.0 / (1.0 + exp(-input[idx]));
    }
}
`;

export const MATMUL = `
@group(0) @binding(0)
var<storage, read> input_a: array<f32>;

@group(0) @binding(1)
var<storage, read> input_b: array<f32>;

@group(0) @binding(2)
var<storage, read_write> output: array<f32>;

@group(0) @binding(3)
var<storage, read> dimensions: array<u32>; // [M, N, K]

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let M = dimensions[0];
    let N = dimensions[1];
    let K = dimensions[2];

    let row = global_id.y;
    let col = global_id.x;

    if (row >= M || col >= N) {
        return;
    }

    var sum = 0.0;
    for (var k = 0u; k < K; k = k + 1u) {
        let a_val = input_a[row * K + k];
        let b_val = input_b[k * N + col];
        sum = sum + a_val * b_val;
    }

    output[row * N + col] = sum;
}
`;
