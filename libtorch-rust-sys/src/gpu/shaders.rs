/// WGSL compute shaders for tensor operations
///
/// These shaders run on the GPU and perform parallel operations on tensors.

/// Element-wise addition: output = a + b
pub const ELEMENTWISE_ADD: &str = r#"
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
"#;

/// Element-wise multiplication: output = a * b
pub const ELEMENTWISE_MUL: &str = r#"
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
"#;

/// Element-wise subtraction: output = a - b
pub const ELEMENTWISE_SUB: &str = r#"
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
        output[idx] = input_a[idx] - input_b[idx];
    }
}
"#;

/// Element-wise division: output = a / b
pub const ELEMENTWISE_DIV: &str = r#"
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
        output[idx] = input_a[idx] / input_b[idx];
    }
}
"#;

/// ReLU activation: output = max(0, x)
pub const RELU: &str = r#"
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
"#;

/// Sigmoid activation: output = 1 / (1 + exp(-x))
pub const SIGMOID: &str = r#"
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
"#;

/// Scalar multiplication: output = input * scalar
pub const SCALAR_MUL: &str = r#"
@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output: array<f32>;

@group(0) @binding(2)
var<uniform> scalar: f32;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < arrayLength(&output)) {
        output[idx] = input[idx] * scalar;
    }
}
"#;

/// Matrix multiplication (tiled algorithm for better cache usage)
/// Note: This is a simplified version. Production would use more sophisticated tiling.
pub const MATMUL_SIMPLE: &str = r#"
struct Dimensions {
    M: u32,  // Rows of A and output
    N: u32,  // Cols of B and output
    K: u32,  // Cols of A, rows of B
}

@group(0) @binding(0)
var<storage, read> matrix_a: array<f32>;

@group(0) @binding(1)
var<storage, read> matrix_b: array<f32>;

@group(0) @binding(2)
var<storage, read_write> output: array<f32>;

@group(0) @binding(3)
var<uniform> dims: Dimensions;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.y;
    let col = global_id.x;

    if (row >= dims.M || col >= dims.N) {
        return;
    }

    var sum = 0.0;
    for (var k = 0u; k < dims.K; k = k + 1u) {
        let a_idx = row * dims.K + k;
        let b_idx = k * dims.N + col;
        sum = sum + matrix_a[a_idx] * matrix_b[b_idx];
    }

    let out_idx = row * dims.N + col;
    output[out_idx] = sum;
}
"#;
