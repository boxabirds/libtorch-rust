/// Demonstrates basic tensor operations in libtorch-rust
use tch::{Device, Kind, Tensor};

fn main() {
    println!("=== LibTorch Rust - Tensor Operations Example ===\n");

    // Create tensors
    println!("1. Creating tensors:");
    let zeros = Tensor::zeros(&[2, 3], Kind::Float, Device::Cpu);
    println!("   Zeros tensor: {:?}", zeros.to_vec_f64());

    let ones = Tensor::ones(&[2, 3], Kind::Float, Device::Cpu);
    println!("   Ones tensor: {:?}", ones.to_vec_f64());

    let from_slice = Tensor::of_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    println!("   From slice: {:?}\n", from_slice.to_vec_f64());

    // Reshape
    println!("2. Reshaping:");
    let matrix = from_slice.reshape(&[2, 3]);
    println!("   Reshaped to [2, 3]: {:?}", matrix.to_vec_f64());
    println!("   Shape: {:?}\n", matrix.size());

    // Arithmetic operations
    println!("3. Arithmetic operations:");
    let a = Tensor::of_slice(&[1.0, 2.0, 3.0]);
    let b = Tensor::of_slice(&[4.0, 5.0, 6.0]);

    let sum = &a + &b;
    println!("   [1, 2, 3] + [4, 5, 6] = {:?}", sum.to_vec_f64());

    let product = &a * &b;
    println!("   [1, 2, 3] * [4, 5, 6] = {:?}", product.to_vec_f64());

    let scaled = &a * 2.0;
    println!("   [1, 2, 3] * 2.0 = {:?}\n", scaled.to_vec_f64());

    // Matrix multiplication
    println!("4. Matrix multiplication:");
    let m1 = Tensor::of_data_size(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let m2 = Tensor::of_data_size(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);

    println!("   Matrix 1 (2x3): {:?}", m1.to_vec_f64());
    println!("   Matrix 2 (3x2): {:?}", m2.to_vec_f64());

    let result = m1.matmul(&m2);
    println!("   Result (2x2): {:?}", result.to_vec_f64());
    println!("   Result shape: {:?}\n", result.size());

    // Transpose
    println!("5. Transpose:");
    let original = Tensor::of_data_size(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    println!("   Original (2x2): {:?}", original.to_vec_f64());

    let transposed = original.transpose(0, 1);
    println!("   Transposed (2x2): {:?}", transposed.to_vec_f64());

    // Broadcasting
    println!("\n6. Broadcasting:");
    let matrix_2d = Tensor::of_data_size(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let bias = Tensor::of_slice(&[10.0, 20.0, 30.0]);

    println!("   Matrix (2x3): {:?}", matrix_2d.to_vec_f64());
    println!("   Bias (3,): {:?}", bias.to_vec_f64());

    let with_bias = &matrix_2d + &bias;
    println!("   Matrix + Bias: {:?}\n", with_bias.to_vec_f64());

    // Activation functions
    println!("7. Activation functions:");
    let values = Tensor::of_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
    println!("   Input: {:?}", values.to_vec_f64());

    let relu_result = values.relu();
    println!("   ReLU: {:?}", relu_result.to_vec_f64());

    let sigmoid_result = values.sigmoid();
    println!("   Sigmoid: {:?}", sigmoid_result.to_vec_f64());

    // Reductions
    println!("\n8. Reductions:");
    let data = Tensor::of_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    println!("   Data: {:?}", data.to_vec_f64());

    let sum_val = data.sum(Kind::Float).to_f64();
    println!("   Sum: {}", sum_val);

    let mean_val = data.mean(Kind::Float).to_f64();
    println!("   Mean: {}", mean_val);

    println!("\n=== Example Complete ===");
}
