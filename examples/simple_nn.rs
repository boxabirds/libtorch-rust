/// Simple neural network example demonstrating libtorch-rust API
use tch::nn::{linear, Module, VarStore};
use tch::{Device, Kind, Tensor};

fn main() {
    println!("=== LibTorch Rust - Simple Neural Network Example ===\n");

    // Create a variable store to manage parameters
    let vs = VarStore::new(Device::Cpu);
    let root = vs.root();

    // Build a simple 2-layer network: 4 -> 8 -> 2
    println!("Building a 2-layer neural network (4 -> 8 -> 2)...");
    let layer1 = linear(&root.sub("layer1"), 4, 8);
    let layer2 = linear(&root.sub("layer2"), 8, 2);

    // Create a batch of 3 input samples with 4 features each
    println!("\nCreating input tensor (batch_size=3, features=4)...");
    let input = Tensor::of_slice(&[
        1.0, 2.0, 3.0, 4.0, // Sample 1
        2.0, 3.0, 4.0, 5.0, // Sample 2
        3.0, 4.0, 5.0, 6.0, // Sample 3
    ])
    .reshape(&[3, 4]);

    println!("Input shape: {:?}", input.size());
    println!("Input data:\n{:?}\n", input.to_vec_f64());

    // Forward pass through layer 1
    println!("Forward pass through layer 1...");
    let hidden = layer1.forward(&input);
    println!("Hidden layer shape: {:?}", hidden.size());

    // Apply ReLU activation
    println!("Applying ReLU activation...");
    let activated = hidden.relu();

    // Forward pass through layer 2
    println!("Forward pass through layer 2...");
    let output = layer2.forward(&activated);

    println!("\nFinal output shape: {:?}", output.size());
    println!("Final output data:\n{:?}\n", output.to_vec_f64());

    // Apply softmax to get probabilities
    println!("Applying softmax to get class probabilities...");
    let probs = output.softmax(1, Kind::Float);
    println!("Class probabilities:\n{:?}\n", probs.to_vec_f64());

    // Count trainable parameters
    let params = vs.trainable_variables();
    let total_params: i64 = params.iter().map(|t| t.numel()).sum();
    println!("Total trainable parameters: {}", total_params);

    println!("\n=== Example Complete ===");
}
