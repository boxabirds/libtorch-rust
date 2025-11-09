/// Verify that training actually improves MNIST model
/// This is a sanity check to ensure the training loop works

use burn::backend::ndarray::{NdArray, NdArrayDevice};
use burn::backend::Autodiff;
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::AdamConfig;

// Import from the library
use libtorch_wasm_trainer::{MnistMLP, MnistTrainer, TrainingConfig};

type Backend = Autodiff<NdArray>;
type Optimizer = OptimizerAdaptor<burn::optim::Adam, MnistMLP<Backend>, Backend>;

fn main() {
    println!("Testing MNIST training...\n");

    let device = NdArrayDevice::Cpu;
    let config = TrainingConfig::new(0.01, 32);
    let model = MnistMLP::new(&device);
    let optim = AdamConfig::new().init();

    let mut trainer: MnistTrainer<Backend, Optimizer> = MnistTrainer {
        model,
        optim,
        config,
        device,
    };

    // Create synthetic "MNIST-like" data
    // In real MNIST, digit 0 might have pixels in a circle pattern
    // We'll create a simple pattern: digit 0 = first 100 pixels are 1.0, rest 0.0
    //                                digit 1 = pixels 100-200 are 1.0, rest 0.0
    // This is a toy problem but should be learnable

    let batch_size = 32;
    let mut images = vec![0.0; batch_size * 784];
    let mut labels = vec![0; batch_size];

    // Half the batch is digit 0, half is digit 1
    for i in 0..batch_size {
        if i < batch_size / 2 {
            // Digit 0: first 100 pixels bright
            labels[i] = 0;
            for j in 0..100 {
                images[i * 784 + j] = 1.0;
            }
        } else {
            // Digit 1: pixels 100-200 bright
            labels[i] = 1;
            for j in 100..200 {
                images[i * 784 + j] = 1.0;
            }
        }
    }

    // Evaluate initial performance
    let (initial_loss, initial_acc) = trainer.eval_batch(&images, &labels);
    println!("Initial performance:");
    println!("  Loss: {:.4}", initial_loss);
    println!("  Accuracy: {:.2}%", initial_acc * 100.0);

    // Train for several epochs
    println!("\nTraining...");
    let epochs = 50;
    for epoch in 0..epochs {
        let loss = trainer.train_batch(&images, &labels);
        if epoch % 10 == 0 {
            let (eval_loss, eval_acc) = trainer.eval_batch(&images, &labels);
            println!("Epoch {}: loss={:.4}, accuracy={:.2}%", epoch, eval_loss, eval_acc * 100.0);
        }
    }

    // Final evaluation
    let (final_loss, final_acc) = trainer.eval_batch(&images, &labels);
    println!("\nFinal performance:");
    println!("  Loss: {:.4}", final_loss);
    println!("  Accuracy: {:.2}%", final_acc * 100.0);

    // Check if training worked
    println!("\nResults:");
    if final_loss < initial_loss {
        println!("✅ Loss decreased: {:.4} -> {:.4}", initial_loss, final_loss);
    } else {
        println!("❌ Loss did NOT decrease: {:.4} -> {:.4}", initial_loss, final_loss);
    }

    if final_acc > initial_acc + 0.2 {
        println!("✅ Accuracy improved significantly: {:.2}% -> {:.2}%",
                 initial_acc * 100.0, final_acc * 100.0);
    } else if final_acc > initial_acc {
        println!("⚠️  Accuracy improved slightly: {:.2}% -> {:.2}%",
                 initial_acc * 100.0, final_acc * 100.0);
    } else {
        println!("❌ Accuracy did NOT improve: {:.2}% -> {:.2}%",
                 initial_acc * 100.0, final_acc * 100.0);
    }

    if final_acc > 0.95 {
        println!("\n🎉 Training works! Model learned the pattern successfully.");
    } else if final_acc > 0.7 {
        println!("\n⚠️  Training partially works, but model could learn better.");
    } else {
        println!("\n❌ Training may not be working correctly.");
    }
}
