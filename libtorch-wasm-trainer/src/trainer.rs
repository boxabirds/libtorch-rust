use burn::optim::{Adam, AdamConfig, Optimizer};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{Data, Tensor};
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

use crate::model::MnistMLP;

/// PyTorch-compatible weight format
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PyTorchWeights {
    pub fc1_weight: Vec<f32>, // [128, 784] flattened
    pub fc1_bias: Vec<f32>,   // [128]
    pub fc2_weight: Vec<f32>, // [10, 128] flattened
    pub fc2_bias: Vec<f32>,   // [10]
}

/// Training configuration
#[wasm_bindgen]
#[derive(Clone, Copy)]
pub struct TrainingConfig {
    learning_rate: f64,
    batch_size: usize,
}

#[wasm_bindgen]
impl TrainingConfig {
    #[wasm_bindgen(constructor)]
    pub fn new(learning_rate: f64, batch_size: usize) -> Self {
        Self {
            learning_rate,
            batch_size,
        }
    }
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-3,
            batch_size: 32,
        }
    }
}

/// MNIST Trainer for browser-based training
pub struct MnistTrainer<B: AutodiffBackend> {
    model: MnistMLP<B>,
    optim: Adam<B::InnerBackend>,
    config: TrainingConfig,
    device: B::Device,
}

impl<B: AutodiffBackend> MnistTrainer<B> {
    /// Create a new trainer
    pub fn new(config: TrainingConfig, device: B::Device) -> Self {
        let model = MnistMLP::new(&device);

        let optim = AdamConfig::new()
            .with_weight_decay(Some(burn::optim::decay::WeightDecayConfig::new(5e-5)))
            .init();

        Self {
            model,
            optim,
            config,
            device,
        }
    }

    /// Train on a single batch
    /// images: flattened [batch_size * 784] array
    /// labels: [batch_size] array of class indices
    /// Returns: loss value
    pub fn train_batch(&mut self, images: &[f32], labels: &[usize]) -> f32 {
        let batch_size = labels.len();

        // Convert images to tensor [batch_size, 784]
        let images_tensor = Tensor::<B, 1>::from_floats(images, &self.device)
            .reshape([batch_size, 784]);

        // Forward pass
        let output = self.model.forward(images_tensor);

        // Compute cross-entropy loss
        let loss = self.cross_entropy_loss(output.clone(), labels);

        // Backward pass
        let grads = loss.backward();

        // Update weights
        let grads = burn::module::AutodiffModule::grad(&self.model, &grads);
        self.model = self.optim.step(self.config.learning_rate, self.model, grads);

        // Return loss value
        loss.into_scalar()
    }

    /// Compute cross-entropy loss
    fn cross_entropy_loss(&self, logits: Tensor<B, 2>, labels: &[usize]) -> Tensor<B, 1> {
        let batch_size = labels.len();

        // Convert labels to tensor
        let labels_data: Vec<i64> = labels.iter().map(|&l| l as i64).collect();
        let labels_tensor = Tensor::<B, 1, burn::tensor::Int>::from_ints(
            labels_data.as_slice(),
            &self.device,
        );

        // Compute log softmax
        let log_probs = logits.log_softmax(1);

        // Negative log likelihood
        let mut total_loss = 0.0;
        let log_probs_data = log_probs.into_data();

        for i in 0..batch_size {
            let label = labels[i];
            let log_prob = log_probs_data.value[i * 10 + label];
            total_loss -= log_prob;
        }

        let loss = total_loss / batch_size as f32;
        Tensor::from_floats([loss], &self.device)
    }

    /// Evaluate on a batch (no gradient tracking)
    pub fn eval_batch(&self, images: &[f32], labels: &[usize]) -> (f32, f32) {
        let batch_size = labels.len();

        // Convert images to tensor [batch_size, 784]
        let images_tensor = Tensor::<B::InnerBackend, 1>::from_floats(images, &self.device)
            .reshape([batch_size, 784]);

        // Forward pass (no gradients)
        let output = self.model.valid().forward(images_tensor);

        // Compute accuracy
        let predictions = output.clone().argmax(1);
        let predictions_data = predictions.into_data();

        let mut correct = 0;
        for i in 0..batch_size {
            if predictions_data.value[i] == labels[i] as i64 {
                correct += 1;
            }
        }
        let accuracy = correct as f32 / batch_size as f32;

        // Compute loss
        let loss = self.cross_entropy_loss_no_grad(output, labels);

        (loss, accuracy)
    }

    /// Cross-entropy loss without gradient tracking
    fn cross_entropy_loss_no_grad(
        &self,
        logits: Tensor<B::InnerBackend, 2>,
        labels: &[usize],
    ) -> f32 {
        let batch_size = labels.len();
        let log_probs = logits.log_softmax(1);
        let log_probs_data = log_probs.into_data();

        let mut total_loss = 0.0;
        for i in 0..batch_size {
            let label = labels[i];
            let log_prob = log_probs_data.value[i * 10 + label];
            total_loss -= log_prob;
        }

        total_loss / batch_size as f32
    }

    /// Export weights in PyTorch-compatible format
    pub fn export_weights(&self) -> PyTorchWeights {
        // Get fc1 weights [128, 784]
        let fc1_weight_tensor = self.model.fc1.weight.val();
        let fc1_weight_data: Data<f32, 2> = fc1_weight_tensor.into_data();
        let fc1_weight: Vec<f32> = fc1_weight_data.value;

        // Get fc1 bias [128]
        let fc1_bias_tensor = self.model.fc1.bias.as_ref().unwrap().val();
        let fc1_bias_data: Data<f32, 1> = fc1_bias_tensor.into_data();
        let fc1_bias: Vec<f32> = fc1_bias_data.value;

        // Get fc2 weights [10, 128]
        let fc2_weight_tensor = self.model.fc2.weight.val();
        let fc2_weight_data: Data<f32, 2> = fc2_weight_tensor.into_data();
        let fc2_weight: Vec<f32> = fc2_weight_data.value;

        // Get fc2 bias [10]
        let fc2_bias_tensor = self.model.fc2.bias.as_ref().unwrap().val();
        let fc2_bias_data: Data<f32, 1> = fc2_bias_tensor.into_data();
        let fc2_bias: Vec<f32> = fc2_bias_data.value;

        PyTorchWeights {
            fc1_weight,
            fc1_bias,
            fc2_weight,
            fc2_bias,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::{NdArray, NdArrayDevice};
    use burn::backend::Autodiff;

    type TestBackend = Autodiff<NdArray<f32>>;

    #[test]
    fn test_trainer_creation() {
        let device = NdArrayDevice::Cpu;
        let config = TrainingConfig::default();
        let _trainer: MnistTrainer<TestBackend> = MnistTrainer::new(config, device);
    }

    #[test]
    fn test_train_batch() {
        let device = NdArrayDevice::Cpu;
        let config = TrainingConfig::default();
        let mut trainer: MnistTrainer<TestBackend> = MnistTrainer::new(config, device);

        // Dummy batch
        let images = vec![0.0; 32 * 784];
        let labels = vec![0; 32];

        let loss = trainer.train_batch(&images, &labels);
        assert!(loss > 0.0);
    }
}
