use burn::module::AutodiffModule;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::optim::{GradientsParams, Optimizer};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{activation, ElementConversion, Int, Tensor};
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
pub struct MnistTrainer<B, O>
where
    B: AutodiffBackend,
    O: Optimizer<MnistMLP<B>, B>,
{
    pub(crate) model: MnistMLP<B>,
    pub(crate) optim: O,
    pub(crate) config: TrainingConfig,
    pub(crate) device: B::Device,
}

impl<B, O> MnistTrainer<B, O>
where
    B: AutodiffBackend,
    O: Optimizer<MnistMLP<B>, B>,
{
    /// Create a new trainer
    pub fn new(config: TrainingConfig, device: B::Device) -> Self
    where
        O: Default,
    {
        let model = MnistMLP::new(&device);
        let optim = O::default();

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

        // Convert labels to tensor
        let labels_i64: Vec<i64> = labels.iter().map(|&l| l as i64).collect();
        let labels_tensor = Tensor::<B, 1, Int>::from_ints(labels_i64.as_slice(), &self.device);

        // Forward pass
        let logits = self.model.forward(images_tensor);

        // Compute cross-entropy loss using Burn's built-in loss
        let loss_fn = CrossEntropyLossConfig::new().init(&self.device);
        let loss = loss_fn.forward(logits, labels_tensor);

        // Backward pass
        let grads = loss.backward();

        // Extract parameter-specific gradients
        let grads = GradientsParams::from_grads(grads, &self.model);

        // Update weights with optimizer (takes ownership and returns updated model)
        let model = std::mem::replace(&mut self.model, MnistMLP::new(&self.device));
        self.model = self.optim.step(self.config.learning_rate, model, grads);

        // Return loss value (convert to f32)
        loss.into_scalar().elem()
    }

    /// Evaluate on a batch (no gradient tracking)
    pub fn eval_batch(&self, images: &[f32], labels: &[usize]) -> (f32, f32) {
        let batch_size = labels.len();

        // Convert images to tensor [batch_size, 784]
        let images_tensor = Tensor::<B::InnerBackend, 1>::from_floats(images, &self.device)
            .reshape([batch_size, 784]);

        // Forward pass (no gradients)
        let logits = self.model.valid().forward(images_tensor);

        // Compute accuracy
        let predictions = logits.clone().argmax(1);
        let predictions_data = predictions.into_data();

        let mut correct = 0;
        for i in 0..batch_size {
            if predictions_data.as_slice::<i64>().unwrap()[i] == labels[i] as i64 {
                correct += 1;
            }
        }
        let accuracy = correct as f32 / batch_size as f32;

        // Compute loss using log_softmax function
        let log_probs = activation::log_softmax(logits, 1);
        let log_probs_data = log_probs.into_data();

        let mut total_loss = 0.0;
        for i in 0..batch_size {
            let label = labels[i];
            let log_prob = log_probs_data.as_slice::<f32>().unwrap()[i * 10 + label];
            total_loss -= log_prob;
        }
        let loss = total_loss / batch_size as f32;

        (loss, accuracy)
    }

    /// Export weights in PyTorch-compatible format
    pub fn export_weights(&self) -> PyTorchWeights {
        // Get model in eval mode for weight extraction
        let model_valid = self.model.valid();

        // Extract fc1 weights [128, 784]
        let fc1_weight_data = model_valid.fc1.weight.to_data();
        let fc1_weight: Vec<f32> = fc1_weight_data.to_vec().unwrap();

        // Extract fc1 bias [128]
        let fc1_bias_data = model_valid.fc1.bias.as_ref().unwrap().to_data();
        let fc1_bias: Vec<f32> = fc1_bias_data.to_vec().unwrap();

        // Extract fc2 weights [10, 128]
        let fc2_weight_data = model_valid.fc2.weight.to_data();
        let fc2_weight: Vec<f32> = fc2_weight_data.to_vec().unwrap();

        // Extract fc2 bias [10]
        let fc2_bias_data = model_valid.fc2.bias.as_ref().unwrap().to_data();
        let fc2_bias: Vec<f32> = fc2_bias_data.to_vec().unwrap();

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
    use burn::optim::adaptor::OptimizerAdaptor;
    use burn::optim::AdamConfig;

    type TestBackend = Autodiff<NdArray>;
    type TestOptimizer = OptimizerAdaptor<burn::optim::Adam, MnistMLP<TestBackend>, TestBackend>;

    #[test]
    fn test_train_batch() {
        let device = NdArrayDevice::Cpu;
        let config = TrainingConfig::default();
        let model = MnistMLP::new(&device);
        let optim = AdamConfig::new().init();

        let mut trainer: MnistTrainer<TestBackend, TestOptimizer> = MnistTrainer {
            model,
            optim,
            config,
            device,
        };

        // Dummy batch
        let images = vec![0.0; 32 * 784];
        let labels = vec![0; 32];

        let loss = trainer.train_batch(&images, &labels);
        assert!(loss > 0.0);
    }

    #[test]
    fn test_eval_batch() {
        let device = NdArrayDevice::Cpu;
        let config = TrainingConfig::default();
        let model = MnistMLP::new(&device);
        let optim = AdamConfig::new().init();

        let trainer: MnistTrainer<TestBackend, TestOptimizer> = MnistTrainer {
            model,
            optim,
            config,
            device,
        };

        // Dummy batch
        let images = vec![0.0; 8 * 784];
        let labels = vec![0; 8];

        let (loss, accuracy) = trainer.eval_batch(&images, &labels);
        assert!(loss > 0.0);
        assert!(accuracy >= 0.0 && accuracy <= 1.0);
    }

    #[test]
    fn test_export_weights() {
        let device = NdArrayDevice::Cpu;
        let config = TrainingConfig::default();
        let model = MnistMLP::new(&device);
        let optim = AdamConfig::new().init();

        let trainer: MnistTrainer<TestBackend, TestOptimizer> = MnistTrainer {
            model,
            optim,
            config,
            device,
        };

        let weights = trainer.export_weights();

        // Check weight shapes
        assert_eq!(weights.fc1_weight.len(), 128 * 784);
        assert_eq!(weights.fc1_bias.len(), 128);
        assert_eq!(weights.fc2_weight.len(), 10 * 128);
        assert_eq!(weights.fc2_bias.len(), 10);
    }
}
