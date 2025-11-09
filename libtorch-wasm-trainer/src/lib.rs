mod model;
mod trainer;

use burn::backend::ndarray::{NdArray, NdArrayDevice};
use burn::backend::Autodiff;
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::AdamConfig;
use trainer::{MnistTrainer, TrainingConfig};
use wasm_bindgen::prelude::*;

use crate::model::MnistMLP;

// Use NdArray backend with autodiff for WASM
// Note: Using NdArray (CPU) initially to get it working
// TODO: Switch to WGPU backend once we solve WASM + WebGPU integration
type WasmBackend = Autodiff<NdArray>;

// Concrete optimizer type for WASM
type WasmOptimizer = OptimizerAdaptor<burn::optim::Adam, MnistMLP<WasmBackend>, WasmBackend>;

/// Initialize panic hook for better error messages in browser
#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
}

/// Browser-based MNIST trainer
#[wasm_bindgen]
pub struct BrowserTrainer {
    trainer: MnistTrainer<WasmBackend, WasmOptimizer>,
}

#[wasm_bindgen]
impl BrowserTrainer {
    /// Create a new trainer
    #[wasm_bindgen(constructor)]
    pub fn new(learning_rate: f64) -> Self {
        let device = NdArrayDevice::Cpu;
        let config = TrainingConfig::new(learning_rate, 32);
        let model = MnistMLP::new(&device);
        let optim = AdamConfig::new().init();

        let trainer = MnistTrainer {
            model,
            optim,
            config,
            device,
        };

        Self { trainer }
    }

    /// Train on a single batch
    ///
    /// # Arguments
    /// * `images` - Flattened image data [batch_size * 784]
    /// * `labels` - Class labels [batch_size]
    ///
    /// # Returns
    /// Loss value for this batch
    #[wasm_bindgen]
    pub fn train_batch(&mut self, images: Vec<f32>, labels: Vec<usize>) -> f32 {
        self.trainer.train_batch(&images, &labels)
    }

    /// Evaluate on a batch (no training)
    ///
    /// # Returns
    /// Array of [loss, accuracy]
    #[wasm_bindgen]
    pub fn eval_batch(&self, images: Vec<f32>, labels: Vec<usize>) -> Vec<f32> {
        let (loss, accuracy) = self.trainer.eval_batch(&images, &labels);
        vec![loss, accuracy]
    }

    /// Export trained weights in PyTorch-compatible JSON format
    #[wasm_bindgen]
    pub fn export_pytorch_weights(&self) -> String {
        let weights = self.trainer.export_weights();
        serde_json::to_string(&weights).unwrap_or_else(|e| {
            format!("{{\"error\": \"Failed to serialize weights: {}\"}}", e)
        })
    }

    /// Export weights as formatted JSON with metadata
    #[wasm_bindgen]
    pub fn export_full(&self, test_accuracy: f32) -> String {
        let weights = self.trainer.export_weights();

        let output = serde_json::json!({
            "model_type": "mnist-mlp",
            "framework": "libtorch-rust (Burn)",
            "architecture": {
                "input_size": 784,
                "hidden_size": 128,
                "output_size": 10
            },
            "weights": {
                "fc1_weight": weights.fc1_weight,
                "fc1_bias": weights.fc1_bias,
                "fc2_weight": weights.fc2_weight,
                "fc2_bias": weights.fc2_bias
            },
            "metadata": {
                "test_accuracy": test_accuracy,
                "source": "browser_training",
                "backend": "burn-ndarray"
            }
        });

        serde_json::to_string_pretty(&output).unwrap()
    }

    /// Get model info for debugging
    #[wasm_bindgen]
    pub fn model_info(&self) -> String {
        "MNIST MLP: 784 → 128 → 10".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    #[wasm_bindgen_test]
    fn test_trainer_creation() {
        let _trainer = BrowserTrainer::new(0.001);
    }

    #[wasm_bindgen_test]
    fn test_train_batch() {
        let mut trainer = BrowserTrainer::new(0.001);
        let images = vec![0.0; 32 * 784];
        let labels = vec![0; 32];

        let loss = trainer.train_batch(images, labels);
        assert!(loss > 0.0);
    }

    #[wasm_bindgen_test]
    fn test_export_weights() {
        let trainer = BrowserTrainer::new(0.001);
        let weights_json = trainer.export_pytorch_weights();
        assert!(!weights_json.is_empty());
    }
}
