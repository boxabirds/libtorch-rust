use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::backend::Backend;
use burn::tensor::{activation, Tensor};

/// MNIST MLP model: 784 → 128 → 10
/// Architecture matches the one used in the browser demo
#[derive(Module, Debug)]
pub struct MnistMLP<B: Backend> {
    pub fc1: Linear<B>,
    pub fc2: Linear<B>,
}

impl<B: Backend> MnistMLP<B> {
    /// Create a new MNIST MLP model
    pub fn new(device: &B::Device) -> Self {
        let fc1 = LinearConfig::new(784, 128).init(device);
        let fc2 = LinearConfig::new(128, 10).init(device);

        Self { fc1, fc2 }
    }

    /// Forward pass
    /// Input: [batch_size, 784]
    /// Output: [batch_size, 10] (logits)
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        // Layer 1: Linear + ReLU
        let x = self.fc1.forward(input);
        let x = activation::relu(x);

        // Layer 2: Linear (output logits)
        self.fc2.forward(x)
    }

    /// Forward pass with classification
    /// Returns class predictions (argmax)
    pub fn predict(&self, input: Tensor<B, 2>) -> Tensor<B, 2, burn::tensor::Int> {
        let logits = self.forward(input);
        logits.argmax(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArrayDevice;
    use burn::backend::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_model_forward() {
        let device = NdArrayDevice::Cpu;
        let model: MnistMLP<TestBackend> = MnistMLP::new(&device);

        // Create dummy input [1, 784]
        let input = Tensor::<TestBackend, 2>::zeros([1, 784], &device);

        // Forward pass
        let output = model.forward(input);

        // Check output shape
        assert_eq!(output.shape().dims, [1, 10]);
    }
}
