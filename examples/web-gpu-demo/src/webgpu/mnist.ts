/**
 * MNIST MLP Model
 *
 * Simple 2-layer neural network for MNIST digit classification
 * Architecture:
 *   - Input: 784 (28x28 pixels)
 *   - Hidden: 128 units with ReLU
 *   - Output: 10 classes with Softmax
 */

import * as ops from './operations';

export interface ModelWeights {
  fc1_weight: Float32Array; // [128, 784]
  fc1_bias: Float32Array;   // [128]
  fc2_weight: Float32Array; // [10, 128]
  fc2_bias: Float32Array;   // [10]
}

export class MNISTModel {
  private device: GPUDevice;
  private weights: ModelWeights;

  constructor(device: GPUDevice, weights: ModelWeights) {
    this.device = device;
    this.weights = weights;
  }

  /**
   * Run forward pass inference
   * @param input Flattened 28x28 image (784 values, normalized 0-1)
   * @returns Probabilities for each digit 0-9
   */
  async predict(input: Float32Array): Promise<Float32Array> {
    if (input.length !== 784) {
      throw new Error('Input must be 784 values (28x28 flattened image)');
    }

    console.log('🔍 MNIST Inference Pipeline:');
    console.log('1. Input statistics:');
    const inputStats = this.getArrayStats(input);
    console.log(`   - Shape: 784 (28×28)`, inputStats);
    console.log(`   - First 10 values:`, Array.from(input.slice(0, 10)).map(v => v.toFixed(3)));

    // Layer 1: fc1 = input @ fc1_weight.T + fc1_bias
    console.log('2. Layer 1 (fc1): 784 → 128');
    const fc1_result = await ops.matmul(
      this.device,
      input,
      this.weights.fc1_weight,
      1,    // M = 1 (batch size)
      128,  // N = 128 (output features)
      784   // K = 784 (input features)
    );
    console.log(`   - Matmul output:`, this.getArrayStats(fc1_result.data));

    // Add bias
    const fc1_biased = new Float32Array(128);
    for (let i = 0; i < 128; i++) {
      fc1_biased[i] = fc1_result.data[i] + this.weights.fc1_bias[i];
    }
    console.log(`   - After bias:`, this.getArrayStats(fc1_biased));
    console.log(`   - First 10 values:`, Array.from(fc1_biased.slice(0, 10)).map(v => v.toFixed(3)));

    // ReLU activation
    console.log('3. ReLU activation:');
    const fc1_activated = await ops.relu(this.device, fc1_biased);
    console.log(`   - After ReLU:`, this.getArrayStats(fc1_activated.data));
    const zerosCount = Array.from(fc1_activated.data).filter(v => v === 0).length;
    console.log(`   - Zeros: ${zerosCount}/128 (${(zerosCount/128*100).toFixed(1)}%)`);

    // Layer 2: fc2 = fc1_activated @ fc2_weight.T + fc2_bias
    console.log('4. Layer 2 (fc2): 128 → 10');
    const fc2_result = await ops.matmul(
      this.device,
      fc1_activated.data,
      this.weights.fc2_weight,
      1,   // M = 1
      10,  // N = 10 (num classes)
      128  // K = 128
    );
    console.log(`   - Matmul output:`, this.getArrayStats(fc2_result.data));

    // Add bias
    const fc2_biased = new Float32Array(10);
    for (let i = 0; i < 10; i++) {
      fc2_biased[i] = fc2_result.data[i] + this.weights.fc2_bias[i];
    }
    console.log(`   - After bias (logits):`, this.getArrayStats(fc2_biased));
    console.log(`   - Logits:`, Array.from(fc2_biased).map(v => v.toFixed(3)));

    // Softmax to get probabilities
    console.log('5. Softmax:');
    const probabilities = await ops.softmax(this.device, fc2_biased);
    console.log(`   - Probabilities:`, this.getArrayStats(probabilities.data));
    console.log(`   - All probs:`, Array.from(probabilities.data).map((v, i) =>
      `${i}:${(v*100).toFixed(1)}%`).join(' '));

    const maxProb = Math.max(...Array.from(probabilities.data));
    const predicted = Array.from(probabilities.data).indexOf(maxProb);
    console.log(`   - Predicted: ${predicted} (${(maxProb*100).toFixed(1)}% confidence)`);
    console.log('');

    return probabilities.data;
  }

  /**
   * Helper: Get array statistics for debugging
   */
  private getArrayStats(arr: Float32Array): { min: number, max: number, mean: number, std: number } {
    const values = Array.from(arr);
    const min = Math.min(...values);
    const max = Math.max(...values);
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const variance = values.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / values.length;
    const std = Math.sqrt(variance);
    return {
      min: parseFloat(min.toFixed(4)),
      max: parseFloat(max.toFixed(4)),
      mean: parseFloat(mean.toFixed(4)),
      std: parseFloat(std.toFixed(4))
    };
  }

  /**
   * Get the predicted digit (0-9) from probabilities
   */
  static getPrediction(probabilities: Float32Array): number {
    let maxIdx = 0;
    let maxProb = probabilities[0];
    for (let i = 1; i < probabilities.length; i++) {
      if (probabilities[i] > maxProb) {
        maxProb = probabilities[i];
        maxIdx = i;
      }
    }
    return maxIdx;
  }
}

/**
 * Load MNIST model weights from URL or local file
 */
export async function loadMNISTWeights(url?: string): Promise<ModelWeights> {
  const modelUrl = url || '/models/mnist-mlp.json';

  const response = await fetch(modelUrl);

  if (!response.ok) {
    throw new Error(
      `❌ Model weights not found at ${modelUrl}\n\n` +
      `To use the MNIST demo, you need to train a model first:\n\n` +
      `1. cd examples/web-gpu-demo\n` +
      `2. pip install torch torchvision\n` +
      `3. python scripts/train_mnist.py\n\n` +
      `This trains a model in ~2-3 minutes and exports weights to public/models/mnist-mlp.json`
    );
  }

  const data = await response.json();

  // Handle different weight formats
  const weights = data.weights || data;

  console.log('✅ Loaded trained model weights');

  return {
    fc1_weight: new Float32Array(weights.fc1_weight),
    fc1_bias: new Float32Array(weights.fc1_bias),
    fc2_weight: new Float32Array(weights.fc2_weight),
    fc2_bias: new Float32Array(weights.fc2_bias),
  };
}
