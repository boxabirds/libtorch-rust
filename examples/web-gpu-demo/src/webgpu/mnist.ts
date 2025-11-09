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

    // Layer 1: fc1 = input @ fc1_weight.T + fc1_bias
    const fc1_result = await ops.matmul(
      this.device,
      input,
      this.weights.fc1_weight,
      1,    // M = 1 (batch size)
      128,  // N = 128 (output features)
      784   // K = 784 (input features)
    );

    // Add bias
    const fc1_biased = new Float32Array(128);
    for (let i = 0; i < 128; i++) {
      fc1_biased[i] = fc1_result.data[i] + this.weights.fc1_bias[i];
    }

    // ReLU activation
    const fc1_activated = await ops.relu(this.device, fc1_biased);

    // Layer 2: fc2 = fc1_activated @ fc2_weight.T + fc2_bias
    const fc2_result = await ops.matmul(
      this.device,
      fc1_activated.data,
      this.weights.fc2_weight,
      1,   // M = 1
      10,  // N = 10 (num classes)
      128  // K = 128
    );

    // Add bias
    const fc2_biased = new Float32Array(10);
    for (let i = 0; i < 10; i++) {
      fc2_biased[i] = fc2_result.data[i] + this.weights.fc2_bias[i];
    }

    // Softmax to get probabilities
    const probabilities = await ops.softmax(this.device, fc2_biased);

    return probabilities.data;
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
 * Load MNIST model weights from URL or create synthetic weights for demo
 */
export async function loadMNISTWeights(url?: string): Promise<ModelWeights> {
  if (url) {
    // Load from URL (HuggingFace, etc.)
    const response = await fetch(url);
    const data = await response.json();
    return {
      fc1_weight: new Float32Array(data.fc1_weight),
      fc1_bias: new Float32Array(data.fc1_bias),
      fc2_weight: new Float32Array(data.fc2_weight),
      fc2_bias: new Float32Array(data.fc2_bias),
    };
  } else {
    // Create synthetic weights for demo (Xavier initialization)
    const createWeight = (rows: number, cols: number) => {
      const scale = Math.sqrt(2.0 / (rows + cols));
      const weights = new Float32Array(rows * cols);
      for (let i = 0; i < weights.length; i++) {
        weights[i] = (Math.random() * 2 - 1) * scale;
      }
      return weights;
    };

    const createBias = (size: number) => {
      return new Float32Array(size); // Initialize to zero
    };

    return {
      fc1_weight: createWeight(128, 784),
      fc1_bias: createBias(128),
      fc2_weight: createWeight(10, 128),
      fc2_bias: createBias(10),
    };
  }
}
