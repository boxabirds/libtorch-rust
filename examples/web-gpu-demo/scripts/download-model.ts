/**
 * Download pre-trained MNIST model weights
 *
 * This script downloads a trained MNIST MLP model from HuggingFace
 * and saves it to public/models/mnist-mlp.json
 */

import { writeFileSync, existsSync, mkdirSync } from 'fs';
import { join } from 'path';

const MODEL_URL = 'https://huggingface.co/datasets/mnist/resolve/main/mnist-mlp-weights.json';
const FALLBACK_URL = 'https://raw.githubusercontent.com/mnielsen/neural-networks-and-deep-learning/master/data/mnist.pkl.gz';
const MODELS_DIR = join(import.meta.dir, '..', 'public', 'models');
const MODEL_PATH = join(MODELS_DIR, 'mnist-mlp.json');

function createSyntheticWeights() {
  console.log('   Creating synthetic weights...');

  const createWeights = (rows: number, cols: number) => {
    const scale = Math.sqrt(2.0 / (rows + cols));
    const weights: number[] = [];
    for (let i = 0; i < rows * cols; i++) {
      weights.push((Math.random() * 2 - 1) * scale);
    }
    return weights;
  };

  const createBias = (size: number) => {
    return new Array(size).fill(0);
  };

  const modelWeights = {
    model_type: 'mnist-mlp',
    architecture: {
      input_size: 784,
      hidden_size: 128,
      output_size: 10,
    },
    weights: {
      fc1_weight: createWeights(128, 784),
      fc1_bias: createBias(128),
      fc2_weight: createWeights(10, 128),
      fc2_bias: createBias(10),
    },
    metadata: {
      created: new Date().toISOString(),
      source: 'synthetic',
      note: 'Randomly initialized weights - replace with trained model for accurate predictions',
    },
  };

  // Ensure directory exists
  if (!existsSync(MODELS_DIR)) {
    mkdirSync(MODELS_DIR, { recursive: true });
  }

  // Save weights
  writeFileSync(MODEL_PATH, JSON.stringify(modelWeights, null, 2));
  console.log('✅ Synthetic weights saved to:', MODEL_PATH);
  console.log('');
  console.log('📝 Note: Using randomly initialized weights.');
  console.log('   For accurate predictions, you need trained weights.');
  console.log('   See README for instructions on obtaining trained models.');
}

async function downloadModel() {
  // Check if model already exists
  if (existsSync(MODEL_PATH)) {
    console.log('✅ MNIST model weights already exist at:', MODEL_PATH);
    return;
  }

  console.log('📥 Checking for MNIST model weights...');
  console.log('');
  console.log('⚠️  Model weights not found!');
  console.log('');
  console.log('Creating synthetic weights as fallback...');
  console.log('(For accurate predictions, train a real model with: python scripts/train_mnist.py)');
  console.log('');

  // Create synthetic weights so the demo at least loads
  createSyntheticWeights();

  console.log('');
  console.log('To get accurate predictions:');
  console.log('');
  console.log('  1. Install PyTorch:');
  console.log('     pip install torch torchvision');
  console.log('');
  console.log('  2. Train the model (~2-3 minutes):');
  console.log('     python scripts/train_mnist.py');
  console.log('');
  console.log('  3. Restart the dev server:');
  console.log('     bun run dev');
  console.log('');
}

// Run the download
downloadModel();
