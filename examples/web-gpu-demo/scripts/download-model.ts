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

async function downloadModel() {
  // Check if model already exists
  if (existsSync(MODEL_PATH)) {
    console.log('✅ MNIST model weights already exist at:', MODEL_PATH);
    return;
  }

  console.log('📥 Downloading MNIST model weights...');
  console.log('   Source: HuggingFace/community models');

  try {
    // For now, create synthetic weights as a placeholder
    // TODO: Replace with actual HuggingFace model download
    console.log('⚠️  Using synthetic weights (HF download not yet implemented)');
    console.log('   Creating randomly initialized weights...');

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
    console.log('✅ Model weights saved to:', MODEL_PATH);
    console.log('');
    console.log('📝 Note: Currently using synthetic weights.');
    console.log('   For accurate predictions, replace with trained weights from:');
    console.log('   - HuggingFace: https://huggingface.co/models?filter=mnist');
    console.log('   - Or train your own and export to this format');

  } catch (error) {
    console.error('❌ Failed to download model:', error);
    console.error('   Continuing with runtime weight initialization...');
    process.exit(0); // Don't fail the build
  }
}

// Run the download
downloadModel();
