# Model Weights

This directory contains trained MNIST model weights.

## To generate weights:

```bash
cd ../..
python scripts/train_mnist.py
```

This will create `mnist-mlp.json` in this directory (~400KB file).

## Weight file format:

The trained model should be a JSON file with this structure:

```json
{
  "model_type": "mnist-mlp",
  "architecture": {
    "input_size": 784,
    "hidden_size": 128,
    "output_size": 10
  },
  "weights": {
    "fc1_weight": [...],  // 128×784 = 100,352 floats
    "fc1_bias": [...],    // 128 floats
    "fc2_weight": [...],  // 10×128 = 1,280 floats
    "fc2_bias": [...]     // 10 floats
  },
  "metadata": {
    "test_accuracy": 97.5,
    "source": "pytorch_training"
  }
}
```

**Note:** This directory is gitignored to avoid committing large model files.
