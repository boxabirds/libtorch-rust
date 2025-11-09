# MNIST Dataset

This directory contains the MNIST dataset in browser-friendly JSON format.

## Downloading the Dataset

Run from the project root:

```bash
bun run download-mnist
```

This will download and process the MNIST dataset, creating:

- `train-subset.json` - 1,000 training samples (good for quick demos)
- `train-batched.json` - 60,000 samples in batches of 32 (recommended for training)
- `test-batched.json` - 10,000 test samples in batches of 100
- `train-full.json` - Full 60,000 training samples (large file, ~140MB)
- `test-full.json` - Full 10,000 test samples (large file, ~23MB)

## File Formats

### Subset/Full Format

```json
{
  "metadata": {
    "num_samples": 1000,
    "image_size": [28, 28],
    "num_classes": 10
  },
  "samples": [
    {
      "image": [0.0, 0.1, ...],  // 784 pixels, normalized to [0, 1]
      "label": 5                  // digit class (0-9)
    }
  ]
}
```

### Batched Format (Recommended)

```json
{
  "metadata": {
    "num_batches": 1875,
    "batch_size": 32,
    "total_samples": 60000
  },
  "batches": [
    {
      "images": [...],  // Flattened array: batch_size * 784
      "labels": [...]   // Array of labels: batch_size
    }
  ]
}
```

## Usage in Browser

```javascript
// Load subset for quick demo
const response = await fetch('/data/mnist/train-subset.json');
const dataset = await response.json();

// Use with BrowserTrainer
for (const sample of dataset.samples) {
  const loss = trainer.train_batch(sample.image, [sample.label]);
}
```

## Dataset Source

Downloaded from: http://yann.lecun.com/exdb/mnist/

Original MNIST database of handwritten digits by Yann LeCun, Corinna Cortes, and Christopher Burges.
