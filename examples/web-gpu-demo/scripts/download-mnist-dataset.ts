/**
 * Download and prepare MNIST dataset for browser-based training
 *
 * This script:
 * 1. Downloads the raw MNIST dataset from Yann LeCun's website
 * 2. Parses the IDX file format
 * 3. Creates browser-friendly JSON files with image data
 * 4. Generates both full dataset and a smaller subset for quick demos
 */

import { writeFileSync, existsSync, mkdirSync } from 'fs';
import { join } from 'path';
import { gunzip } from 'zlib';
import { promisify } from 'util';

const gunzipAsync = promisify(gunzip);

// Using multiple mirror sources for reliability
// Primary: GitHub ossdata-cdn mirror (reliable)
// Fallback: Direct PyTorch mirror
const MNIST_MIRRORS = [
  'https://ossci-datasets.s3.amazonaws.com/mnist/',
  'https://storage.googleapis.com/cvdf-datasets/mnist/',
];

const DATA_DIR = join(import.meta.dir, '..', 'public', 'data');
const MNIST_DIR = join(DATA_DIR, 'mnist');

interface MNISTFile {
  filename: string;
  localName: string;
  type: 'images' | 'labels';
}

const MNIST_FILES: MNISTFile[] = [
  { filename: 'train-images-idx3-ubyte.gz', localName: 'train-images.gz', type: 'images' },
  { filename: 'train-labels-idx1-ubyte.gz', localName: 'train-labels.gz', type: 'labels' },
  { filename: 't10k-images-idx3-ubyte.gz', localName: 'test-images.gz', type: 'images' },
  { filename: 't10k-labels-idx1-ubyte.gz', localName: 'test-labels.gz', type: 'labels' },
];

/**
 * Download a file from URL with mirror fallback
 */
async function downloadFile(filename: string, outputPath: string): Promise<void> {
  for (let i = 0; i < MNIST_MIRRORS.length; i++) {
    const url = MNIST_MIRRORS[i] + filename;
    const mirrorName = i === 0 ? 'primary' : `fallback ${i}`;

    try {
      console.log(`   Downloading from ${mirrorName}: ${filename}...`);
      const response = await fetch(url);

      if (!response.ok) {
        console.log(`   ⚠️  Mirror ${mirrorName} failed: ${response.statusText}`);
        continue; // Try next mirror
      }

      const buffer = await response.arrayBuffer();
      writeFileSync(outputPath, Buffer.from(buffer));
      console.log(`   ✅ Downloaded ${filename} (${(buffer.byteLength / 1024 / 1024).toFixed(1)} MB)`);
      return; // Success!
    } catch (error) {
      console.log(`   ⚠️  Mirror ${mirrorName} error: ${error}`);
      if (i === MNIST_MIRRORS.length - 1) {
        throw new Error(`All mirrors failed for ${filename}`);
      }
    }
  }
}

/**
 * Parse IDX format images file
 */
function parseIDXImages(buffer: Buffer): Float32Array[] {
  const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);

  // Read header
  const magic = view.getInt32(0, false);
  if (magic !== 0x00000803) {
    throw new Error(`Invalid magic number for images: ${magic}`);
  }

  const numImages = view.getInt32(4, false);
  const numRows = view.getInt32(8, false);
  const numCols = view.getInt32(12, false);

  console.log(`   Found ${numImages} images of ${numRows}x${numCols}`);

  const images: Float32Array[] = [];
  const imageSize = numRows * numCols;
  const offset = 16; // Header size

  for (let i = 0; i < numImages; i++) {
    const imageData = new Float32Array(imageSize);

    for (let j = 0; j < imageSize; j++) {
      // Normalize pixel values to [0, 1]
      const pixel = buffer[offset + i * imageSize + j];
      imageData[j] = pixel / 255.0;
    }

    images.push(imageData);
  }

  return images;
}

/**
 * Parse IDX format labels file
 */
function parseIDXLabels(buffer: Buffer): Uint8Array {
  const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);

  // Read header
  const magic = view.getInt32(0, false);
  if (magic !== 0x00000801) {
    throw new Error(`Invalid magic number for labels: ${magic}`);
  }

  const numLabels = view.getInt32(4, false);
  console.log(`   Found ${numLabels} labels`);

  const labels = new Uint8Array(buffer.subarray(8, 8 + numLabels));
  return labels;
}

/**
 * Create a JSON dataset file
 */
function createDatasetJSON(
  images: Float32Array[],
  labels: Uint8Array,
  outputPath: string,
  maxSamples?: number
): void {
  const numSamples = maxSamples ? Math.min(maxSamples, images.length) : images.length;

  console.log(`   Creating JSON with ${numSamples} samples...`);

  const dataset = {
    metadata: {
      num_samples: numSamples,
      image_size: [28, 28],
      num_classes: 10,
      created: new Date().toISOString(),
    },
    samples: [] as Array<{ image: number[]; label: number }>,
  };

  for (let i = 0; i < numSamples; i++) {
    dataset.samples.push({
      image: Array.from(images[i]),
      label: labels[i],
    });
  }

  writeFileSync(outputPath, JSON.stringify(dataset, null, 2));
  const sizeKB = (require('fs').statSync(outputPath).size / 1024).toFixed(1);
  console.log(`   ✅ Saved ${numSamples} samples to ${outputPath} (${sizeKB} KB)`);
}

/**
 * Create batched binary format for faster loading
 */
function createBatchedDataset(
  images: Float32Array[],
  labels: Uint8Array,
  outputPath: string,
  batchSize: number = 32
): void {
  console.log(`   Creating batched dataset (batch size ${batchSize})...`);

  const batches = [];
  for (let i = 0; i < images.length; i += batchSize) {
    const batchImages = images.slice(i, i + batchSize);
    const batchLabels = Array.from(labels.slice(i, i + batchSize));

    // Flatten all images in batch into single array
    const flatImages = new Float32Array(batchImages.length * 784);
    batchImages.forEach((img, idx) => {
      flatImages.set(img, idx * 784);
    });

    batches.push({
      images: Array.from(flatImages),
      labels: batchLabels,
    });
  }

  const dataset = {
    metadata: {
      num_batches: batches.length,
      batch_size: batchSize,
      total_samples: images.length,
      image_size: [28, 28],
      num_classes: 10,
    },
    batches,
  };

  writeFileSync(outputPath, JSON.stringify(dataset));
  const sizeKB = (require('fs').statSync(outputPath).size / 1024).toFixed(1);
  console.log(`   ✅ Saved ${batches.length} batches to ${outputPath} (${sizeKB} KB)`);
}

/**
 * Main download and processing pipeline
 */
async function downloadMNIST(): Promise<void> {
  console.log('📥 Downloading and processing MNIST dataset...\n');

  // Create directories
  if (!existsSync(MNIST_DIR)) {
    mkdirSync(MNIST_DIR, { recursive: true });
  }

  // Check if already downloaded - look for train-subset.json which is always created
  const trainSubsetPath = join(MNIST_DIR, 'train-subset.json');
  if (existsSync(trainSubsetPath)) {
    console.log('✅ MNIST dataset already exists!');
    console.log('   Files: train-subset.json, train-batched.json, test-batched.json, etc.');
    console.log('   To re-download, delete the public/data/mnist directory.\n');
    return;
  }

  try {
    // Download all files
    console.log('1️⃣ Downloading MNIST files...');
    for (const file of MNIST_FILES) {
      const outputPath = join(MNIST_DIR, file.localName);
      await downloadFile(file.filename, outputPath);
    }

    console.log('\n2️⃣ Decompressing and parsing files...');

    // Process training data
    console.log('   Processing training data...');
    const trainImagesGz = Bun.file(join(MNIST_DIR, 'train-images.gz'));
    const trainLabelsGz = Bun.file(join(MNIST_DIR, 'train-labels.gz'));

    const trainImagesBuffer = Buffer.from(await gunzipAsync(Buffer.from(await trainImagesGz.arrayBuffer())));
    const trainLabelsBuffer = Buffer.from(await gunzipAsync(Buffer.from(await trainLabelsGz.arrayBuffer())));

    const trainImages = parseIDXImages(trainImagesBuffer);
    const trainLabels = parseIDXLabels(trainLabelsBuffer);

    // Process test data
    console.log('   Processing test data...');
    const testImagesGz = Bun.file(join(MNIST_DIR, 'test-images.gz'));
    const testLabelsGz = Bun.file(join(MNIST_DIR, 'test-labels.gz'));

    const testImagesBuffer = Buffer.from(await gunzipAsync(Buffer.from(await testImagesGz.arrayBuffer())));
    const testLabelsBuffer = Buffer.from(await gunzipAsync(Buffer.from(await testLabelsGz.arrayBuffer())));

    const testImages = parseIDXImages(testImagesBuffer);
    const testLabels = parseIDXLabels(testLabelsBuffer);

    console.log('\n3️⃣ Creating JSON datasets...');

    // Create small subset for quick testing (1000 samples)
    console.log('   Creating training subset (1000 samples)...');
    createDatasetJSON(trainImages, trainLabels, join(MNIST_DIR, 'train-subset.json'), 1000);

    // Create batched format for efficient browser training
    console.log('   Creating batched training data...');
    createBatchedDataset(trainImages, trainLabels, join(MNIST_DIR, 'train-batched.json'), 32);

    console.log('   Creating batched test data...');
    createBatchedDataset(testImages, testLabels, join(MNIST_DIR, 'test-batched.json'), 100);

    // Also create full JSON files (warning: large!)
    console.log('   Creating full training dataset (60,000 samples - this may take a while)...');
    createDatasetJSON(trainImages, trainLabels, join(MNIST_DIR, 'train-full.json'));

    console.log('   Creating full test dataset (10,000 samples)...');
    createDatasetJSON(testImages, testLabels, join(MNIST_DIR, 'test-full.json'));

    console.log('\n✅ MNIST dataset downloaded and processed successfully!\n');
    console.log('📊 Files created:');
    console.log('   • train-subset.json - 1,000 training samples (good for quick demos)');
    console.log('   • train-batched.json - 60,000 samples in batches of 32');
    console.log('   • test-batched.json - 10,000 test samples in batches of 100');
    console.log('   • train-full.json - Full 60,000 training samples (large file!)');
    console.log('   • test-full.json - Full 10,000 test samples');
    console.log('\n💡 Recommendation: Use train-batched.json for browser training');
    console.log('   It\'s optimized for batch processing and smaller than the full JSON.');

  } catch (error) {
    console.error('\n❌ Error downloading MNIST dataset:', error);
    console.error('\nYou can try again or download manually from:');
    console.error('   https://ossci-datasets.s3.amazonaws.com/mnist/');
    console.error('   https://storage.googleapis.com/cvdf-datasets/mnist/');
    console.error('\nAlternatively, use PyTorch to download:');
    console.error('   python scripts/train_mnist.py  (will download via torchvision)');
    process.exit(1);
  }
}

// Run the download
downloadMNIST();
