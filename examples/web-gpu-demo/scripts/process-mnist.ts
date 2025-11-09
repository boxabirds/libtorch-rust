/**
 * Process already-downloaded MNIST .gz files into JSON format
 */

import { writeFileSync, readFileSync, statSync } from 'fs';
import { join, dirname } from 'path';
import { gunzip } from 'zlib';
import { promisify } from 'util';
import { fileURLToPath } from 'url';

const gunzipAsync = promisify(gunzip);

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const MNIST_DIR = join(__dirname, '..', 'public', 'data', 'mnist');

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
    samples: [] as Array<{ image: number[][]; label: number }>,
  };

  for (let i = 0; i < numSamples; i++) {
    // Convert flat array to 28x28 array
    const image2D: number[][] = [];
    for (let row = 0; row < 28; row++) {
      const rowData: number[] = [];
      for (let col = 0; col < 28; col++) {
        rowData.push(images[i][row * 28 + col]);
      }
      image2D.push(rowData);
    }

    dataset.samples.push({
      image: image2D,
      label: labels[i],
    });
  }

  writeFileSync(outputPath, JSON.stringify(dataset, null, 2));
  const sizeKB = (statSync(outputPath).size / 1024).toFixed(1);
  console.log(`   ✅ Saved ${numSamples} samples to ${outputPath} (${sizeKB} KB)`);
}

async function main() {
  console.log('📦 Processing MNIST files...\n');

  // Process training data
  console.log('1️⃣ Processing training data...');
  const trainImagesGz = readFileSync(join(MNIST_DIR, 'train-images.gz'));
  const trainLabelsGz = readFileSync(join(MNIST_DIR, 'train-labels.gz'));

  const trainImagesBuffer = Buffer.from(await gunzipAsync(trainImagesGz));
  const trainLabelsBuffer = Buffer.from(await gunzipAsync(trainLabelsGz));

  const trainImages = parseIDXImages(trainImagesBuffer);
  const trainLabels = parseIDXLabels(trainLabelsBuffer);

  // Create small subset for quick testing (1000 samples)
  console.log('\n2️⃣ Creating training subset (1000 samples)...');
  createDatasetJSON(trainImages, trainLabels, join(MNIST_DIR, 'train-subset.json'), 1000);

  console.log('\n✅ MNIST dataset processed successfully!\n');
  console.log('📊 Files created:');
  console.log('   • train-subset.json - 1,000 training samples');
}

main().catch(console.error);
