import React, { useState, useEffect } from 'react';

interface TrainingDemoProps {
  gpuInfo: any;
}

interface TrainingMetrics {
  epoch: number;
  batch: number;
  loss: number;
  accuracy?: number;
}

export default function TrainingDemo({ gpuInfo }: TrainingDemoProps) {
  const [wasmLoaded, setWasmLoaded] = useState(false);
  const [wasmError, setWasmError] = useState<string>('');
  const [datasetLoaded, setDatasetLoaded] = useState(false);
  const [training, setTraining] = useState(false);
  const [metrics, setMetrics] = useState<TrainingMetrics[]>([]);
  const [currentMetric, setCurrentMetric] = useState<TrainingMetrics | null>(null);

  useEffect(() => {
    loadWasm();
    checkDataset();
  }, []);

  const loadWasm = async () => {
    try {
      console.log('🔄 Loading WASM module...');
      // Dynamic import of WASM module
      const wasmModule = await import('/wasm/libtorch_wasm_trainer.js');
      await wasmModule.default();

      // Store the module for later use
      (window as any).__wasmTrainer = wasmModule;

      console.log('✅ WASM module loaded successfully');
      console.log('   BrowserTrainer:', wasmModule.BrowserTrainer);
      setWasmLoaded(true);
    } catch (err: any) {
      console.error('❌ Failed to load WASM:', err);
      setWasmError(err.message || 'Failed to load WASM module');
    }
  };

  const checkDataset = async () => {
    try {
      // Check if training data exists
      const response = await fetch('/data/mnist/train-subset.json');
      if (response.ok) {
        console.log('✅ MNIST training dataset found');
        setDatasetLoaded(true);
      } else {
        console.log('⚠️ MNIST training dataset not found');
        setDatasetLoaded(false);
      }
    } catch (err) {
      console.log('⚠️ Could not check for MNIST dataset');
      setDatasetLoaded(false);
    }
  };

  const handleStartTraining = async () => {
    if (!wasmLoaded || !datasetLoaded) return;

    setTraining(true);
    setMetrics([]);
    setCurrentMetric(null);

    try {
      console.log('🚀 Starting browser-based training...');

      // Load training data
      const response = await fetch('/data/mnist/train-subset.json');
      const datasetJSON = await response.json();

      // Extract samples array from dataset structure
      const dataset = datasetJSON.samples || datasetJSON;

      console.log(`📦 Loaded ${dataset.length} training samples`);

      // Initialize WASM trainer
      const wasmModule = (window as any).__wasmTrainer;
      const trainer = new wasmModule.BrowserTrainer(0.001); // learning_rate = 0.001

      console.log('✅ WASM trainer initialized');
      console.log('   Model:', trainer.model_info());

      // Run real training loop
      await trainWithWasm(trainer, dataset);

      // Export trained weights
      console.log('📤 Exporting trained weights...');
      const weightsJson = trainer.export_full(0.0); // TODO: compute actual test accuracy

      // Save to models directory
      console.log('💾 Trained weights ready for download');
      console.log('   Use these for MNIST Inference demo!');

      // Trigger download
      const blob = new Blob([weightsJson], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'mnist-mlp-browser-trained.json';
      a.click();
      URL.revokeObjectURL(url);

      alert('✅ Training complete! Weights downloaded. Replace public/models/mnist-mlp.json with this file.');

    } catch (err: any) {
      console.error('❌ Training failed:', err);
      alert(`Training failed: ${err.message}`);
    } finally {
      setTraining(false);
    }
  };

  // Real WASM training loop
  const trainWithWasm = async (trainer: any, dataset: any[]) => {
    const epochs = 5;
    const batchSize = 32;
    const numSamples = dataset.length;
    const batchesPerEpoch = Math.floor(numSamples / batchSize);

    console.log(`🎯 Training: ${epochs} epochs, ${batchesPerEpoch} batches/epoch`);

    for (let epoch = 0; epoch < epochs; epoch++) {
      console.log(`\n📊 Epoch ${epoch + 1}/${epochs}`);

      // Shuffle dataset
      const shuffled = [...dataset].sort(() => Math.random() - 0.5);

      for (let batchIdx = 0; batchIdx < batchesPerEpoch; batchIdx++) {
        const startIdx = batchIdx * batchSize;
        const endIdx = Math.min(startIdx + batchSize, numSamples);
        const batchData = shuffled.slice(startIdx, endIdx);

        // Prepare batch data
        const images: number[] = [];
        const labels: number[] = [];

        for (const sample of batchData) {
          // Flatten image to 784 values
          const flatImage = sample.image.flat();
          images.push(...flatImage);
          labels.push(sample.label);
        }

        // Train on batch (this calls Rust code!)
        const loss = trainer.train_batch(images, labels);

        // Evaluate occasionally for accuracy
        let accuracy = 0;
        if (batchIdx % 5 === 0) {
          const evalResult = trainer.eval_batch(images, labels);
          accuracy = evalResult[1]; // [loss, accuracy]
        }

        const metric: TrainingMetrics = {
          epoch: epoch + 1,
          batch: batchIdx + 1,
          loss,
          accuracy,
        };

        setCurrentMetric(metric);
        setMetrics(prev => [...prev, metric]);

        // Log progress
        if (batchIdx % 10 === 0) {
          console.log(`  Batch ${batchIdx + 1}/${batchesPerEpoch}: loss=${loss.toFixed(4)}`);
        }

        // Small delay to keep UI responsive
        if (batchIdx % 5 === 0) {
          await new Promise(resolve => setTimeout(resolve, 10));
        }
      }
    }

    console.log('✅ Training complete!');
  };

  return (
    <div style={styles.container}>
      <div style={styles.card}>
        <h2>🎓 Browser-Based Training</h2>
        <p style={styles.description}>
          Train a neural network directly in your browser using Rust + WASM + Burn framework.
          The model trains on MNIST digits and can export PyTorch-compatible weights.
        </p>

        <div style={styles.statusGrid}>
          <div style={styles.statusItem}>
            <span style={styles.statusLabel}>WASM Module:</span>
            <span style={wasmLoaded ? styles.statusSuccess : styles.statusError}>
              {wasmLoaded ? '✅ Loaded' : wasmError ? `❌ ${wasmError}` : '⏳ Loading...'}
            </span>
          </div>
          <div style={styles.statusItem}>
            <span style={styles.statusLabel}>Training Data:</span>
            <span style={datasetLoaded ? styles.statusSuccess : styles.statusWarning}>
              {datasetLoaded ? '✅ Ready (1,000 samples)' : '⚠️ Run: bun run download-mnist'}
            </span>
          </div>
        </div>

        {wasmLoaded && datasetLoaded && (
          <button
            style={{
              ...styles.button,
              ...(training ? styles.buttonDisabled : {}),
            }}
            onClick={handleStartTraining}
            disabled={training}
          >
            {training ? '⏳ Training...' : '▶️ Start Training'}
          </button>
        )}

        {!datasetLoaded && (
          <div style={styles.infoBox}>
            <p><strong>📥 Download Training Data:</strong></p>
            <p>Run this command in the terminal:</p>
            <code style={styles.code}>
              cd examples/web-gpu-demo && bun run download-mnist
            </code>
          </div>
        )}
      </div>

      {currentMetric && (
        <div style={styles.card}>
          <h3>📊 Training Progress</h3>
          <div style={styles.metricsGrid}>
            <div style={styles.metricCard}>
              <div style={styles.metricLabel}>Epoch</div>
              <div style={styles.metricValue}>{currentMetric.epoch}</div>
            </div>
            <div style={styles.metricCard}>
              <div style={styles.metricLabel}>Batch</div>
              <div style={styles.metricValue}>{currentMetric.batch}</div>
            </div>
            <div style={styles.metricCard}>
              <div style={styles.metricLabel}>Loss</div>
              <div style={styles.metricValue}>{currentMetric.loss.toFixed(4)}</div>
            </div>
            <div style={styles.metricCard}>
              <div style={styles.metricLabel}>Accuracy</div>
              <div style={styles.metricValue}>
                {currentMetric.accuracy ? `${(currentMetric.accuracy * 100).toFixed(1)}%` : 'N/A'}
              </div>
            </div>
          </div>

          {metrics.length > 0 && (
            <div style={styles.chartContainer}>
              <h4>Loss Over Time</h4>
              <div style={styles.chart}>
                {metrics.slice(-20).map((m, i) => (
                  <div
                    key={i}
                    style={{
                      ...styles.bar,
                      height: `${Math.min(100, m.loss * 40)}%`,
                    }}
                    title={`Epoch ${m.epoch}, Batch ${m.batch}: Loss ${m.loss.toFixed(4)}`}
                  />
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {training && (
        <div style={styles.card}>
          <h3>ℹ️ What's Happening (REAL Training!)</h3>
          <ul style={styles.infoList}>
            <li>✅ WASM module loaded (Rust + Burn framework)</li>
            <li>✅ Training data loaded ({currentMetric?.epoch}/5 epochs)</li>
            <li>🔥 Forward pass: Rust code computing predictions</li>
            <li>🔥 Backward pass: Automatic differentiation calculating gradients</li>
            <li>🔥 Adam optimizer: Updating 100,480 weights</li>
            <li>⚡ All computation happening in your browser via WebAssembly!</li>
          </ul>
        </div>
      )}

      <div style={styles.card}>
        <h3>🔧 Technical Details</h3>
        <div style={styles.techDetails}>
          <div><strong>Framework:</strong> Burn 0.19 (Rust ML framework)</div>
          <div><strong>Backend:</strong> NdArray (CPU) - WebGPU coming in Milestone 3</div>
          <div><strong>Model:</strong> MLP (784 → 128 → 10)</div>
          <div><strong>Optimizer:</strong> Adam with 0.001 learning rate</div>
          <div><strong>Loss:</strong> Cross-Entropy</div>
          <div><strong>WASM Size:</strong> 2.9 MB (unoptimized)</div>
        </div>
      </div>

      <div style={styles.infoBox}>
        <p><strong>🚀 Training Workflow:</strong></p>
        <ul style={styles.infoList}>
          <li>✅ Click "Start Training" to train a model from scratch</li>
          <li>✅ Watch real-time loss and accuracy (Rust WASM doing backprop!)</li>
          <li>✅ After training, weights auto-download as JSON</li>
          <li>📝 Replace <code style={styles.inlineCode}>public/models/mnist-mlp.json</code> with downloaded file</li>
          <li>🎨 Refresh and use MNIST Inference tab with your trained model!</li>
        </ul>
        <p style={{marginTop: '10px', opacity: 0.8}}><strong>Coming:</strong> WebGPU backend for GPU-accelerated training</p>
      </div>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: 'grid',
    gap: '20px',
  },
  card: {
    background: 'rgba(255, 255, 255, 0.1)',
    backdropFilter: 'blur(10px)',
    borderRadius: '16px',
    padding: '30px',
    border: '1px solid rgba(255, 255, 255, 0.2)',
  },
  description: {
    fontSize: '16px',
    opacity: 0.9,
    marginBottom: '20px',
    lineHeight: '1.6',
  },
  statusGrid: {
    display: 'grid',
    gap: '15px',
    marginBottom: '20px',
  },
  statusItem: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '12px',
    background: 'rgba(255, 255, 255, 0.05)',
    borderRadius: '8px',
  },
  statusLabel: {
    fontWeight: 'bold',
  },
  statusSuccess: {
    color: '#86efac',
  },
  statusError: {
    color: '#fca5a5',
  },
  statusWarning: {
    color: '#fcd34d',
  },
  button: {
    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    color: 'white',
    border: 'none',
    padding: '15px 30px',
    fontSize: '18px',
    borderRadius: '8px',
    cursor: 'pointer',
    fontWeight: 'bold',
    width: '100%',
    transition: 'transform 0.2s',
  },
  buttonDisabled: {
    opacity: 0.6,
    cursor: 'not-allowed',
  },
  metricsGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))',
    gap: '15px',
    marginTop: '20px',
  },
  metricCard: {
    background: 'rgba(255, 255, 255, 0.05)',
    borderRadius: '12px',
    padding: '15px',
    textAlign: 'center',
  },
  metricLabel: {
    fontSize: '12px',
    opacity: 0.7,
    marginBottom: '8px',
    textTransform: 'uppercase',
  },
  metricValue: {
    fontSize: '24px',
    fontWeight: 'bold',
  },
  chartContainer: {
    marginTop: '30px',
  },
  chart: {
    display: 'flex',
    alignItems: 'flex-end',
    height: '100px',
    gap: '2px',
    marginTop: '15px',
  },
  bar: {
    flex: 1,
    background: 'linear-gradient(180deg, #667eea 0%, #764ba2 100%)',
    borderRadius: '2px 2px 0 0',
    transition: 'height 0.3s',
    minHeight: '2px',
  },
  infoBox: {
    background: 'rgba(59, 130, 246, 0.1)',
    border: '1px solid rgba(59, 130, 246, 0.3)',
    borderRadius: '12px',
    padding: '20px',
    marginTop: '20px',
  },
  code: {
    display: 'block',
    background: 'rgba(0, 0, 0, 0.3)',
    padding: '12px',
    borderRadius: '6px',
    fontFamily: 'monospace',
    fontSize: '14px',
    marginTop: '10px',
    overflowX: 'auto',
  },
  infoList: {
    marginLeft: '20px',
    lineHeight: '1.8',
  },
  techDetails: {
    display: 'grid',
    gap: '10px',
    fontSize: '14px',
    marginTop: '15px',
  },
  inlineCode: {
    background: 'rgba(0, 0, 0, 0.3)',
    padding: '2px 6px',
    borderRadius: '4px',
    fontFamily: 'monospace',
    fontSize: '13px',
  },
};
