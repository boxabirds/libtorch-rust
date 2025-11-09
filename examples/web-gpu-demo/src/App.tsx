import React, { useState, useEffect } from 'react';
import { initializeGPU, isWebGPUSupported, GpuDeviceInfo } from './webgpu/device';
import * as ops from './webgpu/operations';

interface DemoResult {
  name: string;
  status: 'pending' | 'running' | 'success' | 'error';
  timeMs?: number;
  throughput?: number;
  gflops?: number;
  error?: string;
  details?: string;
}

export default function App() {
  const [supported, setSupported] = useState(false);
  const [gpuInfo, setGpuInfo] = useState<GpuDeviceInfo | null>(null);
  const [error, setError] = useState<string>('');
  const [results, setResults] = useState<DemoResult[]>([]);
  const [running, setRunning] = useState(false);

  useEffect(() => {
    setSupported(isWebGPUSupported());
  }, []);

  const updateResult = (name: string, update: Partial<DemoResult>) => {
    setResults((prev) =>
      prev.map((r) => (r.name === name ? { ...r, ...update } : r))
    );
  };

  const runDemo = async (name: string, fn: () => Promise<any>) => {
    updateResult(name, { status: 'running' });
    try {
      const result = await fn();
      updateResult(name, {
        status: 'success',
        timeMs: result.timeMs,
        throughput: result.throughput,
        gflops: result.gflops,
        details: result.details,
      });
    } catch (err: any) {
      updateResult(name, { status: 'error', error: err.message });
    }
  };

  const handleInitialize = async () => {
    try {
      const info = await initializeGPU();
      setGpuInfo(info);
    } catch (err: any) {
      setError(err.message);
    }
  };

  const handleRunDemos = async () => {
    if (!gpuInfo) return;

    setRunning(true);
    const device = gpuInfo.device;

    // Initialize demo results
    setResults([
      { name: 'Element-wise Addition', status: 'pending' },
      { name: 'Element-wise Multiplication', status: 'pending' },
      { name: 'Matrix Multiplication', status: 'pending' },
      { name: 'ReLU Activation', status: 'pending' },
      { name: 'Sigmoid Activation', status: 'pending' },
    ]);

    // Demo 1: Element-wise Addition (10M elements)
    await runDemo('Element-wise Addition', async () => {
      const size = 10_000_000;
      const a = new Float32Array(size).map((_, i) => i * 0.1);
      const b = new Float32Array(size).map((_, i) => i * 0.2);
      const result = await ops.elementwiseAdd(device, a, b);
      return {
        ...result,
        details: `${(size / 1_000_000).toFixed(1)}M elements, ${(
          size * 4 / 1_000_000
        ).toFixed(1)} MB each`,
      };
    });

    // Demo 2: Element-wise Multiplication (5M elements)
    await runDemo('Element-wise Multiplication', async () => {
      const size = 5_000_000;
      const a = new Float32Array(size).map((_, i) => Math.sin(i));
      const b = new Float32Array(size).map((_, i) => Math.cos(i));
      const result = await ops.elementwiseMul(device, a, b);
      return {
        ...result,
        details: `${(size / 1_000_000).toFixed(1)}M elements, ${(
          size * 4 / 1_000_000
        ).toFixed(1)} MB each`,
      };
    });

    // Demo 3: Matrix Multiplication (512x512)
    await runDemo('Matrix Multiplication', async () => {
      const size = 512;
      const a = new Float32Array(size * size).map(
        (_, i) => (Math.sin(i * 0.01) + 1) * 0.5
      );
      const b = new Float32Array(size * size).map(
        (_, i) => (Math.cos(i * 0.01) + 1) * 0.5
      );
      const result = await ops.matmul(device, a, b, size, size, size);
      return {
        ...result,
        details: `${size}×${size} matrices, ${(
          size * size * 4 / 1_000_000
        ).toFixed(1)} MB each`,
      };
    });

    // Demo 4: ReLU (8M elements)
    await runDemo('ReLU Activation', async () => {
      const size = 8_000_000;
      const input = new Float32Array(size).map(
        (_, i) => Math.sin(i * 0.001) * 10
      );
      const result = await ops.relu(device, input);
      return {
        ...result,
        details: `${(size / 1_000_000).toFixed(1)}M elements, ${(
          size * 4 / 1_000_000
        ).toFixed(1)} MB`,
      };
    });

    // Demo 5: Sigmoid (6M elements)
    await runDemo('Sigmoid Activation', async () => {
      const size = 6_000_000;
      const input = new Float32Array(size).map(
        (_, i) => (i / size) * 20 - 10
      );
      const result = await ops.sigmoid(device, input);
      return {
        ...result,
        details: `${(size / 1_000_000).toFixed(1)}M elements, ${(
          size * 4 / 1_000_000
        ).toFixed(1)} MB`,
      };
    });

    setRunning(false);
  };

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <h1 style={styles.title}>🚀 LibTorch-Rust WebGPU Demo</h1>
        <p style={styles.subtitle}>
          Browser-based ML with GPU Acceleration
        </p>
      </div>

      {!supported ? (
        <div style={styles.errorCard}>
          <h2>❌ WebGPU Not Supported</h2>
          <p>
            Your browser doesn't support WebGPU. Please use:
          </p>
          <ul style={styles.list}>
            <li>Chrome 113+ or Edge 113+</li>
            <li>Safari 18+ (macOS)</li>
            <li>Firefox Nightly (with flags enabled)</li>
          </ul>
        </div>
      ) : !gpuInfo ? (
        <div style={styles.card}>
          {error ? (
            <>
              <h2>❌ Initialization Failed</h2>
              <p style={styles.error}>{error}</p>
            </>
          ) : (
            <>
              <h2>📱 Ready to Initialize GPU</h2>
              <p>Click below to detect and initialize your GPU device.</p>
              <button style={styles.button} onClick={handleInitialize}>
                Initialize WebGPU
              </button>
            </>
          )}
        </div>
      ) : (
        <>
          <div style={styles.card}>
            <h2>✅ GPU Initialized</h2>
            <div style={styles.infoGrid}>
              <div><strong>Vendor:</strong> {gpuInfo.info.vendor}</div>
              <div><strong>Device:</strong> {gpuInfo.info.device}</div>
              <div><strong>Architecture:</strong> {gpuInfo.info.architecture}</div>
              <div><strong>Description:</strong> {gpuInfo.info.description}</div>
            </div>
            <button
              style={{ ...styles.button, marginTop: '20px' }}
              onClick={handleRunDemos}
              disabled={running}
            >
              {running ? '⏳ Running Demos...' : '▶️ Run GPU Demos'}
            </button>
          </div>

          {results.length > 0 && (
            <div style={styles.resultsContainer}>
              {results.map((result) => (
                <div key={result.name} style={styles.resultCard}>
                  <div style={styles.resultHeader}>
                    <span style={styles.resultName}>{result.name}</span>
                    <span style={styles.resultStatus}>
                      {result.status === 'pending' && '⏸️ Pending'}
                      {result.status === 'running' && '⏳ Running...'}
                      {result.status === 'success' && '✅ Success'}
                      {result.status === 'error' && '❌ Error'}
                    </span>
                  </div>
                  {result.details && (
                    <div style={styles.resultDetails}>{result.details}</div>
                  )}
                  {result.status === 'success' && (
                    <div style={styles.resultMetrics}>
                      <div>
                        <strong>Time:</strong> {result.timeMs?.toFixed(2)} ms
                      </div>
                      {result.throughput && (
                        <div>
                          <strong>Throughput:</strong>{' '}
                          {(result.throughput / 1_000_000).toFixed(1)} M ops/sec
                        </div>
                      )}
                      {result.gflops && (
                        <div>
                          <strong>Performance:</strong>{' '}
                          {result.gflops.toFixed(2)} GFLOPS
                        </div>
                      )}
                    </div>
                  )}
                  {result.error && (
                    <div style={styles.error}>{result.error}</div>
                  )}
                </div>
              ))}
            </div>
          )}
        </>
      )}

      <footer style={styles.footer}>
        <p>
          This demo showcases WebGPU compute shaders running the same operations
          as the Rust implementation. The same WGSL shaders work in both native
          (via wgpu) and browser (via WebGPU API) environments.
        </p>
      </footer>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    color: 'white',
  },
  header: {
    textAlign: 'center',
    marginBottom: '30px',
  },
  title: {
    fontSize: '48px',
    marginBottom: '10px',
    textShadow: '2px 2px 4px rgba(0,0,0,0.3)',
  },
  subtitle: {
    fontSize: '20px',
    opacity: 0.9,
  },
  card: {
    background: 'rgba(255, 255, 255, 0.1)',
    backdropFilter: 'blur(10px)',
    borderRadius: '16px',
    padding: '30px',
    marginBottom: '20px',
    border: '1px solid rgba(255, 255, 255, 0.2)',
  },
  errorCard: {
    background: 'rgba(220, 38, 38, 0.1)',
    backdropFilter: 'blur(10px)',
    borderRadius: '16px',
    padding: '30px',
    border: '1px solid rgba(220, 38, 38, 0.3)',
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
    transition: 'transform 0.2s',
  },
  infoGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
    gap: '15px',
    marginTop: '20px',
  },
  error: {
    color: '#fca5a5',
    marginTop: '10px',
  },
  list: {
    marginTop: '15px',
    marginLeft: '20px',
  },
  resultsContainer: {
    display: 'grid',
    gap: '15px',
  },
  resultCard: {
    background: 'rgba(255, 255, 255, 0.05)',
    borderRadius: '12px',
    padding: '20px',
    border: '1px solid rgba(255, 255, 255, 0.1)',
  },
  resultHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '10px',
  },
  resultName: {
    fontSize: '18px',
    fontWeight: 'bold',
  },
  resultStatus: {
    fontSize: '14px',
    opacity: 0.8,
  },
  resultDetails: {
    fontSize: '14px',
    opacity: 0.7,
    marginBottom: '10px',
  },
  resultMetrics: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
    gap: '10px',
    fontSize: '14px',
  },
  footer: {
    marginTop: '40px',
    textAlign: 'center',
    opacity: 0.7,
    fontSize: '14px',
  },
};
