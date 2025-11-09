import React, { useState, useEffect } from 'react';
import { GpuDeviceInfo } from '../webgpu/device';
import { MNISTModel, loadMNISTWeights } from '../webgpu/mnist';
import DrawingCanvas from './DrawingCanvas';

interface MNISTDemoProps {
  gpuInfo: GpuDeviceInfo;
}

export default function MNISTDemo({ gpuInfo }: MNISTDemoProps) {
  const [model, setModel] = useState<MNISTModel | null>(null);
  const [loading, setLoading] = useState(true);
  const [prediction, setPrediction] = useState<number | null>(null);
  const [probabilities, setProbabilities] = useState<Float32Array | null>(null);
  const [inferenceTime, setInferenceTime] = useState<number>(0);

  useEffect(() => {
    async function initModel() {
      try {
        // Load model weights (currently using synthetic weights for demo)
        const weights = await loadMNISTWeights();
        const mnistModel = new MNISTModel(gpuInfo.device, weights);
        setModel(mnistModel);
        setLoading(false);
      } catch (err) {
        console.error('Failed to load model:', err);
        setLoading(false);
      }
    }

    initModel();
  }, [gpuInfo]);

  const handleImageChange = async (imageData: Float32Array) => {
    if (!model) return;

    // Check if canvas is empty (all zeros)
    const isEmpty = imageData.every(x => x === 0);
    if (isEmpty) {
      setPrediction(null);
      setProbabilities(null);
      return;
    }

    try {
      const startTime = performance.now();
      const probs = await model.predict(imageData);
      const endTime = performance.now();

      setInferenceTime(endTime - startTime);
      setProbabilities(probs);

      const predicted = MNISTModel.getPrediction(probs);
      setPrediction(predicted);
    } catch (err) {
      console.error('Inference error:', err);
    }
  };

  if (loading) {
    return (
      <div style={styles.container}>
        <h2 style={styles.title}>📊 MNIST Digit Recognition</h2>
        <p>Loading model...</p>
      </div>
    );
  }

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <h2 style={styles.title}>📊 MNIST Digit Recognition</h2>
        <p style={styles.subtitle}>
          Draw a digit (0-9) and see GPU-accelerated inference in action
        </p>
      </div>

      <div style={styles.content}>
        <div style={styles.canvasSection}>
          <h3 style={styles.sectionTitle}>Draw Here</h3>
          <DrawingCanvas onImageChange={handleImageChange} />
          <p style={styles.hint}>Use your mouse to draw a digit</p>
        </div>

        <div style={styles.resultsSection}>
          <h3 style={styles.sectionTitle}>Prediction</h3>

          {prediction !== null ? (
            <>
              <div style={styles.predictionBox}>
                <div style={styles.predictedDigit}>{prediction}</div>
                <div style={styles.confidence}>
                  {((probabilities![prediction] || 0) * 100).toFixed(1)}% confident
                </div>
              </div>

              <div style={styles.inferenceTime}>
                ⚡ Inference: {inferenceTime.toFixed(2)}ms
              </div>

              <div style={styles.probabilitiesContainer}>
                <h4 style={styles.probabilitiesTitle}>All Probabilities:</h4>
                <div style={styles.probabilities}>
                  {probabilities && Array.from(probabilities).map((prob, idx) => (
                    <div
                      key={idx}
                      style={{
                        ...styles.probabilityBar,
                        ...(idx === prediction ? styles.probabilityBarActive : {}),
                      }}
                    >
                      <span style={styles.probabilityLabel}>{idx}</span>
                      <div style={styles.probabilityTrack}>
                        <div
                          style={{
                            ...styles.probabilityFill,
                            width: `${prob * 100}%`,
                          }}
                        />
                      </div>
                      <span style={styles.probabilityValue}>
                        {(prob * 100).toFixed(1)}%
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </>
          ) : (
            <div style={styles.emptyState}>
              <p>Draw a digit to see predictions</p>
            </div>
          )}
        </div>
      </div>

      <div style={styles.footer}>
        <h4>🧠 Model Architecture</h4>
        <div style={styles.architecture}>
          <div style={styles.layer}>Input: 784 (28×28)</div>
          <div style={styles.arrow}>→</div>
          <div style={styles.layer}>Dense: 128 + ReLU</div>
          <div style={styles.arrow}>→</div>
          <div style={styles.layer}>Dense: 10 + Softmax</div>
        </div>
        <p style={styles.note}>
          <strong>Note:</strong> Currently using synthetic weights for demo.
          Replace with trained HuggingFace weights for accurate predictions.
        </p>
      </div>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    background: 'rgba(255, 255, 255, 0.05)',
    borderRadius: '16px',
    padding: '30px',
    border: '1px solid rgba(255, 255, 255, 0.1)',
  },
  header: {
    textAlign: 'center',
    marginBottom: '30px',
  },
  title: {
    fontSize: '28px',
    marginBottom: '10px',
  },
  subtitle: {
    opacity: 0.8,
    fontSize: '16px',
  },
  content: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
    gap: '30px',
    marginBottom: '30px',
  },
  canvasSection: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
  },
  resultsSection: {
    display: 'flex',
    flexDirection: 'column',
  },
  sectionTitle: {
    fontSize: '20px',
    marginBottom: '15px',
    textAlign: 'center',
  },
  hint: {
    fontSize: '14px',
    opacity: 0.7,
    marginTop: '10px',
  },
  predictionBox: {
    background: 'rgba(102, 126, 234, 0.2)',
    borderRadius: '12px',
    padding: '30px',
    textAlign: 'center',
    marginBottom: '20px',
  },
  predictedDigit: {
    fontSize: '72px',
    fontWeight: 'bold',
    marginBottom: '10px',
  },
  confidence: {
    fontSize: '20px',
    opacity: 0.9,
  },
  inferenceTime: {
    textAlign: 'center',
    fontSize: '14px',
    marginBottom: '20px',
    opacity: 0.8,
  },
  probabilitiesContainer: {
    background: 'rgba(0, 0, 0, 0.2)',
    borderRadius: '8px',
    padding: '15px',
  },
  probabilitiesTitle: {
    fontSize: '16px',
    marginBottom: '10px',
  },
  probabilities: {
    display: 'flex',
    flexDirection: 'column',
    gap: '8px',
  },
  probabilityBar: {
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
    fontSize: '14px',
  },
  probabilityBarActive: {
    fontWeight: 'bold',
    opacity: 1,
  },
  probabilityLabel: {
    width: '20px',
  },
  probabilityTrack: {
    flex: 1,
    height: '20px',
    background: 'rgba(255, 255, 255, 0.1)',
    borderRadius: '4px',
    overflow: 'hidden',
  },
  probabilityFill: {
    height: '100%',
    background: 'linear-gradient(90deg, #667eea, #764ba2)',
    transition: 'width 0.3s ease',
  },
  probabilityValue: {
    width: '50px',
    textAlign: 'right',
    opacity: 0.8,
  },
  emptyState: {
    textAlign: 'center',
    padding: '60px 20px',
    opacity: 0.6,
  },
  footer: {
    marginTop: '30px',
    padding: '20px',
    background: 'rgba(0, 0, 0, 0.2)',
    borderRadius: '8px',
  },
  architecture: {
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    gap: '15px',
    margin: '15px 0',
    flexWrap: 'wrap',
  },
  layer: {
    padding: '10px 20px',
    background: 'rgba(102, 126, 234, 0.3)',
    borderRadius: '6px',
    fontSize: '14px',
  },
  arrow: {
    fontSize: '20px',
    opacity: 0.6,
  },
  note: {
    fontSize: '14px',
    opacity: 0.7,
    marginTop: '15px',
    textAlign: 'center',
  },
};
