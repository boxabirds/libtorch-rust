import React, { useRef, useState, useEffect } from 'react';

interface DrawingCanvasProps {
  onImageChange: (imageData: Float32Array) => void;
  width?: number;
  height?: number;
}

export default function DrawingCanvas({
  onImageChange,
  width = 280,
  height = 280
}: DrawingCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isDrawing, setIsDrawing] = useState(false);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Initialize with white background
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, width, height);
  }, [width, height]);

  const startDrawing = (e: React.MouseEvent<HTMLCanvasElement>) => {
    setIsDrawing(true);
    draw(e);
  };

  const stopDrawing = () => {
    setIsDrawing(false);
    processImage();
  };

  const draw = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDrawing && e.type !== 'mousedown') return;

    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    ctx.fillStyle = 'black';
    ctx.beginPath();
    ctx.arc(x, y, 10, 0, Math.PI * 2);
    ctx.fill();
  };

  const processImage = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Get image data and resize to 28x28
    const imageData = ctx.getImageData(0, 0, width, height);
    const resized = resizeImageData(imageData, 28, 28);

    // Convert to grayscale and normalize to 0-1
    const normalized = new Float32Array(784);
    for (let i = 0; i < 28 * 28; i++) {
      const pixelIndex = i * 4;
      // Convert RGB to grayscale (inverted: white=0, black=1)
      const gray = 1.0 - (
        (resized.data[pixelIndex] +
         resized.data[pixelIndex + 1] +
         resized.data[pixelIndex + 2]) / 3 / 255
      );
      normalized[i] = gray;
    }

    onImageChange(normalized);
  };

  const resizeImageData = (imageData: ImageData, newWidth: number, newHeight: number): ImageData => {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d')!;

    canvas.width = imageData.width;
    canvas.height = imageData.height;
    ctx.putImageData(imageData, 0, 0);

    const resizedCanvas = document.createElement('canvas');
    const resizedCtx = resizedCanvas.getContext('2d')!;
    resizedCanvas.width = newWidth;
    resizedCanvas.height = newHeight;

    resizedCtx.drawImage(canvas, 0, 0, newWidth, newHeight);
    return resizedCtx.getImageData(0, 0, newWidth, newHeight);
  };

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, width, height);

    // Reset to zeros
    onImageChange(new Float32Array(784));
  };

  return (
    <div style={styles.container}>
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        style={styles.canvas}
        onMouseDown={startDrawing}
        onMouseMove={draw}
        onMouseUp={stopDrawing}
        onMouseLeave={stopDrawing}
      />
      <button onClick={clearCanvas} style={styles.clearButton}>
        Clear
      </button>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: '15px',
  },
  canvas: {
    border: '2px solid rgba(255, 255, 255, 0.3)',
    borderRadius: '8px',
    cursor: 'crosshair',
    background: 'white',
  },
  clearButton: {
    padding: '10px 20px',
    background: 'rgba(255, 255, 255, 0.1)',
    border: '1px solid rgba(255, 255, 255, 0.3)',
    borderRadius: '6px',
    color: 'white',
    cursor: 'pointer',
    fontSize: '14px',
    fontWeight: 'bold',
  },
};
