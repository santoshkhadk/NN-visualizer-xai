import { useRef, useState, useEffect } from "react";

export default function App() {
  const canvasRef = useRef(null);
  const [drawing, setDrawing] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [lastPos, setLastPos] = useState({ x: 0, y: 0 });

  // Initialize black background
  useEffect(() => {
    const ctx = canvasRef.current.getContext("2d");
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, 280, 280);
  }, []);

  const start = (e) => {
    setDrawing(true);
    const rect = canvasRef.current.getBoundingClientRect();
    setLastPos({
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
    });
  };

  const stop = () => setDrawing(false);

  const draw = (e) => {
    if (!drawing) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const ctx = canvasRef.current.getContext("2d");
    ctx.strokeStyle = "white";
    ctx.lineWidth = 16; // Thicker strokes for MNIST
    ctx.lineCap = "round";

    ctx.beginPath();
    ctx.moveTo(lastPos.x, lastPos.y);
    ctx.lineTo(x, y);
    ctx.stroke();

    setLastPos({ x, y });
  };

  const clear = () => {
    const ctx = canvasRef.current.getContext("2d");
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, 280, 280);
    setPrediction(null);
  };

  // Convert canvas → 28x28 normalized pixels
  const getPixels = () => {
  const canvas = canvasRef.current;
  const ctx = canvas.getContext("2d");

  // 1️⃣ Downscale to 28x28 first
  const smallCanvas = document.createElement("canvas");
  smallCanvas.width = 28;
  smallCanvas.height = 28;
  const smallCtx = smallCanvas.getContext("2d");

  // Draw original canvas into 28x28
  smallCtx.drawImage(canvas, 0, 0, 28, 28);

  // 2️⃣ Get image data
  let data = smallCtx.getImageData(0, 0, 28, 28).data;

  // 3️⃣ Convert to grayscale and invert
  let pixels = [];
  for (let i = 0; i < data.length; i += 4) {
    let gray = data[i]; // R channel
    let normalized = 1 - gray / 255; // invert: white=1, black=0
    pixels.push(normalized);
  }

  // 4️⃣ Center the digit
  // Find bounding box of non-zero pixels
  let top = 28, left = 28, bottom = 0, right = 0;
  for (let y = 0; y < 28; y++) {
    for (let x = 0; x < 28; x++) {
      if (pixels[y * 28 + x] > 0) {
        if (x < left) left = x;
        if (x > right) right = x;
        if (y < top) top = y;
        if (y > bottom) bottom = y;
      }
    }
  }

  const digitWidth = right - left + 1;
  const digitHeight = bottom - top + 1;

  const centeredPixels = new Array(28 * 28).fill(0);

  const offsetX = Math.floor((28 - digitWidth) / 2);
  const offsetY = Math.floor((28 - digitHeight) / 2);

  for (let y = top; y <= bottom; y++) {
    for (let x = left; x <= right; x++) {
      const srcIndex = y * 28 + x;
      const dstIndex = (y - top + offsetY) * 28 + (x - left + offsetX);
      centeredPixels[dstIndex] = pixels[srcIndex];
    }
  }

  return centeredPixels; // 784 MNIST-style pixels, centered
};
  const predictDigit = async () => {
    const pixels = getPixels();

    const res = await fetch("http://127.0.0.1:8000/api/predict/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ pixels }),
    });

    const data = await res.json();
    setPrediction(data.prediction);
  };

  return (
    <div style={{ textAlign: "center", marginTop: 30 }}>
      <h1>Draw Digit (MNIST)</h1>

      <canvas
        ref={canvasRef}
        width={280}
        height={280}
        style={{ border: "2px solid white", background: "black" }}
        onMouseDown={start}
        onMouseUp={stop}
        onMouseMove={draw}
      />

      <div style={{ marginTop: 20 }}>
        <button onClick={predictDigit}>Predict</button>
        <button onClick={clear} style={{ marginLeft: 10 }}>
          Clear
        </button>
      </div>

      {prediction !== null && (
        <h2 style={{ marginTop: 20 }}>Prediction: {prediction}</h2>
      )}
    </div>
  );
}