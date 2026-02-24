import { useRef, useState, useEffect } from "react";

export default function App() {
  const canvasRef = useRef(null);
  const [drawing, setDrawing] = useState(false);
  const [prediction, setPrediction] = useState(null);

  // fill black background once
  useEffect(() => {
    const ctx = canvasRef.current.getContext("2d");
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, 280, 280);
  }, []);

  const start = () => setDrawing(true);
  const stop = () => setDrawing(false);

  const draw = (e) => {
    if (!drawing) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const ctx = canvasRef.current.getContext("2d");
    ctx.fillStyle = "white";
    ctx.beginPath();
    ctx.arc(x, y, 8, 0, Math.PI * 2);
    ctx.fill();
  };

  const clear = () => {
    const ctx = canvasRef.current.getContext("2d");
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, 280, 280);
    setPrediction(null);
  };

  // convert canvas → 784 pixels
  const getPixels = () => {
    const small = document.createElement("canvas");
    small.width = 28;
    small.height = 28;

    const ctx = small.getContext("2d");
    ctx.drawImage(canvasRef.current, 0, 0, 28, 28);

    const data = ctx.getImageData(0, 0, 28, 28).data;

    const pixels = [];
    for (let i = 0; i < data.length; i += 4) {
      pixels.push(data[i]); // grayscale
    }
    return pixels;
  };

  const predict = async () => {
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
        <button onClick={predict}>Predict</button>
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