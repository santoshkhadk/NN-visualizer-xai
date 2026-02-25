import React, { useRef, useState } from "react";
import CanvasDraw from "./components/canva";
import PredictionList from "./components/predictionList";
import CorrectionBox from "./components/correctionBox";

function App() {
  const canvasRef = useRef(null);
  const [drawing, setDrawing] = useState(false);
  const [predictions, setPredictions] = useState(null);

  const startDrawing = () => setDrawing(true);

  const stopDrawing = () => {
    setDrawing(false);
    canvasRef.current.getContext("2d").beginPath();
  };

  const draw = (e) => {
    if (!drawing) return;

    const ctx = canvasRef.current.getContext("2d");
    const rect = canvasRef.current.getBoundingClientRect();

    ctx.lineWidth = 15;
    ctx.lineCap = "round";
    ctx.strokeStyle = "white";

    ctx.lineTo(e.clientX - rect.left, e.clientY - rect.top);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(e.clientX - rect.left, e.clientY - rect.top);
  };

  const clearCanvas = () => {
    const ctx = canvasRef.current.getContext("2d");
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, 280, 280);
    setPredictions(null);
  };

  const predictDigit = async () => {
    const dataURL = canvasRef.current.toDataURL();

    try {
      const res = await fetch("http://localhost:8000/api/predict_digit/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: dataURL }),
      });

      const result = await res.json();
      if (result.predictions) setPredictions(result.predictions);
    } catch (e) {
      console.error(e);
    }
  };

  return (
    <div style={{ textAlign: "center", color: "white", fontFamily: "sans-serif" }}>
      <h1>Draw a Digit</h1>

      <CanvasDraw
        canvasRef={canvasRef}
        startDrawing={startDrawing}
        stopDrawing={stopDrawing}
        draw={draw}
        predictDigit={predictDigit}
        clearCanvas={clearCanvas}
      />

      <PredictionList predictions={predictions} />

      <CorrectionBox canvasRef={canvasRef} onTrained={predictDigit} />
    </div>
  );
}

export default App;