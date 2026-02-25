import React, { useRef, useState } from "react";

function App() {
  const canvasRef = useRef(null);
  const [drawing, setDrawing] = useState(false);
  const [prediction, setPrediction] = useState(null);

  const startDrawing = e => setDrawing(true);
  const stopDrawing = e => {
    setDrawing(false);
    const ctx = canvasRef.current.getContext("2d");
    ctx.beginPath(); // reset path
  };

  const draw = e => {
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
    setPrediction(null);
  };

  const predictDigit = async () => {
    const canvas = canvasRef.current;
    const dataURL = canvas.toDataURL("image/png");

    const res = await fetch("http://localhost:8000/api/predict_digit/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: dataURL })
    });

    const result = await res.json();
    if (result.prediction !== undefined) {
      setPrediction(result.prediction);
    } else {
      console.error(result.error);
    }
  };

  return (
    <div style={{ textAlign: "center", marginTop: "20px" }}>
      <h1>Draw a digit</h1>
      <canvas
        ref={canvasRef}
        width={280}
        height={280}
        style={{ border: "2px solid white", background: "black", cursor: "crosshair" }}
        onMouseDown={startDrawing}
        onMouseUp={stopDrawing}
        onMouseMove={draw}
      />
      <div style={{ marginTop: "10px" }}>
        <button onClick={predictDigit}>Predict</button>
        <button onClick={clearCanvas} style={{ marginLeft: "10px" }}>Clear</button>
      </div>
      {prediction !== null && <h2>Prediction: {prediction}</h2>}
    </div>
  );
}

export default App;