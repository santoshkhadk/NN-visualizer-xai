import React, { useRef, useState } from "react";

function App() {
  const canvasRef = useRef(null);
  const [drawing, setDrawing] = useState(false);
  const [predictions, setPredictions] = useState(null);

  const startDrawing = (e) => setDrawing(true);

  const stopDrawing = () => {
    setDrawing(false);
    const ctx = canvasRef.current.getContext("2d");
    ctx.beginPath(); // reset path
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
    const canvas = canvasRef.current;
    const dataURL = canvas.toDataURL("image/png");

    try {
      const res = await fetch("http://localhost:8000/api/predict_digit/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: dataURL }),
      });

      const result = await res.json();
      console.log(result.predictions)
      if (result.predictions) {
        setPredictions(result.predictions);
      } else {
        console.error(result.error);
      }
    } catch (err) {
      console.error("Error:", err);
    }
  };

  return (
    <div style={{ textAlign: "center", marginTop: "20px", color: "white", fontFamily: "sans-serif" }}>
      <h1>Draw a Digit</h1>
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
        <button onClick={predictDigit} style={{ padding: "10px 20px" }}>Predict</button>
        <button onClick={clearCanvas} style={{ padding: "10px 20px", marginLeft: "10px" }}>Clear</button>
      </div>

     {predictions && predictions.length > 0 && (
  <div style={{ marginTop: "20px", background:"red" }}>
    <h2>Top 3 Predictions:</h2>
   <ul style={{ listStyle: "none", padding: 0, fontSize: "1.2em", color: "white" }}>
  {predictions.map((p, idx) => (
    <li key={idx} style={{ color: "white" }}>
      Digit <strong>{p.digit}</strong> → Probability: <strong>{(p.probability * 100).toFixed(1)}%</strong>
    </li>
  ))}
</ul>
  </div>
)}
    </div>
  );
}

export default App;