import React from "react";

export default function CanvasDraw({
  canvasRef,
  startDrawing,
  stopDrawing,
  draw,
  predictDigit,
  clearCanvas,
}) {
  return (
    <div style={{ textAlign: "center" }}>
      <canvas
        ref={canvasRef}
        width={280}
        height={280}
        style={{
          border: "2px solid white",
          background: "black",
          cursor: "crosshair",
        }}
        onMouseDown={startDrawing}
        onMouseUp={stopDrawing}
        onMouseMove={draw}
      />

      <div style={{ marginTop: "10px" }}>
        <button onClick={predictDigit} style={{ padding: "10px 20px" }}>
          Predict
        </button>
        <button
          onClick={clearCanvas}
          style={{ padding: "10px 20px", marginLeft: "10px" }}
        >
          Clear
        </button>
      </div>
    </div>
  );
}