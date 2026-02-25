import React, { useState } from "react";

// Utility function to convert 2D array heatmap to Base64 PNG
// You can use canvas to do this
const convertHeatmapToBase64 = (heatmap) => {
  const size = heatmap.length; // assume square 28x28
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d");

  const imageData = ctx.createImageData(size, size);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const value = Math.min(Math.max(heatmap[y][x], 0), 1) * 255; // normalize 0-255
      const index = (y * size + x) * 4;
      imageData.data[index] = 255;        // red channel
      imageData.data[index + 1] = 0;      // green channel
      imageData.data[index + 2] = 0;      // blue channel
      imageData.data[index + 3] = value;  // alpha channel
    }
  }

  ctx.putImageData(imageData, 0, 0);
  return canvas.toDataURL("image/png").split(",")[1]; // base64 only
};

const ExplainHeatmap = ({ canvasRef }) => {
  const [heatmap, setHeatmap] = useState(null);
  const [loading, setLoading] = useState(false);

  const explainDigit = async () => {
    if (!canvasRef.current) return;

    const dataURL = canvasRef.current.toDataURL();

    setLoading(true);
    try {
      const res = await fetch("http://localhost:8000/api/explain_digit/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: dataURL }),
      });

      const result = await res.json();
      if (result.heatmap) {
        setHeatmap(result.heatmap);
      } else {
        console.error(result.error);
      }
    } catch (err) {
      console.error("Error fetching heatmap:", err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ marginTop: "20px" }}>
      <button onClick={explainDigit} style={{ padding: "10px 20px" }}>
        Explain
      </button>

      {loading && <p>Loading heatmap...</p>}

      {heatmap && (
        <div style={{ marginTop: "10px" }}>
          <h3>Saliency Heatmap</h3>
          <img
            src={`data:image/png;base64,${convertHeatmapToBase64(heatmap)}`}
            alt="Heatmap"
            style={{ width: 280, height: 280, imageRendering: "pixelated", border: "2px solid white" }}
          />
        </div>
      )}
    </div>
  );
};

export default ExplainHeatmap;