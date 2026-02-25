import React, { useState } from "react";

function ExplainHeatmap({ canvasRef }) {
  const [heatmap, setHeatmap] = useState(null);

  const explainDigit = async () => {
    const dataURL = canvasRef.current.toDataURL();

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
  };

  return (
    <div style={{ marginTop: "20px" }}>
      <button onClick={explainDigit}>Explain Prediction</button>

      {heatmap && (
        <div style={{ marginTop: "15px" }}>
          <h3>Pixel Importance Heatmap</h3>
          <img
            src={`data:image/png;base64,${heatmap}`}
            alt="Heatmap"
            style={{ border: "2px solid white", width: "200px" }}
          />
        </div>
      )}
    </div>
  );
}

export default ExplainHeatmap;