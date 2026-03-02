import React, { useState } from "react";
import Plot from "react-plotly.js";

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
      setHeatmap(result.heatmap); // this is 28x28 matrix
    } else {
      console.error(result.error);
    }
  };

  return (
    <div style={{ marginTop: "20px" }}>
      <button onClick={explainDigit}>Explain Prediction</button>

      {heatmap && (
        <div style={{ marginTop: "20px" }}>
          <h3>Pixel Importance Heatmap</h3>

          <Plot
            data={[
              {
                z: heatmap,          // 28x28 matrix
                type: "heatmap",
                colorscale: "Jet",
              },
            ]}
            layout={{
              width: 400,
              height: 400,
              xaxis: { visible: false },
              yaxis: { visible: false, autorange: "reversed" },
              margin: { l: 20, r: 20, t: 20, b: 20 },
            }}
            config={{ displayModeBar: false }}
          />
        </div>
      )}
    </div>
  );
}

export default ExplainHeatmap;