import React from "react";
import Plot from "react-plotly.js";

function SaliencyHeatmap({ heatmap }) {
  if (!heatmap) return null;

  return (
    <div style={{ width: "400px", margin: "auto" }}>
      <Plot
        data={[
          {
            z: heatmap,          // 28x28 matrix
            type: "heatmap",
            colorscale: "Jet",   // nice red-yellow-blue look
          }
        ]}
        layout={{
          title: "Pixel Importance Heatmap",
          width: 400,
          height: 400,
          xaxis: { visible: false },
          yaxis: { visible: false, autorange: "reversed" },
          margin: { l: 20, r: 20, t: 40, b: 20 }
        }}
        config={{ displayModeBar: false }}
      />
    </div>
  );
}

export default SaliencyHeatmap;