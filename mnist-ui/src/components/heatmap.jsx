import React, { useState } from "react";
import Plot from "react-plotly.js";

function ExplainHeatmap({ canvasRef }) {
  const [pixelHeatmap, setPixelHeatmap] = useState(null);
  const [hiddenActivations, setHiddenActivations] = useState(null);
  const [outputProbs, setOutputProbs] = useState(null);
  const [topNeuron, setTopNeuron] = useState(null);
  const [predictedClass, setPredictedClass] = useState(null);

  const explainDigit = async () => {
    const dataURL = canvasRef.current.toDataURL();

    try {
      const res = await fetch("http://localhost:8000/api/explain_digit/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: dataURL }),
      });

      const result = await res.json();

      if (result.heatmap) {
        setPixelHeatmap(result.heatmap);
        setHiddenActivations(result.hidden_activations[0]); // flatten 1D if needed
        setOutputProbs(result.output_probs[0]);
        setTopNeuron(result.top_neuron);
        setPredictedClass(result.predicted_class);
      } else {
        console.error(result.error);
      }
    } catch (err) {
      console.error("Error fetching explanation:", err);
    }
  };

  return (
    <div style={{ marginTop: "20px" }}>
      <button onClick={explainDigit}>Explain Prediction</button>

      {pixelHeatmap && hiddenActivations && (
        <div style={{ display: "flex", gap: "40px", marginTop: "20px" }}>
          {/* Pixel Importance Heatmap */}
          <div>
            <h3>Pixel Importance Heatmap</h3>
            <Plot
              data={[
                {
                  z: pixelHeatmap,
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

          {/* Hidden Neuron Activations */}
          <div>
            <h3>Hidden Layer Activations</h3>
            <Plot
              data={[
                {
                  y: hiddenActivations,
                  type: "bar",
                  marker: { color: "orange" },
                },
              ]}
              layout={{
                width: 400,
                height: 400,
                yaxis: { title: "Activation" },
                xaxis: { title: "Neuron Index" },
                margin: { l: 50, r: 20, t: 20, b: 50 },
              }}
              config={{ displayModeBar: false }}
            />
            <p>
              <strong>Top neuron:</strong> {topNeuron} <br />
              <strong>Predicted class:</strong> {predictedClass}
            </p>
          </div>
        </div>
      )}

      {/* Optional: Output probabilities bar chart */}
      {outputProbs && (
        <div style={{ marginTop: "20px" }}>
          <h3>Output Probabilities</h3>
          <Plot
            data={[
              {
                x: [...Array(10).keys()],
                y: outputProbs,
                type: "bar",
                marker: { color: "green" },
              },
            ]}
            layout={{
              width: 820,
              height: 300,
              yaxis: { title: "Probability" },
              xaxis: { title: "Digit" },
              margin: { l: 50, r: 20, t: 20, b: 50 },
            }}
            config={{ displayModeBar: false }}
          />
        </div>
      )}
    </div>
  );
}

export default ExplainHeatmap;