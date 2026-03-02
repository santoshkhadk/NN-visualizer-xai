import React, { useState } from "react";
import Plot from "react-plotly.js";

function ExplainHeatmap({ canvasRef }) {
  // Initialize states as empty arrays to avoid null errors
  const [pixelHeatmap, setPixelHeatmap] = useState([]);
  const [topNeuronMaps, setTopNeuronMaps] = useState([]);
  const [selectedNeuron, setSelectedNeuron] = useState(0);
  const [hiddenActivations, setHiddenActivations] = useState([]);
  const [topNeurons, setTopNeurons] = useState([]);
  const [neuronClassContribs, setNeuronClassContribs] = useState([]);
  const [outputProbs, setOutputProbs] = useState([]);
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
      if (result.pixel_heatmap) {
        setPixelHeatmap(result.pixel_heatmap || []);
        setTopNeuronMaps(result.top_neuron_maps || []);
        setNeuronClassContribs(result.neuron_class_contribs || []);
        setSelectedNeuron(0);
        setHiddenActivations(result.hidden_activations?.[0] || []);
        setTopNeurons(result.top_neurons || []);
        setOutputProbs(result.output_probs?.[0] || []);
        setPredictedClass(result.predicted_class ?? null);
      } else {
        console.error(result.error);
      }
    } catch (err) {
      console.error("Error fetching explanation:", err);
    }
  };

  // Highlight top neurons in red
  const getNeuronColors = () => {
    if (!hiddenActivations.length) return [];
    const top3 = new Set(topNeurons.slice(0, 3));
    return hiddenActivations.map((_, idx) => (top3.has(idx) ? "red" : "orange"));
  };

  return (
    <div style={{ marginTop: "20px" }}>
      <button onClick={explainDigit}>Explain Prediction</button>

      {/* Only render if pixel heatmap and hidden activations exist */}
      {pixelHeatmap.length > 0 && hiddenActivations.length > 0 && (
        <div style={{ display: "flex", gap: "40px", marginTop: "20px" }}>
          {/* Left Column: Pixel Heatmaps */}
          <div>
            <h3>Pixel Importance Heatmap</h3>
            <Plot
              data={[{ z: pixelHeatmap, type: "heatmap", colorscale: "Jet" }]}
              layout={{
                width: 400,
                height: 400,
                xaxis: { visible: false },
                yaxis: { visible: false, autorange: "reversed" },
                margin: { l: 20, r: 20, t: 20, b: 20 },
              }}
              config={{ displayModeBar: false }}
            />

            <h4>Top Neuron Pixel Contributions</h4>
            <div style={{ marginBottom: "10px" }}>
              {topNeuronMaps.map((_, idx) => (
                <button
                  key={idx}
                  onClick={() => setSelectedNeuron(idx)}
                  style={{
                    marginRight: "5px",
                    background: idx === selectedNeuron ? "#ff6666" : "#ccc",
                  }}
                >
                  Neuron {topNeurons[idx]}
                </button>
              ))}
            </div>

            {topNeuronMaps.length > 0 && topNeuronMaps[selectedNeuron] && (
              <Plot
                data={[
                  {
                    z: topNeuronMaps[selectedNeuron],
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
            )}

            {/* Neuron class contributions */}
            {neuronClassContribs.length > 0 && topNeurons.length > 0 && (
              <div style={{ marginTop: "20px" }}>
                <h4>Neuron Contribution to Predicted Class</h4>
                <Plot
                  data={[
                    {
                      x: topNeurons.map((n) => `Neuron ${n}`),
                      y: neuronClassContribs,
                      type: "bar",
                      marker: { color: "blue" },
                    },
                  ]}
                  layout={{ width: 400, height: 200, yaxis: { title: "Contribution" } }}
                  config={{ displayModeBar: false }}
                />
              </div>
            )}
          </div>

          <div>
            <h3>Hidden Layer Activations</h3>
            <Plot
              data={[
                { y: hiddenActivations, type: "bar", marker: { color: getNeuronColors() } },
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
              <strong>Predicted class:</strong> {predictedClass} <br />
              <strong>Top neurons:</strong> {topNeurons.join(", ")}
            </p>
          </div>
        </div>
      )}

      {/* Output Probabilities */}
      {outputProbs.length > 0 && (
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