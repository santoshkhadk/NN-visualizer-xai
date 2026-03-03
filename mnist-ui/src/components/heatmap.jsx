import React, { useState } from "react";
import Plot from "react-plotly.js";

function ExplainHeatmap({ canvasRef }) {
  const [pixelHeatmap, setPixelHeatmap] = useState([]);
  const [topNeuronMaps, setTopNeuronMaps] = useState([]);
  const [selectedNeuron, setSelectedNeuron] = useState(0);
  const [hiddenActivations, setHiddenActivations] = useState([]);
  const [topNeurons, setTopNeurons] = useState([]);
  const [neuronClassContribs, setNeuronClassContribs] = useState([]);
  const [outputProbs, setOutputProbs] = useState([]);
  const [predictedClass, setPredictedClass] = useState(null);

  // 🔥 NEW STATE for processed 28x28 image
  const [processedImage, setProcessedImage] = useState([]);

  const [deactivated, setDeactivated] = useState([]);

  const explainDigit = async () => {
    const dataURL = canvasRef.current.toDataURL();

    try {
      const res = await fetch("http://localhost:8000/api/explain_digit/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image: dataURL,
          deactivate_neurons: deactivated
        }),
      });

      const result = await res.json();

     if (result.explanation) {
  const exp = result.explanation;

  setPixelHeatmap(exp.pixel_heatmap || []);
  setTopNeuronMaps(exp.top_neuron_maps || []);
  setNeuronClassContribs(exp.neuron_class_contribs || []);
  setSelectedNeuron(0);
  setHiddenActivations(exp.hidden_activations?.[0] || []);
  setTopNeurons(exp.top_neurons || []);
  setOutputProbs(exp.output_probs );
  setPredictedClass(exp.predicted_class ?? null);

  // 🔥 SET processed image
  setProcessedImage(result.processed_image || []);
} else {
  console.error(result.error);
}

    } catch (err) {
      console.error("Error fetching explanation:", err);
    }
  };

  const getNeuronColors = () => {
    if (!hiddenActivations.length) return [];
    const top3 = new Set(topNeurons.slice(0, 3));
    return hiddenActivations.map((_, idx) =>
      top3.has(idx)
        ? (deactivated.includes(idx) ? "gray" : "red")
        : "orange"
    );
  };

  const toggleDeactivate = (n) => {
    if (deactivated.includes(n)) {
      setDeactivated(deactivated.filter(x => x !== n));
    } else {
      setDeactivated([...deactivated, n]);
    }
  };

  return (
    <div style={{ marginTop: "20px" }}>
      <button onClick={explainDigit}>Explain Prediction</button>

      {/* 🔥 SHOW PROCESSED 28x28 IMAGE */}
      {processedImage.length > 0 && (
        <div style={{ marginTop: "20px" }}>
          <h3>Processed 28×28 Image (Model Input)</h3>
          <Plot
            data={[{
              z: processedImage,
              type: "heatmap",
              colorscale: "grey" ,
             interpolation: 'none',
              origin: 'upper',
              
            }]}
            layout={{
               width: 400,
                height: 400,
              xaxis: { visible: false },
              yaxis: { visible: false, autorange: "reversed" },
              margin: { l: 20, r: 20, t: 20, b: 20 }
            }}
            config={{ displayModeBar: false }}
          />
        </div>
      )}

      {pixelHeatmap.length > 0 && hiddenActivations.length > 0 && (
        <div style={{ display: "flex", gap: "40px", marginTop: "20px" }}>
          
          {/* LEFT COLUMN */}
          <div>
            <h3>Pixel Importance Heatmap</h3>
            <Plot
              data={[{ z: pixelHeatmap, type: "heatmap", colorscale: "Jet" }]}
              layout={{
                width: 400,
                height: 400,
                xaxis: { visible: false },
                yaxis: { visible: false, autorange: "reversed" },
                margin: { l: 20, r: 20, t: 20, b: 20 }
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
                    background: idx === selectedNeuron ? "#ff6666" : "#ccc"
                  }}
                >
                  Neuron {topNeurons[idx]}
                </button>
              ))}
            </div>

            {topNeuronMaps.length > 0 && topNeuronMaps[selectedNeuron] && (
              <Plot
                data={[{
                  z: topNeuronMaps[selectedNeuron],
                  type: "heatmap",
                  colorscale: "Jet"
                }]}
                layout={{
                  width: 400,
                  height: 400,
                  xaxis: { visible: false },
                  yaxis: { visible: false, autorange: "reversed" },
                  margin: { l: 20, r: 20, t: 20, b: 20 }
                }}
                config={{ displayModeBar: false }}
              />
            )}
          </div>

          {/* RIGHT COLUMN */}
          <div>
            <h3>Hidden Layer Activations</h3>
            <Plot
              data={[{
                y: hiddenActivations,
                type: "bar",
                marker: { color: getNeuronColors() }
              }]}
              layout={{
                width: 400,
                height: 400,
                yaxis: { title: "Activation" },
                xaxis: { title: "Neuron Index" },
                margin: { l: 50, r: 20, t: 20, b: 50 }
              }}
              config={{ displayModeBar: false }}
            />

            <p>
              <strong>Predicted class:</strong> {predictedClass}
              <br />
              <strong>Top neurons:</strong> {topNeurons.join(", ")}
            </p>

            <div  style={{ background: "red" }}>
              <h4>Deactivate Neurons</h4>
              {topNeurons.map(n => (
                <label key={n} style={{ display: "block" }}>
                  <input
                    type="checkbox"
                    checked={deactivated.includes(n)}
                    onChange={() => toggleDeactivate(n)}
                  />
                  Neuron {n}
                </label>
              ))}
            </div>
          </div>
        </div>
      )}

      {outputProbs.length > 0 && (
        <div style={{ marginTop: "20px" }}>
          <h3>Output Probabilities</h3>
          <Plot
            data={[{
              x: [...Array(10).keys()],
              y: outputProbs,
              type: "bar",
              marker: { color: "green" }
            }]}
            layout={{
              width: 820,
              height: 300,
              yaxis: { title: "Probability" },
              xaxis: { title: "Digit" },
              margin: { l: 50, r: 20, t: 20, b: 50 }
            }}
            config={{ displayModeBar: false }}
          />
        </div>
      )}
    </div>
  );
}

export default ExplainHeatmap;