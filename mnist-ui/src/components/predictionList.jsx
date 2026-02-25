import React from "react";

export default function PredictionList({ predictions }) {
  if (!predictions || predictions.length === 0) return null;

  return (
    <div style={{ marginTop: "20px", background: "#333", padding: "10px", borderRadius: "8px" }}>
      <h2>Top 3 Predictions</h2>
      <ul style={{ listStyle: "none", padding: 0 }}>
        {predictions.map((p, idx) => (
          <li key={idx}>
            Digit <strong>{p.digit}</strong> →
            <strong> {(p.probability * 100).toFixed(1)}%</strong>
          </li>
        ))}
      </ul>
    </div>
  );
}