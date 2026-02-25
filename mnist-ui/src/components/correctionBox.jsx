import React, { useState } from "react";

export default function CorrectionBox({ canvasRef, onTrained }) {
  const [correctDigit, setCorrectDigit] = useState("");
  const [msg, setMsg] = useState("");

  const sendCorrection = async () => {
    if (correctDigit === "") {
      alert("Enter correct digit");
      return;
    }

    try {
      const res = await fetch("http://localhost:8000/api/correct_digit/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image: canvasRef.current.toDataURL(),
          label: parseInt(correctDigit),
        }),
      });

      const data = await res.json();

      if (res.ok) {
        setMsg("✅ Model trained!");
        setCorrectDigit("");
        onTrained && onTrained(); // re-predict
      } else {
        setMsg("❌ Training failed");
      }
    } catch (e) {
      setMsg("Server error");
    }
  };

  return (
    <div style={{ marginTop: "20px", background: "#444", padding: "10px", borderRadius: "8px" }}>
      <h3>Wrong prediction?</h3>

      <input
        type="number"
        min="0"
        max="9"
        value={correctDigit}
        onChange={(e) => setCorrectDigit(e.target.value)}
        placeholder="Correct digit"
        style={{ padding: "6px", marginRight: "10px" }}
      />

      <button onClick={sendCorrection}>Train Model</button>

      {msg && <p>{msg}</p>}
    </div>
  );
}