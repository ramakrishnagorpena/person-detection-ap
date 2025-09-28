import { useState } from "react";
import axios from "axios";

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [outputUrl, setOutputUrl] = useState("");
  const [counts, setCounts] = useState({ left: 0, right: 0 });

  const handleUpload = async () => {
    if (!file) return;
    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await axios.post(
        "https://person-detection-ap.onrender.com/detect",
        formData,
        {
          headers: { "Content-Type": "multipart/form-data" },
        }
      );
      setCounts({ left: res.data.left_count, right: res.data.right_count });
      setOutputUrl(
        `https://person-detection-ap.onrender.com/output?t=${Date.now()}`
      );
    } catch (err) {
      console.error(err);
      alert("Error uploading image.");
    }
  };

  return (
    <div style={{ textAlign: "center", padding: "2rem" }}>
      <h1>YOLO11 Person Detection</h1>
      <input
        type="file"
        onChange={(e) => e.target.files?.[0] && setFile(e.target.files[0])}
      />
      <button
        onClick={handleUpload}
        style={{
          marginLeft: "10px",
          padding: "6px 12px",
          backgroundColor: "#007bff",
          color: "#fff",
          border: "none",
          borderRadius: "6px",
          cursor: "pointer",
        }}
      >
        Detect
      </button>

      {outputUrl && (
        <div style={{ marginTop: "20px" }}>
          <p style={{ fontSize: "18px" }}>
            Left Side: <b>{counts.left}</b> | Right Side: <b>{counts.right}</b>
          </p>
          <img
            src={outputUrl}
            alt="Detection Result"
            style={{ maxWidth: "80%", borderRadius: "10px", marginTop: "10px" }}
          />
        </div>
      )}
    </div>
  );
}

export default App;
