import { Link, useParams } from "react-router-dom";
import { useEffect, useRef, useState } from "react";

function Translator() {
  const { lang } = useParams();

  const videoRef = useRef(null);
  const streamRef = useRef(null);

  const [cameraOn, setCameraOn] = useState(false);
  const [error, setError] = useState("");

  const getLanguageName = () => {
    if (lang === "english") return "English Alphabet";
    if (lang === "arabic") return "Arabic Sign Language";
    return "Unknown Language";
  };

  const getDatasetName = () => {
    if (lang === "english") return "English Alphabet Dataset / Model";
    if (lang === "arabic") return "Arabic Sign Language Dataset / Model";
    return "No dataset loaded";
  };

  const startCamera = async () => {
    try {
      setError("");

      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false,
      });

      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }

      setCameraOn(true);
    } catch (err) {
      setError("Could not access the camera. Please allow camera permission.");
      console.error(err);
    }
  };

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    setCameraOn(false);
  };

  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, []);

  return (
    <div className="translator-page">
      <div className="translator-container">
        <div className="translator-header">
          <h1>Live Translator</h1>
          <p>
            The selected language determines which dataset or model will be
            loaded.
          </p>
        </div>

        <div className="translator-layout">
          <div className="camera-box">
            <h2>Camera Feed</h2>

            <div className="camera-placeholder">
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                style={{
                  width: "100%",
                  height: "100%",
                  borderRadius: "18px",
                  objectFit: "cover",
                  display: cameraOn ? "block" : "none",
                }}
              />

              {!cameraOn && <span>Camera will appear here</span>}
            </div>

            <div className="button-group">
              <button className="btn btn-primary" onClick={startCamera}>
                Start Camera
              </button>

              <button className="btn btn-secondary" onClick={stopCamera}>
                Stop Camera
              </button>
            </div>

            {error && (
              <p style={{ color: "salmon", marginTop: "14px" }}>{error}</p>
            )}
          </div>

          <div className="result-box">
            <h2>Translation Details</h2>

            <div className="result-item">
              <div className="result-label">Selected Language</div>
              <div className="result-value">{getLanguageName()}</div>
            </div>

            <div className="result-item">
              <div className="result-label">Dataset / Model</div>
              <div className="result-value">{getDatasetName()}</div>
            </div>

            <div className="result-item">
              <div className="result-label">Current Detection</div>
              <div className="result-value">No gesture detected yet</div>
            </div>

            <div className="result-item">
              <div className="result-label">Translated Text</div>
              <div className="result-value">---</div>
            </div>

            <Link to="/" className="back-link">
              ← Back to Language Selection
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Translator;