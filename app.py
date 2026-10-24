
"""Flask app for the credit-card fraud detector."""
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE = Path(__file__).resolve().parent
ARTIFACTS = BASE / "artifacts"

# Load once at startup (not on every request)
model = joblib.load(ARTIFACTS / "fraud_model.joblib")
config = json.loads((ARTIFACTS / "config.json").read_text())

FEATURES = config["feature_order"]          # Time, V1..V28, Amount (same order as training)
THRESHOLD = float(config["threshold"])      # chosen on validation data
MODEL_NAME = config["model_name"]

samples_file = ARTIFACTS / "sample_transactions.json"
SAMPLES = json.loads(samples_file.read_text()) if samples_file.exists() else {}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024   # reject huge request bodies


def build_row(payload):
    """Validate the request body and return a one-row DataFrame in training column order."""
    if not isinstance(payload, dict):
        raise ValueError("Send a JSON object with the feature values.")

    missing = [f for f in FEATURES if f not in payload]
    if missing:
        shown = ", ".join(missing[:5]) + ("..." if len(missing) > 5 else "")
        raise ValueError(f"Missing features: {shown}")

    try:
        row = {f: float(payload[f]) for f in FEATURES}
    except (TypeError, ValueError):
        raise ValueError("All feature values must be numbers.")

    if not np.all(np.isfinite(list(row.values()))):
        raise ValueError("Feature values must be finite numbers.")

    return pd.DataFrame([row], columns=FEATURES)


@app.get("/")
def index():
    return render_template(
        "index.html",
        model_name=MODEL_NAME,
        threshold=round(THRESHOLD, 3),
        has_samples=bool(SAMPLES),
    )


@app.get("/api/sample/<kind>")
def sample(kind):
    if kind not in SAMPLES:
        return jsonify(error="No sample available."), 404
    return jsonify(SAMPLES[kind])


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)
    try:
        X = build_row(payload)
    except ValueError as e:
        return jsonify(error=str(e)), 400

    proba = float(model.predict_proba(X)[0, 1])
    return jsonify(
        fraud_probability=round(proba, 4),
        is_fraud=bool(proba >= THRESHOLD),
        threshold=round(THRESHOLD, 4),
        model=MODEL_NAME,
    )


@app.get("/health")
def health():
    return jsonify(status="ok", model=MODEL_NAME, n_features=len(FEATURES))


if __name__ == "__main__":
    # Local testing only. In Docker the app is served by gunicorn.
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)), debug=False)