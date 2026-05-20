"""
Alzheimer's Diagnosis Prediction API
Flask backend that loads the trained XGBoost models and serves predictions.
"""

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import os

app = Flask(__name__, static_folder=".")
CORS(app)

# ── Load model bundle ──────────────────────────────────────────────────────────
MODEL_PATH = os.environ.get("MODEL_PATH", "xgb_models.pkl")

try:
    trainedModels = joblib.load(MODEL_PATH)
    print(f"✓ Models loaded from {MODEL_PATH}")
except FileNotFoundError:
    print(f"✗ Could not find {MODEL_PATH}. Run the notebook first to generate it.")
    trainedModels = None

# ── Tier definitions (must match notebook) ────────────────────────────────────
MRIBILATERAL  = ["BILATERAL_HIPP_VOL","BILATERAL_ERC_VOL","BILATERAL_BA35_VOL","BILATERAL_PHC_VOL","BILATERAL_SULCUS_VOL"]
MMSE          = ["MMSCORE"]
CDR           = ["CDMEMORY","CDCARE"]
DEMOGRAPHICS  = ["AGE","PTGENDER","PTEDUCAT","PTNOTRT","GENOTYPE"]
PET           = ["PHC_CENTILOIDS"]

TIERS = {
    1: DEMOGRAPHICS,
    2: DEMOGRAPHICS + MRIBILATERAL,
    3: DEMOGRAPHICS + MMSE,
    4: DEMOGRAPHICS + CDR,
    5: DEMOGRAPHICS + MMSE + CDR,
    6: DEMOGRAPHICS + MMSE + CDR + MRIBILATERAL,
    7: DEMOGRAPHICS + MMSE + CDR + MRIBILATERAL + PET,
}

TIER_LABELS = {
    1: "Demographics Only",
    2: "Demographics + MRI",
    3: "Demographics + MMSE",
    4: "Demographics + CDR",
    5: "Demographics + MMSE + CDR",
    6: "Demographics + MRI + MMSE + CDR",
    7: "Demographics + MRI + MMSE + CDR + Amyloid PET",
}

CATEGORY = {0: "CN", 1: "MCI", 2: "AD"}


def run_tier(tier: int, patient_input: pd.Series) -> dict:
    current_model = trainedModels[tier]
    features      = current_model["FEATURES"]
    vec           = np.array([patient_input[f] for f in features], dtype=float).reshape(1, -1)
    model         = current_model["MODEL"]
    probability   = model.predict_proba(vec)[0]
    prediction    = int(np.argmax(probability))
    score         = float(probability[1] * 50 + probability[2] * 100)
    confidence    = float(np.max(probability) * 100)
    return {
        "probability": probability.tolist(),
        "prediction":  prediction,
        "score":       round(score, 1),
        "confidence":  round(confidence, 1),
    }


def patient_prediction(patient_input: dict) -> dict:
    patient_data = pd.Series(patient_input)

    # Find highest applicable tier
    best_tier = None
    for tier in sorted(TIERS.keys(), reverse=True):
        features = TIERS[tier]
        if all(
            feature in patient_data.index and pd.notna(patient_data[feature])
            for feature in features
        ):
            best_tier = tier
            break

    if best_tier is None:
        return {"error": "Insufficient Features — at minimum AGE, PTGENDER, PTEDUCAT, PTNOTRT, and GENOTYPE are required."}

    primary = run_tier(best_tier, patient_data)

    # Run all applicable tiers
    results = {}
    for tier in TIERS:
        features = TIERS[tier]
        if all(
            feature in patient_data.index and pd.notna(patient_data[feature])
            for feature in features
        ):
            results[tier] = run_tier(tier, patient_data)

    predictions_across_tiers = [r["prediction"] for r in results.values()]
    agree      = len(set(predictions_across_tiers)) == 1
    consensus  = "Stable" if agree else "Unstable — review tier breakdown"

    return {
        "activeTier":     best_tier,
        "tierLabel":      TIER_LABELS[best_tier],
        "predictedClass": CATEGORY[primary["prediction"]],
        "score":          round(primary["score"], 1),
        "confidence":     round(primary["confidence"], 1),
        "pCN":            round(float(primary["probability"][0]) * 100, 1),
        "pMCI":           round(float(primary["probability"][1]) * 100, 1),
        "pAD":            round(float(primary["probability"][2]) * 100, 1),
        "consensus":      consensus,
        "tierBreakdown": {
            str(tier): {
                "label":      TIER_LABELS[tier],
                "prediction": CATEGORY[r["prediction"]],
                "score":      r["score"],
                "confidence": r["confidence"],
                "pCN":        round(float(r["probability"][0]) * 100, 1),
                "pMCI":       round(float(r["probability"][1]) * 100, 1),
                "pAD":        round(float(r["probability"][2]) * 100, 1),
            }
            for tier, r in results.items()
        },
    }


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if trainedModels is None:
        return jsonify({"error": "Model file not loaded. Please run the notebook first."}), 503

    payload = request.get_json(force=True)
    if not payload:
        return jsonify({"error": "No JSON payload received."}), 400

    # Convert to float where provided, skip empty strings / None
    cleaned = {}
    for k, v in payload.items():
        if v is not None and v != "":
            try:
                cleaned[k] = float(v)
            except (ValueError, TypeError):
                pass

    result = patient_prediction(cleaned)
    return jsonify(result)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "modelsLoaded": trainedModels is not None})


if __name__ == "__main__":
    app.run(debug=True, port=5050)
