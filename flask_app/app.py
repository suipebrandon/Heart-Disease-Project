from datetime import datetime
import json
import os
import sqlite3
import tempfile
from pathlib import Path

from flask import Flask, jsonify, redirect, render_template, request, url_for
import joblib


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
MODEL_PATH = PROJECT_DIR / "heart_disease_model.pkl"
SCALER_PATH = PROJECT_DIR / "scaler.pkl"
DB_PATH = (
    Path(tempfile.gettempdir()) / "heart_predictions.db"
    if os.environ.get("VERCEL")
    else BASE_DIR / "heart_predictions.db"
)

app = Flask(__name__)

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except FileNotFoundError as exc:
    raise SystemExit(
        "Error: heart_disease_model.pkl or scaler.pkl not found. "
        "Please run the training notebook first."
    ) from exc


FEATURE_COLUMNS = [
    "age",
    "trestbps",
    "chol",
    "thalch",
    "oldpeak",
    "ca",
    "sex_Male",
    "cp_atypical angina",
    "cp_non-anginal",
    "cp_typical angina",
    "fbs_True",
    "restecg_normal",
    "restecg_st-t abnormality",
    "exang_True",
    "slope_flat",
    "slope_upsloping",
    "thal_normal",
    "thal_reversable defect",
]

NUMERIC_FIELDS = {
    "age": {"label": "Age", "min": 1, "max": 120},
    "trestbps": {"label": "Resting blood pressure", "min": 70, "max": 250},
    "chol": {"label": "Serum cholesterol", "min": 80, "max": 700},
    "thalch": {"label": "Maximum heart rate", "min": 60, "max": 250},
    "oldpeak": {"label": "ST depression", "min": 0, "max": 10},
    "ca": {"label": "Number of major vessels", "min": 0, "max": 3},
}

CATEGORICAL_FIELDS = {
    "sex": {"Male", "Female"},
    "cp": {"typical", "atypical", "non-anginal", "asymptomatic"},
    "fbs": {"False", "True"},
    "restecg": {"normal", "st-t", "lv-hyper"},
    "exang": {"False", "True"},
    "slope": {"upsloping", "flat", "downsloping"},
    "thal": {"normal", "fixed", "reversable"},
}

DEFAULT_FORM = {
    "age": "50",
    "sex": "Male",
    "cp": "typical",
    "trestbps": "120",
    "chol": "200",
    "thalch": "150",
    "fbs": "False",
    "restecg": "normal",
    "exang": "False",
    "oldpeak": "1.0",
    "slope": "upsloping",
    "ca": "0",
    "thal": "normal",
}


def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                inputs_json TEXT NOT NULL,
                risk_label TEXT NOT NULL,
                risk_band TEXT NOT NULL,
                probability REAL NOT NULL
            )
            """
        )


def validate_payload(payload):
    cleaned = {}
    errors = []

    for field, rules in NUMERIC_FIELDS.items():
        raw_value = payload.get(field)
        if raw_value in (None, ""):
            errors.append(f"{rules['label']} is required.")
            continue

        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            errors.append(f"{rules['label']} must be a valid number.")
            continue

        if value < rules["min"] or value > rules["max"]:
            errors.append(
                f"{rules['label']} must be between {rules['min']} and {rules['max']}."
            )
            continue

        cleaned[field] = value

    for field, allowed_values in CATEGORICAL_FIELDS.items():
        value = payload.get(field)
        if value not in allowed_values:
            errors.append(f"{field} has an invalid selection.")
            continue
        cleaned[field] = value

    return cleaned, errors


def build_model_features(cleaned):
    data = {
        "age": cleaned["age"],
        "trestbps": cleaned["trestbps"],
        "chol": cleaned["chol"],
        "thalch": cleaned["thalch"],
        "oldpeak": cleaned["oldpeak"],
        "ca": cleaned["ca"],
        "sex_Male": 1 if cleaned["sex"] == "Male" else 0,
        "cp_atypical angina": 1 if cleaned["cp"] == "atypical" else 0,
        "cp_non-anginal": 1 if cleaned["cp"] == "non-anginal" else 0,
        "cp_typical angina": 1 if cleaned["cp"] == "typical" else 0,
        "fbs_True": 1 if cleaned["fbs"] == "True" else 0,
        "restecg_normal": 1 if cleaned["restecg"] == "normal" else 0,
        "restecg_st-t abnormality": 1 if cleaned["restecg"] == "st-t" else 0,
        "exang_True": 1 if cleaned["exang"] == "True" else 0,
        "slope_flat": 1 if cleaned["slope"] == "flat" else 0,
        "slope_upsloping": 1 if cleaned["slope"] == "upsloping" else 0,
        "thal_normal": 1 if cleaned["thal"] == "normal" else 0,
        "thal_reversable defect": 1 if cleaned["thal"] == "reversable" else 0,
    }
    return [[data[column] for column in FEATURE_COLUMNS]]


def classify_risk(probability):
    if probability >= 0.7:
        return {
            "label": "High Risk",
            "band": "high",
            "alert_class": "danger",
            "message": (
                "The model predicts a high likelihood of heart disease. "
                "A qualified healthcare professional should review this case."
            ),
        }
    if probability >= 0.4:
        return {
            "label": "Moderate Risk",
            "band": "moderate",
            "alert_class": "warning",
            "message": (
                "The model predicts a moderate likelihood of heart disease. "
                "Clinical review and follow-up testing may be useful."
            ),
        }
    return {
        "label": "Low Risk",
        "band": "low",
        "alert_class": "success",
        "message": (
            "The model predicts a lower likelihood of heart disease based on "
            "the submitted values."
        ),
    }


def explain_prediction(cleaned):
    explanations = []

    if cleaned["cp"] == "asymptomatic":
        explanations.append("Asymptomatic chest pain is commonly associated with higher risk cases.")
    if cleaned["exang"] == "True":
        explanations.append("Exercise-induced angina was present.")
    if cleaned["oldpeak"] >= 2:
        explanations.append("ST depression is elevated, which can indicate exercise-related cardiac stress.")
    if cleaned["ca"] >= 2:
        explanations.append("The number of major vessels is relatively high.")
    if cleaned["thal"] == "reversable":
        explanations.append("A reversible thalassemia defect was selected.")
    if cleaned["thalch"] < 120:
        explanations.append("Maximum heart rate achieved is relatively low.")
    if cleaned["trestbps"] >= 140:
        explanations.append("Resting blood pressure is in a high range.")
    if cleaned["chol"] >= 240:
        explanations.append("Serum cholesterol is in a high range.")

    if not explanations:
        explanations.append(
            "No major rule-based risk indicators were highlighted from the submitted values."
        )

    return explanations


def predict_from_payload(payload):
    cleaned, errors = validate_payload(payload)
    if errors:
        return None, errors

    patient_df = build_model_features(cleaned)
    patient_scaled = scaler.transform(patient_df)
    probability = float(model.predict_proba(patient_scaled)[0][1])
    risk = classify_risk(probability)

    result = {
        "inputs": cleaned,
        "risk_label": risk["label"],
        "risk_band": risk["band"],
        "alert_class": risk["alert_class"],
        "probability": probability,
        "probability_percent": probability * 100,
        "message": risk["message"],
        "explanations": explain_prediction(cleaned),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    return result, []


def save_prediction(result):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO predictions (
                created_at, inputs_json, risk_label, risk_band, probability
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                result["timestamp"],
                json.dumps(result["inputs"]),
                result["risk_label"],
                result["risk_band"],
                result["probability"],
            ),
        )


def get_recent_predictions(limit=25):
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, created_at, inputs_json, risk_label, risk_band, probability
            FROM predictions
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    predictions = []
    for row in rows:
        inputs = json.loads(row["inputs_json"])
        predictions.append(
            {
                "id": row["id"],
                "report_number": f"HDR-{row['id']:06d}",
                "created_at": row["created_at"],
                "inputs": inputs,
                "risk_label": row["risk_label"],
                "risk_band": row["risk_band"],
                "probability_percent": row["probability"] * 100,
            }
        )
    return predictions


def clear_prediction_history():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM predictions")


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    errors = []
    form_values = DEFAULT_FORM.copy()

    if request.method == "POST":
        form_values.update(request.form.to_dict())
        result, errors = predict_from_payload(form_values)
        if result:
            save_prediction(result)

    return render_template(
        "index.html",
        form_values=form_values,
        result=result,
        errors=errors,
    )


@app.route("/history")
def history():
    return render_template("history.html", predictions=get_recent_predictions())


@app.route("/history/clear", methods=["POST"])
def clear_history():
    clear_prediction_history()
    return redirect(url_for("history"))


@app.route("/model-info")
def model_info():
    return render_template("model_info.html")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"errors": ["Request body must be a valid JSON object."]}), 400

    result, errors = predict_from_payload(payload)
    if errors:
        return jsonify({"errors": errors}), 400

    save_prediction(result)
    return jsonify(
        {
            "risk": result["risk_label"],
            "risk_band": result["risk_band"],
            "probability": round(result["probability"], 4),
            "probability_percent": round(result["probability_percent"], 2),
            "message": result["message"],
            "explanations": result["explanations"],
            "timestamp": result["timestamp"],
        }
    )


init_db()


if __name__ == "__main__":
    app.run(debug=True, port=5000)
