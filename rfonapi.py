from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)

try:
    model = joblib.load("crop_risk_model.pkl")
    feature_columns = joblib.load("model_features.pkl")
except Exception as e:
    print("Error loading model:", e)
    model = None
    feature_columns = None

def predict_risk(temp, humidity, co2, crop_type, crop_stage):
    input_dict = {
        "temperature": temp,
        "humidity": humidity,
        "co2": co2,
        "crop_type": crop_type,
        "crop_stage": crop_stage
    }

    df = pd.DataFrame([input_dict])
    df = pd.get_dummies(df)
    df = df.reindex(columns=feature_columns, fill_value=0)

    prediction = model.predict(df)[0]
    return float(prediction)

@app.route("/predict", methods=["POST"])
def predict():

    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.get_json()

    if not data:
        return jsonify({"error": "Invalid JSON input"}), 400

    try:
        temp = float(data.get("temperature"))
        humidity = float(data.get("humidity"))
        co2 = float(data.get("co2"))
        crop_type = str(data.get("crop_type")).lower()
        crop_stage = str(data.get("crop_stage")).lower()
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid input format"}), 400

    try:
        risk_score = predict_risk(temp, humidity, co2, crop_type, crop_stage)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    if risk_score < 0.3:
        status = "Optimal"
    elif risk_score < 0.6:
        status = "Warning"
    else:
        status = "Critical"

    return jsonify({
        "risk_score": round(risk_score, 4),
        "status": status
    })
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)