from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)

model = joblib.load("crop_risk_model.pkl")
feature_columns = joblib.load("model_features.pkl")

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

    return float(model.predict(df)[0])

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    temp = float(data["temperature"])
    humidity = float(data["humidity"])
    co2 = float(data["co2"])
    crop_type = data["crop_type"].lower()
    crop_stage = data["crop_stage"].lower()

    risk_score = predict_risk(temp, humidity, co2, crop_type, crop_stage)

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
    app.run(debug=True)
