import joblib
import pandas as pd

# Load model and feature structure
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

    # Align columns with training
    df = df.reindex(columns=feature_columns, fill_value=0)

    risk = model.predict(df)[0]
    return risk


if __name__ == "__main__":

    print("\n--- Indoor Crop Risk Prediction ---")

    temp = float(input("Enter Temperature (°C): "))
    humidity = float(input("Enter Humidity (%): "))
    co2 = float(input("Enter CO2 (ppm): "))
    crop_type = input("Enter Crop Type (tomato/capsicum/cucumber/lettuce/strawberry): ").lower()
    crop_stage = input("Enter Crop Stage (vegetative/flowering/fruiting): ").lower()

    risk_score = predict_risk(temp, humidity, co2, crop_type, crop_stage)

    print("\nRisk Score:", round(risk_score, 4))

    if risk_score < 0.3:
        print("Status: Optimal")
    elif risk_score < 0.6:
        print("Status: Warning")
    else:
        print("Status: Critical")
