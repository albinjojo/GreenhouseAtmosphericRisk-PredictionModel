Got it. Below is the **final README in proper Markdown format**, clean and ready to paste directly into `README.md`.

No commentary. Just the file content.

---

```markdown
# Indoor Environmental Risk Prediction System

## Overview

A crop-aware environmental risk prediction system designed for indoor and polyhouse farming environments.

The system evaluates environmental suitability using:

- Temperature (°C)
- Humidity (%)
- CO₂ concentration (ppm)
- Crop Type
- Crop Growth Stage

It produces:

- Continuous risk score (0–1)
- Environmental status classification (Optimal / Warning / Critical)
- Parameter deviation detection (HIGH / LOW)

This project demonstrates end-to-end machine learning integration, including dataset generation, model training, API deployment, and frontend communication.

---

## System Architecture

User / Sensor Input  
→ Flask API  
→ Random Forest Model  
→ Risk Evaluation + Deviation Logic  
→ Frontend (HTML / React)

All components run locally during development.

---

## Dataset

**Dataset File:**  
`indoor_crop_risk_dataset_final.csv`

### Characteristics

- 6000 samples  
- 5 crops × 3 growth stages × 400 samples each  
- Synthetic but agronomically structured  
- Balanced across crop-stage combinations  
- Continuous regression target  

### Crops Included

- Tomato  
- Capsicum  
- Cucumber  
- Lettuce  
- Strawberry  

### Growth Stages

- Vegetative  
- Flowering  
- Fruiting  

### Data Schema

```

temperature | humidity | co2 | crop_type | crop_stage | risk_score

```

### Target Variable

`risk_score` (continuous value in range 0–1)

Risk values were generated using weighted normalized deviation from crop-stage-specific optimal environmental centers.

---

## Machine Learning Model

**Model Type:** RandomForestRegressor  
**Framework:** scikit-learn  

### Final Optimized Configuration

- `n_estimators = 120`
- `max_depth = 15`
- `min_samples_split = 5`
- `min_samples_leaf = 2`
- `random_state = 42`

### Performance

- RMSE ≈ 0.08  
- R² ≈ 0.80  
- Serialized Model Size ≈ 3 MB (compressed)

The model was optimized from an initial 125 MB configuration to a lightweight 3 MB deployable model while maintaining acceptable predictive performance.

---

## Risk Modeling Approach

The system separates responsibilities:

1. **ML Model** → Estimates environmental risk magnitude.
2. **Logic Layer** → Detects parameter deviation direction (HIGH / LOW).

This ensures clean separation between prediction and interpretation without hard-coded threshold rules inside the model.

---

## Project Structure

```

ecogrow-model/
├── indoor_crop_risk_dataset_final.csv
├── randomforesttrain.py
├── testrf.py
├── app.py
├── crop_risk_model.pkl
├── model_features.pkl
└── README.md

````

---

## File Responsibilities

### `randomforesttrain.py`

- Loads dataset
- Applies one-hot encoding
- Splits data into train/test sets
- Trains optimized RandomForestRegressor
- Evaluates model performance
- Saves compressed model file
- Saves feature ordering metadata

---

### `testrf.py`

Command-line testing interface.

- Accepts environmental inputs
- Loads trained model
- Predicts risk score
- Displays status classification
- Detects HIGH / LOW parameter deviations

Used for local validation and debugging.

---

### `app.py`

Flask API implementation.

- Exposes `/predict` endpoint
- Accepts JSON input
- Returns structured JSON response
- Handles model inference
- Designed for frontend integration

---

## Running the System

### Requirements

Python 3.9+

Install dependencies:

```bash
pip install pandas scikit-learn joblib flask flask-cors
````

---

### Training the Model

```bash
python randomforesttrain.py
```

Outputs:

* `crop_risk_model.pkl`
* `model_features.pkl`

---

### Running CLI Test

```bash
python testrf.py
```

---

### Running Flask API

```bash
python app.py
```

Endpoint:

```
POST http://127.0.0.1:5000/predict
```

Example Request:

```json
{
  "temperature": 30,
  "humidity": 65,
  "co2": 350,
  "crop_type": "tomato",
  "crop_stage": "fruiting"
}
```

Example Response:

```json
{
  "risk_score": 0.44,
  "status": "Warning"
}
```

---

## Limitations

* Dataset is synthetic.
* No time-series modeling.
* No real-world sensor calibration.
* No automatic environmental control integration.

---

## Intended Use

* Academic ML projects
* Controlled-environment farming prototypes
* ML + Flask backend integration practice
* Deployment and model optimization demonstration

```

---

If you want, I can now also give you:

- A short professional one-line GitHub repository description  
- Or recommended GitHub topics for better visibility  
- Or a deployment-ready version including Docker section.
```
