# Indoor Environmental Risk Prediction System

## Overview

A crop-aware environmental risk prediction system developed for indoor and polyhouse farming environments.

The system estimates environmental suitability using:

* Temperature (°C)
* Humidity (%)
* CO₂ concentration (ppm)
* Crop Type
* Crop Growth Stage

It outputs:

* Continuous risk score (0–1)
* Environmental status classification (Optimal / Warning / Critical)

The model is designed for educational research, ML deployment practice, and smart farming prototyping.

---

## About the Project

This system is built on a structured, crop-stage specific dataset representing environmental conditions for common indoor crops.

A Random Forest Regressor is used to estimate environmental risk based on deviation from optimal crop-stage conditions.

The project demonstrates:

* Tabular regression modeling
* Categorical feature encoding
* Model optimization for deployment

---

## Dataset Architecture

**Dataset File:**
`indoor_crop_risk_dataset_final.csv`

### Key Characteristics

* 6000 total samples
* 5 crops × 3 growth stages × 400 samples each
* Synthetic but agronomically structured
* Balanced across crop-stage combinations
* Continuous risk target

### Crops Included

* Tomato
* Capsicum
* Cucumber
* Lettuce
* Strawberry

### Growth Stages

* Vegetative
* Flowering
* Fruiting

### Data Schema

temperature | humidity | co2 | crop_type | crop_stage | risk_score

Input Features:

* temperature
* humidity
* co2
* crop_type
* crop_stage

Output Label:

* risk_score (continuous value in range 0–1)

---

## Risk Modeling Approach

Risk score is computed during dataset generation using weighted normalized deviation from optimal crop-stage environmental centers.

The model learns this mapping without hard-coded threshold rules.

The system separates:

* Risk estimation (ML model)
* Parameter deviation detection (logic layer)

---

## Project Structure

```
ecogrow-model/
├── indoor_crop_risk_dataset_final.csv
├── crop_risk_model.pkl
├── model_features.pkl
├── randomforesttrain.py
├── testrf.py
└── README.md
```

---

## File Responsibilities

### randomforesttrain.py

* Loads dataset
* Applies one-hot encoding
* Trains RandomForestRegressor
* Evaluates performance
* Serializes model
* Saves feature ordering metadata

---

### testrf.py

CLI testing interface

* Accepts environmental input from terminal
* Loads trained model
* Predicts risk score
* Displays environmental status
* Detects parameter deviation (HIGH / LOW)

Used for:

* Local validation
* Model testing without API

---

## Machine Learning Design

Model Type: RandomForestRegressor
Framework: scikit-learn

### Final Configuration

* n_estimators = 120
* max_depth = 15
* min_samples_split = 5
* min_samples_leaf = 2
* random_state = 42

### Performance

* RMSE ≈ 0.08
* R² ≈ 0.80
* Model size ≈ 3 MB (compressed)

The model was optimized to reduce size from 125 MB to 3 MB while maintaining acceptable predictive accuracy.

---

## Prediction Pipeline

1. User provides environmental values
2. Input converted into encoded feature vector
3. Random Forest predicts risk_score
4. Risk categorized into status level
5. Deviation logic identifies parameter direction
6. Result returned to CLI or API

---

## Running the System

### Requirements

Python 3.9+

Install dependencies:

```
pip install pandas scikit-learn joblib flask flask-cors
```

---

### Training the Model

```
python randomforesttrain.py
```

Outputs:

* crop_risk_model.pkl
* model_features.pkl

---

### Running CLI Version

```
python testrf.py
```

Used for:

* Local validation
* Manual testing

---

## System Architecture

User / Sensor
→ Flask Backend
→ Random Forest Model
→ Risk Evaluation
→ React Frontend

All components can run locally on the same machine.

---

## Limitations

* Dataset is synthetic
* No real sensor time-series modeling
* No disease or pest modeling
* No actuator control integration

---

## Intended Use

* Academic ML projects
* Smart farming prototypes
* ML web integration demonstrations
* Backend model deployment practice
