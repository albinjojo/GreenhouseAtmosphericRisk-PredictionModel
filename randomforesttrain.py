import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("indoor_crop_risk_dataset_final.csv")

# One-hot encode categorical columns
df = pd.get_dummies(df, columns=["crop_type", "crop_stage"], drop_first=True)

# Separate features and target
X = df.drop("risk_score", axis=1)
y = df["risk_score"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Optimized Model (Smaller)
# -----------------------------
model = RandomForestRegressor(
    n_estimators=120,        # reduced trees
    max_depth=15,            # limited depth
    min_samples_split=5,     # avoid tiny splits
    min_samples_leaf=2,      # avoid tiny leaves
    random_state=42,
    n_jobs=-1
)

# Train model
model.fit(X_train, y_train)

# -----------------------------
# Evaluate
# -----------------------------
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("RMSE:", rmse)
print("R2 Score:", r2)

# -----------------------------
# Save model (compressed)
# -----------------------------
joblib.dump(model, "crop_risk_model.pkl", compress=3)
joblib.dump(X.columns.tolist(), "model_features.pkl")

# Check file size
size_mb = os.path.getsize("crop_risk_model.pkl") / (1024 * 1024)
print("Model Size (MB):", round(size_mb, 2))
