from pathlib import Path

import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

print("Loading processed data...")

X_train = pd.read_csv(
    DATA_DIR / "X_train.csv"
)

y_train = pd.read_csv(
    DATA_DIR / "y_train.csv"
).values.ravel()


print("Training data:", X_train.shape)
print("Target data:", y_train.shape)


# --------------------------------------------------
# MODEL
# --------------------------------------------------

print("\nTraining Random Forest model...")

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_split=5,
    random_state=42,
    class_weight="balanced"
)


# --------------------------------------------------
# TRAIN
# --------------------------------------------------

model.fit(
    X_train,
    y_train
)


# --------------------------------------------------
# SAVE MODEL
# --------------------------------------------------

MODEL_PATH = MODEL_DIR / "student_performance_model.pkl"

joblib.dump(
    model,
    MODEL_PATH
)


# --------------------------------------------------
# OUTPUT
# --------------------------------------------------

print("\n======================================")
print(" MODEL TRAINING COMPLETED!")
print("======================================")

print("Model saved successfully!")
print("Model path:", MODEL_PATH)