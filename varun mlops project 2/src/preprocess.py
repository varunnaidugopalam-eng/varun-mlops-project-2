from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

RAW_DATA = BASE_DIR / "data" / "raw" / "StudentsPerformance.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"


# Create processed folder if it doesn't exist
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

print("Loading dataset...")

df = pd.read_csv(RAW_DATA)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# --------------------------------------------------
# CREATE TARGET
# --------------------------------------------------

print("\nCreating target variable...")

df["average_score"] = (
    df["math score"]
    + df["reading score"]
    + df["writing score"]
) / 3

# Pass = 1 if average score >= 50
# Pass = 0 if average score < 50

df["pass"] = (df["average_score"] >= 50).astype(int)


# --------------------------------------------------
# FEATURES AND TARGET
# --------------------------------------------------

X = df[
    [
        "gender",
        "race/ethnicity",
        "parental level of education",
        "lunch",
        "test preparation course",
    ]
]

y = df["pass"]


# --------------------------------------------------
# ENCODE CATEGORICAL FEATURES
# --------------------------------------------------

print("Encoding categorical features...")

X = pd.get_dummies(
    X,
    drop_first=True,
    dtype=int
)


# --------------------------------------------------
# TRAIN TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# SAVE PROCESSED DATA
# --------------------------------------------------

print("\nSaving processed data...")

X_train.to_csv(
    PROCESSED_DIR / "X_train.csv",
    index=False
)

X_test.to_csv(
    PROCESSED_DIR / "X_test.csv",
    index=False
)

y_train.to_csv(
    PROCESSED_DIR / "y_train.csv",
    index=False
)

y_test.to_csv(
    PROCESSED_DIR / "y_test.csv",
    index=False
)


# --------------------------------------------------
# METADATA
# --------------------------------------------------

metadata = {
    "original_rows": int(df.shape[0]),
    "original_columns": int(df.shape[1]),
    "training_rows": int(X_train.shape[0]),
    "testing_rows": int(X_test.shape[0]),
    "features": int(X_train.shape[1]),
    "target": "pass",
    "target_definition": "1 if average score >= 50, otherwise 0"
}

import json

with open(
    PROCESSED_DIR / "dataset_metadata.json",
    "w"
) as f:
    json.dump(
        metadata,
        f,
        indent=4
    )


# --------------------------------------------------
# OUTPUT
# --------------------------------------------------

print("\n======================================")
print(" PREPROCESSING COMPLETED!")
print("======================================")

print("Training data:", X_train.shape)
print("Testing data :", X_test.shape)

print("\nFiles created:")
print("X_train.csv")
print("X_test.csv")
print("y_train.csv")
print("y_test.csv")
print("dataset_metadata.json")