from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data" / "processed"

MLFLOW_DB = BASE_DIR / "mlflow.db"


# --------------------------------------------------
# MLFLOW TRACKING
# --------------------------------------------------

mlflow.set_tracking_uri(
    "sqlite:///" + str(MLFLOW_DB)
)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

print("Loading processed data...")

X_train = pd.read_csv(
    DATA_DIR / "X_train.csv"
)

X_test = pd.read_csv(
    DATA_DIR / "X_test.csv"
)

y_train = pd.read_csv(
    DATA_DIR / "y_train.csv"
).values.ravel()

y_test = pd.read_csv(
    DATA_DIR / "y_test.csv"
).values.ravel()


print("Training data:", X_train.shape)
print("Testing data :", X_test.shape)


# --------------------------------------------------
# MLFLOW EXPERIMENT
# --------------------------------------------------

mlflow.set_experiment(
    "Student_Performance_Prediction"
)


# --------------------------------------------------
# MODEL
# --------------------------------------------------

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_split=5,
    random_state=42,
    class_weight="balanced"
)


# --------------------------------------------------
# MLFLOW RUN
# --------------------------------------------------

with mlflow.start_run(
    run_name="random_forest_student_performance"
):

    print("\nTraining Random Forest model...")

    model.fit(
        X_train,
        y_train
    )


    # --------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------

    predictions = model.predict(
        X_test
    )


    # --------------------------------------------------
    # METRICS
    # --------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )


    # --------------------------------------------------
    # LOG PARAMETERS
    # --------------------------------------------------

    mlflow.log_param(
        "model",
        "RandomForestClassifier"
    )

    mlflow.log_param(
        "n_estimators",
        200
    )

    mlflow.log_param(
        "max_depth",
        8
    )

    mlflow.log_param(
        "min_samples_split",
        5
    )

    mlflow.log_param(
        "random_state",
        42
    )


    # --------------------------------------------------
    # LOG METRICS
    # --------------------------------------------------

    mlflow.log_metric(
        "accuracy",
        accuracy
    )

    mlflow.log_metric(
        "precision",
        precision
    )

    mlflow.log_metric(
        "recall",
        recall
    )

    mlflow.log_metric(
        "f1_score",
        f1
    )


    # --------------------------------------------------
    # LOG MODEL
    # --------------------------------------------------

    print("\nLogging model to MLflow...")

    mlflow.sklearn.log_model(
        model,
        name="model",
        skops_trusted_types=[
            "sklearn.tree._tree.Tree"
        ]
    )


    # --------------------------------------------------
    # OUTPUT
    # --------------------------------------------------

    print("\n======================================")
    print(" MLFLOW TRAINING COMPLETED!")
    print("======================================")

    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)

    print("\nMLflow Database:")
    print(MLFLOW_DB)