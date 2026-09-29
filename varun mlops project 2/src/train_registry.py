from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score


BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data" / "processed"

MLFLOW_DB = BASE_DIR / "mlflow.db"


# MLflow tracking database
mlflow.set_tracking_uri(
    "sqlite:///" + str(MLFLOW_DB)
)


# Load data
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
print("Testing data:", X_test.shape)


# MLflow experiment
mlflow.set_experiment(
    "Student_Performance_Registry"
)


# Model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_split=5,
    random_state=42,
    class_weight="balanced"
)


# Train and register
with mlflow.start_run(
    run_name="student_performance_registry"
) as run:

    print("\nTraining Random Forest model...")

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    print("\nF1 Score:", f1)

    # Parameters
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

    # Metric
    mlflow.log_metric(
        "f1_score",
        f1
    )

    # Register model
    model_name = "Student_Performance_Model"

    print("\nLogging model to MLflow...")

    mlflow.sklearn.log_model(
        model,
        name="model",
        registered_model_name=model_name
    )

    print("\n======================================")
    print(" MODEL REGISTERED SUCCESSFULLY!")
    print("======================================")

    print("Run ID:", run.info.run_id)
    print("Registered Model:", model_name)
    print("F1 Score:", f1)
    print("MLflow Database:", MLFLOW_DB)
    