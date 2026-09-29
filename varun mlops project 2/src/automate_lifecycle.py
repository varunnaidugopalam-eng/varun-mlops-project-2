from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

MLFLOW_DB = BASE_DIR / "mlflow.db"


# --------------------------------------------------
# MLflow CONNECTION
# --------------------------------------------------

mlflow.set_tracking_uri(
    "sqlite:///" + str(MLFLOW_DB)
)

client = MlflowClient()


# --------------------------------------------------
# REGISTERED MODEL
# --------------------------------------------------

MODEL_NAME = "Student_Performance_Model"


print("\n======================================")
print(" MODEL LIFECYCLE CHECK")
print("======================================")


# --------------------------------------------------
# GET MODEL VERSIONS
# --------------------------------------------------

versions = client.search_model_versions(
    f"name='{MODEL_NAME}'"
)


if not versions:
    print("No registered model versions found.")
    raise SystemExit(1)


print("\nModel name:", MODEL_NAME)

print("\nAvailable Model Versions:")

for version in sorted(
    versions,
    key=lambda x: int(x.version),
    reverse=True
):

    print(
        f"Version: {version.version} | "
        f"Run ID: {version.run_id}"
    )


# --------------------------------------------------
# LATEST VERSION
# --------------------------------------------------

latest_version = max(
    versions,
    key=lambda x: int(x.version)
)


print(
    "\nLatest model version:",
    latest_version.version
)


# --------------------------------------------------
# MODEL URI
# --------------------------------------------------

model_uri = (
    f"models:/{MODEL_NAME}/"
    f"{latest_version.version}"
)

print(
    "Model URI:",
    model_uri
)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

try:

    model = mlflow.pyfunc.load_model(
        model_uri
    )

    print("\nModel loaded successfully!")

except Exception as e:

    print("\nModel loading failed!")
    print(e)
    raise SystemExit(1)


# --------------------------------------------------
# FINAL OUTPUT
# --------------------------------------------------

print("\n======================================")
print(" MODEL LIFECYCLE CHECK COMPLETED!")
print("======================================")

print(
    "Registered Model :",
    MODEL_NAME
)

print(
    "Current Version  :",
    latest_version.version
)

print(
    "Model Status     : READY FOR USE"
)