import subprocess
import sys


def run_script(script):

    print("\n======================================")
    print(f"Running: {script}")
    print("======================================")

    result = subprocess.run(
        [sys.executable, script]
    )

    if result.returncode != 0:

        print(
            f"\nERROR: {script} failed!"
        )

        sys.exit(1)


# --------------------------------------------------
# PIPELINE
# --------------------------------------------------

print("\n======================================")
print(" STUDENT PERFORMANCE MLOPS PIPELINE")
print("======================================")


print("\n[1] Running preprocessing...")

run_script(
    "src/preprocess.py"
)


print("\n[2] Training model...")

run_script(
    "src/train.py"
)


print("\n[3] Running model evaluation...")

run_script(
    "src/evaluate.py"
)


print("\n[4] Running MLflow training...")

run_script(
    "src/train_mlflow.py"
)


print("\n[5] Registering model...")

run_script(
    "src/train_registry.py"
)


print("\n[6] Checking model lifecycle...")

run_script(
    "src/automate_lifecycle.py"
)


print("\n======================================")
print(" PIPELINE COMPLETED SUCCESSFULLY!")
print("======================================")