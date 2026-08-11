import os
import sys
import subprocess
import json


BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

PERFORMANCE_REPORT = os.path.join(
    BASE_DIR,
    "monitoring",
    "performance_report.json"
)

DRIFT_REPORT = os.path.join(
    BASE_DIR,
    "monitoring",
    "drift_report.json"
)


def check_retraining_condition():

    retrain_reasons = []

    # Check performance


    if os.path.exists(PERFORMANCE_REPORT):

        with open(PERFORMANCE_REPORT, "r") as f:
            performance = json.load(f)

        if performance.get("retrain_required", False):
            retrain_reasons.append(
                "Model performance dropped"
            )

    else:
        print("Performance report not found.")

    # Check data drift


    if os.path.exists(DRIFT_REPORT):

        with open(DRIFT_REPORT, "r") as f:
            drift = json.load(f)

        if drift.get("drift_detected", False):
            retrain_reasons.append(
                "Data drift detected"
            )

    else:
        print("Drift report not found.")

    return retrain_reasons


def trigger_retraining():

    print("=" * 60)
    print("RETRAINING TRIGGER")
    print("=" * 60)

    reasons = check_retraining_condition()

    if not reasons:

        print("\nNo retraining condition detected.")
        print("Model does not require retraining.")

        return False

    print("\nRetraining condition detected.")

    for reason in reasons:
        print(f"- {reason}")

    print("\nStarting retraining pipeline...")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.retrain"
        ],
        cwd=BASE_DIR
    )

    if result.returncode != 0:

        print("\nRetraining failed.")
        return False

    print("\nRetraining completed successfully.")

    return True


if __name__ == "__main__":

    try:

        retraining_triggered = trigger_retraining()

        #Send Result to GitHub Actions
        github_output=os.environ.get("GITHUB_OUTPUT")

        if github_output:
            with open(github_output, "a") as f:
                f.write(f"retraining_triggered={'true' if retraining_triggered else 'false'}\n")

        if retraining_triggered:
            print("RETRAINING_TRIGGERED")
        else:
            print("NO_RETRAINING_REQUIRED")

    except Exception as e:

        print(f"Retraining trigger failed: {e}")
        sys.exit(1)