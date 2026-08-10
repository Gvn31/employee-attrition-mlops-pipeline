import os
import json
import subprocess
import sys



# Project paths


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MONITORING_DIR = os.path.join(BASE_DIR,"monitoring")

PERFORMANCE_REPORT = os.path.join(MONITORING_DIR,"performance_report.json")

DRIFT_REPORT = os.path.join(MONITORING_DIR,"drift_report.json")

RETRAIN_SCRIPT = os.path.join(BASE_DIR,"src","retrain.py")

# Check performance

def check_performance():

    if not os.path.exists(PERFORMANCE_REPORT):

        print("Performance report not found.")
        return False

    with open(PERFORMANCE_REPORT,"r") as file:
        report = json.load(file)

    retraining_required = report.get("retraining_required",False)

    if retraining_required:

        print("Performance drop detected.")

        return True

    print("Model performance is healthy.")

    return False


# --------------------------------------------------
# Check data drift
# --------------------------------------------------

def check_drift():

    if not os.path.exists(DRIFT_REPORT):
        print("Drift report not found.")

        return False

    with open(DRIFT_REPORT,"r") as file:
        report = json.load(file)

    drift_detected = report.get(
        "drift_detected",
        False
    )

    if drift_detected:
        print("Data drift detected.")

        return True
    
    print("No significant data drift detected.")

    return False



# Trigger retraining

def trigger_retraining():

    print("=" * 60)
    print("RETRAINING TRIGGER")
    print("=" * 60)

    performance_drop = check_performance()
    drift_detected = check_drift()


    # Trigger condition
    
    if performance_drop or drift_detected:

        print("\nRetraining condition detected.")

        if performance_drop:
            print("- Performance degradation detected")

        if drift_detected:
            print("- Data drift detected")

        print("\nStarting retraining pipeline...")

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.retrain"
            ],
            cwd=BASE_DIR
        )

        if result.returncode == 0:
            print("\nRetraining completed successfully.")

        else:
            print("\nRetraining failed.")

            return False

    else:
        print("\nNo retraining condition detected.")

        print("Model does not require retraining.")
    print("=" * 60)
    return True



# Main

if __name__ == "__main__":

    try:
        retraining_triggered = trigger_retraining()

        if retraining_triggered:
            print("RETRAINING_TRIGGERED")
        else:
            print("NO_RETRAINING_REQUIRED")

    except Exception as e:

        print(
            f"Retraining trigger failed: {e}"
        )

        sys.exit(1)