import os
import json
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)



# Project paths


BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

LOG_FILE = os.path.join(
    BASE_DIR,
    "logs",
    "prediction_logs.csv"
)

MONITORING_DIR = os.path.join(
    BASE_DIR,
    "monitoring"
)

REPORT_FILE = os.path.join(
    MONITORING_DIR,
    "performance_report.json"
)



# Performance threshold
F1_THRESHOLD = 0.70

# Performance check


def check_performance():

    print("=" * 60)
    print("Production Model Performance Check")
    print("=" * 60)

    # Check prediction log
    if not os.path.exists(LOG_FILE):
        print("Prediction log not found.")
        return False

    # Load prediction logs
    logs = pd.read_csv(LOG_FILE)

    print(f"Total prediction records: {len(logs)}")

    # Check required columns
    required_columns = [
        "Prediction",
        "Actual_Attrition",
        "Stay_Probability",
        "Leave_Probability"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in logs.columns
    ]

    if missing_columns:

        print(
            f"Missing required columns: {missing_columns}"
        )

        return False

   
    # Use only predictions with known actual outcomes
   
    evaluation_data = logs.dropna(
        subset=["Actual_Attrition"]
    ).copy()

    print(
        f"Records with actual outcomes: "
        f"{len(evaluation_data)}"
    )

    # Not enough data
    if len(evaluation_data) < 2:

        print(
            "Not enough production feedback data "
            "for performance evaluation."
        )

        return False

  
    # Convert prediction text to binary values

    evaluation_data["Predicted_Attrition"] = (
        evaluation_data["Prediction"]
        .apply(
            lambda x: 1
            if "leave" in str(x).lower()
            else 0
        )
    )

    evaluation_data["Actual_Attrition"] = (
        evaluation_data["Actual_Attrition"]
        .astype(int)
    )

    y_true = evaluation_data[
        "Actual_Attrition"
    ]

    y_pred = evaluation_data[
        "Predicted_Attrition"
    ]


    # Calculate metrics


    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    # ROC-AUC
    try:

        y_probability = evaluation_data[
            "Leave_Probability"
        ]

        roc_auc = roc_auc_score(
            y_true,
            y_probability
        )

    except ValueError:

        roc_auc = None


    # Determine performance status

    performance_drop = f1 < F1_THRESHOLD

    if performance_drop:
        status = "PERFORMANCE_DROP"
        retraining_required = True
    else:
        status = "HEALTHY"
        retraining_required = False

   
    report = {

        "status": status,
        "retraining_required": retraining_required,
        "evaluation_records": int(len(evaluation_data)),

        "thresholds": {"f1_threshold": F1_THRESHOLD},

        "metrics": {

            "accuracy": round(
                float(accuracy),
                4
            ),

            "precision": round(
                float(precision),
                4
            ),

            "recall": round(
                float(recall),
                4
            ),

            "f1_score": round(
                float(f1),
                4
            ),

            "roc_auc": (
                round(float(roc_auc), 4)
                if roc_auc is not None
                else None)

        }

    }


    # Save report

    with open(REPORT_FILE,"w") as file:
        json.dump(report,file, indent=4)

    # Print results

    print("\nProduction Performance")

    print(f"Accuracy:{accuracy:.4f}")

    print(f"Precision:{precision:.4f}")

    print(f"Recall:{recall:.4f}")

    print(f"F1 Score:{f1:.4f}")

    if roc_auc is not None:

        print(f"ROC-AUC:{roc_auc:.4f}")

    print("\nF1 Threshold:{F1_THRESHOLD:.2f}")

    print(f"Status:{status}")

    print(f"Retraining:{retraining_required}")

    print(
        f"\nReport saved to:"
        f"\n{REPORT_FILE}")

    print("=" * 60)

    return retraining_required

# Main
if __name__ == "__main__":

    try:
        check_performance()

    except Exception as e:
        print(f"Performance check failed: {e}")