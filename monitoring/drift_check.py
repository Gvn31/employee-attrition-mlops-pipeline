import os
import sys
import json
import pandas as pd

# Project paths

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, BASE_DIR)

from src.feature_engineering import transform_features


REFERENCE_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "emp_attrition_features.parquet"
)

CURRENT_FILE = os.path.join(
    BASE_DIR,
    "logs",
    "prediction_logs.csv"
)

REPORT_FILE = os.path.join(
    BASE_DIR,
    "monitoring",
    "drift_report.json"
)

# Configuration

DRIFT_THRESHOLD = 0.20


# Load reference data

def load_reference_data():
    """
    Load the reference feature dataset used for
    production data drift comparison.
    """

    print("Loading reference data...")

    if not os.path.exists(REFERENCE_FILE):
        raise FileNotFoundError(
            f"Reference feature file not found: "
            f"{REFERENCE_FILE}"
        )

    reference_df = pd.read_parquet(
        REFERENCE_FILE
    )

    print(
        f"Reference rows: {len(reference_df)}"
    )

    return reference_df


# Handle missing production data

def handle_missing_production_data():
    """
    Handle the initial production state where
    no prediction logs are available yet.
    """

    print("Prediction log file not found.")
    print("No production predictions are available yet.")
    print("Skipping data drift check.")

    report = {
        "drift_detected": False,
        "drift_threshold": DRIFT_THRESHOLD,
        "total_features": 0,
        "drifted_features": 0,
        "drifted_feature_names": [],
        "feature_results": {},
        "status": "NO_PRODUCTION_DATA"
    }

    save_report(report)

    return False


# Load production prediction logs

def load_production_data():
    """
    Load production prediction logs.

    Returns:
        DataFrame or None if production data
        is not available yet.
    """

    print("Loading production prediction logs...")

    if not os.path.exists(CURRENT_FILE):
        return None

    current_raw_df = pd.read_csv(
        CURRENT_FILE
    )

    print(
        f"Current rows: {len(current_raw_df)}"
    )

    if current_raw_df.empty:
        print("Prediction log file is empty.")
        print("No production data available for drift detection.")
        return None

    return current_raw_df


# Prepare production data

def prepare_production_data(current_raw_df):
    """
    Remove prediction/output columns and apply
    the same feature engineering used during training.
    """

    print("Preparing current production data...")

    columns_to_drop = [
        "Timestamp",
        "Prediction",
        "Stay_Probability",
        "Leave_Probability"
    ]

    current_raw_df = current_raw_df.drop(
        columns=[
            column
            for column in columns_to_drop
            if column in current_raw_df.columns
        ]
    )

    print("Transforming current production data...")

    current_df = transform_features(
        current_raw_df,
        training=False
    )

    return current_df


# Select common features

def select_common_features(
    reference_df,
    current_df
):
    """
    Select features available in both reference
    and current production datasets.
    """

    excluded_columns = [
        "Employee ID",
        "event_timestamp",
        "Attrition"
    ]

    reference_features = [
        column
        for column in reference_df.columns
        if column not in excluded_columns
    ]

    current_features = [
        column
        for column in current_df.columns
        if column not in excluded_columns
    ]

    common_features = sorted(
        set(reference_features)
        & set(current_features)
    )

    if not common_features:
        raise ValueError(
            "No common features found between "
            "reference and current production data."
        )

    print(
        f"Common features found: "
        f"{len(common_features)}"
    )

    reference_data = reference_df[
        common_features
    ].copy()

    current_data = current_df[
        common_features
    ].copy()

    return (
        reference_data,
        current_data,
        common_features
    )


# Align data types

def align_data_types(
    reference_data,
    current_data,
    common_features
):
    """
    Convert comparable features to numeric values.
    """

    for column in common_features:

        reference_data[column] = pd.to_numeric(
            reference_data[column],
            errors="coerce"
        )

        current_data[column] = pd.to_numeric(
            current_data[column],
            errors="coerce"
        )

    return (
        reference_data,
        current_data
    )


# Select valid features

def select_valid_features(
    reference_data,
    current_data,
    common_features
):
    """
    Keep only features that contain usable observations
    in both reference and production datasets.
    """

    valid_features = []

    for column in common_features:

        reference_count = (
            reference_data[column]
            .notna()
            .sum()
        )

        current_count = (
            current_data[column]
            .notna()
            .sum()
        )

        if (
            reference_count > 0
            and current_count > 0
        ):
            valid_features.append(column)

    if not valid_features:
        raise ValueError(
            "No valid features available for "
            "drift calculation."
        )

    return valid_features


# Calculate data drift

def calculate_drift(
    reference_data,
    current_data,
    valid_features
):
    """
    Calculate drift using the standardized mean
    difference between reference and production data.
    """

    print(
        f"Comparing {len(valid_features)} features..."
    )

    drift_results = {}
    drifted_features = []

    for column in valid_features:

        reference_mean = (
            reference_data[column].mean()
        )

        current_mean = (
            current_data[column].mean()
        )

        reference_std = (
            reference_data[column].std()
        )

        if (
            pd.isna(reference_std)
            or reference_std == 0
        ):
            drift_score = 0.0

        else:
            drift_score = abs(
                current_mean - reference_mean
            ) / reference_std

        drift_results[column] = {
            "reference_mean": float(
                reference_mean
            ),
            "current_mean": float(
                current_mean
            ),
            "drift_score": float(
                drift_score
            )
        }

        if drift_score > DRIFT_THRESHOLD:
            drifted_features.append(
                column
            )

    drift_detected = (
        len(drifted_features) > 0
    )

    return (
        drift_detected,
        drifted_features,
        drift_results
    )


# Create drift report

def create_report(
    drift_detected,
    drifted_features,
    drift_results,
    total_features
):
    """
    Create the data drift monitoring report.
    """

    return {
        "drift_detected": drift_detected,
        "drift_threshold": DRIFT_THRESHOLD,
        "total_features": total_features,
        "drifted_features": len(
            drifted_features
        ),
        "drifted_feature_names": (
            drifted_features
        ),
        "feature_results": drift_results,
        "status": (
            "DRIFT_DETECTED"
            if drift_detected
            else "HEALTHY"
        )
    }


# Save report

def save_report(report):
    """
    Save the drift monitoring report.
    """

    os.makedirs(
        os.path.dirname(REPORT_FILE),
        exist_ok=True
    )

    with open(
        REPORT_FILE,
        "w"
    ) as file:

        json.dump(
            report,
            file,
            indent=4
        )

    print(
        f"\nReport saved to: "
        f"{REPORT_FILE}"
    )


# Print results

def print_results(
    drift_detected,
    drifted_features,
    total_features
):
    """
    Print drift monitoring results.
    """

    print("\nDATA DRIFT CHECK")

    print(
        f"Total features   : "
        f"{total_features}"
    )

    print(
        f"Drifted features : "
        f"{len(drifted_features)}"
    )

    print(
        f"Drift detected   : "
        f"{drift_detected}"
    )

    if drifted_features:

        print("\nDrifted features:")

        for feature in drifted_features:
            print(
                f" - {feature}"
            )

    else:

        print(
            "\nNo significant data drift detected."
        )

    print(
        f"\nReport saved to: "
        f"{REPORT_FILE}"
    )


# Main drift check

def check_data_drift():
    """
    Execute the complete data drift monitoring pipeline.

    Returns:
        True  -> Drift detected
        False -> No drift / no production data
    """

    print("Production Data Drift Check")

    # Load reference data

    reference_df = load_reference_data()

    # Load production data

    current_raw_df = load_production_data()

    # Fresh deployment with no predictions yet

    if current_raw_df is None:
        return handle_missing_production_data()

    # Prepare production data

    current_df = prepare_production_data(
        current_raw_df
    )

    # Select common features

    (
        reference_data,
        current_data,
        common_features
    ) = select_common_features(
        reference_df,
        current_df
    )

    # Align data types

    (
        reference_data,
        current_data
    ) = align_data_types(
        reference_data,
        current_data,
        common_features
    )

    # Select usable features

    valid_features = select_valid_features(
        reference_data,
        current_data,
        common_features
    )

    reference_data = reference_data[valid_features]

    current_data = current_data[valid_features]

    # Calculate drift

    (
        drift_detected,
        drifted_features,
        drift_results
    ) = calculate_drift(
        reference_data,
        current_data,
        valid_features
    )

    # Create report

    report = create_report(
        drift_detected,
        drifted_features,
        drift_results,
        len(valid_features)
    )

    # Save report

    save_report(report)

    # Print results

    print_results(
        drift_detected,
        drifted_features,
        len(valid_features)
    )

    return drift_detected


# Main

if __name__ == "__main__":

    try:
        drift_detected = check_data_drift()
        if drift_detected:
            print("\nData drift detected.")
        else:
            print("\nNo data drift detected.")

    except Exception as e:

        print(
            f"\nData drift check failed: {e}"
        )
        sys.exit(1)
