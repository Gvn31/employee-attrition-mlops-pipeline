import os
import json
import pandas as pd
import sys

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


#
# Load reference data


print("Loading reference data...")

reference_df = pd.read_parquet(
    REFERENCE_FILE
)

print(
    f"Reference rows: {len(reference_df)}"
)



# Load production prediction logs


print("Loading production prediction logs...")

current_raw_df = pd.read_csv(
    CURRENT_FILE
)

print(
    f"Current rows: {len(current_raw_df)}"
)



# Prepare current production data


# Remove prediction/output information.
# These are not input features.

columns_to_drop = [
    "Timestamp",
    "Prediction",
    "Stay_Probability",
    "Leave_Probability"
]

current_raw_df = current_raw_df.drop(
    columns=[
        col
        for col in columns_to_drop
        if col in current_raw_df.columns
    ]
)



# Apply the same feature engineering
# used by the production API


print("Transforming current production data...")

current_df = transform_features(
    current_raw_df,
    training=False
)


# Select comparable feature columns


excluded_columns = [
    "Employee ID",
    "event_timestamp",
    "Attrition"
]

reference_features = [
    col
    for col in reference_df.columns
    if col not in excluded_columns
]

current_features = [
    col
    for col in current_df.columns
    if col not in excluded_columns
]


# Keep only features available in both datasets.

common_features = sorted(
    set(reference_features)
    & set(current_features)
)

if not common_features:
    raise ValueError(
        "No common features found between "
        "reference and current data."
    )


reference_data = reference_df[
    common_features
].copy()

current_data = current_df[
    common_features
].copy()


# Align data types


for column in common_features:

    reference_data[column] = pd.to_numeric(
        reference_data[column],
        errors="coerce"
    )

    current_data[column] = pd.to_numeric(
        current_data[column],
        errors="coerce"
    )


# Remove columns that contain no usable
# current production observations.

valid_features = []

for column in common_features:

    if (
        reference_data[column].notna().sum() > 0
        and current_data[column].notna().sum() > 0
    ):
        valid_features.append(column)


reference_data = reference_data[
    valid_features
]

current_data = current_data[
    valid_features
]


# Calculate drift


print(
    f"Comparing {len(valid_features)} features..."
)

drift_results = {}

drifted_features = []

for column in valid_features:

    reference_mean = reference_data[
        column
    ].mean()

    current_mean = current_data[
        column
    ].mean()

    reference_std = reference_data[
        column
    ].std()

    if pd.isna(reference_std) or reference_std == 0:
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
        drifted_features.append(column)


# Overall drift decision

drift_detected = (
    len(drifted_features) > 0
)


# Create report


report = {
    "drift_detected": drift_detected,
    "drift_threshold": DRIFT_THRESHOLD,
    "total_features": len(valid_features),
    "drifted_features": len(
        drifted_features
    ),
    "drifted_feature_names": drifted_features,
    "feature_results": drift_results
}


# Save report

with open(
    REPORT_FILE,
    "w"
) as file:

    json.dump(
        report,
        file,
        indent=4
    )


# Console output

print("\n" + "=" * 60)
print("DATA DRIFT CHECK")
print("=" * 60)

print(
    f"Total features : {len(valid_features)}"
)

print(
    f"Drifted features : "
    f"{len(drifted_features)}"
)

print(
    f"Drift detected : "
    f"{drift_detected}"
)

if drifted_features:

    print("\nDrifted features:")

    for feature in drifted_features:
        print(f" - {feature}")

else:

    print(
        "\nNo significant data drift detected."
    )

print(
    f"\nReport saved to: {REPORT_FILE}"
)

print("=" * 60)