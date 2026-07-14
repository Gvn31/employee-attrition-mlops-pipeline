import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib


def transform_features(df, training=True):
    """
    Perform feature engineering on the given dataframe.

    Args:
        df (DataFrame): Input dataframe.
        training (bool): True for training, False for prediction.

    Returns:
        DataFrame: Feature engineered dataframe.
    """

    # Encode target variable
    if training and "Attrition" in df.columns:
        df["Attrition"] = df["Attrition"].map({"Stayed": 0, "Left": 1})

    # One-Hot Encoding
    categorical_cols = [
        "Gender",
        "Job Role",
        "Work-Life Balance",
        "Job Satisfaction",
        "Performance Rating",
        "Overtime",
        "Education Level",
        "Marital Status",
        "Job Level",
        "Company Size",
        "Remote Work",
        "Leadership Opportunities",
        "Innovation Opportunities",
        "Company Reputation",
        "Employee Recognition"
    ]

    df = pd.get_dummies(df,columns=categorical_cols,drop_first=False,dtype=int)

    numerical_cols = [
        "Age",
        "Years at Company",
        "Monthly Income",
        "Number of Promotions",
        "Distance from Home",
        "Number of Dependents",
        "Company Tenure (In Months)"
    ]

    if training:

        scaler = StandardScaler()
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        joblib.dump(scaler, "../models/scaler.pkl")
        feature_columns = df.drop(columns=["Attrition"]).columns.tolist()
        joblib.dump(feature_columns, "../models/feature_columns.pkl")

    else:
        scaler = joblib.load("../models/scaler.pkl")
        df[numerical_cols] = scaler.transform(df[numerical_cols])
        feature_columns = joblib.load("../models/feature_columns.pkl")

        for column in feature_columns:
            if column not in df.columns:
                df[column] = 0
        df = df[feature_columns]

    return df


def feature_engineering():
    """
    Perform feature engineering on the cleaned employee attrition dataset.

    Returns:
        None
    """

    print("Loading data...")
    df = pd.read_csv("../data/processed/emp_attrition_cleaned.csv")
    print("Applying feature engineering...")
    df = transform_features(df, training=True)
    print("Saving feature engineered dataset...")
    df.to_csv("../data/processed/emp_attrition_features.csv",index=False)

    print("Feature engineering completed successfully!")


if __name__ == "__main__":
    try:
        feature_engineering()

    except Exception as e:
        print(f"Error: {e}")