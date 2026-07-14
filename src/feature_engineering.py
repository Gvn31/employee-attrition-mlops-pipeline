import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib


def feature_engineering():
    """
    Perform feature engineering on the cleaned employee attrition dataset.

    This function performs the following steps:
    - Loads the cleaned dataset.
    - Encodes the target variable.
    - Applies One-Hot Encoding to categorical features.
    - Scales numerical features using StandardScaler.
    - Saves the feature-engineered dataset.

    Returns:
        None
    """
    print("Loading data...")
    df = pd.read_csv("../data/processed/emp_attrition_cleaned.csv")

    print("Encoding features...")

    # Map target variable
    if "Attrition" in df.columns:
        df["Attrition"] = df["Attrition"].map({"Stayed": 0,"Left": 1})

    # Encode categorical columns
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
    df = pd.get_dummies(df,columns=categorical_cols,drop_first=True,dtype=int)

    print("Scaling numerical features...")
    standard_scaler = StandardScaler()
    numerical_cols = [
        "Age",
        "Years at Company",
        "Monthly Income",
        "Number of Promotions",
        "Distance from Home",
        "Number of Dependents",
        "Company Tenure (In Months)"
    ]
    df[numerical_cols] = standard_scaler.fit_transform(df[numerical_cols])
    # Save scaler
    joblib.dump(standard_scaler, "../models/scaler.pkl")
    print("Scaler saved successfully!")

    # Save feature column names
    feature_columns = df.drop(columns=["Attrition"]).columns.tolist()
    joblib.dump(feature_columns, "../models/feature_columns.pkl")
    print("Feature columns saved successfully!")

    print("Saving feature engineered dataset...")
    df.to_csv("../data/processed/emp_attrition_features.csv",index=False)

    print("Feature engineering completed successfully!")



if __name__ == "__main__":
    try:
        feature_engineering()
    except Exception as e:
        print(f"Error: {e}")

