import pandas as pd
import joblib

from feature_engineering import transform_features


def predict():
    """
    Predict employee attrition using the trained XGBoost model.

    Returns:
        None
    """

    print("Loading trained model...")
    model = joblib.load("../models/xgboost.pkl")

    print("Loading sample data...")

    employee = {
        "Age": 30,
        "Years at Company": 5,
        "Monthly Income": 8000,
        "Number of Promotions": 2,
        "Distance from Home": 10,
        "Number of Dependents": 2,
        "Company Tenure (In Months)": 60,

        "Gender": "Male",
        "Job Role": "Technology",
        "Work-Life Balance": "Good",
        "Job Satisfaction": "High",
        "Performance Rating": "High",
        "Overtime": "No",
        "Education Level": "Master's Degree",
        "Marital Status": "Single",
        "Job Level": "Mid",
        "Company Size": "Medium",
        "Remote Work": "Yes",
        "Leadership Opportunities": "Yes",
        "Innovation Opportunities": "Yes",
        "Company Reputation": "Good",
        "Employee Recognition": "High"
    }

    df = pd.DataFrame([employee])

    # Apply feature engineering
    df = transform_features(df, training=False)


    print("\nPredicting...")

    prediction = model.predict(df)
    probability = model.predict_proba(df)

    print("\nPrediction Result")

    if prediction[0] == 1:
        return "Employee is likely to leave the company."
    else:
        return "Employee is likely to stay in the company."


if __name__ == "__main__":
    try:
        result = predict()
        print(result)

    except Exception as e:
        print(f"Error: {e}")