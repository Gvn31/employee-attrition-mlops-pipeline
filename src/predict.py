import pandas as pd
import joblib


def predict():
    """
    Predict employee attrition using the trained XGBoost model.

    Returns:
        None
    """

    print("Loading trained model...")
    model = joblib.load("../models/xgboost.pkl")

    print("Loading preprocessing files...")
    scaler = joblib.load("../models/scaler.pkl")
    feature_columns = joblib.load("../models/feature_columns.pkl")

    print("Loading sample data...")

    employee = {
        "Age": 30,
        "Years at Company": 5,
        "Monthly Income": 50000,
        "Number of Promotions": 2,
        "Distance from Home": 10,
        "Number of Dependents": 2,
        "Company Tenure (In Months)": 60,

        "Gender_Male": 1,
        "Job Role_Healthcare": 0,
        "Job Role_Media": 0,
        "Job Role_Technology": 1,

        #Remaining features will be set to 0 or added automatically
    }

    # Convert dictionary to DataFrame
    df = pd.DataFrame([employee])

    # Add missing feature columns
    for column in feature_columns:
        if column not in df.columns:
            df[column] = 0

    # Arrange columns in the same order used during training
    df = df[feature_columns]

    # Scale numerical columns
    numerical_columns = [
        "Age",
        "Years at Company",
        "Monthly Income",
        "Number of Promotions",
        "Distance from Home",
        "Number of Dependents",
        "Company Tenure (In Months)"
    ]

    df[numerical_columns] = scaler.transform(df[numerical_columns])

    # Make prediction
    prediction = model.predict(df)

    print("\nPrediction Result")

    if prediction[0] == 1:
        print("Employee is likely to leave the company.")
    else:
        print("Employee is likely to stay in the company.")


if __name__ == "__main__":
    try:
        predict()

    except Exception as e:
        print(f"Error: {e}")