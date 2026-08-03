import pandas as pd
import mlflow
import mlflow.pyfunc
import os

from feature_engineering import transform_features

# Base project directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def predict():
    """
    Predict employee attrition using the trained XGBoost model.

    Returns:
        None
    """

    print("Loading trained model...")
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    model = mlflow.pyfunc.load_model(
        model_uri="models:/employee_attrition_model/latest"
    )

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

    prediction = int(model.predict(df)[0])
   

    print("\nPrediction Result")

    if prediction == 1:
        return "Employee is likely to leave the company."
    else:
        return "Employee is likely to stay in the company."


if __name__ == "__main__":
    try:
        result = predict()
        print(result)

    except Exception as e:
        print(f"Error: {e}")