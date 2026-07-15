from fastapi import FastAPI
import pandas as pd
import joblib

from feature_engineering import transform_features

# Create FastAPI application
app = FastAPI(
    title="Employee Attrition Prediction API",
    description="Predict whether an employee is likely to leave the company.",
    version="1.0"
)

# Load trained model
print("Loading trained model...")
model = joblib.load("../models/xgboost.pkl")


@app.get("/")
def home():
    """
    Home endpoint.
    """

    return {
        "message": "Employee Attrition Prediction API is running successfully!"
    }


@app.post("/predict")
def predict(employee: dict):
    """
    Predict employee attrition.
    """

    # Convert input to DataFrame
    df = pd.DataFrame([employee])

    # Apply feature engineering
    df = transform_features(df, training=False)

    # Predict
    prediction = model.predict(df)
    probability = model.predict_proba(df)

    if prediction[0] == 1:
        result = "Employee is likely to leave the company."
    else:
        result = "Employee is likely to stay in the company."

    return {
        "prediction": result,
        "stay_probability": round(float(probability[0][0]), 4),
        "leave_probability": round(float(probability[0][1]), 4)
    }