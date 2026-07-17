from fastapi import FastAPI
import pandas as pd
import joblib
import os
from datetime import datetime

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

#Log file path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_FILE = os.path.join(BASE_DIR, "logs", "prediction_logs.csv")


def log_prediction(input_data, prediction, stay_prob, leave_prob):
    log_data = input_data.copy()

    log_data["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_data["Prediction"] = prediction
    log_data["Stay_Probability"] = round(stay_prob, 4)
    log_data["Leave_Probability"] = round(leave_prob, 4)

    df = pd.DataFrame([log_data])

    # Create file with headers if it doesn't exist
    if not os.path.exists(LOG_FILE) or os.path.getsize(LOG_FILE) == 0:
        df.to_csv(LOG_FILE, index=False)
    else:
        #Append without headers
        df.to_csv(LOG_FILE, mode="a", header=False, index=False)

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

    #Convert probabilitites
    stay_prob = probability[0][0]
    leave_prob = probability[0][1]


    if prediction[0] == 1:
        result = "Employee is likely to leave the company."
    else:
        result = "Employee is likely to stay in the company."

    #Log prediction
    log_prediction(employee, result, stay_prob, leave_prob)

    return {
        "prediction": result,
        "stay_probability": round(float(probability[0][0]), 4),
        "leave_probability": round(float(probability[0][1]), 4)
    }