from fastapi import FastAPI
import pandas as pd
import mlflow
import mlflow.sklearn
import os
from datetime import datetime
import joblib
import json
from src.feature_engineering import transform_features

# Create FastAPI application
app = FastAPI(
    title="Employee Attrition Prediction API",
    description="Predict whether an employee is likely to leave the company.",
    version="1.0"
)

# Base project directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load trained model
print("Loading trained model...")
# mlflow.set_tracking_uri("http://host.docker.internal:5000")

champion_path=os.path.join(
    BASE_DIR,"evaluation","champion_model.json"
    )

if not os.path.exists(champion_path):
    raise FileNotFoundError(
        "champion_model.json not found."
    )
with open(champion_path,"r") as f:
    champion=json.load(f)


model_map={
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree" : "decision_tree.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl"}

model_name=champion["best_model"]

if model_name not in model_map:
    raise ValueError(f"Unknown champion model: {model_name}")
model =joblib.load(
    os.path.join(BASE_DIR,"models",model_map[model_name])
)

# Log file path
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
        # Append without headers
        df.to_csv(LOG_FILE, mode="a", header=False, index=False)


@app.get("/")
def home():
    return {
        "message": "Employee Attrition Prediction API is running successfully!"
    }


@app.post("/predict")
def predict(employee: dict):

    # Convert input to DataFrame
    df = pd.DataFrame([employee])

    # Apply feature engineering
    df = transform_features(df, training=False)

    # Predict
    prediction = model.predict(df)
    probability = model.predict_proba(df)

    # Convert probabilities
    stay_prob = probability[0][0]
    leave_prob = probability[0][1]

    if prediction[0] == 1:
        result = "Employee is likely to leave the company."
    else:
        result = "Employee is likely to stay in the company."

    # Log prediction
    log_prediction(employee, result, stay_prob, leave_prob)

    return {
        "prediction": result,
        "stay_probability": round(float(stay_prob), 4),
        "leave_probability": round(float(leave_prob), 4)
    }