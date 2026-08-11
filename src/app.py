from fastapi import FastAPI
import pandas as pd
import mlflow
import mlflow.sklearn
import os
from datetime import datetime
import joblib
import json
from src.feature_engineering import transform_features
from prometheus_client import Counter,Histogram,Gauge,Info
from prometheus_fastapi_instrumentator import Instrumentator
import time
from fastapi import HTTPException



# Create FastAPI application
app = FastAPI(
    title="Employee Attrition Prediction API",
    description="Predict whether an employee is likely to leave the company.",
    version="1.0"
)

# Prometheus metrics
Instrumentator().instrument(app).expose(app)

# Base project directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Log file path
LOG_FILE = os.path.join(BASE_DIR, "logs", "prediction_logs.csv")
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

#Prometheus custom metrics
REQUEST_COUNT = Counter(
    "prediction_requests_total",
    "Total prediction requests"
    )
PREDICTION_COUNT=Counter(
    "predictions_total",
    "Total successful predictions"
)
FAILED_REQUESTS=Counter(
    "failed_predictions_total",
    "Total failed predictions requests"
)
PREDICTION_LATENCY=Histogram(
    "prediction_latency_seconds",
    "Prediction latency"
)

MODEL_LOADED=Gauge(
    "model_loaded",
    "Whether Model is loaded"
)

MODEL_INFO=Info(
    "model_info",
    "Current Champion Model"
)

STAY_PREDICTION=Counter(
    "stay_predictions_total",
    "Total stay predictions"
)

LEAVE_PREDICTION=Counter(
    "leave_predictions_total",
    "Total leave predictions"
)

PREDICTION_CONFIDENCE=Histogram(
    "prediction_confidence",
    "Prediction confidence"
)

FEATURE_ENGINEERING_FAILURES=Counter(
    "feature_engineering_failures_total",
    "Feature engineering failures"
)

MODEL_HEALTH=Gauge(
    "model_health",
    "Current health status"
)

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

MODEL_LOADED.set(1)
MODEL_HEALTH.set(1)
MODEL_INFO.info({
    "name": model_name
})


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
@app.get("/health")
def health():

    if model is None:

        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )

    return {
        "status": "healthy",
        "service": "Employee Attrition Prediction API",
        "model_loaded":True,
        "time_stamp": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S")

    }

@app.get("/ready")
def ready():

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )

    return {
        "ready": True
    }
@app.post("/predict")
def predict(employee: dict):

    REQUEST_COUNT.inc()
    start_time = time.time()

    try:
        # Get Employee ID for tracking
        employee_id = employee.get("Employee ID")

        # Convert input to DataFrame
        df = pd.DataFrame([employee])

        # Employee ID is only for tracking, not for prediction
        if "Employee ID" in df.columns:
            df = df.drop(columns=["Employee ID"])

        try:
            # Apply feature engineering
            df = transform_features(df, training=False)

            
        except Exception as e:
            FEATURE_ENGINEERING_FAILURES.inc()
            raise e

        #Model prediction
        try:
            prediction = model.predict(df)
            probability = model.predict_proba(df)

            #Model Success
            MODEL_HEALTH.set(1)

        except Exception as e:

            # Model Exception failed
            MODEL_HEALTH.set(0)

            raise HTTPException(
                status_code=500,
                detail="Model prediction failed"
            )

        # Convert probabilities
        stay_prob = probability[0][0]
        leave_prob = probability[0][1]


        #Prediction confidence
        PREDICTION_CONFIDENCE.observe(
            max(stay_prob,leave_prob)
        )


        #Prediction Result 
        if prediction[0] == 1:
            LEAVE_PREDICTION.inc()
            result = "Employee is likely to leave the company."
        else:
            STAY_PREDICTION.inc()
            result = "Employee is likely to stay in the company."


        #Successful prediction
        PREDICTION_COUNT.inc()

        # Log prediction

        log_prediction(
            employee,result,stay_prob,leave_prob
        )
        return{
            "prediction": result,
            "stay_probability": round(float(stay_prob), 4),
            "leave_probability": round(float(leave_prob), 4)
        }


    except HTTPException:
        FAILED_REQUESTS.inc()
        raise

    except Exception as e:
        FAILED_REQUESTS.inc()
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:
        #Record latence for every request
        PREDICTION_LATENCY.observe(
            time.time() - start_time
            )


@app.post("/feedback")
def feedback(employee_id: int, actual_attrition: int):

    if actual_attrition not in [0, 1]:
        raise HTTPException(
            status_code=400,
            detail="actual_attrition must be 0 or 1"
        )

    if not os.path.exists(LOG_FILE):
        raise HTTPException(
            status_code=404,
            detail="Prediction log not found"
        )

    logs = pd.read_csv(LOG_FILE)

    if "Employee ID" not in logs.columns:
        raise HTTPException(
            status_code=400,
            detail="Employee ID not available in prediction logs"
        )

    matching_rows = logs[
        logs["Employee ID"] == employee_id
    ]

    if matching_rows.empty:
        raise HTTPException(
            status_code=404,
            detail="Prediction for employee not found"
        )

    logs.loc[
        logs["Employee ID"] == employee_id,
        "Actual_Attrition"
    ] = actual_attrition

    logs.to_csv(
        LOG_FILE,
        index=False
    )

    return {
        "message": "Actual outcome recorded successfully",
        "Employee ID": employee_id,
        "Actual_Attrition": actual_attrition
    }
 