# Employee Attrition MLOps Pipeline

An end-to-end **MLOps pipeline for Employee Attrition Prediction** that covers data preparation, feature engineering, model training, evaluation, model tracking, deployment, monitoring, and automated retraining.

The project uses **DVC, Feast, MLflow, FastAPI, Docker, GitHub Actions, Prometheus, and Grafana**.

---

## Project Objective

Predict whether an employee is likely to leave an organization based on demographic and workplace-related information.

**Problem Type:** Binary Classification

**Target:** `Attrition`

**Classes:**
- Stayed
- Left

---

## MLOps Architecture

```text
Raw Data
   ↓
Data Preprocessing
   ↓
DVC
   ↓
Feature Engineering
   ↓
Feast Feature Store
   ↓
Model Training
   ↓
Model Evaluation
   ↓
MLflow
   ↓
FastAPI + Docker
   ↓
Monitoring
   ↓
Drift / Performance Check
   ↓
Retraining
   ↓
Redeployment
```

---

## CI/CD Pipeline

GitHub Actions automates the complete MLOps workflow.

```text
Code / Data Change
       ↓
   Tests
       ↓
Preprocessing
       ↓
Feature Engineering
       ↓
    Feast
       ↓
Model Training
       ↓
Model Evaluation
       ↓
   MLflow
       ↓
Docker Build
       ↓
FastAPI Deployment
       ↓
API Health Check
       ↓
Data Drift Check
       ↓
Performance Check
       ↓
Retraining if Required
       ↓
Redeployment
```

---

## Technologies

- **Python**
- **Pandas / NumPy**
- **Scikit-learn**
- **XGBoost**
- **DVC**
- **Feast**
- **MLflow**
- **FastAPI**
- **Docker / Docker Compose**
- **Prometheus**
- **Grafana**
- **GitHub Actions**

---

## Machine Learning Pipeline

### Data Preprocessing

The preprocessing pipeline performs:

- Duplicate removal
- Missing value handling
- Categorical value correction
- Timestamp generation

Output:

```text
data/processed/emp_attrition_cleaned.csv
```

### Feature Engineering

- One-hot encoding
- Numerical feature scaling
- Feature alignment

Generated artifacts:

```text
models/scaler.pkl
models/feature_columns.pkl
```

Feature datasets:

```text
data/processed/emp_attrition_features.csv
data/processed/emp_attrition_features.parquet
```

### Models

The following models are trained and compared:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

`GridSearchCV` is used for hyperparameter tuning.

The model with the best F1 score is selected as the **Champion Model**.

---

## DVC

DVC is used for dataset and pipeline versioning.

```text
Raw Data
   ↓
Processed Data
   ↓
Feature Data
```

Main DVC files:

```text
dvc.yaml
dvc.lock
```

---

## Feast Feature Store

Feast is used to manage and retrieve training features.

The training pipeline uses the:

```text
employee_service
```

Feature Store configuration is applied using:

```bash
cd feature_repo/feature_repo
feast apply
```

---

## MLflow

MLflow is used for:

- Experiment tracking
- Parameter logging
- Metric logging
- Model artifact storage
- Model registration

The champion model is registered as:

```text
employee_attrition_model
```

---

## Model Evaluation

Models are evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix
- Classification Report

The selected champion model is stored in:

```text
evaluation/champion_model.json
```

---

## FastAPI

The trained model is deployed as a REST API.

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API status |
| GET | `/health` | Health check |
| POST | `/predict` | Employee attrition prediction |

Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

Example prediction response:

```json
{
    "prediction": "Left",
    "stay_probability": 0.23,
    "leave_probability": 0.77
}
```

---

## Docker

The application is containerized using Docker.

### Build

```cmd
docker build -t employee-attrition-api .
```

### Docker Compose

```cmd
docker compose up -d --build
```

---

## Monitoring

The project includes:

### API Health Monitoring

```text
GET /health
```

### Prediction Logging

Predictions are stored in:

```text
logs/prediction_logs.csv
```

### Data Drift Monitoring

Implemented using:

```text
monitoring/drift_check.py
```

Report:

```text
monitoring/drift_report.json
```

### Model Performance Monitoring

Implemented using:

```text
monitoring/performance_check.py
```

Report:

```text
monitoring/performance_report.json
```

---

## Automated Retraining

Retraining is triggered when:

```text
Data Drift Detected
        OR
Model Performance Drops
```

Retraining logic:

```text
monitoring/retrain_trigger.py
```

Retraining pipeline:

```text
src/retrain.py
```

After successful retraining, the updated model is redeployed.

---

## GitHub Actions CI/CD

Workflow:

```text
.github/workflows/pipeline.yml
```

The workflow handles:

- Dependency installation
- Data verification
- Testing
- Preprocessing
- Feature engineering
- Feast
- Model training
- Model evaluation
- MLflow
- Docker build
- FastAPI deployment
- API health check
- Data drift monitoring
- Performance monitoring
- Automatic/manual retraining
- Redeployment

The workflow can run through:

- Relevant changes pushed to `main`
- Scheduled execution
- Manual workflow dispatch

---

## Project Structure

```text
employee-attrition-mlops-pipeline/
│
├── .github/
│   └── workflows/
│       └── pipeline.yml
│
├── data/
│   ├── raw/
│   │   └── emp_attrition_csv.csv
│   │
│   └── processed/
│       ├── emp_attrition_cleaned.csv
│       ├── emp_attrition_features.csv
│       └── emp_attrition_features.parquet
│
├── evaluation/
│   └── champion_model.json
│
├── feature_repo/
│   └── feature_repo/
│
├── logs/
│   └── prediction_logs.csv
│
├── models/
│   ├── decision_tree.pkl
│   ├── feature_columns.pkl
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── scaler.pkl
│   └── xgboost.pkl
│
├── monitoring/
│   ├── drift_check.py
│   ├── drift_report.json
│   ├── performance_check.py
│   ├── performance_report.json
│   └── retrain_trigger.py
│
├── notebooks/
│   └── EDA.ipynb
│
├── src/
│   ├── app.py
│   ├── evaluate.py
│   ├── feature_engineering.py
│   ├── predict.py
│   ├── preprocessing.py
│   ├── retrain.py
│   └── train.py
│
├── Dockerfile
├── docker-compose.yaml
├── dvc.yaml
├── dvc.lock
├── prometheus.yaml
├── requirements.txt
├── .gitignore
├── .dockerignore
├── LICENSE
└── README.md
```

---

## Local Setup

### Clone Repository

```cmd
git clone <repository-url>
cd employee-attrition-mlops-pipeline
```

### Create Environment

```cmd
conda create -n employee-mlops python=3.12
conda activate employee-mlops
```

### Install Dependencies

```cmd
pip install -r requirements.txt
```

### Run the Application

```cmd
docker compose up -d --build
```

### Open Swagger

```text
http://127.0.0.1:8000/docs
```

---

## Project Highlights

- End-to-end MLOps pipeline
- Data versioning with DVC
- Feature Store with Feast
- Multiple ML models
- Hyperparameter tuning
- Champion model selection
- MLflow experiment tracking
- MLflow Model Registry
- FastAPI deployment
- Docker containerization
- Prediction logging
- Data drift monitoring
- Model performance monitoring
- Automatic retraining
- GitHub Actions CI/CD
- Prometheus and Grafana

---

## Future Improvements

- Cloud deployment
- Kubernetes
- Advanced model monitoring
- Model explainability
- API authentication
- Automated rollback