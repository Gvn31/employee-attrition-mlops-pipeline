# Employee Attrition MLOps Pipeline

## Production-Grade End-to-End MLOps Pipeline

An end-to-end **MLOps pipeline for Employee Attrition Prediction**, designed to demonstrate the complete machine learning lifecycle — from data ingestion and versioning to feature engineering, model training, evaluation, model registry, deployment, monitoring, and automated retraining.

The project follows a production-oriented architecture using **DVC, Feast, MLflow, Docker, FastAPI, GitHub Actions, and monitoring tools**.

---

# Project Objective

Employee attrition is a major challenge for organizations because it can increase hiring costs, reduce productivity, and result in the loss of organizational knowledge.

The objective of this project is to build a machine learning system capable of predicting whether an employee is likely to leave the organization.

The pipeline also demonstrates how the trained model can be:

- Versioned
- Evaluated
- Registered
- Deployed as an API
- Monitored for drift and performance
- Automatically retrained when required

---

# Machine Learning Task

**Problem Type:** Binary Classification

**Target Variable:** `Attrition`

### Classes

- `Stayed`
- `Left`

---

# Production MLOps Architecture

```text
Data Ingestion
      ↓
Data Engineering
      ↓
DVC Data Versioning
      ↓
Feature Engineering
      ↓
Feast Feature Store
      ↓
Model Training
      ↓
Model Evaluation
      ↓
MLflow Model Registry
      ↓
Docker
      ↓
FastAPI
      ↓
Monitoring
      ↓
Data Drift / Performance Detection
      ↓
Automatic Retraining
      ↓
Redeployment
```

---

# End-to-End CI/CD Workflow

The complete pipeline is automated using **GitHub Actions**.

```text
Code Push / Scheduled Run
          ↓
     Run Tests
          ↓
      DVC Data
          ↓
   Data Pipeline
          ↓
 Feature Engineering
          ↓
    Feast Apply
          ↓
    Train Models
          ↓
  Evaluate Models
          ↓
   MLflow Registry
          ↓
   Build Docker Image
          ↓
      Deploy API
          ↓
    Health Check
          ↓
    Data Drift
          ↓
 Model Performance
          ↓
 Automatic / Manual
     Retraining
          ↓
      Redeploy
```

---

# MLOps Phases

| Phase | Component | Status |
|---|---|---|
| Phase 1 | Data Ingestion | ✅ |
| Phase 2 | Data Engineering & Versioning | ✅ |
| Phase 3 | Feature Engineering | ✅ |
| Phase 4 | Feature Store – Feast | ✅ |
| Phase 5 | Model Training | ✅ |
| Phase 6 | Model Evaluation | ✅ |
| Phase 7 | Model Registry – MLflow | ✅ |
| Phase 8 | Deployment – FastAPI + Docker | ✅ |
| Phase 9 | Monitoring | ✅ |
| Phase 10 | CI/CD & Automated Retraining | ✅* |

> *Final status should be considered complete after the GitHub Actions workflow successfully executes the complete pipeline.*

---

# Technologies Used

### Data & Machine Learning

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost

### Feature Store

- Feast

### Experiment Tracking & Model Registry

- MLflow

### API & Deployment

- FastAPI
- Uvicorn
- Docker
- Docker Compose

### Data Versioning

- DVC

### Monitoring

- Evidently AI
- Prometheus
- Grafana
- Custom drift monitoring
- Model performance monitoring
- Prediction logging

### CI/CD

- Git
- GitHub
- GitHub Actions

---

# Project Structure

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
│   ├── champion_model.json
│   ├── metrics.json
│   └── classification_reports.json
│
├── feature_repo/
│   └── feature_repo/
│       ├── feature_store.yaml
│       ├── feature_views.py
│       └── ...
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
├── .dockerignore
├── .dvcignore
├── .gitignore
├── dvc.yaml
├── dvc.lock
├── LICENSE
├── prometheus.yaml
├── requirements.txt
└── README.md
```

---

# Data Pipeline

The preprocessing pipeline performs:

- Loading the raw employee dataset
- Duplicate removal
- Missing value handling
- Categorical value correction
- Timestamp generation
- Saving the cleaned dataset

Output:

```text
data/processed/emp_attrition_cleaned.csv
```

---

# DVC Data Versioning

DVC is used to version the machine learning data pipeline and datasets.

The workflow maintains reproducibility between:

```text
Raw Data
   ↓
Processed Data
   ↓
Feature Data
   ↓
Model
```

The DVC pipeline is defined using:

```text
dvc.yaml
dvc.lock
```

---

# Feature Engineering

The feature engineering pipeline performs:

### Categorical Encoding

One-hot encoding is applied to categorical features.

### Numerical Scaling

Numerical features are standardized using `StandardScaler`.

### Feature Alignment

The generated feature columns are stored and reused during prediction.

Generated artifacts include:

```text
models/scaler.pkl
models/feature_columns.pkl
```

Feature datasets:

```text
data/processed/emp_attrition_features.csv
data/processed/emp_attrition_features.parquet
```

---

# Feature Store – Feast

Feast is used as the feature store for managing and retrieving machine learning features.

The training pipeline retrieves historical features through the Feast feature service:

```text
employee_service
```

The workflow applies the Feast configuration before model training:

```bash
feast apply
```

This allows the training pipeline to retrieve the appropriate historical features for model development.

---

# Model Training

The pipeline trains and compares multiple classification models.

### Models

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

### Hyperparameter Tuning

GridSearchCV is used for hyperparameter optimization for the tree-based models.

Models are evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

The model with the highest F1 score is selected as the champion model.

---

# Model Evaluation

The evaluation pipeline generates model performance information including:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix
- Classification Report

Evaluation artifacts are stored under:

```text
evaluation/
```

The selected champion model is recorded in:

```text
evaluation/champion_model.json
```

---

# MLflow

MLflow is used for experiment tracking and model management.

The pipeline records:

- Model parameters
- Evaluation metrics
- Model artifacts
- Experiment runs
- Best model information

The trained champion model is registered using MLflow Model Registry.

```text
Employee Attrition Prediction
             ↓
        MLflow Runs
             ↓
       Model Registry
             ↓
 employee_attrition_model
```

---

# FastAPI Deployment

The trained model is exposed through a REST API using FastAPI.

### Endpoints

#### GET `/`

Returns API status.

#### GET `/health`

Used for application health monitoring.

#### POST `/predict`

Accepts employee information and returns an attrition prediction.

Example response:

```json
{
  "prediction": "Left",
  "stay_probability": 0.23,
  "leave_probability": 0.77
}
```

---

# Prediction Monitoring

Prediction requests are logged for monitoring purposes.

The prediction log contains:

- Timestamp
- Prediction
- Stay Probability
- Leave Probability

Logs are stored in:

```text
logs/prediction_logs.csv
```

The application automatically creates the `logs` directory when required.

---

# Docker Deployment

The FastAPI application is containerized using Docker.

### Build

```cmd
docker build -t employee-attrition-api .
```

### Run

```cmd
docker run -p 8000:8000 employee-attrition-api
```

The project also uses Docker Compose to manage the application and supporting services.

```cmd
docker compose up -d
```

---

# Monitoring

The project implements multiple monitoring components.

### API Health

The CI/CD pipeline verifies:

```text
http://localhost:8000/health
```

### Data Drift

The drift monitoring pipeline checks for changes in the input data distribution.

Implemented using:

```text
monitoring/drift_check.py
```

### Model Performance

Model performance is periodically checked using:

```text
monitoring/performance_check.py
```

### Prediction Logs

Prediction requests are stored in:

```text
logs/prediction_logs.csv
```

---

# Automated Retraining

The pipeline supports both **automatic and manual model retraining**.

Retraining can be triggered when:

```text
Data Drift Detected
        OR
Model Performance Drops
```

The retraining condition is evaluated by:

```text
monitoring/retrain_trigger.py
```

The retraining process executes:

```text
src/retrain.py
```

After successful retraining, the updated model is redeployed automatically.

```text
Monitoring
    ↓
Condition Detected
    ↓
Retraining
    ↓
Model Evaluation
    ↓
Redeployment
```

---

# CI/CD with GitHub Actions

The complete MLOps pipeline is automated using:

```text
.github/workflows/pipeline.yml
```

The workflow performs:

1. Repository checkout
2. Python environment setup
3. Dependency installation
4. Raw data verification
5. Code compilation checks
6. MLflow startup
7. Data preprocessing
8. Processed-data verification
9. Feature engineering
10. Feature-data verification
11. Feast feature store application
12. Model training
13. Trained-model verification
14. Model evaluation
15. Docker image building
16. FastAPI deployment
17. API health monitoring
18. Data drift monitoring
19. Model performance monitoring
20. Automatic/manual retraining
21. Retrained-model redeployment

The workflow can run through:

- Push to `main`
- Scheduled execution
- Manual workflow dispatch

---

# Manual Retraining

Manual retraining can be triggered from GitHub Actions using the workflow dispatch option.

The workflow supports:

```text
retrain = true
```

This allows the model to be retrained without waiting for an automatic drift or performance trigger.

---

# Local Setup

### Clone the repository

```cmd
git clone <repository-url>
cd employee-attrition-mlops-pipeline
```

### Create environment

```cmd
conda create -n employee-mlops python=3.12
conda activate employee-mlops
```

### Install dependencies

```cmd
pip install -r requirements.txt
```

### Start the API

```cmd
uvicorn src.app:app --reload
```

---

# Swagger API Documentation

After starting FastAPI, open:

```text
http://127.0.0.1:8000/docs
```

This provides the interactive Swagger UI for testing the prediction API.

---

# Running the MLOps Pipeline Locally

### Data preprocessing

```cmd
python src/preprocessing.py
```

### Feature engineering

```cmd
python src/feature_engineering.py
```

### Apply Feast

```cmd
cd feature_repo\feature_repo
feast apply
cd ..\..
```

### Train models

```cmd
python src/train.py
```

### Evaluate models

```cmd
python src/evaluate.py
```

### Start the application

```cmd
docker compose up -d --build
```

---

# Key Features

- End-to-end MLOps workflow
- DVC data versioning
- Automated data preprocessing
- Feature engineering
- Feast Feature Store
- Multiple ML models
- Hyperparameter tuning
- MLflow experiment tracking
- MLflow model registry
- Champion model selection
- FastAPI REST API
- Docker containerization
- Prediction logging
- Data drift monitoring
- Model performance monitoring
- Automatic retraining
- Manual retraining
- Automated redeployment
- GitHub Actions CI/CD
- Prometheus and Grafana monitoring infrastructure

---

# Future Improvements

The current implementation covers the core production-oriented MLOps workflow.

Possible future improvements include:

- Kubernetes deployment
- Cloud deployment
- Online Feast feature serving
- Advanced alerting
- Model explainability
- Authentication and API security
- Advanced observability dashboards
- Automated rollback of failed model deployments

---

# Architecture

```text
                    EMPLOYEE ATTRITION MLOps PIPELINE

 Data
  │
  ▼
┌───────────────┐
│ Data Ingestion│
└───────┬───────┘
        ▼
┌────────────────┐
│ DVC / Data     │
│ Engineering    │
└───────┬────────┘
        ▼
┌──────────────────┐
│ Feature          │
│ Engineering      │
└───────┬──────────┘
        ▼
┌──────────────────┐
│ Feast Feature    │
│ Store            │
└───────┬──────────┘
        ▼
┌──────────────────┐
│ Model Training   │
│ LR / DT / RF /   │
│ XGBoost          │
└───────┬──────────┘
        ▼
┌──────────────────┐
│ Model Evaluation │
└───────┬──────────┘
        ▼
┌──────────────────┐
│ MLflow Model     │
│ Registry         │
└───────┬──────────┘
        ▼
┌──────────────────┐
│ Docker +         │
│ FastAPI          │
└───────┬──────────┘
        ▼
┌──────────────────┐
│ Monitoring       │
│ Drift / Perf /   │
│ Predictions      │
└───────┬──────────┘
        │
        ▼
┌──────────────────┐
│ Retraining       │
│ Trigger          │
└───────┬──────────┘
        │
        └──────────────► Redeploy
```

---

# Project Goal

This project demonstrates how a machine learning model can be transformed into a **complete automated MLOps system**, covering the lifecycle from data preparation to production deployment, monitoring, and automated retraining.
