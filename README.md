# Employee Attrition MLOps Pipeline

Production-Grade End-to-End MLOps Pipeline for Employee Attrition Prediction using DVC, Feast, MLflow, FastAPI, Docker, AWS (S3, ECR, EC2), GitHub Actions, Prometheus and Grafana.

An end-to-end MLOps pipeline for Employee Attrition Prediction that covers data preparation, feature engineering, model training, evaluation, experiment tracking, deployment, monitoring, automated retraining, and cloud deployment.

The project uses DVC, Feast, MLflow, FastAPI, Docker, AWS S3, AWS ECR, AWS EC2, GitHub Actions, Prometheus, and Grafana.

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
DVC + AWS S3
   ↓
Data Preprocessing
   ↓
Feature Engineering
   ↓
Feast Feature Store
   ↓
Model Training
   ↓
Model Evaluation
   ↓
MLflow Tracking
   ↓
Docker Image Build
   ↓
Amazon ECR
   ↓
Amazon EC2 Deployment
   ↓
FastAPI Service
   ↓
Prometheus + Grafana
   ↓
Data Drift Monitoring
   ↓
Performance Monitoring
   ↓
Automatic Retraining
   ↓
Redeployment
```
---

## CI/CD Pipeline

GitHub Actions automates the complete MLOps workflow.

```text
Code / Data Change
        ↓
GitHub Actions
        ↓
DVC Pull From AWS S3
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
MLflow Tracking
        ↓
Docker Build
        ↓
Push Image To Amazon ECR
        ↓
Deploy To Amazon EC2
        ↓
API Health Check
        ↓
Data Drift Monitoring
        ↓
Performance Monitoring
        ↓
Retraining (If Required)
        ↓
Redeployment
```

---

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- DVC
- Feast
- MLflow
- FastAPI
- Docker
- Docker Compose
- GitHub Actions
- Prometheus
- Grafana
- AWS S3
- AWS ECR
- AWS EC2
- AWS IAM


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

## DVC and AWS S3

DVC is used for dataset versioning while Amazon S3 acts as the remote storage backend.

```text
Raw Data
   ↓
DVC Tracking
   ↓
AWS S3 Storage
   ↓
GitHub Actions dvc pull
   ↓
Pipeline Execution

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


## AWS Deployment

The application is deployed on AWS using:

- Amazon S3 for DVC remote storage
- Amazon ECR for Docker image registry
- Amazon EC2 for application hosting
- IAM for secure access control

Deployment Flow:

```text
GitHub Actions
      ↓
Build Docker Image
      ↓
Push To Amazon ECR
      ↓
SSH To EC2
      ↓
Pull Latest Image
      ↓
Docker Compose Restart
      ↓
Health Check

```

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
│   │   └── emp_attrition_csv.csv.dvc
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
├── README.md
│
└── AWS Infrastructure
    ├── Amazon S3 (DVC Remote Storage)
    ├── Amazon ECR (Docker Image Registry)
    ├── Amazon EC2 (Application Hosting)
    └── IAM (Access Management)

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
http://<EC2-PUBLIC-IP>:8000/docs
```

---

## Project Highlights


- End-to-End MLOps Pipeline
- Binary Classification Problem
- Data Versioning with DVC
- Remote Storage using AWS S3
- Feature Store using Feast
- Multiple ML Models
- Hyperparameter Tuning with GridSearchCV
- Champion Model Selection
- MLflow Experiment Tracking
- MLflow Model Registry
- FastAPI REST API
- Docker Containerization
- Amazon ECR Image Registry
- Amazon EC2 Deployment
- Prediction Logging
- Data Drift Monitoring
- Model Performance Monitoring
- Automatic Retraining
- GitHub Actions CI/CD
- Prometheus Monitoring
- Grafana Dashboards

---

## Future Improvements

- Kubernetes (EKS)
- Terraform Infrastructure as Code
- Blue-Green Deployment
- Automated Rollback
- Advanced Model Monitoring
- SHAP-based Explainability
- API Authentication & Authorization
- Multi-Environment Deployment (Dev / Staging / Production)