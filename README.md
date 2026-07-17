# Employee Attrition MLOps Pipeline

## Production-Grade Employee Attrition Prediction using MLOps

An end-to-end **Production-Grade MLOps Pipeline** that predicts whether an employee is likely to leave an organization based on demographic and workplace-related information.

This project demonstrates:
- Data Engineering
- Feature Engineering
- Model Training
- Model Evaluation
- MLflow Experiment Tracking
- FastAPI Deployment
- Docker Containerization
- Prediction Monitoring
- GitHub Actions CI/CD

---

# Project Objective

Employee attrition is a major challenge for organizations because it increases hiring costs, reduces productivity, and causes knowledge loss.

The objective of this project is to build a Machine Learning model capable of predicting employee attrition so that HR teams can identify at-risk employees and take proactive retention measures.

---

# Machine Learning Task

**Problem Type:** Binary Classification

**Target Variable:** Attrition

**Classes**
- Stayed
- Left

---

# Project Workflow

```text
Repository Setup
        ↓
Problem Definition
        ↓
Exploratory Data Analysis
        ↓
Data Preprocessing
        ↓
Feature Engineering
        ↓
Model Training
        ↓
Model Evaluation
        ↓
MLflow Experiment Tracking
        ↓
FastAPI Deployment
        ↓
Docker Containerization
        ↓
Prediction Monitoring
        ↓
GitHub Actions CI/CD
```

---

# Production MLOps Workflow

| Phase | Status |
|--------|--------|
| Phase 1 – Data Ingestion | ✅ |
| Phase 2 – Data Engineering | ✅ |
| Phase 3 – Feature Engineering | ✅ |
| Phase 4 – Model Training | ✅ |
| Phase 5 – Model Evaluation | ✅ |
| Phase 6 – Model Registry (MLflow) | ✅ |
| Phase 7 – Deployment (FastAPI + Docker) | ✅ |
| Phase 8 – Monitoring | ✅ |
| Phase 9 – CI/CD Automation | ✅ |

---

# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- MLflow
- FastAPI
- Uvicorn
- Docker
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
│       └── emp_attrition_feature_engineered.csv
│
├── logs/
│   └── prediction_logs.csv
│
├── mlruns/
│
├── models/
│   ├── decision_tree.pkl
│   ├── feature_columns.pkl
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── scaler.pkl
│   └── xgboost.pkl
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
│   └── train.py
│
├── Dockerfile
├── .dockerignore
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

# Machine Learning Pipeline

## Data Engineering
- Data Cleaning
- Missing Value Handling
- Train/Test Split
- Data Validation

## Feature Engineering
- One-Hot Encoding
- Feature Scaling
- Feature Alignment

## Models Trained
- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

## Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix

## Best Model

**XGBoost Classifier**

---

# MLflow

- Experiment Tracking
- Parameter Logging
- Metric Logging
- Model Artifact Storage
- Model Comparison

---

# FastAPI

**GET /** - API status

**POST /predict** - Predict employee attrition

---

# Docker

### Build the Docker image

```cmd
docker build -t employee-attrition-api .
```

### Run the Docker container

```cmd
docker run -p 8000:8000 employee-attrition-api
```
---

# Monitoring

Prediction logs include:
- Timestamp
- Prediction
- Stay Probability
- Leave Probability

Stored in:

```text
logs/prediction_logs.csv
```

---

# CI/CD

GitHub Actions automates:
- Repository Checkout
- Python Setup
- Dependency Installation
- Library Verification
- Docker Image Build

### Workflow:

```text
.github/workflows/pipeline.yml
```

---

# Installation

Open **Command Prompt (CMD)** and run:

```cmd
git clone <repository-url>
cd employee-attrition-mlops-pipeline
conda create -n venv python=3.12
conda activate venv/
pip install -r requirements.txt
uvicorn src.app:app --reload
```

## Swagger:
After starting the FastAPI server, open:

```text
http://127.0.0.1:8000/docs

```

---

# Features

- End-to-End Production-Grade MLOps Pipeline
- MLflow Experiment Tracking
- FastAPI REST API
- Docker Deployment
- Prediction Monitoring
- GitHub Actions CI/CD

---

# Future Improvements

- Feature Store (Feast)
- Data Drift Detection
- Prometheus & Grafana
- Kubernetes Deployment
- Automated Model Retraining

---

