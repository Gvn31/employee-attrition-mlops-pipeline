import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from xgboost import XGBClassifier
import mlflow
import mlflow.sklearn
import joblib

def train_model():

    '''
        Train multiple machine learning models, log experiments
    using MLflow, and save the trained models.

    Models:
    - Logistic Regression
    - Decision Tree
    - Random Forest
    - XGBoost

    Returns:
        None
    
    '''

    mlflow.set_tracking_uri("../mlruns")
    mlflow.set_experiment("Employee Attrition Prediction")

    # Load dataset
    print("Loading dataset...")
    df=pd.read_csv("../data/processed/emp_attrition_features.csv")

    # Split dataset
    print("Splitting Datasets........")
    x=df.drop(columns=["Attrition"])
    y=df["Attrition"]

    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)

#Logistic Regression
    with mlflow.start_run(run_name="Logistic Regression"):

        logistic_model = LogisticRegression(random_state=42)    
        logistic_model.fit(x_train, y_train)

        y_pred = logistic_model.predict(x_test)
        y_prob = logistic_model.predict_proba(x_test)[:, 1]

        mlflow.log_param("model", "Logistic Regression")
        mlflow.log_param("random_state", 42)

        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("precision", precision_score(y_test, y_pred))
        mlflow.log_metric("recall", recall_score(y_test, y_pred))
        mlflow.log_metric("f1_score", f1_score(y_test, y_pred))
        mlflow.log_metric("roc_auc", roc_auc_score(y_test, y_prob))

        mlflow.sklearn.log_model(logistic_model, "model")
        mlflow.log_param("model","Logistic Regression")
        print("Logistic Regression model trained successfully!")

#Decision Tree
    print("Training Decision Tree model...")
    decision_tree = DecisionTreeClassifier(random_state=42)

    dt_param_grid = {
        "criterion": ["gini","entropy"],
        "max_depth": [5,10,20,None],
        "min_samples_split": [2,5,10],
        "min_samples_leaf": [1,2,4]
    }

    dt_grid_search = GridSearchCV(
        estimator=decision_tree,
        param_grid=dt_param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1
    )

    dt_grid_search.fit(x_train, y_train)
    best_dt_model = dt_grid_search.best_estimator_

    with mlflow.start_run(run_name="Decision Tree"):

        y_pred = best_dt_model.predict(x_test)
        y_prob = best_dt_model.predict_proba(x_test)[:, 1]

        mlflow.log_params(dt_grid_search.best_params_)

        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("precision", precision_score(y_test, y_pred))
        mlflow.log_metric("recall", recall_score(y_test, y_pred))
        mlflow.log_metric("f1_score", f1_score(y_test, y_pred))
        mlflow.log_metric("roc_auc", roc_auc_score(y_test, y_prob))

        mlflow.sklearn.log_model(best_dt_model, "model")
        mlflow.log_param("model","Decision Tree")

        print("Decision Tree model trained successfully!")
    
#Random Forest
    print("Training Random Forest model...")

    random_forest = RandomForestClassifier(random_state=42)

    rf_param_grid = {
        "n_estimators": [100,200,300],
        "max_depth": [10,20,None],
        "min_samples_split": [2,5],
        "min_samples_leaf": [1,2,4]
    }

    rf_grid_search = GridSearchCV(
        estimator=random_forest,
        param_grid=rf_param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1
    )
    rf_grid_search.fit(x_train, y_train)
    best_rf_model = rf_grid_search.best_estimator_
    with mlflow.start_run(run_name="Random Forest"):
        y_pred = best_rf_model.predict(x_test)
        y_prob = best_rf_model.predict_proba(x_test)[:, 1]

        mlflow.log_params(rf_grid_search.best_params_)

        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("precision", precision_score(y_test, y_pred))
        mlflow.log_metric("recall", recall_score(y_test, y_pred))
        mlflow.log_metric("f1_score", f1_score(y_test, y_pred))
        mlflow.log_metric("roc_auc", roc_auc_score(y_test, y_prob))

        mlflow.sklearn.log_model(best_rf_model, "model")
        mlflow.log_param("model","Random Forest")

        print("Random Forest model trained successfully!")

#XGBoost

    print("Training XGBoost model...")
    xgb_model = XGBClassifier(random_state=42,eval_metric="logloss")
    xgb_param_grid = {
        "n_estimators": [100,200,300],
        "max_depth": [3,5,7],
        "learning_rate": [0.01,0.05,0.1],
        "subsample": [0.8,1.0],
        "colsample_bytree": [0.8,1.0],
        "gamma": [0,0.1,0.3],
        "min_child_weight": [1,3]
    }

    xgb_grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=xgb_param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1
    )
    xgb_grid_search.fit(x_train, y_train)
    best_xgb_model = xgb_grid_search.best_estimator_

    with mlflow.start_run(run_name="XGBoost"):

        y_pred = best_xgb_model.predict(x_test)
        y_prob = best_xgb_model.predict_proba(x_test)[:, 1]

        mlflow.log_params(xgb_grid_search.best_params_)

        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("precision", precision_score(y_test, y_pred))
        mlflow.log_metric("recall", recall_score(y_test, y_pred))
        mlflow.log_metric("f1_score", f1_score(y_test, y_pred))
        mlflow.log_metric("roc_auc", roc_auc_score(y_test, y_prob))

        mlflow.sklearn.log_model(best_xgb_model, "model")
        mlflow.log_param("model","XGBoost")

        print("XGBoost model trained successfully!")

    joblib.dump(logistic_model,"../models/logistic_regression.pkl")
    joblib.dump(best_dt_model,"../models/decision_tree.pkl")
    joblib.dump(best_rf_model,"../models/random_forest.pkl")
    joblib.dump(best_xgb_model,"../models/xgboost.pkl")

    print("Training completed successfully!")


if __name__ == "__main__":
    try:
        train_model()

    except Exception as e:
        print(f"Error: {e}")