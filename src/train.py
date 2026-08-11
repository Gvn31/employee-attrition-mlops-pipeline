import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from feast import FeatureStore

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
import os


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

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
    MODEL_DIR = os.path.join(BASE_DIR, "models")
    MLRUNS_DIR = os.path.join(BASE_DIR, "mlruns")
    os.makedirs(MODEL_DIR, exist_ok=True)

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Employee Attrition Prediction")
    registered_model_name="employee_attrition_model"
    model_results={}

    print("Loading Feature Store...")

    store = FeatureStore(
        repo_path=os.path.join(BASE_DIR, "feature_repo", "feature_repo")
    )

    print("Loading entity dataframe...")

    entity_df = pd.read_parquet(
            os.path.join(DATA_DIR, "emp_attrition_features.parquet"))

    entity_df["event_timestamp"] = pd.to_datetime(
    entity_df["event_timestamp"],
    utc=True)


    training_df = store.get_historical_features(entity_df=entity_df[
                [
                    "Employee ID",
                    "event_timestamp", 
                ]
            ],features=store.get_feature_service("employee_service"),
        ).to_df()



    labels = entity_df[
        [
            "Employee ID",
            "event_timestamp",
            "Attrition"
        ]
    ]

    training_df=training_df.merge(
        labels, on=["Employee ID","event_timestamp"],
          how="left"
    )
    # print(training_df.columns.tolist())
    # print(training_df["Attrition"].isnull().sum())
    print("Splitting datasets...")

    x = training_df.drop(
            columns=[
                "Employee ID",
                "event_timestamp",
                "Attrition"])

    y = training_df["Attrition"]
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42,stratify=y,)

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

        model_results["Logistic Regression"] = {
            'model': logistic_model,
            'f1' : f1_score(y_test, y_pred),
            
        }
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

        model_results["Decision Tree"] = {
            'model': best_dt_model,
            'f1' : f1_score(y_test, y_pred),
        }
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

        model_results["Random Forest"] = {
            'model': best_rf_model,
            'f1' : f1_score(y_test, y_pred),
        }

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

        model_results["XGBoost"] = {
            'model': best_xgb_model,
            'f1' : f1_score(y_test, y_pred),
        }

        mlflow.log_param("model","XGBoost")

        print("XGBoost model trained successfully!")

#Model Registry
    best_model_name = max(
        model_results,
        key=lambda x: model_results[x]["f1"]
    )

    best_model = model_results[best_model_name]["model"]

    print(f"\nBest Model: {best_model_name}")
    print(f"F1 Score : {model_results[best_model_name]['f1']:.4f}")

    with mlflow.start_run(run_name="Best Model Registry"):

        mlflow.log_param("best_model", best_model_name)
        mlflow.log_metric(
            "best_f1_score",
            model_results[best_model_name]["f1"]
        )

        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="best_model",
            registered_model_name=registered_model_name,
        )

    print(f"{best_model_name} registered successfully!"
          f"as {registered_model_name}")

    joblib.dump(
    logistic_model,os.path.join(MODEL_DIR, "logistic_regression.pkl"))

    joblib.dump(
        best_dt_model,os.path.join(MODEL_DIR, "decision_tree.pkl"))

    joblib.dump(
    best_rf_model,os.path.join(MODEL_DIR, "random_forest.pkl"))

    joblib.dump(
    best_xgb_model,os.path.join(MODEL_DIR, "xgboost.pkl"))

    print("Training completed successfully!")


if __name__ == "__main__":
    try:
        train_model()

    except Exception as e:
        print(f"Error: {e}")
        raise