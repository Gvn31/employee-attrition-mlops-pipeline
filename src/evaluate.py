import pandas as pd
import joblib
import os
import json
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

from feast import FeatureStore

# Base project directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")
EVALUATION_DIR=os.path.join(BASE_DIR, "evaluation")
CONFUSION_MATRIX_DIR=os.path.join(EVALUATION_DIR, "confusion_matrices")
ROC_CURVE_DIR=os.path.join(EVALUATION_DIR, "roc_curves")

os.makedirs(EVALUATION_DIR, exist_ok=True)
os.makedirs(CONFUSION_MATRIX_DIR, exist_ok=True)
os.makedirs(ROC_CURVE_DIR, exist_ok=True)

def evaluate_model():
    """
    Evaluate all trained classification models.

    Returns:
        None
    """

    # Load dataset
    print("Loading Feature Engineered Dataset...")
    print("Loading Feature Store...")

    store = FeatureStore(
        repo_path=os.path.join(BASE_DIR, "feature_repo", "feature_repo")
    )

    print("Loading entity dataframe...")

    entity_df = pd.read_parquet(
        os.path.join(DATA_DIR, "emp_attrition_features.parquet")
    )

    entity_df["event_timestamp"] = pd.to_datetime(
    entity_df["event_timestamp"],
    utc=True)

    evaluation_df = store.get_historical_features(
        entity_df=entity_df[
            [
                "Employee ID",
                "event_timestamp",
                
            ]
        ],
        features=store.get_feature_service("employee_service")
    ).to_df()

    labels = entity_df[
        [
        "Employee ID",
        "event_timestamp",
        "Attrition"
        ]
    ]

    evaluation_df= evaluation_df.merge(
        labels,on=["Employee ID","event_timestamp"],
        how="left"
    )

   
    # Split dataset
    print("Splitting Dataset...")

    x = evaluation_df.drop(
        columns=[
            "Employee ID",
            "event_timestamp",
            "Attrition"
        ]
    )

    y = evaluation_df["Attrition"]

    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)

    # Load models
    print("Loading Trained Models...")

    models = {
            "Logistic Regression": joblib.load(os.path.join(MODEL_DIR, "logistic_regression.pkl")),
            "Decision Tree": joblib.load(os.path.join(MODEL_DIR, "decision_tree.pkl") ),
            "Random Forest": joblib.load(os.path.join(MODEL_DIR, "random_forest.pkl")),
            "XGBoost": joblib.load(os.path.join(MODEL_DIR, "xgboost.pkl"))
        }

    best_model = None
    best_f1 = 0
    all_metrics = {}
    all_reports = {}

    for model_name, model in models.items():

        print(f"\nEvaluating {model_name}...")

        y_pred = model.predict(x_test)
        y_prob = model.predict_proba(x_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        metrics={
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "roc_auc": float(roc_auc)
        }
        all_metrics[model_name] = metrics
        all_reports[model_name] = classification_report(y_test, y_pred,)

        print(f"Accuracy  : {accuracy:.4f}")
        print(f"Precision : {precision:.4f}")
        print(f"Recall    : {recall:.4f}")
        print(f"F1 Score  : {f1:.4f}")
        print(f"ROC-AUC   : {roc_auc:.4f}")

        print("\nConfusion Matrix")
        print(confusion_matrix(y_test, y_pred))

        print("\nClassification Report")
        print(classification_report(y_test, y_pred))

        #Save Confusion Matrix
        ConfusionMatrixDisplay.from_predictions(y_test,y_pred)
        plt.savefig(os.path.join(
            CONFUSION_MATRIX_DIR,
                f"{model_name.lower().replace(" ","_")}.png"
                ))
        plt.clf()
        plt.close()

        #Save ROC Curve
        RocCurveDisplay.from_predictions(y_test, y_prob)
        plt.savefig(os.path.join(
            ROC_CURVE_DIR, 
                f"{model_name.lower().replace(" ","_")}.png"
                ))
        plt.clf()
        plt.close()
        

        if f1 > best_f1:
                best_f1 = f1
                best_model = model_name

    #Save Metrics JSON
    with open(os.path.join(EVALUATION_DIR,"metrics.json"
            ),"w"
    ) as f:
        json.dump(all_metrics, f, indent=4)
        
    #Save Classification Report
    with open(os.path.join(EVALUATION_DIR,"classification_reports.json"), 
                       "w"
    )as f:
        json.dump(all_reports,f,indent=4)


   

    print("\nChampion Model")
    print(f"Best Model : {best_model}")
    print(f"Best F1 Score : {best_f1:.4f}")

    champion={
        "best_model": best_model,
        "best_f1_score": float(best_f1)
        
    }
    with open(
        os.path.join(
        EVALUATION_DIR,
        "champion_model.json"),"w"
    )as f:
        json.dump(champion,f,indent=4)

    print("Evaluation Artifacts saved successfully!")
    print(f"Location: {EVALUATION_DIR}")

if __name__ == "__main__":
    try:
        evaluate_model()

    except Exception as e:
        print(f"Error: {e}")