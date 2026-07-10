import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


def evaluate_model():
    """
    Evaluate all trained classification models.

    Returns:
        None
    """

    # Load dataset
    print("Loading Feature Engineered Dataset...")
    df = pd.read_csv("../data/processed/emp_attrition_features.csv")

    # Split dataset
    print("Splitting Dataset...")

    x = df.drop(columns=["Attrition"])
    y = df["Attrition"]

    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)

    # Load models
    print("Loading Trained Models...")

    models = {
        "Logistic Regression": joblib.load("../models/logistic_regression.pkl"),
        "Decision Tree": joblib.load("../models/decision_tree.pkl"),
        "Random Forest": joblib.load("../models/random_forest.pkl"),
        "XGBoost": joblib.load("../models/xgboost.pkl")
    }

    best_model = None
    best_f1 = 0

    for model_name, model in models.items():

        print(f"\nEvaluating {model_name}...")

        y_pred = model.predict(x_test)
        y_prob = model.predict_proba(x_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)

        print(f"Accuracy  : {accuracy:.4f}")
        print(f"Precision : {precision:.4f}")
        print(f"Recall    : {recall:.4f}")
        print(f"F1 Score  : {f1:.4f}")
        print(f"ROC-AUC   : {roc_auc:.4f}")

        print("\nConfusion Matrix")
        print(confusion_matrix(y_test, y_pred))

        print("\nClassification Report")
        print(classification_report(y_test, y_pred))

        if f1 > best_f1:
            best_f1 = f1
            best_model = model_name

    print("\nChampion Model")
    print(f"Best Model : {best_model}")
    print(f"Best F1 Score : {best_f1:.4f}")


if __name__ == "__main__":
    try:
        evaluate_model()

    except Exception as e:
        print(f"Error: {e}")