import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV


def train_model():
    """
    Train multiple classification models using the
    feature-engineered employee attrition dataset.

    Models:
    - Logistic Regression
    - Decision Tree
    - Random Forest
    - XGBoost

    Returns:
        None
    """

    # Load feature engineered dataset
    print("Loading dataset...")
    df = pd.read_csv("../data/processed/emp_attrition_features.csv")

    # Split dataset
    print("Splitting dataset...")

    x = df.drop(columns=["Attrition"])
    y = df["Attrition"]

    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)

    # Logistic Regression
    print("Training Logistic Regression Model...")

    logistic_model = LogisticRegression(random_state=42)
    logistic_model.fit(x_train, y_train)

    print("Logistic Regression Training Complete!")

    # Decision Tree
    print("Training Decision Tree Model...")

    decision_tree = DecisionTreeClassifier(random_state=42)

    dt_param_grid = {
        "criterion": ["gini", "entropy"],
        "max_depth": [5, 10, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
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

    print("Decision Tree Best Parameters:")
    print(dt_grid_search.best_params_)

    print("Decision Tree Training Complete!")

    # Random Forest

    print("Training Random Forest Model...")

    random_forest = RandomForestClassifier(random_state=42)

    rf_param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2, 4]
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

    print("Random Forest Best Parameters:")
    print(rf_grid_search.best_params_)

    print("Random Forest Training Complete!")

    # XGBoost

    print("Training XGBoost Model...")

    xgb_model = XGBClassifier(
        random_state=42,
        eval_metric="logloss"
    )

    xgb_param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "gamma": [0, 0.1, 0.3],
        "min_child_weight": [1, 3]
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

    print("XGBoost Best Parameters:")
    print(xgb_grid_search.best_params_)

    print("XGBoost Training Complete!")

    # Save Models
    print("Saving Trained Models...")

    joblib.dump(logistic_model, "../models/logistic_regression.pkl")
    joblib.dump(best_dt_model, "../models/decision_tree.pkl")
    joblib.dump(best_rf_model, "../models/random_forest.pkl")
    joblib.dump(best_xgb_model, "../models/xgboost.pkl")

    print("All models saved successfully!")


if __name__ == "__main__":
    try:
        train_model()

    except Exception as e:
        print(f"Error: {e}")