import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,GridSearchCV

def train_model():

    """
    Train Logistic Regression and Random Forest models
    using the feature-engineered employee attrition dataset.

    Returns:
        None
    """

    # Load the preprocessed data
    df = pd.read_csv("../data/processed/emp_attrition_features.csv")

    # Split the data into training and testing sets
    print("Splitting dataset...")

    x = df.drop(columns=['Attrition'])
    y = df['Attrition']
    
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)

    

    #Create and train the model
    #Logistic Regression
    print("Training Logistic Regression Model...")
    model1=LogisticRegression(random_state=42)
    model1.fit(x_train,y_train)

    print("Training Complete!")

    #Random Forest
    print("Training Random Forest Model...")

    model2=RandomForestClassifier(random_state=42)
    param_grid = {
        "n_estimators": [100,200,300,500],
        "max_depth": [10,20,30,None],
        "min_samples_split": [2,5,10],
        "min_samples_leaf": [1,2,4]
    }

    # Grid Search
    grid_search = GridSearchCV(
        estimator=model2,
        param_grid=param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1
    )
    grid_search.fit(x_train, y_train)

    best_rf_model = grid_search.best_estimator_
    print("Best Parameters:", grid_search.best_params_)



    print("Training Complete!")

    #Save the trained models
    print("Saving Trained Models")
    joblib.dump(
        model1,"../models/logistic_regression.pkl"
    )

    joblib.dump(
        best_rf_model,"../models/random_forest.pkl"
    )
    print("Models saved successfully!")


if __name__ == "__main__":
    try:
        train_model()
    except Exception as e:
        print(f"Error: {e}")