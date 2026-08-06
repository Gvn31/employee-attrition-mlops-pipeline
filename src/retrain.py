from src.feature_engineering import feature_engineering
from src.train import train_model
from src.evaluate import evaluate_model


def retrain_model():
    """"
    Complete Retraining Pipeline.

    Steps:
    1. Feature Engineering
    2. Model Training
    3. Model Evaluation
    """

    print("*"*60)
    print("Retraining Employee Attrition Model...")
    print("*"*60)

    print("Step 1: Feature Engineering")
    feature_engineering()

    print("\n Step 2: Model Training")
    train_model()

    print("\n Step 3: Model Evaluation")
    evaluate_model()

    print("\n Retraining Pipeline Completed")


if __name__ == "__main__":

    try:
        retrain_model()

    except Exception as e:
        print(f"Error: {e}")