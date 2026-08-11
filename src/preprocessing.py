import os
import numpy as np
import pandas as pd
from datetime import datetime

# Base project directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DATA = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "emp_attrition_csv.csv"
)

PROCESSED_DATA = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "emp_attrition_cleaned.csv"
)


def preprocess_data():
    """
    Clean the raw employee attrition dataset.

    This function performs the following preprocessing steps:
    - Loads the raw dataset.
    - Removes duplicate records.
    - Handles missing values using the median.
    - Corrects spelling and formatting issues in categorical values.
    - Removes irrelevant features.
    - Saves the cleaned dataset to the processed folder.

    Returns:
        None
    """

    print("Loading Dataset...")
    df = pd.read_csv(RAW_DATA)

    print("Cleaning Dataset...")

    # Drop Duplicates
    df.drop_duplicates(inplace=True)

    # Handle Missing Values
    df["Distance from Home"] = df["Distance from Home"].fillna(df["Distance from Home"].median())

    df["Company Tenure (In Months)"] = df["Company Tenure (In Months)"].fillna(df["Company Tenure (In Months)"].median())

    # Spelling Correction
    df["Education Level"] = df["Education Level"].str.strip()
    df["Education Level"] = df["Education Level"].str.replace("Masterâ€™s Degree","Master's Degree")
    df["Education Level"] = df["Education Level"].str.replace("Bachelorâ€™s Degree","Bachelor's Degree")

    # Keep Employee_ID for Feast Entity
    # df.drop(["Employee ID"], axis=1, inplace=True)

    df["event_timestamp"] = pd.Timestamp.now()

    #Create processed directory if it doesn't exist
    os.makedirs(os.path.dirname(PROCESSED_DATA), exist_ok=True)
    # Save Cleaned Dataset
    df.to_csv(PROCESSED_DATA, index=False)
   

    print("Dataset saved successfully!")


if __name__ == "__main__":
    try:
        preprocess_data()

    except Exception as e:
        print(f"Error: {e}")
        raise