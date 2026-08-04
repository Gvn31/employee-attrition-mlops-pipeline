import os
import pandas as pd

from evidently import Report
from evidently.presets import DataDriftPreset
from evidently.ui.workspace import Workspace


def monitor():
    #Base project directory
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    #Reference dataset(tarining data)
    REFERENCE_DATA=os.path.join(BASE_DIR, "data", "processed", "emp_attrition_features.parquet")


    #Current production data
    CURRENT_DATA=os.path.join(BASE_DIR,"logs","prediction_logs.csv")

    #Reports Directory
    REPORT_DIR=os.path.join(BASE_DIR, "monitoring", "reports")

    WORKSPACE_PATH=os.path.join(BASE_DIR, "evidently_workspace")
    PROJECT_ID="019fcb74-03a7-765c-a08b-4ceeb0989498"

    os.makedirs(REPORT_DIR, exist_ok=True)


    print("Loading reference dataset.........")
    reference_data=pd.read_parquet(REFERENCE_DATA)

    if not os.path.exists(CURRENT_DATA):
        raise FileNotFoundError(
            "Prediction logs not found. Run the FastAPI app first.")
    
    print("Loading current production dataset.........")
    current_data=pd.read_csv(CURRENT_DATA)



    #Remove monitoring-specific columns
    current_data=current_data.drop(
        columns=["Timestamp","Prediction","Stay_Probability","Leave_Probability"],
        errors="ignore"
    )
    if current_data.empty:
        raise ValueError(
            "Prediction logs not found. Generate predictions first.")


    common_columns=reference_data.columns.intersection(current_data.columns)
    reference_data=reference_data[common_columns]
    current_data=current_data[common_columns]

    print("Generating data drift report.........")

    drift_report=Report(
        metrics=[DataDriftPreset()]
    )

    snapshot = drift_report.run(
    reference_data=reference_data,
    current_data=current_data
    )

    print("Uploading report to Evidently workspace.........")
    workspace=Workspace(WORKSPACE_PATH)
    workspace.add_run(
        PROJECT_ID,snapshot,name="Employee Attrition Drift Report"
    )
    print("Report Uploaded Successfully!")

    report_path = os.path.join(REPORT_DIR, "data_drift_report.html")
    snapshot.save_html(report_path) 

    print(f"Report saved at: {report_path}")

if __name__=="__main__":
    try:
        monitor()

    except Exception as e:
        print(f"Error: {e}")