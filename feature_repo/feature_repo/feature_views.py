from datetime import timedelta
from feast import Entity, FeatureService, FeatureView, Field, FileSource
from feast import ValueType 
from feast.types import Float32, Int64
import os

employee = Entity(
    name="employee",
    join_keys=["Employee ID"],
    value_type=ValueType.INT64,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

employee_source = FileSource(
    name="employee_source",
    path=os.path.join(
        BASE_DIR,"data","processed","emp_attrition_features.parquet",
    ),
    event_timestamp_column="event_timestamp",
)

employee_feature_view = FeatureView(
    name="employee_features",
    entities=[employee],
    ttl=timedelta(days=365),
    schema=[

        # Numerical Features
        Field(name="Age", dtype=Float32),
        Field(name="Years at Company", dtype=Float32),
        Field(name="Monthly Income", dtype=Float32),
        Field(name="Number of Promotions", dtype=Float32),
        Field(name="Distance from Home", dtype=Float32),
        Field(name="Number of Dependents", dtype=Float32),
        Field(name="Company Tenure (In Months)", dtype=Float32),

        # Gender
        Field(name="Gender_Female", dtype=Int64),
        Field(name="Gender_Male", dtype=Int64),

        # Job Role
        Field(name="Job Role_Education", dtype=Int64),
        Field(name="Job Role_Finance", dtype=Int64),
        Field(name="Job Role_Healthcare", dtype=Int64),
        Field(name="Job Role_Media", dtype=Int64),
        Field(name="Job Role_Technology", dtype=Int64),

        # Work-Life Balance
        Field(name="Work-Life Balance_Excellent", dtype=Int64),
        Field(name="Work-Life Balance_Fair", dtype=Int64),
        Field(name="Work-Life Balance_Good", dtype=Int64),
        Field(name="Work-Life Balance_Poor", dtype=Int64),

        # Job Satisfaction
        Field(name="Job Satisfaction_High", dtype=Int64),
        Field(name="Job Satisfaction_Low", dtype=Int64),
        Field(name="Job Satisfaction_Medium", dtype=Int64),
        Field(name="Job Satisfaction_Very High", dtype=Int64),

        # Performance Rating
        Field(name="Performance Rating_Average", dtype=Int64),
        Field(name="Performance Rating_Below Average", dtype=Int64),
        Field(name="Performance Rating_High", dtype=Int64),
        Field(name="Performance Rating_Low", dtype=Int64),

        # Overtime
        Field(name="Overtime_No", dtype=Int64),
        Field(name="Overtime_Yes", dtype=Int64),

        # Education
        Field(name="Education Level_Associate Degree", dtype=Int64),
        Field(name="Education Level_Bachelor's Degree", dtype=Int64),
        Field(name="Education Level_High School", dtype=Int64),
        Field(name="Education Level_Master's Degree", dtype=Int64),
        Field(name="Education Level_PhD", dtype=Int64),

        # Marital Status
        Field(name="Marital Status_Divorced", dtype=Int64),
        Field(name="Marital Status_Married", dtype=Int64),
        Field(name="Marital Status_Single", dtype=Int64),

        # Job Level
        Field(name="Job Level_Entry", dtype=Int64),
        Field(name="Job Level_Mid", dtype=Int64),
        Field(name="Job Level_Senior", dtype=Int64),

        # Company Size
        Field(name="Company Size_Large", dtype=Int64),
        Field(name="Company Size_Medium", dtype=Int64),
        Field(name="Company Size_Small", dtype=Int64),

        # Remote Work
        Field(name="Remote Work_No", dtype=Int64),
        Field(name="Remote Work_Yes", dtype=Int64),

        # Leadership
        Field(name="Leadership Opportunities_No", dtype=Int64),
        Field(name="Leadership Opportunities_Yes", dtype=Int64),

        # Innovation
        Field(name="Innovation Opportunities_No", dtype=Int64),
        Field(name="Innovation Opportunities_Yes", dtype=Int64),

        # Company Reputation
        Field(name="Company Reputation_Excellent", dtype=Int64),
        Field(name="Company Reputation_Fair", dtype=Int64),
        Field(name="Company Reputation_Good", dtype=Int64),
        Field(name="Company Reputation_Poor", dtype=Int64),

        # Employee Recognition
        Field(name="Employee Recognition_High", dtype=Int64),
        Field(name="Employee Recognition_Low", dtype=Int64),
        Field(name="Employee Recognition_Medium", dtype=Int64),
        Field(name="Employee Recognition_Very High", dtype=Int64),
    ],
    source=employee_source,
)

employee_service = FeatureService(
    name="employee_service",
    features=[employee_feature_view],
)