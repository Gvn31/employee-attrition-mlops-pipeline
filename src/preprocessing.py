import numpy as np
import pandas as pd
try:
    
     #Load Dataset
    df=pd.read_csv(r"../data/raw/emp_attrition_csv.csv")

    #Drop Duplicates
    df.drop_duplicates(inplace=True)

    #Handle Missing Values
    df['Distance from Home']=df['Distance from Home'].fillna(df['Distance from Home'].median())
    df['Company Tenure (In Months)']=df['Company Tenure (In Months)'].fillna(df['Company Tenure (In Months)'].median())

     #Spelling Correction
    df['Education Level']=df['Education Level'].str.strip()
    df['Education Level']=df['Education Level'].str.replace("Masterâ€™s Degree","Master's Degree")
    df['Education Level']=df['Education Level'].str.replace("Bachelorâ€™s Degree","Bachelor's Degree")

    #Drop Irrelevant Features
    df.drop(['Employee ID'],axis=1,inplace=True)


    #Save Cleaned Dataset
    df.to_csv(r"../data/processed/emp_attrition_cleaned.csv",index=False)
    
except Exception as e:
    print(f"Error: {e}")

