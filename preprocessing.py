# preprocessing.py
import numpy as np
import pandas as pd
import pickle

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("svm_model.pkl", "rb") as f:
    model = pickle.load(f)

# List of features used during training — copy-paste from training
feature_order = [
    "satisfaction_level", "last_evaluation", "number_project", "average_monthly_hours",
    "time_spent", "work_accident", "promotion_last_5years",
    'dept_IT', 'dept_RandD', 'dept_accounting', 'dept_hr', 'dept_management',
    'dept_marketing', 'dept_product_mng', 'dept_sales', 'dept_support', 'dept_technical',
    'salary_high', 'salary_low', 'salary_medium'
]

def preprocess_input(user_input_df):
    # Reorder columns to match training
    user_input_df = user_input_df.reindex(columns=feature_order)

    # Scale
    scaled_array = scaler.transform(user_input_df)
    return scaled_array, model


