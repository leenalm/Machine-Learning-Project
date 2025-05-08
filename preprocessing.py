import pandas as pd
import pickle

# Load trained scaler and model
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("svm_model.pkl", "rb") as f:
    model = pickle.load(f)

# This MUST match what was used in training (after one-hot + drop)
expected_columns = [
    "satisfaction_level", "last_evaluation", "number_project",
    "average_monthly_hours", "time_spent", "work_accident", "promotion_last_5years",
    'dept_IT', 'dept_RandD', 'dept_accounting', 'dept_hr', 'dept_management',
    'dept_marketing', 'dept_product_mng', 'dept_sales', 'dept_support', 'dept_technical',
    'salary_high', 'salary_low', 'salary_medium'
]

def preprocess_input(user_input_df):
    # Ensure missing columns are filled with 0 (e.g., one-hot not selected)
    for col in expected_columns:
        if col not in user_input_df.columns:
            user_input_df[col] = 0

    # Reorder columns to match training
    user_input_df = user_input_df[expected_columns]

    # Transform
    scaled_array = scaler.transform(user_input_df)

    return scaled_array, model


