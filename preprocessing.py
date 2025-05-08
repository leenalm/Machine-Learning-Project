import pandas as pd
import pickle

# Load the pre-fitted scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

def preprocess_input(user_input_dict):
    # Create DataFrame from input dictionary
    df = pd.DataFrame([user_input_dict])

    # Ensure column order is the same as during training
    column_order = [
        'Age', 'BusinessTravel', 'DailyRate', 'Department', 'DistanceFromHome',
        'Education', 'EducationField', 'EmployeeCount', 'EmployeeNumber',
        'EnvironmentSatisfaction', 'Gender', 'HourlyRate', 'JobInvolvement',
        'JobLevel', 'JobRole', 'JobSatisfaction', 'MaritalStatus',
        'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 'Over18',
        'OverTime', 'PercentSalaryHike', 'PerformanceRating',
        'RelationshipSatisfaction', 'StandardHours', 'StockOptionLevel',
        'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',
        'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
        'YearsWithCurrManager'
    ]
    df = df[column_order]

    # Apply the same scaler as used during training
    df_scaled = scaler.transform(df)
    return df_scaled




