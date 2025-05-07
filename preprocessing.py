import numpy as np
import pickle

NUMERICAL_FEATURES = [
    "Age", "DailyRate", "DistanceFromHome", "Education", "EmployeeCount",
    "EmployeeNumber", "EnvironmentSatisfaction", "HourlyRate", "JobInvolvement",
    "JobLevel", "RelationshipSatisfaction", "StandardHours", "StockOptionLevel",
    "TotalWorkingYears", "TrainingTimesLastYear", "WorkLifeBalance",
    "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion",
    "YearsWithCurrManager"
]

# Load the scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

def preprocess_input(user_input_dict):
    input_values = [user_input_dict[feature] for feature in NUMERICAL_FEATURES]
    input_array = np.array([input_values])
    return scaler.transform(input_array)
