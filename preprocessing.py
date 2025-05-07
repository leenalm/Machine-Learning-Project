import numpy as np
import pickle

# Define the exact numerical features used during training
NUMERICAL_FEATURES = [
    "Age", "DailyRate", "DistanceFromHome", "Education", "EmployeeCount",
    "EmployeeNumber", "EnvironmentSatisfaction", "HourlyRate", "JobInvolvement",
    "JobLevel", "RelationshipSatisfaction", "StandardHours", "StockOptionLevel",
    "TotalWorkingYears", "TrainingTimesLastYear", "WorkLifeBalance",
    "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion",
    "YearsWithCurrManager"
]

# Load scaler (StandardScaler) trained on numerical data
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

def preprocess_input(user_input_dict):
    """
    Convert dictionary input into a scaled NumPy array using the trained StandardScaler.
    
    Parameters:
        user_input_dict (dict): Keys must match NUMERICAL_FEATURES
    
    Returns:
        np.ndarray: Scaled input (1 row, 20 columns)
    """
    input_values = [user_input_dict[feature] for feature in NUMERICAL_FEATURES]
    input_array = np.array([input_values])  # Shape (1, n_features)
    scaled_array = scaler.transform(input_array)
    return scaled_array
