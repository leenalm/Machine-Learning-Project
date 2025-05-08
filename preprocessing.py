import numpy as np
import pickle

# Load saved objects
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("features.pkl", "rb") as f:
    FEATURES = pickle.load(f)

def preprocess_input(user_input: dict):
    """
    Prepares the input using all features, in correct order.
    """
    input_values = [user_input[feature] for feature in FEATURES]
    input_array = np.array(input_values).reshape(1, -1)
    return scaler.transform(input_array)

