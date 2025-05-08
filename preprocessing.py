# preprocessing.py
import numpy as np
import pickle

# Load the scaler and model
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("svm_model.pkl", "rb") as f:
    model = pickle.load(f)

def preprocess_input(user_input_df):
    # Ensure the input is in the same format and column order
    input_array = user_input_df.values.reshape(1, -1)  # assume DataFrame with right columns
    scaled_array = scaler.transform(input_array)
    return scaled_array, model


