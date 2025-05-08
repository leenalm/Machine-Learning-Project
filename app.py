import streamlit as st
import pickle
from preprocessing import preprocess_input

# Load model
with open("svm_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("features.pkl", "rb") as f:
    FEATURES = pickle.load(f)

st.title("Employee Attrition Prediction")

# User input form
user_input = {}
for feature in FEATURES:
    user_input[feature] = st.number_input(feature, value=0)

if st.button("Predict"):
    try:
        scaled_input = preprocess_input(user_input)
        prediction = model.predict(scaled_input)
        probability = model.predict_proba(scaled_input)[0][1]

        if prediction[0] == 1:
            st.error(f" This employee is **likely to leave**. (Probability: {probability:.2f})")
        else:
            st.success(f" This employee is **likely to stay**. (Probability: {probability:.2f})")
    except Exception as e:
        st.error(f"Error: {str(e)}")



