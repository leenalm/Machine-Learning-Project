import streamlit as st
import pickle
import os
from preprocessing import preprocess_input

# Load model
current_dir = os.path.dirname(__file__)
model_path = os.path.join(current_dir, "svm_model.pkl")

with open(model_path, "rb") as f:
    model = pickle.load(f)

# App Title
st.title("Employee Attrition Prediction App")
st.markdown("Enter employee details to predict if the employee will likely **stay** or **leave**.")

# Sidebar inputs
st.sidebar.header("Employee Details")
user_input = {
    "Age": st.sidebar.slider("Age", 18, 60, 30),
    "DailyRate": st.sidebar.slider("Daily Rate", 100, 1500, 800),
    "DistanceFromHome": st.sidebar.slider("Distance From Home (km)", 1, 30, 10),
    "Education": st.sidebar.slider("Education (1=Below College, 5=Doctor)", 1, 5, 3),
    "EmployeeCount": 1,
    "EmployeeNumber": st.sidebar.slider("Employee Number", 1, 2068, 1001),
    "EnvironmentSatisfaction": st.sidebar.slider("Environment Satisfaction (1=Low, 4=High)", 1, 4, 3),
    "HourlyRate": st.sidebar.slider("Hourly Rate", 30, 100, 60),
    "JobInvolvement": st.sidebar.slider("Job Involvement (1=Low, 4=High)", 1, 4, 3),
    "JobLevel": st.sidebar.slider("Job Level", 1, 5, 2),
    "RelationshipSatisfaction": st.sidebar.slider("Relationship Satisfaction (1=Low, 4=High)", 1, 4, 3),
    "StandardHours": 80,
    "StockOptionLevel": st.sidebar.slider("Stock Option Level", 0, 3, 1),
    "TotalWorkingYears": st.sidebar.slider("Total Working Years", 0, 40, 10),
    "TrainingTimesLastYear": st.sidebar.slider("Training Times Last Year", 0, 6, 3),
    "WorkLifeBalance": st.sidebar.slider("Work-Life Balance (1=Bad, 4=Best)", 1, 4, 3),
    "YearsAtCompany": st.sidebar.slider("Years at Company", 0, 40, 5),
    "YearsInCurrentRole": st.sidebar.slider("Years in Current Role", 0, 18, 4),
    "YearsSinceLastPromotion": st.sidebar.slider("Years Since Last Promotion", 0, 15, 2),
    "YearsWithCurrManager": st.sidebar.slider("Years With Current Manager", 0, 17, 3)
}

# Prediction
if st.button("Predict Attrition"):
    scaled_input = preprocess_input(user_input)
    prediction = model.predict(scaled_input)[0]

    if prediction == 1:
        st.error("This employee is **likely to leave** the company.")
    else:
        st.success("This employee is **likely to stay** at the company.")


