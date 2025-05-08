import streamlit as st
import pandas as pd
import pickle
from preprocessing import preprocess_data

# Load the trained model and scaler
with open('svm_model.pkl', 'rb') as model_file:
    svm_model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Streamlit UI elements to take user input for all 34 features
st.title("Employee Attrition Prediction")

# Create input fields for all features
features = {
    'Age': st.number_input("Age", min_value=18, max_value=100, value=30),
    'BusinessTravel': st.selectbox("BusinessTravel", [0, 1, 2]),
    'DailyRate': st.number_input("DailyRate", min_value=0, value=500),
    'Department': st.selectbox("Department", [0, 1, 2]),
    'DistanceFromHome': st.number_input("DistanceFromHome", min_value=0, value=5),
    'Education': st.number_input("Education", min_value=1, max_value=5, value=3),
    'EducationField': st.selectbox("EducationField", [0, 1, 2, 3, 4]),
    'EmployeeCount': st.number_input("EmployeeCount", min_value=1, value=1),
    'EmployeeNumber': st.number_input("EmployeeNumber", min_value=1, value=1000),
    'EnvironmentSatisfaction': st.selectbox("EnvironmentSatisfaction", [1, 2, 3, 4]),
    'Gender': st.selectbox("Gender", [0, 1]),
    'HourlyRate': st.number_input("HourlyRate", min_value=10, value=20),
    'JobInvolvement': st.selectbox("JobInvolvement", [1, 2, 3, 4]),
    'JobLevel': st.number_input("JobLevel", min_value=1, max_value=5, value=2),
    'JobRole': st.selectbox("JobRole", [0, 1, 2, 3, 4, 5, 6]),
    'JobSatisfaction': st.selectbox("JobSatisfaction", [1, 2, 3, 4]),
    'MaritalStatus': st.selectbox("MaritalStatus", [0, 1, 2]),
    'MonthlyIncome': st.number_input("MonthlyIncome", min_value=1000, value=5000),
    'MonthlyRate': st.number_input("MonthlyRate", min_value=1000, value=15000),
    'NumCompaniesWorked': st.number_input("NumCompaniesWorked", min_value=0, value=3),
    'Over18': st.selectbox("Over18", [0, 1]),
    'OverTime': st.selectbox("OverTime", [0, 1]),
    'PercentSalaryHike': st.number_input("PercentSalaryHike", min_value=0, value=10),
    'PerformanceRating': st.selectbox("PerformanceRating", [1, 2, 3, 4]),
    'RelationshipSatisfaction': st.selectbox("RelationshipSatisfaction", [1, 2, 3, 4]),
    'StandardHours': st.number_input("StandardHours", min_value=1, value=40),
    'StockOptionLevel': st.number_input("StockOptionLevel", min_value=0, max_value=3, value=1),
    'TotalWorkingYears': st.number_input("TotalWorkingYears", min_value=0, value=10),
    'TrainingTimesLastYear': st.number_input("TrainingTimesLastYear", min_value=0, value=1),
    'WorkLifeBalance': st.selectbox("WorkLifeBalance", [1, 2, 3, 4]),
    'YearsAtCompany': st.number_input("YearsAtCompany", min_value=0, value=5),
    'YearsInCurrentRole': st.number_input("YearsInCurrentRole", min_value=0, value=3),
    'YearsSinceLastPromotion': st.number_input("YearsSinceLastPromotion", min_value=0, value=2),
    'YearsWithCurrManager': st.number_input("YearsWithCurrManager", min_value=0, value=2),
}

# Convert input into DataFrame
input_data = pd.DataFrame([features])

# Preprocess the input data (scaling, encoding, etc.)
processed_data = preprocess_data(input_data)

# Predict the Attrition using the SVM model
prediction = svm_model.predict(processed_data)

# Display the prediction result
if prediction[0] == 1:
    st.write("Prediction: Employee is likely to leave.")
else:
    st.write("Prediction: Employee is likely to stay.")




