import streamlit as st
import pickle
from preprocessing import preprocess_input

# Load trained model
with open("svm_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Employee Attrition Prediction App")

# Define input fields for all 34 features (excluding Attrition)
user_input = {
    'Age': st.number_input("Age", min_value=18, max_value=60, value=30),
    'BusinessTravel': st.selectbox("BusinessTravel", [0, 1, 2]),
    'DailyRate': st.number_input("DailyRate", min_value=100, max_value=1500, value=800),
    'Department': st.selectbox("Department", [0, 1, 2]),
    'DistanceFromHome': st.number_input("DistanceFromHome", min_value=0, value=5),
    'Education': st.selectbox("Education", [1, 2, 3, 4, 5]),
    'EducationField': st.selectbox("EducationField", [0, 1, 2, 3, 4, 5]),
    'EmployeeCount': st.number_input("EmployeeCount", min_value=1, value=1),
    'EmployeeNumber': st.number_input("EmployeeNumber", min_value=1, value=100),
    'EnvironmentSatisfaction': st.selectbox("EnvironmentSatisfaction", [1, 2, 3, 4]),
    'Gender': st.selectbox("Gender", [0, 1]),
    'HourlyRate': st.number_input("HourlyRate", min_value=30, max_value=100, value=60),
    'JobInvolvement': st.selectbox("JobInvolvement", [1, 2, 3, 4]),
    'JobLevel': st.selectbox("JobLevel", [1, 2, 3, 4, 5]),
    'JobRole': st.selectbox("JobRole", list(range(9))),
    'JobSatisfaction': st.selectbox("JobSatisfaction", [1, 2, 3, 4]),
    'MaritalStatus': st.selectbox("MaritalStatus", [0, 1, 2]),
    'MonthlyIncome': st.number_input("MonthlyIncome", min_value=1000, value=5000),
    'MonthlyRate': st.number_input("MonthlyRate", min_value=1000, value=10000),
    'NumCompaniesWorked': st.number_input("NumCompaniesWorked", min_value=0, value=1),
    'Over18': st.selectbox("Over18", [0]),  # Constant in dataset
    'OverTime': st.selectbox("OverTime", [0, 1]),
    'PercentSalaryHike': st.number_input("PercentSalaryHike", min_value=10, max_value=25, value=15),
    'PerformanceRating': st.selectbox("PerformanceRating", [1, 2, 3, 4]),
    'RelationshipSatisfaction': st.selectbox("RelationshipSatisfaction", [1, 2, 3, 4]),
    'StandardHours': st.selectbox("StandardHours", [80]),  # Constant in dataset
    'StockOptionLevel': st.selectbox("StockOptionLevel", [0, 1, 2, 3]),
    'TotalWorkingYears': st.number_input("TotalWorkingYears", min_value=0, value=10),
    'TrainingTimesLastYear': st.number_input("TrainingTimesLastYear", min_value=0, value=2),
    'WorkLifeBalance': st.selectbox("WorkLifeBalance", [1, 2, 3, 4]),
    'YearsAtCompany': st.number_input("YearsAtCompany", min_value=0, value=5),
    'YearsInCurrentRole': st.number_input("YearsInCurrentRole", min_value=0, value=3),
    'YearsSinceLastPromotion': st.number_input("YearsSinceLastPromotion", min_value=0, value=1),
    'YearsWithCurrManager': st.number_input("YearsWithCurrManager", min_value=0, value=3)
}

if st.button("Predict"):
    processed_input = preprocess_input(user_input)
    prediction = model.predict(processed_input)[0]
    st.success(f"Predicted Attrition: {'Yes' if prediction == 1 else 'No'}")





