# app.py
import streamlit as st
import pandas as pd
from preprocessing import preprocess_input

st.title("Employee Attrition Predictor")

# Collect input
satisfaction_level = st.slider("Satisfaction Level", 0.0, 1.0, 0.5)
last_evaluation = st.slider("Last Evaluation", 0.0, 1.0, 0.5)
number_project = st.slider("Number of Projects", 2, 7, 4)
average_monthly_hours = st.slider("Average Monthly Hours", 90, 310, 160)
time_spent = st.slider("Time Spent at Company (Years)", 1, 10, 3)
work_accident = st.selectbox("Work Accident", [0, 1])
promotion_last_5years = st.selectbox("Promotion in Last 5 Years", [0, 1])
dept = st.selectbox("Department", ['sales', 'technical', 'support', 'IT', 'product_mng', 'marketing', 'RandD', 'accounting', 'hr', 'management'])
salary = st.selectbox("Salary", ['low', 'medium', 'high'])

# Manual one-hot encoding (you can also use same encoder used during training)
dept_dummies = [1 if dept == d else 0 for d in ['IT', 'RandD', 'accounting', 'hr', 'management', 'marketing', 'product_mng', 'sales', 'support', 'technical']]
salary_dummies = [1 if salary == s else 0 for s in ['high', 'low', 'medium']]

user_input = [
    satisfaction_level,
    last_evaluation,
    number_project,
    average_monthly_hours,
    time_spent,
    work_accident,
    promotion_last_5years,
    *dept_dummies,
    *salary_dummies
]

columns = [
    "satisfaction_level", "last_evaluation", "number_project", "average_monthly_hours",
    "time_spent", "work_accident", "promotion_last_5years",
    'dept_IT', 'dept_RandD', 'dept_accounting', 'dept_hr', 'dept_management', 'dept_marketing',
    'dept_product_mng', 'dept_sales', 'dept_support', 'dept_technical',
    'salary_high', 'salary_low', 'salary_medium'
]

user_input_df = pd.DataFrame([user_input], columns=columns)

if st.button("Predict"):
    scaled_input, model = preprocess_input(user_input_df)
    prediction = model.predict(scaled_input)[0]

    if prediction == 1:
        st.error("This employee is **likely to leave**.")
    else:
        st.success("This employee is **likely to stay**.")



