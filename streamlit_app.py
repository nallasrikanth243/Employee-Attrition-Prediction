# streamlit_app.py

import streamlit as st
import pandas as pd
import joblib

# Load saved components
model = joblib.load("models/attrition_model.pkl")
label_encoders = joblib.load("models/label_encoders.pkl")
features = joblib.load("models/features.pkl")

st.title("🧠 Employee Attrition Prediction")
st.write("Predict if an employee is likely to leave based on their profile")

# Collect user input
user_input = {}

# Simple numeric fields
numeric_fields = [
    'Age', 'DailyRate', 'DistanceFromHome', 'HourlyRate', 'MonthlyIncome',
    'MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike', 'TotalWorkingYears',
    'TrainingTimesLastYear', 'YearsAtCompany', 'YearsInCurrentRole',
    'YearsSinceLastPromotion', 'YearsWithCurrManager'
]

for field in numeric_fields:
    user_input[field] = st.number_input(field, min_value=0, max_value=100, value=1)

# Categorical fields (just a few examples)
categorical_fields = {
    'BusinessTravel': ['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'],
    'Department': ['Sales', 'Research & Development', 'Human Resources'],
    'EducationField': ['Life Sciences', 'Other', 'Medical', 'Marketing', 'Technical Degree', 'Human Resources'],
    'Gender': ['Male', 'Female'],
    'JobRole': ['Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manufacturing Director', 'Healthcare Representative', 'Manager', 'Sales Representative', 'Research Director', 'Human Resources'],
    'MaritalStatus': ['Single', 'Married', 'Divorced'],
    'OverTime': ['Yes', 'No']
}

for field, options in categorical_fields.items():
    choice = st.selectbox(field, options)
    user_input[field] = label_encoders[field].transform([choice])[0]

# WorkLifeBalance, EnvironmentSatisfaction, etc.
satisfaction_fields = ['JobSatisfaction', 'WorkLifeBalance', 'EnvironmentSatisfaction', 'JobInvolvement', 'PerformanceRating']
for field in satisfaction_fields:
    user_input[field] = st.slider(field + " (1-4)", min_value=1, max_value=4, value=3)

# Build input dataframe
input_df = pd.DataFrame([user_input], columns=features)

# Predict
if st.button("Predict Attrition"):
    prediction = model.predict(input_df)[0]
    st.success("This employee is likely to {}leave.".format("" if prediction == 1 else "not "))
