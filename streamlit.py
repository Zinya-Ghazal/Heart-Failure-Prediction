import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load the model
model = joblib.load('heart_failure_model.joblib')

# Feature columns expected by the model
feature_columns = ['Age', 'Gender', 'Life.Style', 'Sleep', 'Depression', 'Hyperlipi', 'thal', 'slope']

# Function to predict
def predict_heart_failure(data):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform([data])
    prediction = model.predict(data_scaled)
    return "High Risk of Heart Failure" if prediction == 1 else "Low Risk of Heart Failure"

# Streamlit UI
st.title("Heart Failure Prediction")
st.write("Enter the following details to predict the risk of heart failure:")

# Collect user input for each feature
age = st.number_input("Age", min_value=0, max_value=120, value=50)
gender = st.selectbox("Gender", options=["Male", "Female"])
lifestyle = st.slider("Life.Style (1-10)", min_value=1, max_value=10, value=5)
sleep = st.slider("Sleep (Hours)", min_value=0, max_value=24, value=7)
depression = st.selectbox("Depression (0 - No, 1 - Yes)", options=[0, 1])
hyperlipi = st.selectbox("Hyperlipidemia (0 - No, 1 - Yes)", options=[0, 1])
thal = st.selectbox("Thalassemia (0 - Normal, 1 - Fixed Defect, 2 - Reversible Defect)", options=[0, 1, 2])
slope = st.slider("Slope (1-3)", min_value=1, max_value=3, value=2)

# Map gender to numeric
gender = 1 if gender == "Male" else 0

# Prepare the data
input_data = [age, gender, lifestyle, sleep, depression, hyperlipi, thal, slope]

# Button for prediction
if st.button("Predict"):
    result = predict_heart_failure(input_data)
    st.success(f"Prediction: {result}")
