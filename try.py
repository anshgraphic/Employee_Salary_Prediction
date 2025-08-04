import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="Employee Salary Predictor", layout="centered")

try:
    model = joblib.load('model.joblib')
    preprocessor = joblib.load('preprocessor.joblib')
    # st.success("Model and Preprocessor loaded successfully!") 
except FileNotFoundError:
    st.error("Error: Model or Preprocessor files not found. "
             "Please ensure 'model.joblib' "
             "and 'preprocessor.joblib' are in the same directory.")
    st.stop() # Stop the app if files are not found

try:
    gender_categories = preprocessor.named_transformers_['cat'].categories_[0].tolist()
    education_categories = preprocessor.named_transformers_['cat'].categories_[1].tolist()
    job_title_categories = preprocessor.named_transformers_['cat'].categories_[2].tolist()

    if 'Other' not in job_title_categories:
        job_title_categories.append('Other')

except Exception as e:
    st.error(f"Error extracting categories from preprocessor: {e}")
    st.stop()

st.title("Employee Salary Predictor") # Removed emoji
st.markdown("Enter employee details to get a salary prediction.")

# Input fields for user
st.header("Employee Details")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", min_value=18, max_value=60, value=30)
    gender = st.selectbox("Gender", gender_categories)
    education_level = st.selectbox("Education Level", education_categories)

with col2:
    years_of_experience = st.slider("Years of Experience", min_value=0.0, max_value=42.0, value=5.0, step=0.5)
    job_title = st.selectbox("Job Title", job_title_categories)

if st.button("Predict Salary"):
    # --- Input Validation Logic ---
    min_working_age = 18 # This can remain 18, as it aligns with the new age slider min
    if (age - years_of_experience) < min_working_age:
        st.error(f"Invalid input: An individual with {age} years of age cannot have {years_of_experience} years of experience. "
                 f"This implies a professional start before age {min_working_age}. Please adjust the values.")
    else:
        input_data = pd.DataFrame([[age, gender, education_level, job_title, years_of_experience]],
                                  columns=['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience'])

        try:
            processed_input = preprocessor.transform(input_data)
            # Make prediction
            predicted_salary = model.predict(processed_input)[0]

            st.success(f"### Predicted Monthly Salary: ₹{predicted_salary:,.2f}")
            st.balloons() # Fun animation
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.warning("Please ensure all inputs are valid and the preprocessor/model are correctly loaded.")
