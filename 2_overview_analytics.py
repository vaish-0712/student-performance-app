import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.header("Placement/Risk Overview Table")

# --- LOAD MODEL AND DATA ---
model = joblib.load("best_model.pkl")
df = pd.read_excel("student_exam_prediction_dataset_extended copy.xlsx")
df['extracurricular_participation_encoded'] = df['extracurricular_participation'].map({'No': 0, 'Yes': 1})

features = ['study_hours_per_day', 'attendance_percentage', 'mental_health_rating', 'sleep_hours', 'extracurricular_participation_encoded']

placement_thresholds = {"High": 85, "Medium": 70, "Low": 0}

# --- Overview Table ---
at_risk = df.copy()
at_risk["Predicted Score"] = model.predict(at_risk[features])
at_risk["Placement Category"] = pd.cut(
    at_risk["Predicted Score"],
    bins=[-np.inf, placement_thresholds["Medium"], placement_thresholds["High"], np.inf],
    labels=["Low", "Medium", "High"]
)
st.dataframe(
    at_risk[["student_id", "Predicted Score", "Placement Category", "final_exam_marks", "python_marks", "mathematics_marks", "dbms_marks"]],
    use_container_width=True
)
