import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import random
import warnings
warnings.filterwarnings("ignore")

model = joblib.load("best_model.pkl")
df = pd.read_excel("student_exam_prediction_dataset_extended copy.xlsx")
df['extracurricular_participation_encoded'] = df['extracurricular_participation'].map({'No':0, 'Yes':1})
features = [
    'study_hours_per_day', 'attendance_percentage', 'mental_health_rating', 
    'sleep_hours', 'extracurricular_participation_encoded'
]

st.header("Student Performance Prediction and Explainability")

with st.sidebar:
    st.header("Select Student and Adjust Inputs")
    student_ids = df["student_id"].unique()
    selected_id = st.selectbox("Select Student", student_ids)
    student_row = df[df["student_id"] == selected_id].iloc[0]

    study_hours = st.slider("Study Hours per Day", 0, 12, int(student_row["study_hours_per_day"]))
    attendance = st.slider("Attendance Percentage", 0, 100, int(student_row["attendance_percentage"]))
    sleep_hours = st.slider("Sleep Hours per Night", 0, 12, int(student_row["sleep_hours"]))
    mental_health = st.slider("Mental Health Rating (1-10)", 1, 10, int(student_row["mental_health_rating"]))
    extracurricular = st.selectbox(
        "Extracurricular Participation",
        ['No', 'Yes'],
        index=0 if student_row['extracurricular_participation'] == 'No' else 1
    )
    run_pred = st.button("Predict")

# --- Student Profile: Three Sets of Exam Marks ---
st.markdown("### Student Profile")
col1, col2 = st.columns(2)
with col1:
    st.write(f"Age: {student_row['age']}")
    st.write(f"Gender: {student_row['gender']}")
    st.write(f"Diet Quality: {student_row['diet_quality']}")
    st.write(f"Exercise Frequency: {student_row['exercise_frequency']}/week")
    st.write(f"Extracurricular: {student_row['extracurricular_participation']}")

with col2:
    st.write(f"Study Hours: {student_row['study_hours_per_day']} hrs/day")
    st.write(f"Attendance: {student_row['attendance_percentage']}%")
    st.write(f"Sleep: {student_row['sleep_hours']} hrs/night")
    st.write(f"Mental Health: {student_row['mental_health_rating']}/10")
    st.write(f"Internet Quality: {student_row['internet_quality']}")
    st.write(f"Parental Education: {student_row['parental_education_level']}")

mark1, mark2, mark3 = st.columns(3)
mark1.metric("Python 1/2/3", f"{student_row['python_marks']} / {student_row['python_marks_2']} / {student_row['python_marks_3']}")
mark2.metric("Math 1/2/3", f"{student_row['mathematics_marks']} / {student_row['mathematics_marks_2']} / {student_row['mathematics_marks_3']}")
mark3.metric("DBMS 1/2/3", f"{student_row['dbms_marks']} / {student_row['dbms_marks_2']} / {student_row['dbms_marks_3']}")

st.metric("Final Exam", f"{student_row['final_exam_marks']}")

st.divider()

# --- Prediction and Explainability ---
if run_pred:
    extra_enc = 1 if extracurricular == 'Yes' else 0
    input_data = np.array([[study_hours, attendance, mental_health, sleep_hours, extra_enc]])
    pred = float(model.predict(input_data)[0])
    pred = np.clip(pred, 0, 100)

    st.subheader("Prediction Result")
    st.metric("Predicted Final Exam Score", f"{pred:.2f}%")

    # SHAP explainability
    background = df[features].sample(min(100, len(df)), random_state=42).values
    explainer = shap.TreeExplainer(model, background)
    shap_values = explainer.shap_values(input_data)
    shap_vals = np.array(shap_values).flatten()  # Ensured 1-D array for feature coloring!

    colors = ['green' if val > 0 else 'red' for val in shap_vals]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(features, shap_vals, color=colors)
    ax.set_xlabel('Impact on Score')
    ax.set_title('Feature Impact on Predicted Score')
    st.pyplot(fig)

    expl_text = []
    for feat, val in zip(features, shap_vals):
        expl_text.append(
            f"- Higher **{feat.replace('_', ' ')}** {'increases' if val > 0 else 'decreases'} predicted score by {abs(val):.2f} points."
        )
    st.markdown("### Explanation Summary\n" + "\n".join(expl_text))

    # --- Personalized Suggestions ---
    tips = []
    if study_hours < 2:
        tips.append("📖 Increase your study hours gradually to at least 2-3 hours per day for sustained improvement.")
    if attendance < 75:
        tips.append("📝 Regular class attendance is critical. Aim for 75%+ attendance to maximize learning opportunities.")
    if mental_health < 5:
        tips.append("🧠 Consider reaching out for mental health support; well-being is key to better performance.")
    if sleep_hours < 6:
        tips.append("💤 Try to get at least 6-7 hours of sleep nightly for better focus and retention.")
    if extracurricular == "No":
        tips.append("🎭 Participating in extracurricular activities can boost confidence and holistic development.")
    if not tips:
        tips = ["👍 Keep up the good work! Maintain your current habits for continued success."]
    st.subheader("Personalized Improvement Suggestions")
    for t in tips:
        st.write(t)

# --- Subject Quiz Section ---
subject_quiz = {
    "Python": [
        {"question": "What is the keyword to define a function in Python?", "answer": "def"},
        {"question": "What is the output of print(2 ** 3)?", "answer": "8"}
    ],
    "Mathematics": [
        {"question": "What is the derivative of x^2?", "answer": "2x"},
        {"question": "What is the value of π (pi) rounded to two decimals?", "answer": "3.14"}
    ],
    "DBMS": [
        {"question": "Which language is used to create and modify database tables? (abbr.)", "answer": "DDL"},
        {"question": "What does SQL stand for?", "answer": "Structured Query Language"}
    ]
}

if 'quiz_selected' not in st.session_state:
    st.session_state.quiz_selected = {subj: random.choice(qs) for subj, qs in subject_quiz.items()}
if 'quiz_answers' not in st.session_state:
    st.session_state.quiz_answers = {subj: "" for subj in subject_quiz}
if 'quiz_submitted' not in st.session_state:
    st.session_state.quiz_submitted = False

with st.expander("Take a Subject Quiz!"):
    st.write("Answer all subject questions, then submit to see your results:")
    for subj in subject_quiz:
        qtxt = st.session_state.quiz_selected[subj]['question']
        st.session_state.quiz_answers[subj] = st.text_input(f"{subj}: {qtxt}", key=f"ans_{subj}")
    if st.button("Submit All Answers"):
        st.session_state.quiz_submitted = True
    if st.session_state.quiz_submitted:
        results = []
        score = 0
        for subj in subject_quiz:
            correct = st.session_state.quiz_selected[subj]['answer'].strip().lower()
            user_ans = st.session_state.quiz_answers[subj].strip().lower()
            if user_ans == correct:
                results.append(f"✅ {subj}: Correct!")
                score += 1
            else:
                results.append(f"❌ {subj}: Incorrect. Correct answer: {st.session_state.quiz_selected[subj]['answer']}")
        st.markdown("#### Quiz Results")
        for res in results:
            st.write(res)
        st.success(f"Your score: {score} / {len(subject_quiz)}")
        if st.button("Try Another Quiz"):
            st.session_state.quiz_selected = {subj: random.choice(qs) for subj, qs in subject_quiz.items()}
            st.session_state.quiz_answers = {subj: "" for subj in subject_quiz}
            st.session_state.quiz_submitted = False
