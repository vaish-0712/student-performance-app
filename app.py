import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import warnings
import random
warnings.filterwarnings("ignore")


# --- LOAD DATASET & MODEL ---
MODEL_FILE = "best_model.pkl"
DATA_FILE = "student_exam_prediction_dataset_extended copy.xlsx"

model = joblib.load(MODEL_FILE)
df = pd.read_excel(DATA_FILE)
df['extracurricular_participation_encoded'] = df['extracurricular_participation'].map({'No': 0, 'Yes': 1})

features = [
    "study_hours_per_day",
    "attendance_percentage",
    "mental_health_rating",
    "sleep_hours",
    "extracurricular_participation_encoded"
]

# --- Prediction Category Thresholds ---
placement_thresholds = {"High": 85, "Medium": 70, "Low": 0}

# --- SIDEBAR: SELECT STUDENT ---
st.set_page_config(page_title="Student Analytics Admin", layout="wide")
st.sidebar.title("Admin Controls")

# Use full dataset, no filtering
filtered_df = df.copy()

# --- SHAP SUMMARY PLOT ---
background = filtered_df[features].sample(min(100, len(filtered_df)), random_state=42)
explainer = shap.TreeExplainer(model, background)
shap_values = explainer.shap_values(background)

st.subheader("Overall Feature Importance Summary (SHAP)")
shap.summary_plot(shap_values, background, plot_type='bar', show=False)
fig_summary = plt.gcf()
st.pyplot(fig_summary)

# --- Session State Management for Roll Number Selection ---
student_ids = filtered_df["student_id"].unique()
if 'selected_id' not in st.session_state or st.session_state.selected_id not in student_ids:
    st.session_state.selected_id = student_ids[0] if len(student_ids) > 0 else None

selected_id = st.sidebar.selectbox(
    "Select Student",
    student_ids,
    index=list(student_ids).index(st.session_state.selected_id) if st.session_state.selected_id in student_ids else 0
)
st.session_state.selected_id = selected_id
student_row = filtered_df[filtered_df["student_id"] == selected_id].iloc[0]

# --- Sidebar Academic Inputs ---
st.sidebar.header("Adjust Key Academic Inputs")
study_hours = st.sidebar.slider("Study Hours per Day", 0, 12, int(student_row["study_hours_per_day"]))
attendance = st.sidebar.slider("Attendance Percentage", 0, 100, int(student_row["attendance_percentage"]))
sleep_hours = st.sidebar.slider("Sleep Hours per Night", 0, 12, int(student_row["sleep_hours"]))
mental_health = st.sidebar.slider("Mental Health Rating (1-10)", 1, 10, int(student_row["mental_health_rating"]))
extracurricular = st.sidebar.selectbox(
    "Extracurricular Participation",
    ['No', 'Yes'],
    index=0 if student_row['extracurricular_participation'] == 'No' else 1
)
run_pred = st.sidebar.button("Predict & Explain")

st.title("Student Performance Prediction and Analysis")

# --- STUDENT CARD ---
st.markdown("### Student Profile")
profile_col1, profile_col2 = st.columns([2, 2])
with profile_col1:
    st.markdown(f"🧑 **Age:** {student_row['age']}")
    st.markdown(f"⚧️ **Gender:** {student_row['gender']}")
    # Removed 'department' and 'semester' display lines
    st.markdown(f"🥗 **Diet Quality:** {student_row['diet_quality']}")
    st.markdown(f"🏃‍♂️ **Exercise:** {student_row['exercise_frequency']}/week")
    st.markdown(f"🎭 **Extracurricular:** {student_row['extracurricular_participation']}")
    st.markdown(f"🌐 **Internet Quality:** {student_row['internet_quality']}")
    st.markdown(f"🎓 **Parental Ed.:** {student_row['parental_education_level']}")
with profile_col2:
    st.metric("Study Hours", f"{student_row['study_hours_per_day']} hrs/day")
    st.metric("Attendance", f"{student_row['attendance_percentage']}%")
    st.metric("Sleep", f"{student_row['sleep_hours']} hrs/night")
    st.metric("Mental Health", f"{student_row['mental_health_rating']}/10")
    st.metric(" Final Exam", f"{student_row['final_exam_marks']}")

mark1, mark2, mark3 = st.columns(3)
mark1.metric("Python 1/2/3", f"{student_row['python_marks']} / {student_row['python_marks_2']} / {student_row['python_marks_3']}")
mark2.metric("Math 1/2/3", f"{student_row['mathematics_marks']} / {student_row['mathematics_marks_2']} / {student_row['mathematics_marks_3']}")
mark3.metric("DBMS 1/2/3", f"{student_row['dbms_marks']} / {student_row['dbms_marks_2']} / {student_row['dbms_marks_3']}")

st.divider()

# --- MAIN PREDICTION INTERFACE ---
if run_pred:
    extra_enc = 1 if extracurricular == 'Yes' else 0
    input_vals = np.array([[study_hours, attendance, mental_health, sleep_hours, extra_enc]])
    pred = float(model.predict(input_vals)[0])
    pred = np.clip(pred, 0, 100)

    # Placement Category
    placement_cat = "Low"
    if pred >= placement_thresholds["High"]:
        placement_cat = "High"
    elif pred >= placement_thresholds["Medium"]:
        placement_cat = "Medium"

    st.subheader("Prediction Results")
    colA, colB = st.columns(2)
    colA.metric("Predicted Academic Score", f"{pred:.2f}%")
    colB.metric("Predicted Placement Category", placement_cat)

    st.divider()
    st.subheader("Explainability & Influential Features")
    background_vals = filtered_df[features].sample(min(100, len(filtered_df)), random_state=42).values
    explainer = shap.TreeExplainer(model, background_vals)
    shap_vals = explainer.shap_values(input_vals)
    shap_vals = np.array(shap_vals).flatten()
    colors = ['green' if val > 0 else 'red' for val in shap_vals]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(features, shap_vals, color=colors)
    ax.set_xlabel('Impact on Prediction')
    ax.set_title('Key Influences on Academic Score')
    st.pyplot(fig)
    expl_text = []
    for feat, val in zip(features, shap_vals):
        expl_text.append(
            f"- Higher **{feat.replace('_', ' ')}** {'increases' if val > 0 else 'decreases'} predicted score by {abs(val):.2f} points."
        )
    st.markdown("### Explanation Summary\n" + "\n".join(expl_text))

    # --- Risk Alert ---
    if placement_cat == "Low" or pred < 60:
        st.error(
            "⚠️ This student is at academic risk. Immediate intervention recommended (counseling, mentoring, or academic support)."
        )
    elif placement_cat == "Medium":
        st.warning("⚠️ This student is in the moderate risk zone. Suggest proactive check-ins.")
    else:
        st.success("🎉 This student is on track for successful placement!")

st.subheader("Personalized Improvement Suggestions")
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
for t in tips:
    st.write(t)

# --- Subject Quiz ---
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
    st.session_state.quiz_selected = {
        subj: random.choice(qs) for subj, qs in subject_quiz.items()
    }
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
            st.session_state.quiz_selected = {
                subj: random.choice(qs) for subj, qs in subject_quiz.items()
            }
            st.session_state.quiz_answers = {subj: "" for subj in subject_quiz}
            st.session_state.quiz_submitted = False

# --- Admin Table of All Students ---
st.divider()
st.header("📊 Placement/Risk Overview Table")
at_risk = filtered_df.copy()
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
  