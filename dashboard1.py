   import streamlit as st
import pandas as pd
import joblib
import datetime
import random

# =====================
# Load Models & Encoders
# =====================
priority_model = joblib.load("priority_xgboost.pkl")
priority_label_encoder = joblib.load("priority_label_encoder.pkl")
priority_vectorizer = joblib.load("priority_tfidf_vectorizer.pkl")

task_model = joblib.load("nb_task_classifier.joblib")
task_label_encoder = joblib.load("nb_label_encoder.joblib")
task_vectorizer = joblib.load("task_tfidf_vectorizer.pkl")

# =====================
# Load Dataset
# =====================
@st.cache_data
def load_data():
    return pd.read_csv("cleaned_final_task_dataset (1).csv")

df = load_data()

# =====================
# Improved workload assignment
# =====================
def assign_user_with_check(pred_category, urgency_score=0):
    user_workload_df = df.groupby("assigned_user")["user_current_load"].mean().reset_index()
    user_workload_df.rename(columns={"user_current_load": "avg_workload"}, inplace=True)

    # Filter users who worked in this category
    matching_users = df[df["category"] == pred_category]["assigned_user"].unique()
    matching_users_filtered = user_workload_df[user_workload_df["assigned_user"].isin(matching_users)].copy()

    if matching_users_filtered.empty:
        # Fallback: all users
        matching_users_filtered = user_workload_df.copy()

    matching_users_filtered["urgency_score"] = urgency_score
    matching_users_filtered["combined_score"] = matching_users_filtered["avg_workload"] + urgency_score

    # Pick randomly among top 3 candidates
    top_candidates = matching_users_filtered.sort_values("combined_score").head(3)
    assigned_user = random.choice(top_candidates["assigned_user"].tolist())
    return assigned_user

# =====================
# Streamlit UI
# =====================
st.title("🧠 AI Task Assignment Dashboard")

# Dropdown for selecting task from dataset
task_options = df["task_description"].dropna().unique().tolist()
selected_task = st.selectbox("Or select an existing task:", [""] + task_options)

with st.form("task_form"):
    task_desc = st.text_area("📝 Enter Task Description", value=selected_task if selected_task else "")
    deadline = st.date_input("📅 Deadline", min_value=datetime.date.today())
    submitted = st.form_submit_button("Predict & Assign")

if submitted:
    if not task_desc.strip():
        st.error("Task description cannot be empty!")
    else:
        # Vectorize
        task_vector = task_vectorizer.transform([task_desc])
        priority_vector = priority_vectorizer.transform([task_desc])

        # Predictions
        pred_priority_enc = priority_model.predict(priority_vector)[0]
        pred_priority = priority_label_encoder.inverse_transform([pred_priority_enc])[0]

        pred_category_enc = task_model.predict(task_vector)[0]
        pred_category = task_label_encoder.inverse_transform([pred_category_enc])[0]

        # Deadline urgency
        today = datetime.date.today()
        days_left = (deadline - today).days
        urgency_score = max(0, 10 - days_left)

        # Assign user using improved logic
        assigned_user = assign_user_with_check(pred_category, urgency_score)

        # Show results
        st.success(f"✅ Task Assigned to: **{assigned_user}**")
        st.info(f"🔺 Priority: **{pred_priority}** | 📁 Category: **{pred_category}** | 🗓 Days to Deadline: {days_left}")
