!pip install streamlit pandas scikit-learn joblib xgboost

import streamlit as st
import pandas as pd
import datetime
import joblib

# =======================
# Load Models & Encoders
# =======================
# Priority prediction
priority_model = joblib.load("priority_xgboost.pkl")
priority_label_encoder = joblib.load("priority_label_encoder.pkl")
priority_vectorizer = joblib.load("priority_tfidf_vectorizer.pkl")

# Task classification (Naive Bayes)
task_model = joblib.load("nb_task_classifier.joblib")
task_label_encoder = joblib.load("nb_label_encoder.joblib")
task_vectorizer = joblib.load("task_tfidf_vectorizer.pkl")

# =======================
# Load Dataset
# =======================
@st.cache_data
def load_data():
    return pd.read_csv("final_task_dataset_balanced.csv")

df = load_data()

# =======================
# Workload Balancing Function
# =======================
def assign_user(predicted_category, urgency_score):
    # Filter users who handled predicted category
    category_users = df[df["category"] == predicted_category]["assigned_user"].unique()
    user_workload_df = df.groupby("assigned_user")["user_current_load"].mean().reset_index()
    user_workload_df.rename(columns={"user_current_load": "avg_workload"}, inplace=True)
    matching_users = user_workload_df[user_workload_df["assigned_user"].isin(category_users)].copy()

    # If no matching users, fallback to all
    if matching_users.empty:
        matching_users = user_workload_df.copy()

    # Compute combined score
    matching_users["urgency_score"] = urgency_score
    matching_users["combined_score"] = matching_users["avg_workload"] + urgency_score

    # Sort by combined score (lower is better)
    assigned_user = matching_users.sort_values("combined_score").iloc[0]["assigned_user"]
    return assigned_user, matching_users

# =======================
# Streamlit UI
# =======================
st.title("🧠 AI Task Assignment Dashboard")

with st.form("task_form"):
    task_desc = st.text_area("📝 Enter Task Description")
    deadline = st.date_input("📅 Deadline", min_value=datetime.date.today())
    submitted = st.form_submit_button("Predict & Assign")

if submitted:
    if not task_desc.strip():
        st.error("⚠️ Task description cannot be empty!")
    else:
        # Vectorize input
        task_vector = task_vectorizer.transform([task_desc])
        priority_vector = priority_vectorizer.transform([task_desc])

        # Predictions
        pred_priority_enc = priority_model.predict(priority_vector)[0]
        pred_priority = priority_label_encoder.inverse_transform([pred_priority_enc])[0]

        pred_category_enc = task_model.predict(task_vector)[0]
        pred_category = task_label_encoder.inverse_transform([pred_category_enc])[0]

        # Compute urgency score
        today = datetime.date.today()
        days_left = (deadline - today).days
        urgency_score = max(0, 10 - days_left)

        # Assign user
        assigned_user, debug_info = assign_user(pred_category, urgency_score)

        # Show result
        st.success(f"✅ Task Assigned to: **{assigned_user}**")
        st.info(f"🔺 Priority: **{pred_priority}** | 📁 Category: **{pred_category}** | 🗓 Days Left: {days_left}")

        st.write("### Debug Info (Matching Users):")
        st.dataframe(debug_info)

