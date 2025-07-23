import streamlit as st
import pandas as pd
import joblib
import datetime

# =======================
# Load Models & Encoders
# =======================
priority_model = joblib.load("priority_xgboost.pkl")
priority_label_encoder = joblib.load("priority_label_encoder.pkl")
priority_vectorizer = joblib.load("priority_tfidf_vectorizer.pkl")

task_model = joblib.load("nb_task_classifier.joblib")
task_label_encoder = joblib.load("nb_label_encoder.joblib")
task_vectorizer = joblib.load("task_tfidf_vectorizer.pkl")

# NEW: Workload assignment model
user_model = joblib.load("user_assignment_model.pkl")
user_label_encoder = joblib.load("user_label_encoder.pkl")
priority_encoder_for_user = joblib.load("priority_encoder_for_user.pkl")
category_encoder_for_user = joblib.load("category_encoder_for_user.pkl")

# Dataset
@st.cache_data
def load_data():
    return pd.read_csv("cleaned_final_task_dataset (1).csv")

df = load_data()

# =======================
# Streamlit UI
# =======================
st.title("🧠 AI Task Assignment Dashboard (ML‑based User Prediction)")

# Task input
task_options = df['task_description'].dropna().unique().tolist()
selected_task = st.selectbox("Select a Task from Dataset (or type your own):", [""] + task_options)
task_desc = st.text_area("📝 Or enter a new Task Description", value=selected_task if selected_task else "")
deadline = st.date_input("📅 Deadline", min_value=datetime.date.today())

if st.button("Predict & Assign"):
    if not task_desc.strip():
        st.warning("⚠️ Please enter or select a task description.")
    else:
        # Vectorize task description
        task_vector = task_vectorizer.transform([task_desc])
        priority_vector = priority_vectorizer.transform([task_desc])

        # Predict priority & category
        pred_priority_enc = priority_model.predict(priority_vector)[0]
        pred_priority = priority_label_encoder.inverse_transform([pred_priority_enc])[0]

        pred_category_enc = task_model.predict(task_vector)[0]
        pred_category = task_label_encoder.inverse_transform([pred_category_enc])[0]

        # Prepare features for user assignment model
        urgency_score = max(0, 10 - (deadline - datetime.date.today()).days)
        avg_load = df.groupby("assigned_user")["user_current_load"].mean().median()
        past_behavior = df["past_behavior_score"].median()
        time_taken = df["time_taken_(hours)"].median()

        user_features = pd.DataFrame([[
            priority_encoder_for_user.transform([pred_priority])[0],
            category_encoder_for_user.transform([pred_category])[0],
            avg_load,
            past_behavior,
            time_taken
        ]], columns=["priority_encoded", "category_encoded", "user_current_load", "past_behavior_score", "time_taken_(hours)"])

        # Predict assigned user
        pred_user_enc = user_model.predict(user_features)[0]
        assigned_user = user_label_encoder.inverse_transform([pred_user_enc])[0]

        # Display results
        st.success(f"✅ Task Assigned to: **{assigned_user}**")
        st.info(f"🔺 Priority: **{pred_priority}** | 📁 Category: **{pred_category}** | 🗓 Deadline: {deadline}")

        # Optional: Show model confidence (top 3 users)
        probs = user_model.predict_proba(user_features)[0]
        top3_idx = probs.argsort()[-3:][::-1]
        st.write("### Top 3 Candidate Users:")
        for i in top3_idx:
            st.write(f"- **{user_label_encoder.inverse_transform([i])[0]}** ({probs[i]*100:.2f}%)")
