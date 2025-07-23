import streamlit as st
import pandas as pd
import datetime
import joblib

# =====================
# 🔹 Load Models & Encoders
# =====================
priority_model = joblib.load("priority_xgboost (1).pkl")
priority_label_encoder = joblib.load("priority_label_encoder (1).pkl")
priority_vectorizer = joblib.load("priority_tfidf_vectorizer (1).pkl")

category_model = joblib.load("naive_bayes_task_classifier.joblib")  # Naive Bayes
category_label_encoder = joblib.load("task_label_encoder (1).pkl")
task_vectorizer = joblib.load("task_tfidf_vectorizer.pkl")

# =====================
# 🔹 Load Dataset
# =====================
@st.cache_data
def load_data():
    return pd.read_csv("cleaned_final_task_dataset (1).csv")

df = load_data()

# =====================
# 🔹 Workload-based User Assignment
# =====================
def assign_user_with_check(pred_category, urgency_score=0):
    # Group by workload
    user_workload_df = df.groupby("assigned_user")["user_current_load"].mean().reset_index()
    user_workload_df.rename(columns={"user_current_load": "avg_workload"}, inplace=True)

    # Filter only users who handled this category
    matching_users = df[df["category"] == pred_category]["assigned_user"].unique()
    matching_users_filtered = user_workload_df[user_workload_df["assigned_user"].isin(matching_users)].copy()

    if matching_users_filtered.empty:
        return "No available user", None

    # Compute combined score
    matching_users_filtered["urgency_score"] = urgency_score
    matching_users_filtered["combined_score"] = matching_users_filtered["avg_workload"] + urgency_score

    # Choose lowest combined score
    assigned_user = matching_users_filtered.sort_values("combined_score").iloc[0]["assigned_user"]
    return assigned_user, matching_users_filtered

# =====================
# 🔹 Streamlit UI
# =====================
st.title("🧠 AI Task Assignment Dashboard with Cross-Check")

with st.form("task_form"):
    task_desc = st.text_area("📝 Enter Task Description")
    deadline = st.date_input("📅 Deadline", min_value=datetime.date.today())
    submitted = st.form_submit_button("Predict & Assign")

if submitted:
    if not task_desc.strip():
        st.warning("⚠️ Please enter a task description.")
    else:
        # 🔸 Vectorize task description
        task_vector_priority = priority_vectorizer.transform([task_desc])
        task_vector_category = task_vectorizer.transform([task_desc])

        # 🔸 Predict priority
        pred_priority_enc = priority_model.predict(task_vector_priority)[0]
        pred_priority = priority_label_encoder.inverse_transform([pred_priority_enc])[0]

        # 🔸 Predict category
        pred_category_enc = category_model.predict(task_vector_category)[0]
        pred_category = category_label_encoder.inverse_transform([pred_category_enc])[0]

        # 🔸 Compute urgency
        today = datetime.date.today()
        days_left = (deadline - today).days
        urgency_score = max(0, 10 - days_left)

        # 🔸 Assign user
        assigned_user, matching_users_filtered = assign_user_with_check(pred_category, urgency_score)

        # Display results
        if assigned_user != "No available user":
            st.success(f"✅ Task Assigned to: **{assigned_user}**")
            st.info(f"🔺 Priority: **{pred_priority}** | 📁 Category: **{pred_category}** | 🗓 Days to Deadline: {days_left}")
        else:
            st.warning("⚠️ No suitable user found.")

        # =====================
        # 🔹 Cross-Check Feature
        # =====================
        st.subheader("🔍 Cross-Check: Has this user worked in this category?")
        user_past_tasks = df[(df["assigned_user"] == assigned_user) & (df["category"] == pred_category)]
        if not user_past_tasks.empty:
            st.success(f"✅ {assigned_user} has experience in **{pred_category}** ({len(user_past_tasks)} tasks).")
            st.dataframe(user_past_tasks[["task_id", "task_description", "status"]].head(5))
        else:
            st.error(f"⚠️ {assigned_user} has **no prior tasks** in category **{pred_category}**.")

