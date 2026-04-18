import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# Page config
st.set_page_config(layout="wide")

# ---------------- CSS (Premium Dark UI) ----------------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
}

h1, h2, h3, h4 {
    color: #e2e8f0;
}

.stButton>button {
    background: linear-gradient(90deg, #6366f1, #8b5cf6);
    color: white;
    border-radius: 12px;
    height: 3em;
    width: 100%;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
model = joblib.load("models/xgb_model.pkl")
columns = joblib.load("models/columns.pkl")

# ---------------- TITLE ----------------
st.title("💼 Employee Performance Predictor")

# ---------------- LAYOUT ----------------
left, right = st.columns([1,2])

# ================= LEFT: INPUTS =================
with left:
    st.markdown("## 🧾 Employee Inputs")

    age = st.slider("Age", 20, 60)
    experience = st.slider("Experience (Years)", 1, 20)
    salary = st.number_input("Salary", value=30000)
    training_hours = st.slider("Training Hours", 10, 100)
    department = st.selectbox("Department", ["HR", "Tech", "Sales"])

    if st.button("🚀 Predict Performance"):
        input_data = pd.DataFrame([[age, experience, salary, training_hours,
                                    int(department=="HR"),
                                    int(department=="Tech"),
                                    int(department=="Sales")]],
                                  columns=columns)

        input_data = input_data.reindex(columns=columns, fill_value=0)

        prediction = model.predict(input_data)[0]
        result = ["Low", "Medium", "High"][prediction]

        st.success(f"Predicted Performance: {result}")

# ================= RIGHT: DASHBOARD =================
with right:
    st.markdown("## 📊 Dashboard")

    # KPI Cards
    col1, col2, col3 = st.columns(3)

    col1.metric("Avg Salary", "₹50K", "+5%")
    col2.metric("High Performers", "120", "+12%")
    col3.metric("Training Impact", "High")

    st.markdown("### 📈 Performance Insights")

    # Load data
    data = pd.read_csv("data/employee_data.csv")

    # 🔥 FIX: Convert encoded columns → readable department
    data["department"] = data[[
        "department_HR",
        "department_Tech",
        "department_Sales"
    ]].idxmax(axis=1)

    data["department"] = data["department"].str.replace("department_", "")

    # Plot
    fig = px.histogram(
        data,
        x="performance",
        color="department",
        template="plotly_dark"
    )

    st.plotly_chart(fig, use_container_width=True)