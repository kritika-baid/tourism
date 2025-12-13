import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# ------------------ CONFIG ------------------
MODEL_REPO = "kritika25/tourismmodel"
MODEL_FILE = "best_model.joblib"

st.set_page_config(
    page_title="Tourism Conversion Predictor",
    page_icon="🌍",
    layout="centered"
)

# ------------------ LOAD MODEL ------------------
st.title("🌍 Tourism Package Conversion Prediction")
st.markdown(
    "This app predicts whether a customer is **likely to purchase a tourism package** "
    "based on their profile and interaction details."
)

model_path = hf_hub_download(
    repo_id=MODEL_REPO,
    filename=MODEL_FILE,
    repo_type="model"
)

model = joblib.load(model_path)
FEATURES = list(model.feature_names_in_)

# ------------------ SIDEBAR INPUT ------------------
st.sidebar.header("🧾 Customer Information")

def user_input():
    data = {}

    st.sidebar.subheader("👤 Personal Details")
    data["Age"] = st.sidebar.slider("Age", 18, 80, 30)
    data["Gender"] = st.sidebar.selectbox("Gender", ["Male", "Female"])
    data["MaritalStatus"] = st.sidebar.selectbox(
        "Marital Status", ["Single", "Married", "Divorced"]
    )

    st.sidebar.subheader("🌆 Location & Travel")
    data["CityTier"] = st.sidebar.selectbox("City Tier", [1, 2, 3])
    data["NumberOfTrips"] = st.sidebar.slider("Number of Trips", 0, 20, 2)
    data["Passport"] = st.sidebar.selectbox("Has Passport?", ["No", "Yes"])

    st.sidebar.subheader("📞 Sales Interaction")
    data["DurationOfPitch"] = st.sidebar.slider("Pitch Duration (mins)", 0, 60, 10)
    data["NumberOfFollowups"] = st.sidebar.slider("Follow-ups", 0, 10, 2)
    data["PitchSatisfactionScore"] = st.sidebar.selectbox(
        "Pitch Satisfaction Score", [1, 2, 3, 4, 5]
    )

    st.sidebar.subheader("💰 Financial Details")
    data["MonthlyIncome"] = st.sidebar.number_input(
        "Monthly Income", min_value=5000, max_value=300000, value=50000
    )
    data["OwnCar"] = st.sidebar.selectbox("Owns a Car?", ["No", "Yes"])

    # ------------------ ENCODING ------------------
    data["Gender"] = 1 if data["Gender"] == "Male" else 0
    data["Passport"] = 1 if data["Passport"] == "Yes" else 0
    data["OwnCar"] = 1 if data["OwnCar"] == "Yes" else 0

    # Fill missing model features safely
    final_data = {}
    for col in FEATURES:
        final_data[col] = data.get(col, 0)

    return pd.DataFrame([final_data])

input_df = user_input()

# ------------------ PREDICTION ------------------
st.markdown("---")

if st.button("🔮 Predict Conversion"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("📊 Prediction Result")

    if prediction == 1:
        st.success(
            f"✅ **Customer is likely to purchase the package**\n\n"
            f"📈 Probability: **{probability:.2%}**"
        )
    else:
        st.error(
            f" **Customer is unlikely to purchase the package**\n\n"
            f"📉 Probability: **{probability:.2%}**"
        )

# ------------------ DEBUG VIEW ------------------
with st.expander("🔍 View Model Input Data"):
    st.dataframe(input_df)
