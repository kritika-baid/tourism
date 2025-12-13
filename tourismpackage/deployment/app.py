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

st.title("🌍 Tourism Package Conversion Prediction")
st.markdown(
    "Predict whether a customer is **likely to purchase a tourism package** "
    "based on their profile and interaction details."
)

# ------------------ LOAD MODEL ------------------
model_path = hf_hub_download(
    repo_id=MODEL_REPO,
    filename=MODEL_FILE,
    repo_type="model"
)
model = joblib.load(model_path)

# ------------------ GET EXPECTED FEATURES ------------------
preprocessor = model.named_steps.get("columntransformer") or model.named_steps.get("preprocessor")
numeric_cols = list(preprocessor.transformers_[0][2])
categorical_cols = list(preprocessor.transformers_[1][2])

# ------------------ SIDEBAR INPUT ------------------
st.sidebar.header("Customer Information")
input_data = {}

# ---------- Numeric Inputs (NO HARD MAX) ----------
for col in numeric_cols:
    input_data[col] = st.sidebar.number_input(
        label=col,
        min_value=0,
        value=0,
        step=1
    )

# ---------- Categorical Inputs ----------
CATEGORICAL_OPTIONS = {
    "Gender": ["Female", "Male"],
    "MaritalStatus": ["Single", "Married", "Divorced", "Unmarried", "Unknown"],
    "TypeofContact": ["Self Enquiry", "Company Invited", "Unknown"],
    "Occupation": ["Salaried", "Small Business", "Free Lancer", "Unknown"],
    "ProductPitched": ["Basic", "Standard", "Deluxe", "Unknown"],
    "Designation": ["Executive", "Manager", "Senior Manager", "Unknown"],
}

for col in categorical_cols:
    options = CATEGORICAL_OPTIONS.get(col, ["Unknown"])
    input_data[col] = st.sidebar.selectbox(col, options)

input_df = pd.DataFrame([input_data])

# ------------------ PREDICTION ------------------
st.markdown("---")

if st.button("Predict Conversion"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")
    if prediction == 1:
        st.success(f"✅ Customer is likely to purchase the package\n\nProbability: {probability:.2%}")
    else:
        st.error(f"❌ Customer is unlikely to purchase the package\n\nProbability: {probability:.2%}")
