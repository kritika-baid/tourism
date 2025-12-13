import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# ---------------- CONFIG ----------------
MODEL_REPO = "kritika25/tourismmodel"
MODEL_FILE = "best_model.joblib"

st.set_page_config(
    page_title="Tourism Package Conversion Predictor",
    page_icon="🌍",
    layout="centered"
)

st.title("🌍 Tourism Package Conversion Prediction")
st.markdown(
    "Predict whether a customer is likely to purchase a tourism package "
    "based on their profile and interaction details."
)

# ---------------- LOAD MODEL ----------------
model_path = hf_hub_download(
    repo_id=MODEL_REPO,
    filename=MODEL_FILE,
    repo_type="model"
)
model = joblib.load(model_path)

# ---------------- GET FEATURES ----------------
preprocessor = model.named_steps.get("columntransformer") or model.named_steps.get("preprocessor")
numeric_cols = list(preprocessor.transformers_[0][2])
categorical_cols = list(preprocessor.transformers_[1][2])
expected_cols = numeric_cols + categorical_cols

# ---------------- UI DEFINITIONS ----------------
DROPDOWNS = {
    "Gender": ["Female", "Male"],
    "TypeofContact": ["Self Enquiry", "Company Invited"],
    "Occupation": ["Salaried", "Small Business", "Free Lancer"],
    "ProductPitched": ["Basic", "Standard", "Deluxe"],
    "MaritalStatus": ["Single", "Married", "Divorced", "Unmarried"],
    "Designation": ["Executive", "Manager", "Senior Manager"],
    "CityTier": [1, 2, 3],
    "PreferredPropertyStar": [1, 2, 3, 4, 5],
    "PitchSatisfactionScore": [1, 2, 3, 4, 5],
    "Passport": [0, 1],
    "OwnCar": [0, 1],
}

# ------------------ SIDEBAR INPUT ------------------
st.sidebar.header("Customer Information")

ENCODED_DROPDOWNS = {
    "Gender": {"Female": 0, "Male": 1},
    "TypeofContact": {"Self Enquiry": 0, "Company Invited": 1},
    "Occupation": {"Salaried": 0, "Small Business": 1, "Free Lancer": 2},
    "ProductPitched": {"Basic": 0, "Standard": 1, "Deluxe": 2},
    "MaritalStatus": {"Single": 0, "Married": 1, "Divorced": 2, "Unmarried": 3},
}

input_data = {}

for col in expected_cols:

    # AGE with minimum value
    if col == "Age":
        input_data[col] = st.sidebar.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=30
        )

    # Other numeric fields
    elif col in numeric_cols:
        input_data[col] = st.sidebar.number_input(
            col,
            value=0
        )

    # Categorical fields (already handled via dropdowns earlier)
    else:
        input_data[col] = st.sidebar.text_input(col, "Unknown")

input_df = pd.DataFrame([input_data])


# ---------------- PREDICTION ----------------
st.markdown("---")

if st.button("Predict Conversion"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.success(f" Customer is likely to purchase the package\n\nProbability: {probability:.2%}")
    else:
        st.error(f" Customer is unlikely to purchase the package\n\nProbability: {probability:.2%}")
