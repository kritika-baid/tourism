import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# ------------------ CONFIG ------------------
MODEL_REPO = "kritika25/tourismmodel"
MODEL_FILE = "best_model.joblib"
DATASET_REPO = "kritika25/tourismproject"
TRAIN_CSV = "Xtrain.csv"

st.set_page_config(page_title="Tourism Conversion Predictor", page_icon="🌍", layout="wide")
st.title("🌍 Tourism Package Conversion Predictor")
st.markdown(
    "Predict whether a customer is **likely to purchase a tourism package** "
    "based on their profile, interactions, and financial details."
)

# ------------------ LOAD MODEL ------------------
model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE, repo_type="model")
model = joblib.load(model_path)

# ------------------ LOAD TRAINING CSV ------------------
train_csv_path = hf_hub_download(repo_id=DATASET_REPO, filename=TRAIN_CSV, repo_type="dataset")
train_df = pd.read_csv(train_csv_path)

# Identify numeric and categorical columns
numeric_cols = train_df.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = train_df.select_dtypes(include=["object"]).columns.tolist()

# ------------------ SIDEBAR INPUT ------------------
st.sidebar.header("Customer Input")
input_data = {}

# Numeric sliders with dynamic min/max from training CSV
for col in numeric_cols:
    min_val = int(train_df[col].min())
    max_val = int(train_df[col].max())
    mean_val = int(train_df[col].mean())
    input_data[col] = st.sidebar.number_input(col, min_value=min_val, max_value=max_val, value=mean_val, step=1)

# Categorical dropdowns from training CSV
for col in categorical_cols:
    options = sorted(train_df[col].dropna().unique())
    input_data[col] = st.sidebar.selectbox(col, options, index=0)

input_df = pd.DataFrame([input_data])

# ------------------ MAIN UI TABS ------------------
tab1, tab2, tab3 = st.tabs(["Personal Info", "Interaction Info", "Financial Info"])

with tab1:
    st.subheader("Personal Information")
    personal_cols = ["Age", "Gender", "MaritalStatus", "NumberOfChildrenVisiting"]
    personal_cols = [c for c in personal_cols if c in input_df.columns]
    st.dataframe(input_df[personal_cols].T.rename(columns={0:"Input"}))

with tab2:
    st.subheader("Interaction Information")
    interaction_cols = ["TypeofContact", "CityTier", "DurationOfPitch",
                        "Occupation", "NumberOfPersonVisiting", "NumberOfFollowups",
                        "ProductPitched", "PreferredPropertyStar", "PitchSatisfactionScore",
                        "Designation"]
    interaction_cols = [c for c in interaction_cols if c in input_df.columns]
    st.dataframe(input_df[interaction_cols].T.rename(columns={0:"Input"}))

with tab3:
    st.subheader("Financial Information")
    financial_cols = ["MonthlyIncome", "Passport", "ProdTaken", "NumberOfTrips", "OwnCar"]
    financial_cols = [c for c in financial_cols if c in input_df.columns]
    st.dataframe(input_df[financial_cols].T.rename(columns={0:"Input"}))

st.markdown("---")

# ------------------ PREDICTION ------------------
if st.button("Predict Conversion"):
    try:
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        st.subheader("Prediction Result")
        if prediction == 1:
            st.success(f" Customer is likely to purchase the package\nProbability: {probability:.2%}")
        else:
            st.error(f"Customer is unlikely to purchase the package\nProbability: {probability:.2%}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
