import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

# ---------------- CONFIG ----------------
MODEL_REPO = "kritika25/tourismmodel"
MODEL_FILE = "best_model.joblib"
DATASET_REPO = "kritika25/tourismproject"
TRAIN_CSV = "Xtrain.csv"
LABEL_ENCODERS_FILE = "label_encoders.joblib"

st.set_page_config(page_title="Tourism Conversion Predictor", page_icon="🌍", layout="wide")
st.title("🌍 Tourism Package Conversion Predictor")
st.markdown(
    "Predict whether a customer is **likely to purchase a tourism package** "
    "based on their profile, interactions, and financial details."
)

# ---------------- LOAD MODEL ----------------
model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE, repo_type="model")
model = joblib.load(model_path)

# ---------------- LOAD TRAINING CSV ----------------
train_csv_path = hf_hub_download(repo_id=DATASET_REPO, filename=TRAIN_CSV, repo_type="dataset")
train_df = pd.read_csv(train_csv_path)

# ---------------- LOAD LABEL ENCODERS ----------------
encoders_path = hf_hub_download(repo_id=MODEL_REPO, filename=LABEL_ENCODERS_FILE, repo_type="model")
label_encoders = joblib.load(encoders_path)

# Identify numeric and categorical columns
numeric_cols = train_df.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = list(label_encoders.keys())

# ---------------- SIDEBAR INPUT ----------------
st.sidebar.header("Customer Input")
input_data = {}

# Numeric inputs with min/max from training data
for col in numeric_cols:
    min_val = int(train_df[col].min())
    max_val = int(train_df[col].max())
    mean_val = int(train_df[col].mean())
    input_data[col] = st.sidebar.number_input(col, min_value=min_val, max_value=max_val, value=mean_val, step=1)

# Categorical dropdowns
for col in categorical_cols:
    options = sorted([str(cls) for cls in label_encoders[col].classes_])
    input_data[col] = st.sidebar.selectbox(col, options, index=0)

input_df = pd.DataFrame([input_data])

# ---------------- APPLY LABEL ENCODERS ----------------
for col in categorical_cols:
    le = label_encoders[col]
    input_df[col] = le.transform([input_df[col][0]])

# Ensure numeric columns are float
input_df[numeric_cols] = input_df[numeric_cols].astype(float)

# ---------------- DISPLAY INPUTS ----------------
st.subheader("Customer Input Summary")
st.dataframe(input_df.T.rename(columns={0:"Input"}))

st.markdown("---")

# ---------------- PREDICTION ----------------
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
