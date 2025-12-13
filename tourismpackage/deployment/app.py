import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

#  Load Model From Hugging Face

MODEL_REPO = "kritika25/tourismmodel"
MODEL_FILE = "best_model.joblib"

st.title("Tourism Customer Conversion Prediction App")
st.write("Predict whether a customer will purchase the tourism package.")

# Download model
model_path = hf_hub_download(
    repo_id=MODEL_REPO,
    filename=MODEL_FILE,
    repo_type="model"
)

model = joblib.load(model_path)


FEATURES = list(model.feature_names_in_)

st.sidebar.header("Enter Customer Details")

def user_input():
    values = {}

    for col in FEATURES:
        # Numeric fields
        if any(keyword in col.lower() for keyword in 
               ["age", "duration", "number", "monthly", "income", "score"]):
            values[col] = st.sidebar.number_input(col, min_value=0, value=1)

        # Binary yes/no
        elif any(keyword in col.lower() for keyword in 
                 ["passport", "owncar", "owns_car", "car"]):
            values[col] = st.sidebar.selectbox(col, [0, 1])

        # Gender
        elif col.lower() == "gender":
            gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
            values[col] = 1 if gender == "Male" else 0

        # Generic categorical fields
        else:
            values[col] = st.sidebar.text_input(col, "")

    return pd.DataFrame([values])

input_df = user_input()


if st.button("Predict"):
    try:
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        st.subheader("🔍 Prediction Result")

        if prediction == 1:
            st.success(f"Customer WILL take the package (Probability: {probability:.2f})")
        else:
            st.error(f" Customer will NOT take the package (Probability: {probability:.2f})")

    except Exception as e:
        st.error(f"Prediction error: {e}")


st.subheader(" User Input Summary")
st.write(input_df)
