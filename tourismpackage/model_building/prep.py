import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from huggingface_hub import HfApi, login

# Login to HuggingFace (token must be set as environment variable)
api = HfApi(token=os.getenv("HF_TOKEN"))

# Dataset path from HuggingFace or local
DATASET_PATH = "hf://datasets/kritika25/tourismproject/tourism.csv"

# Load dataset
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Drop unnecessary columns
df.drop(columns=['Unnamed: 0', 'CustomerID'], inplace=True)

# Encode all categorical variables
label_encoder = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = label_encoder.fit_transform(df[col])

# Set the target column
target_col = "ProdTaken"

# Split into features X and target y
X = df.drop(columns=[target_col])
y = df[target_col]

# Train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Save files
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Train-test splits saved.")

# List of files to upload
files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

# Upload to HuggingFace repo
for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path,   # upload under same name
       repo_id="kritika25/tourismproject",
      repo_type="dataset"

    )

print("All files uploaded successfully.")
