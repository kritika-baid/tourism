import os
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier

from huggingface_hub import HfApi, HfFolder
from huggingface_hub.utils import RepositoryNotFoundError


# ------------------------ CONFIG ------------------------

HF_TOKEN = os.getenv("HF_TOKEN")   # MUST be set in Colab
DATA_REPO = "kritika25/tourismproject"
MODEL_REPO = "kritika25/tourismmodel"

HfFolder.save_token(HF_TOKEN)
api = HfApi(token=HF_TOKEN)


# ------------------------ HF MODEL REPO ------------------------

try:
    api.repo_info(repo_id=MODEL_REPO, repo_type="model")
    print("✓ Model repo exists")
except RepositoryNotFoundError:
    print("✗ Model repo missing → creating...")
    api.create_repo(
        repo_id=MODEL_REPO,
        repo_type="model",
        private=False
    )
    print("✓ Model repo created!")


# ------------------------ MLflow Setup ------------------------

mlflow.set_tracking_uri("mlruns")
mlflow.set_experiment("tourism_model_experiments")


# ------------------------ LOAD DATA ------------------------

print("Loading dataset from HuggingFace Hub...")

Xtrain = pd.read_csv(f"hf://datasets/{DATA_REPO}/Xtrain.csv")
Xtest  = pd.read_csv(f"hf://datasets/{DATA_REPO}/Xtest.csv")
ytrain = pd.read_csv(f"hf://datasets/{DATA_REPO}/ytrain.csv")
ytest  = pd.read_csv(f"hf://datasets/{DATA_REPO}/ytest.csv")

print("✓ Data loaded successfully!")


# ------------------------ PIPELINE ------------------------

numeric_cols = Xtrain.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = Xtrain.select_dtypes(include=['object']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]
)

print("✓ Preprocessing pipeline created!")


# ------------------------ MODELS ------------------------

models = {
    "decision_tree": {
        "model": DecisionTreeClassifier(),
        "params": {"decisiontreeclassifier__max_depth": [3, 5, 7, None]}
    },
    "bagging": {
        "model": BaggingClassifier(),
        "params": {"baggingclassifier__n_estimators": [10, 30, 50]}
    },
    "random_forest": {
        "model": RandomForestClassifier(),
        "params": {
            "randomforestclassifier__n_estimators": [50, 100],
            "randomforestclassifier__max_depth": [5, 10, None]
        }
    }
}

best_model = None
best_score = 0
best_name = ""


# ------------------------ TRAINING LOOP ------------------------

for name, cfg in models.items():

    with mlflow.start_run(run_name=name):

        print(f"\n Training Model: {name}")

        # FULL PIPELINE = preprocess + model
        pipe = make_pipeline(preprocessor, cfg["model"])

        grid = GridSearchCV(
            estimator=pipe,
            param_grid=cfg["params"],
            cv=3,
            scoring="accuracy"
        )

        grid.fit(Xtrain, ytrain.values.ravel())

        preds = grid.predict(Xtest)
        acc = accuracy_score(ytest, preds)

        # Log best params + metrics
        mlflow.log_params(grid.best_params_)
        mlflow.log_metric("accuracy", acc)

        print(f"✓ {name} Accuracy = {acc}")

        # Track best model
        if acc > best_score:
            best_score = acc
            best_model = grid.best_estimator_
            best_name = name

        # Save model in MLflow
        mlflow.sklearn.log_model(grid.best_estimator_, name)


# ------------------------ SAVE + UPLOAD BEST MODEL ------------------------

print(f"\n Best Model = {best_name} | Accuracy = {best_score}")

joblib.dump(best_model, "best_model.joblib")

api.upload_file(
    path_or_fileobj="best_model.joblib",
    path_in_repo="best_model.joblib",
    repo_id=MODEL_REPO,
    repo_type="model"
)

print( "Best model successfully uploaded to HuggingFace Hub!")
