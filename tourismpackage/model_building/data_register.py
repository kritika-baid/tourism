from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os


api = HfApi(token=os.getenv("HF_TOKEN"))

repo_id = "kritika25/tourismproject"

repo_type = "dataset"


api = HfApi(token=os.getenv("HF_TOKEN"))


# Step 1: Check if the space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Space '{repo_id}' created.")

# Step 2: Upload folder
print("Uploading data...")
api.upload_folder(
    folder_path="tourismpackage/data",
    repo_id=repo_id,
    repo_type=repo_type,
)
print("Upload completed!")
