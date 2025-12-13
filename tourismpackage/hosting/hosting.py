from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="tourismpackage/deployment",     # the local folder containing your files
    repo_id="kritika25/tourismpackage",          # the target repo
    repo_type="space",                      # dataset, model, or space
)
