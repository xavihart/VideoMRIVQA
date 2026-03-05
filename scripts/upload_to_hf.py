import os
from huggingface_hub import HfApi, create_repo

# Configuration
LOCAL_DIR = "/storage/ice-shared/ae8803che/hxue/data/eccv/VideoMRIVQA/checkpoints/qwen3_vl_8b_brains_combined_full/checkpoint-2000"
REPO_NAME = "qwen-mri-step2000"
USERNAME = "t1an"
REPO_ID = f"{USERNAME}/{REPO_NAME}"

api = HfApi()

print(f"Creating repository: {REPO_ID}...")
try:
    create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)
    print("Repository created or already exists.")
except Exception as e:
    print(f"Error creating repository: {e}")

print(f"Uploading files from {LOCAL_DIR} to {REPO_ID}...")

# Files to ignore (usually training states not needed for inference)
IGNORE_PATTERNS = [
    "rng_state_*.pth",
    "optimizer.pt",
    "scheduler.pt",
    "trainer_state.json",
    "training_args.bin",
    "global_step*",
    "latest",
    "zero_to_fp32.py"
]

try:
    api.upload_folder(
        folder_path=LOCAL_DIR,
        repo_id=REPO_ID,
        repo_type="model",
        ignore_patterns=IGNORE_PATTERNS
    )
    print(f"Successfully uploaded checkpoint to https://huggingface.co/{REPO_ID}")
except Exception as e:
    print(f"Error uploading files: {e}")
