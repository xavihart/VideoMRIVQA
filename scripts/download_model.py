import os
from huggingface_hub import snapshot_download

model_id = "Qwen/Qwen3-VL-8B-Instruct"
local_dir = "/storage/ice-shared/ae8803che/hxue/data/checkpoint/Qwen3-VL-8B-Instruct"

print(f"Starting download of {model_id} to {local_dir}...")
os.makedirs(local_dir, exist_ok=True)

snapshot_download(
    repo_id=model_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    resume_download=True
)

print("Download complete!")
