from huggingface_hub import snapshot_download
import sys

try:
    snapshot_download(
        repo_id="Qwen/Qwen3-VL-8B-Instruct",
        local_dir="/storage/ice-shared/ae8803che/hxue/data/checkpoint/Qwen3-VL-8B-Instruct",
        ignore_patterns=["*.msgpack", "*.h5", "*.tflite", "*.ot"]
    )
    print("Download successful")
except Exception as e:
    print(f"Error during download: {e}")
    sys.exit(1)
