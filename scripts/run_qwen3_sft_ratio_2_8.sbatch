#!/bin/bash
#SBATCH --job-name=qwen3_ratio_2_8
#SBATCH --partition=coe-gpu
#SBATCH --qos=coe-grade
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=1000G
#SBATCH --time=03:00:00
#SBATCH --output=/storage/ice-shared/ae8803che/hxue/data/eccv/VideoMRIVQA/logs/qwen3_ratio_2_8_%j.log
#SBATCH --error=/storage/ice-shared/ae8803che/hxue/data/eccv/VideoMRIVQA/logs/qwen3_ratio_2_8_%j.err

# Load modules and activate env
source ~/.bashrc
enter_project_vlm
export TRANSFORMERS_VIDEO_BACKEND="av"

# Ensure logs directory exists
mkdir -p /storage/ice-shared/ae8803che/hxue/data/eccv/VideoMRIVQA/logs/

# Start nvitop monitoring in the background
/storage/ice-shared/ae8803che/hxue/data/eccv/VideoMRIVQA/.venv/bin/nvitop --mode monitor --interval 5 --logger-format text --logger-file /storage/ice-shared/ae8803che/hxue/data/eccv/VideoMRIVQA/logs/gpu_usage_2_8_%j.txt &
MONITOR_PID=$!

# Run the SFT script
cd /storage/ice-shared/ae8803che/hxue/data/eccv/VideoMRIVQA/scripts/
chmod +x qwen3_vl_8b_brainsknees_combined_ratio_2_8.sh
./qwen3_vl_8b_brainsknees_combined_ratio_2_8.sh

# Kill monitoring after training
kill $MONITOR_PID
