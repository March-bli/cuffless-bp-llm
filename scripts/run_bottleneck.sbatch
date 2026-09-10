#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --job-name=llm_bp_bn
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bli92@sheffield.ac.uk
#SBATCH --output=/users/acp25bl/bishe/bn_%j.out
#SBATCH --error=/users/acp25bl/bishe/bn_%j.err

echo "=== 作业开始 $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python3 /users/acp25bl/bishe/llm_bp_bottleneck.py

echo "=== 作业结束 $(date) ==="
