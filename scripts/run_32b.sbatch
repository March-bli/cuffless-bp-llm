#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --job-name=llm_32b
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bli92@sheffield.ac.uk
#SBATCH --output=/users/acp25bl/bishe/llm32b_%j.out
#SBATCH --error=/users/acp25bl/bishe/llm32b_%j.err

echo "=== 32B 实验开始 $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd /users/acp25bl/bishe
python3 llm_bp_32b.py
echo "=== 32B 实验结束 $(date) ==="
