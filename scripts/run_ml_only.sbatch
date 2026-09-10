#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=03:00:00
#SBATCH --job-name=ml_baseline
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bli92@sheffield.ac.uk
#SBATCH --output=/users/acp25bl/bishe/ml_%j.out
#SBATCH --error=/users/acp25bl/bishe/ml_%j.err

echo "=== 传统 ML 基线开始 $(date) ==="
cd /users/acp25bl/bishe
python3 ml_baseline_full.py
echo "=== 传统 ML 完成 $(date) ==="
