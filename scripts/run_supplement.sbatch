#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --job-name=supplement
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bli92@sheffield.ac.uk
#SBATCH --output=/users/acp25bl/bishe/supp_%j.out
#SBATCH --error=/users/acp25bl/bishe/supp_%j.err

echo "=== 补充实验开始 $(date) ==="
cd /users/acp25bl/bishe

echo "[1/2] XGBoost K 校准开始 $(date)"
python3 ml_calibration_full.py
echo "[1/2] XGBoost K 校准完成 $(date)"

echo "[2/2] 语义描述 K 消融开始 $(date)"
python3 llm_bp_semantic_ablation.py
echo "[2/2] 语义描述 K 消融完成 $(date)"

echo "=== 补充实验结束 $(date) ==="
