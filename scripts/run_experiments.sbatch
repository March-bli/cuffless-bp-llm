#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --job-name=bishe_exp
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bli92@sheffield.ac.uk
#SBATCH --output=/users/acp25bl/bishe/exp_%j.out
#SBATCH --error=/users/acp25bl/bishe/exp_%j.err

echo "=== 实验 Pipeline 开始 $(date) ==="
cd /users/acp25bl/bishe

# 特征已存在（/mnt/parscratch/users/acp25bl/processed/pulsedb_features_full.npz），跳过特征提取

echo "[1/4] 传统 ML 开始 $(date)"
python3 ml_baseline_full.py
echo "[1/4] 传统 ML 完成 $(date)"

echo "[2/4] LLM few-shot 消融开始 $(date)"
python3 llm_bp_local.py
echo "[2/4] LLM 消融完成 $(date)"

echo "[3/4] 语义描述实验开始 $(date)"
python3 llm_bp_semantic.py
echo "[3/4] 语义描述完成 $(date)"

echo "[4/4] 语义瓶颈实验开始 $(date)"
python3 llm_bp_bottleneck.py
echo "[4/4] 语义瓶颈完成 $(date)"

echo "=== 实验 Pipeline 结束 $(date) ==="
