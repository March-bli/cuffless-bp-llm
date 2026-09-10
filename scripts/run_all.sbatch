#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --job-name=bishe_full
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bli92@sheffield.ac.uk
#SBATCH --output=/users/acp25bl/bishe/full_%j.out
#SBATCH --error=/users/acp25bl/bishe/full_%j.err

echo "=== 完整 Pipeline 开始 $(date) ==="

cd /users/acp25bl/bishe

# 0. 安装 h5py 和 xgboost（GPU 节点 Python 3.9）
echo "[0/5] 安装依赖..."
python3 -m pip install --user h5py xgboost scikit-learn -q 2>&1 | tail -1

# 1. 特征提取
echo "[1/5] 特征提取开始 $(date)"
python3 build_feature_dataset.py
echo "[1/5] 特征提取完成 $(date)"

# 2. 传统 ML 基线
echo "[2/5] 传统 ML 开始 $(date)"
python3 ml_baseline_full.py
echo "[2/5] 传统 ML 完成 $(date)"

# 3. LLM few-shot 消融
echo "[3/5] LLM 消融实验开始 $(date)"
python3 llm_bp_local.py
echo "[3/5] LLM 消融完成 $(date)"

# 4. 语义描述实验
echo "[4/5] 语义描述实验开始 $(date)"
python3 llm_bp_semantic.py
echo "[4/5] 语义描述完成 $(date)"

# 5. 语义瓶颈实验
echo "[5/5] 语义瓶颈实验开始 $(date)"
python3 llm_bp_bottleneck.py
echo "[5/5] 语义瓶颈完成 $(date)"

echo "=== 完整 Pipeline 结束 $(date) ==="
