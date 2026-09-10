#!/bin/bash
#SBATCH --dependency=afterany:11339007
#SBATCH --partition=hp-cpu
#SBATCH --time=00:10:00
#SBATCH --job-name=notify_bp
#SBATCH --output=/users/acp25bl/bishe/notify_%j.out

RESULT_FILE=/users/acp25bl/bishe/llm_bp_11339007.out
RESULT_JSON=/users/acp25bl/bishe/llm_bp_local_results.json

{
echo "Hi Boyuan,"
echo ""
echo "你的 LLM-BP 实验（作业 11339007）已经在 HPC 上跑完了。"
echo ""
echo "=== 结果摘要 ==="
grep -E "zero-shot|few-shot|SBP MAE|DBP MAE|Saved|Done!" "$RESULT_FILE" | tail -25
echo ""
echo "=== 文件位置（HPC） ==="
echo "完整输出: $RESULT_FILE"
echo "结果 JSON: $RESULT_JSON"
echo ""
echo "Best regards,"
echo "HPC Auto-Notifier"
} | mail -s "LLM-BP 实验完成通知 (Job 11339007)" bli92@sheffield.ac.uk

echo "邮件已发送"
