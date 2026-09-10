#!/bin/bash
# 等待实验作业 11339007 完成，然后发邮件通知
JOB=11339007
RESULT_FILE=/users/acp25bl/bishe/llm_bp_${JOB}.out

# 轮询作业状态
while squeue -j $JOB 2>/dev/null | grep -q $JOB; do
    sleep 60
done
sleep 10

{
echo "Hi Boyuan,"
echo ""
echo "你的 LLM-BP 实验（作业 $JOB）已经在 HPC 上跑完了。"
echo ""
echo "=== 结果摘要 ==="
grep -E "zero-shot|few-shot|SBP MAE|DBP MAE|Saved|Done!" "$RESULT_FILE" | tail -25
echo ""
echo "=== 文件位置（HPC） ==="
echo "完整输出: $RESULT_FILE"
echo "结果 JSON: /users/acp25bl/bishe/llm_bp_local_results.json"
echo ""
echo "Best regards,"
echo "HPC Auto-Notifier"
} | mail -s "LLM-BP 实验完成通知 (Job $JOB)" bli92@sheffield.ac.uk

echo "$(date) 邮件已发送" >> /users/acp25bl/bishe/notify_done.log
