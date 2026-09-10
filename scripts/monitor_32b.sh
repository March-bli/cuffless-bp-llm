#!/bin/bash
# 等 32B 模型下载完成，自动提交实验
LOG=/users/acp25bl/bishe/dl_32b.log
MON=/users/acp25bl/bishe/monitor_32b.log
echo "=== 32B 监控开始 $(date) ===" >> $MON

# 等下载完成（最多 3 小时）
for i in $(seq 1 180); do
    if grep -q "MODEL_32B_DOWNLOAD_DONE" "$LOG" 2>/dev/null; then
        echo "下载完成 $(date)" >> $MON
        JOB=$(sbatch /users/acp25bl/bishe/run_32b.sbatch | awk '{print $4}')
        echo "已提交 32B 实验: $JOB $(date)" >> $MON
        exit 0
    fi
    sleep 60
done
echo "下载超时（3小时）$(date)" >> $MON
echo "Hi Boyuan, 32B 模型下载超时，请检查 HPC 日志 dl_32b.log" | mail -s "32B 下载超时" bli92@sheffield.ac.uk
