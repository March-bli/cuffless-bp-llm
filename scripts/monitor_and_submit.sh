#!/bin/bash
# 等待 Box 下载完成 → 解压 → 提交完整 Pipeline
LOG=/users/acp25bl/bishe/pipeline_monitor.log
echo "=== 监控开始 $(date) ===" >> $LOG

BOX=/mnt/parscratch/users/acp25bl/pulsedb/box
SEG=/mnt/parscratch/users/acp25bl/pulsedb/Segment_Files

# 等待 Box 下载脚本结束
while pgrep -f "box_download.sh" > /dev/null; do
    sleep 600
done
echo "Box 下载结束 $(date)" >> $LOG

# 检查分卷完整性
N_MIMIC=$(ls $BOX/PulseDB_MIMIC.zip.* 2>/dev/null | wc -l)
N_VITAL=$(ls $BOX/PulseDB_Vital.zip.* 2>/dev/null | wc -l)
echo "MIMIC 分卷: $N_MIMIC/16, Vital 分卷: $N_VITAL/10" >> $LOG

if [ "$N_MIMIC" -ge 16 ] && [ "$N_VITAL" -ge 10 ]; then
    echo "分卷完整，开始解压 $(date)" >> $LOG

    # 解压 MIMIC
    cd $BOX
    cat PulseDB_MIMIC.zip.* > /tmp/MIMIC_full.zip 2>>$LOG
    mkdir -p $SEG/PulseDB_MIMIC
    unzip -o -q /tmp/MIMIC_full.zip -d $SEG/PulseDB_MIMIC/ >> $LOG 2>&1
    rm -f /tmp/MIMIC_full.zip
    echo "MIMIC 解压完成 $(date)" >> $LOG

    # 解压 Vital
    cat PulseDB_Vital.zip.* > /tmp/Vital_full.zip 2>>$LOG
    mkdir -p $SEG/PulseDB_Vital
    unzip -o -q /tmp/Vital_full.zip -d $SEG/PulseDB_Vital/ >> $LOG 2>&1
    rm -f /tmp/Vital_full.zip
    echo "Vital 解压完成 $(date)" >> $LOG

    # 统计文件数
    N=$(find $SEG -name "*.mat" 2>/dev/null | wc -l)
    echo "解压后 .mat 文件数: $N" >> $LOG

    if [ "$N" -ge 4500 ]; then
        JOB=$(sbatch /users/acp25bl/bishe/run_all.sbatch | awk '{print $4}')
        echo "文件数达标，提交 Pipeline: $JOB $(date)" >> $LOG
    else
        echo "❌ 文件数不足（$N < 4500）" >> $LOG
        echo "Hi Boyuan, Box 下载解压后文件数不足（$N），请检查。" | mail -s "PulseDB 解压异常" bli92@sheffield.ac.uk
    fi
else
    echo "❌ 分卷不完整（MIMIC $N_MIMIC, Vital $N_VITAL）" >> $LOG
    echo "Hi Boyuan, Box 分卷下载不完整（MIMIC $N_MIMIC/16, Vital $N_VITAL/10）。" | mail -s "PulseDB 下载不完整" bli92@sheffield.ac.uk
fi
