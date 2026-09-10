#!/bin/bash
# PulseDB 完整数据下载（Box 分卷，替代失效的 SharePoint）
set -u
DEST=/mnt/parscratch/users/acp25bl/pulsedb/box
LOG=$DEST/download.log
mkdir -p "$DEST"
cd "$DEST"

download() {
    local name="$1" url="$2"
    if [ -f "$name" ]; then
        local sz=$(stat -c%s "$name" 2>/dev/null || echo 0)
        echo "$(date +%T) $name 已存在 ($sz bytes)" >> "$LOG"
    else
        echo "$(date +%T) 下载 $name ..." >> "$LOG"
        curl -L -C - --max-time 3600 -o "$name" "$url" >> "$LOG" 2>&1
        echo "$(date +%T) $name 完成 ($(stat -c%s "$name") bytes)" >> "$LOG"
    fi
}

# MIMIC 16 卷
download "PulseDB_MIMIC.zip.001" "https://rutgers.box.com/shared/static/7l8n3tn9tr0602tdss1x7e3uliahlibp.001"
download "PulseDB_MIMIC.zip.002" "https://rutgers.box.com/shared/static/zco48rvz5dog72970679foen6hct15c8.002"
download "PulseDB_MIMIC.zip.003" "https://rutgers.box.com/shared/static/x22qpmelx6sz3wgkm5qyc0eis429361f.003"
download "PulseDB_MIMIC.zip.004" "https://rutgers.box.com/shared/static/xj25sqnluiz6s4z8tzzm5phk00ohp6e8.004"
download "PulseDB_MIMIC.zip.005" "https://rutgers.box.com/shared/static/dxus2lsoop02chaspnwipwrf0g4wmenr.005"
download "PulseDB_MIMIC.zip.006" "https://rutgers.box.com/shared/static/rts6sj441laenm2sy1qcemg7ke4om3j6.006"
download "PulseDB_MIMIC.zip.007" "https://rutgers.box.com/shared/static/vor4hjllld7a0c3nzef8uptbb4ut3koo.007"
download "PulseDB_MIMIC.zip.008" "https://rutgers.box.com/shared/static/a2qg2p4ebyrooji3z88djlokji65tlf3.008"
download "PulseDB_MIMIC.zip.009" "https://rutgers.box.com/shared/static/uh6kbiuqgnib5wakiv6o35gkpusyamc7.009"
download "PulseDB_MIMIC.zip.010" "https://rutgers.box.com/shared/static/h6eyhkkx48pf3ce3th1clwj43hn98j5c.010"
download "PulseDB_MIMIC.zip.011" "https://rutgers.box.com/shared/static/e93dp94hxpkas45yc59n289s2wvkafgi.011"
download "PulseDB_MIMIC.zip.012" "https://rutgers.box.com/shared/static/iuvyuw7dmlxvbjvt53dj49wqn3gelqni.012"
download "PulseDB_MIMIC.zip.013" "https://rutgers.box.com/shared/static/qxx6tjz8c3778601ib3icu6o1rranmc7.013"
download "PulseDB_MIMIC.zip.014" "https://rutgers.box.com/shared/static/ip2ninwqj8437l9fyffjprnk90ptnx9k.014"
download "PulseDB_MIMIC.zip.015" "https://rutgers.box.com/shared/static/yrtbo0lg8mjhaw624iw9bbhk1obbocwd.015"
download "PulseDB_MIMIC.zip.016" "https://rutgers.box.com/shared/static/wmzndowgfa5xi3tvtqahxkld3ngdyjds.016"

# Vital 10 卷
download "PulseDB_Vital.zip.001" "https://rutgers.box.com/shared/static/vtxoksmn7emeaxypb2prywgwscuefoqa.001"
download "PulseDB_Vital.zip.002" "https://rutgers.box.com/shared/static/euzkek7c3xoy62jisheuxqar7z5y8xig.002"
download "PulseDB_Vital.zip.003" "https://rutgers.box.com/shared/static/49lngo0benxfjw193jnqz9tctlyb3qam.003"
download "PulseDB_Vital.zip.004" "https://rutgers.box.com/shared/static/jf4fwgkmhry20mf5tcg9t0wxvky64um0.004"
download "PulseDB_Vital.zip.005" "https://rutgers.box.com/shared/static/2lgxysbskfuapsaan4jypvmm8316fdkc.005"
download "PulseDB_Vital.zip.006" "https://rutgers.box.com/shared/static/x27ktb4qsx43razwo4tjmxq9v1ro0x3y.006"
download "PulseDB_Vital.zip.007" "https://rutgers.box.com/shared/static/q0t36fikgf3pimhvnerwwnovfr0umtp8.007"
download "PulseDB_Vital.zip.008" "https://rutgers.box.com/shared/static/ihckx2g0f981g5yz2x8v5rgwndl6yebw.008"
download "PulseDB_Vital.zip.009" "https://rutgers.box.com/shared/static/y8j14h8tvi5b3du8nap9dnura1omfrk6.009"
download "PulseDB_Vital.zip.010" "https://rutgers.box.com/shared/static/fu0m9tx33jkxywq32shh0g8dg3not15u.010"

echo "=== 所有分卷下载完成 $(date) ===" >> "$LOG"
