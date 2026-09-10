#!/bin/bash
# 在 GPU 节点上安装 Python 3.9 依赖
set -e
echo "=== 升级 pip ==="
python3 -m pip install --user --upgrade pip 2>&1 | tail -2

echo "=== 安装 torch + transformers ==="
python3 -m pip install --user torch transformers numpy scipy 2>&1 | tail -5

echo "=== 验证 ==="
python3 -c "import torch; print('torch', torch.__version__, 'CUDA', torch.cuda.is_available())"
python3 -c "import transformers; print('transformers', transformers.__version__)"
echo "INSTALL_DONE"
