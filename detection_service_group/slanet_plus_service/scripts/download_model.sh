#!/usr/bin/env bash
# 下载 SLANet_plus 推理模型到本地 ./models/SLANet_plus
#
# 前置：pip install modelscope
# 说明：模型很小（inference.pdiparams 约 7.3MB），下载很快
set -euo pipefail

MODEL_ID="PaddlePaddle/SLANet_plus"
TARGET_DIR="${1:-${LOCAL_MODEL_DIR:-./models/SLANet_plus}}"

mkdir -p "${TARGET_DIR}"

# ModelScope 官方 CLI 会在当前目录建出同名子目录，这里用 --local_dir 直接落到目标目录
if command -v modelscope >/dev/null 2>&1; then
    modelscope download --model "${MODEL_ID}" --local_dir "${TARGET_DIR}"
else
    echo "未找到 modelscope 命令，正在通过 pip 安装..." >&2
    python3 -m pip install -U modelscope
    python3 -m modelscope.cli.cli download --model "${MODEL_ID}" --local_dir "${TARGET_DIR}"
fi

echo "下载完成，目录内容："
ls -lh "${TARGET_DIR}"

python3 "$(dirname "$0")/check_model.py" "${TARGET_DIR}"
