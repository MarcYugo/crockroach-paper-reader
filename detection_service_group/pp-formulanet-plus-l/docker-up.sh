#!/usr/bin/env bash
# 一键部署（PP-FormulaNet_plus-L）
#
# 自动判断宿主机有没有可用的 NVIDIA GPU，决定是否叠加 GPU 直通文件：
#   * 有 GPU → docker-compose.yaml + docker-compose.gpu.yaml
#   * 无 GPU → 只跑 docker-compose.yaml，容器内会自动用 CPU
#
# 用法：
#   ./docker-up.sh                 # 构建并后台启动
#   FORCE_CPU=1 ./docker-up.sh     # 强制按 CPU 部署（即使机器上有 GPU）
#   ./docker-up.sh --build         # 额外参数会透传给 docker compose
set -euo pipefail

cd "$(dirname "$0")"

COMPOSE_FILES=(-f docker-compose.yaml)

if [ "${FORCE_CPU:-0}" != "1" ] \
  && command -v nvidia-smi >/dev/null 2>&1 \
  && nvidia-smi -L >/dev/null 2>&1; then
  echo "[docker-up] 检测到 NVIDIA GPU，叠加 docker-compose.gpu.yaml"
  COMPOSE_FILES+=(-f docker-compose.gpu.yaml)
else
  echo "[docker-up] 未检测到可用 NVIDIA GPU，按 CPU 方式启动（应用会自动选择 cpu）"
fi

exec docker compose "${COMPOSE_FILES[@]}" up -d --build "$@"
