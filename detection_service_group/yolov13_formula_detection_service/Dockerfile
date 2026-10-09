# syntax=docker/dockerfile:1

# =============================================================================
# YOLOv13 公式检测服务镜像
#
# 功能：接收 PDF 渲染后的页面图片，输出检测框、类别、置信度以及框内区域图像（JSON）。
# 推理逻辑复用仓库根目录的 inference.py。
#
# 构建上下文为当前仓库根目录，因为镜像内需要 inference.py 与 yolov13/ 源码：
#   docker build -f Dockerfile -t yolov13-formula-detection:latest .
# 推荐直接使用 docker-compose.yml（已配置好上下文与 GPU）。
# =============================================================================

FROM python:3.11-slim-bookworm

# cu121 = 默认 GPU 版；纯 CPU 部署可传 --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
ARG TORCH_VERSION=2.4.0
ARG TORCHVISION_VERSION=0.19.0
# ARG MODEL_PATH=/app/yolov13_formula_detection_service/models/yolov13_arxivformula_sft/yolov13s_ep5_bs32_lr_0.01.pt
# 1 = 安装 wheels/ 中的 FlashAttention-2（Ampere 及以上 GPU 提速，可选）
ARG INSTALL_FLASH_ATTN=0

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app:/app/yolov13 \
    YOLO_CONFIG_DIR=/home/appuser/.config/Ultralytics \
    MODEL_PATH=$MODEL_PATH \
    SERVICE_HOST=0.0.0.0 \
    SERVICE_PORT=8000 \
    LOG_LEVEL=info

# opencv（libgl*/libglib）与 torch（libgomp）运行期所需的系统库
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 curl \
 && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 appuser

WORKDIR /app

# 1) PyTorch：按 TORCH_INDEX_URL 决定 cuda12.1 或 cpu 版本
COPY requirements.txt /app/yolov13_formula_detection_service/requirements.txt
RUN pip install --no-cache-dir --extra-index-url "${TORCH_INDEX_URL}" \
        "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}"

# 2) 其余依赖（opencv / Pillow / PyMuPDF / matplotlib / fastapi / uvicorn 等）
RUN pip install --no-cache-dir -r /app/yolov13_formula_detection_service/requirements.txt

# 3) 可选 FlashAttention-2：bind 挂载不写入镜像层，未启用时不产生体积开销（需要 BuildKit，Docker 23+ 默认开启）
RUN --mount=type=bind,source=wheels,target=/tmp/wheels \
    if [ "${INSTALL_FLASH_ATTN}" = "1" ]; then \
        pip install --no-cache-dir "einops==0.8.0" \
            /tmp/wheels/flash_attn-2.8.3+cu12torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl; \
    else \
        echo "跳过 FlashAttention（INSTALL_FLASH_ATTN=${INSTALL_FLASH_ATTN}，推理将使用 PyTorch SDPA 回退）"; \
    fi

# 4) 代码：inference.py + 内置 yolov13 源码 + 服务代码
COPY --chown=appuser:appuser inference.py /app/inference.py
COPY --chown=appuser:appuser yolov13 /app/yolov13
COPY --chown=appuser:appuser ./app.py ./config.py ./detector.py ./schemas.py /app/yolov13_formula_detection_service/
RUN mkdir -p /app/yolov13_formula_detection_service/models \
 && chown appuser:appuser /app/yolov13_formula_detection_service/models

# 5) 权重（单独一层，替换权重时无需重装依赖）
# COPY --chown=appuser:appuser yolov13_formula_detection_service/models /app/yolov13_formula_detection_service/models

# 工作目录设为服务目录：`python -m uvicorn app:app` 依赖 cwd 找到 app/detector/schemas/config
WORKDIR /app/yolov13_formula_detection_service

USER appuser

EXPOSE 9000

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${SERVICE_PORT:-${PORT:-9000}}/health" || exit 1

# 端口优先取 SERVICE_PORT，其次 PORT（docker run -e PORT=9000 也生效），默认 9000。
# Uvicorn CLI 的 --log-level 仅接受小写值，兼容整套部署 .env 中的 LOG_LEVEL=INFO。
CMD ["sh", "-c", "exec python -m uvicorn app:app --host ${SERVICE_HOST} --port ${SERVICE_PORT:-${PORT:-9000}} --log-level \"$(printf '%s' \"${LOG_LEVEL}\" | tr '[:upper:]' '[:lower:]')\""]
