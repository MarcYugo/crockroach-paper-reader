# syntax=docker/dockerfile:1
#
# SLANet_plus 表格结构识别服务
# ----------------------------
# 基础镜像可用构建参数切换：
#   * 默认 ubuntu:22.04 + CPU 版 Paddle，适用于无 GPU 环境
#   * GPU 部署需使用 CUDA runtime 基础镜像、GPU 版 Paddle，并向容器挂载 NVIDIA 驱动
ARG BASE_IMAGE=ubuntu:22.04
FROM ${BASE_IMAGE}

# 保持兼容：旧脚本里仍可能用到这个参数
ARG REQUIRE_CUDA=true
# gpu: 安装 paddlepaddle-gpu（运行时需要可用的 NVIDIA 驱动）
# cpu: 安装 paddlepaddle（仅 CPU）
ARG PADDLE_VARIANT=cpu
# PaddleOCR 与模型权重是配套发布的，默认用模型卡验证过的 3.0.3
ARG PADDLEOCR_VERSION=3.0.3
# pip 源，国内可改成 https://pypi.tuna.tsinghua.edu.cn/simple
ARG PIP_INDEX=https://pypi.org/simple

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Shanghai \
    LANG=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH=/opt/venv/bin:$PATH \
    # 模型只从本地挂载目录读取，禁止容器内联网下载 / 联网校验
    PADDLE_PDX_MODEL_SOURCE=BOS \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PADDLE_PDX_CACHE_HOME=/app/.cache/paddlex \
    # ---- 运行时默认值（compose / k8s 可覆盖）----
    MODEL_NAME=SLANet_plus \
    MODEL_DIR=/models/SLANet_plus \
    # auto：运行时自动探测设备，有 GPU 用 gpu:0，没有则用 cpu
    DEVICE=auto \
    PORT=9002 \
    WEB_CONCURRENCY=1 \
    PREDICTOR_POOL_SIZE=2 \
    MAX_QUEUE=64 \
    QUEUE_TIMEOUT=30 \
    CPU_THREADS=4 \
    MAX_BATCH_SIZE=8 \
    MAX_IMAGE_MB=20 \
    TEMP_DIR=/dev/shm/slanet-plus \
    LOG_LEVEL=INFO

# 运行期依赖：Python、OpenCV 需要的图形库、libgomp、健康检查用的 curl
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        ca-certificates curl git python3 python3-pip python3-venv \
        libgomp1 libglib2.0-0 libgl1 libsm6 libxext6 libxrender1; \
    rm -rf /var/lib/apt/lists/*; \
    python3 -m venv /opt/venv; \
    /opt/venv/bin/python -m pip install --upgrade pip; \
    ln -sf /opt/venv/bin/python /usr/local/bin/python; \
    python --version

WORKDIR /app

# 先装依赖，最大化利用镜像层缓存
COPY requirements.txt ./
RUN set -eux; \
    # 在独立 venv 中安装依赖，避免 Ubuntu 22.04 系统 Python 的 externally-managed 限制。
    python -m pip install --upgrade pip -i "${PIP_INDEX}"; \
    # ★ 清理旧版 PyYAML（如果环境中存在）
    rm -rf /usr/lib/python3/dist-packages/yaml \
           /usr/lib/python3/dist-packages/_yaml*.so \
           /usr/lib/python3/dist-packages/PyYAML-*.egg-info \
           /usr/lib/python3/dist-packages/PyYAML-*-py3.egg-info; \
    # 关键修正：PaddleOCR 3.0.3 需要与 paddlex 3.0.3 一致，避免运行期出现
    # "PaddlePredictorOption.__init__() takes 1 positional argument but 2 were given"。
    python -m pip install -i "${PIP_INDEX}" \
        "paddleocr==${PADDLEOCR_VERSION}" \
        "paddlex==${PADDLEOCR_VERSION}" \
        "langchain>=0.2,<0.3"; \
    python -m pip install -i "${PIP_INDEX}" -r requirements.txt; \
    if [ "${PADDLE_VARIANT}" = "cpu" ]; then \
        python -m pip install -i "${PIP_INDEX}" "paddlepaddle==3.0.0"; \
    else \
        python -m pip install -i "https://www.paddlepaddle.org.cn/packages/stable/cu126/" \
            paddlepaddle-gpu==3.0.0; \
    fi; \
    python -c "import yaml; print('PyYAML', yaml.__version__, 'from', yaml.__file__)"; \
    if [ "${REQUIRE_CUDA}" = "true" ]; then \
        echo "CUDA device selection is deferred to runtime; the GPU Paddle variant still requires NVIDIA drivers."; \
    fi

# 业务代码
COPY app ./app
COPY scripts ./scripts

# 非 root 运行；模型挂载目录只读，无需写权限
ENV HOME=/home/appuser \
    TMPDIR=/tmp
RUN set -eux; \
    if ! id -u appuser >/dev/null 2>&1; then useradd -m -u 1000 -s /bin/bash appuser; fi; \
    mkdir -p /app/logs /app/.cache /dev/shm/slanet-plus; \
    chown -R appuser:appuser /app /dev/shm/slanet-plus

USER appuser

EXPOSE 9002

# 模型加载需要时间，start-period 给足
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT}/health" || exit 1

# 用 sh -c 是为了让 $PORT / $WEB_CONCURRENCY 在运行时展开
# 并发要点：多进程（WEB_CONCURRENCY）会成倍占用显存，GPU 场景建议保持 1，
# 靠 PREDICTOR_POOL_SIZE 在进程内做并发。
CMD ["sh", "-c", "exec python3 -m uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers ${WEB_CONCURRENCY} --timeout-keep-alive 65 --backlog 2048"]
