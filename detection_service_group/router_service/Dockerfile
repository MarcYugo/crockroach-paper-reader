# syntax=docker/dockerfile:1
#
# router_service —— 公式/表格识别流水线路由服务
# ----------------------------------------------
# 本服务只做 HTTP 转发与编排（生产者-消费者流水线），不做模型推理，
# 因此不需要 torch / paddle / GPU，镜像很小、启动很快。
#
# 三个下游服务通过环境变量指定（compose 中用 service 名即可）：
#   DETECTION_URL / FORMULA_URL / TABLE_URL
#
# 构建：
#   docker build -t router-service:1.0.0 .
# 国内加速：
#   docker build --build-arg PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple -t router-service:1.0.0 .

ARG BASE_IMAGE=python:3.11-slim-bookworm
FROM ${BASE_IMAGE}

# pip 源，国内可改成 https://pypi.tuna.tsinghua.edu.cn/simple
ARG PIP_INDEX=https://pypi.org/simple

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Shanghai \
    LANG=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # ---- 下游服务（compose / k8s 覆盖）----
    DETECTION_URL=http://yolov13-formula-detection:9000 \
    FORMULA_URL=http://pp-formulanet-plus-l:9001 \
    TABLE_URL=http://slanet-plus:9002 \
    # ---- 运行时默认值 ----
    SERVICE_HOST=0.0.0.0 \
    SERVICE_PORT=9003 \
    LOG_LEVEL=INFO \
    MAX_CONCURRENT_REQUESTS=2 \
    MAX_QUEUE=64 \
    QUEUE_TIMEOUT=30 \
    REQUEST_TIMEOUT=300 \
    SESSION_TTL=1800 \
    FORMULA_WORKERS=2 \
    TABLE_WORKERS=1 \
    FORMULA_BATCH_SIZE=4 \
    TABLE_BATCH_SIZE=1 \
    BATCH_WINDOW_MS=20 \
    POOL_MAX_SIZE=512 \
    POOL_OVERFLOW=block \
    HTTP_CONNECT_TIMEOUT=10 \
    HTTP_READ_TIMEOUT=300 \
    HTTP_MAX_CONNECTIONS=32 \
    FORCE_RETURN_CROPS=true \
    DETECT_CONF=0.25 \
    DETECT_IOU=0.7 \
    DETECT_IMGSZ=640 \
    DETECT_MAX_DET=300 \
    DETECT_BATCH=8 \
    DETECT_DEDUP=true \
    DETECT_DEDUP_THR=0.8 \
    DETECT_CROP_FORMAT=png \
    DETECT_CROP_QUALITY=90 \
    DETECT_CROP_PADDING=0

# 健康检查与调试需要 curl
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends ca-certificates curl; \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先装依赖，最大化利用镜像层缓存
COPY requirements.txt ./
RUN python -m pip install --upgrade pip -i "${PIP_INDEX}" \
 && python -m pip install -i "${PIP_INDEX}" -r requirements.txt

# 业务代码
COPY app ./app
COPY scripts ./scripts

# 非 root 运行
RUN set -eux; \
    useradd --create-home --uid 1000 --shell /bin/bash appuser; \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 9003

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${SERVICE_PORT}/health" || exit 1

# 用 sh -c 让环境变量在运行时展开，并将 Uvicorn 日志级别规范为小写
CMD ["sh", "-c", "exec python -m uvicorn app.main:app --host ${SERVICE_HOST} --port ${SERVICE_PORT} --log-level \"$(printf '%s' \"${LOG_LEVEL}\" | tr '[:upper:]' '[:lower:]')\" --timeout-keep-alive 65 --backlog 2048"]
