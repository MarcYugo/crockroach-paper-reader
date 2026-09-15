# ============================================================
#  paper-trans-to-html 应用镜像
#  构建（在项目根目录或 docker/ 下都可以，见 docker/README.md）：
#     cd docker && docker compose build
#  说明：
#   - 只打包代码与依赖，不含 data/、config.json、mongo_configuration/.env（机密与运行时数据都在外面）
#   - 容器内监听 0.0.0.0:8000（run.py 的 APP_HOST/APP_PORT 控制），否则端口映射不到
# ============================================================
FROM python:3.11-slim

# Python 行为：立刻输出日志、不写 .pyc、pip 不留缓存
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TZ=Asia/Shanghai \
    APP_HOST=0.0.0.0 \
    APP_PORT=8000

WORKDIR /app

# 1) 依赖单独一层，改代码不会重装依赖
#    torch/torchvision 的 CPU 版与 surya-ocr 都在 requirements.txt 里锁好了
#    （--extra-index-url + torch==2.14.0+cpu / torchvision==0.29.0+cpu），
#    所以这里一条命令即可，不必再单独装 torch。
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 2) 应用代码（frontend 由 FastAPI 静态托管，tools 便于容器内自检/迁移）
COPY run.py config.example.json ./
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY tools/ ./tools/

# 3) 非 root 用户运行；data/ 先建好并授权，这样命名卷首次挂载也会继承该属主
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/data \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# 健康检查：/api/health 无需登录、不碰数据库
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3).status == 200 else 1)"

CMD ["python", "run.py"]
