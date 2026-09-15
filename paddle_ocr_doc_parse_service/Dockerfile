# syntax=docker/dockerfile:1

# ============================================================
#  PaddleOCR-VL-1.6 文档解析模型 —— vLLM 推理服务镜像
#
#  基础镜像: vLLM 官方镜像(已装好 vLLM + CUDA 运行库, 不需要自己编译)
#      GPU: vllm/vllm-openai:<tag>              默认
#      CPU: vllm/vllm-openai-cpu:<tag>-x86_64   实验性(很慢, 仅用于验证)
#    → 用构建参数 BASE_IMAGE 切换, 两个 docker-compose 文件里已分别指定。
#
#  ⚠ 本镜像**不**编译 vLLM 源码: 官方镜像里已装好 vLLM(v0.29.0) 及预编译的
#    CUDA kernel。源码编译 vLLM 需要 devel 镜像 + nvcc + 1~2 小时编译, 而且
#    编译产物仍要求驱动版本匹配, 也仍会踩 ②(b) 的 cuda-compat 坑 —— 没必要。
#
#  角色: 镜像 = 推理服务端。容器主进程 = vLLM 的 OpenAI 兼容 HTTP 服务
#        (基础镜像自带 ENTRYPOINT, 本文件不覆盖; 启动参数由 compose 的
#         command / docker run 的尾参传入)。
#
#  权重: **不打进镜像**。运行时把宿主机 checkpoints/PaddleOCR-VL-1.6 以只读
#        方式挂载到容器 /models/PaddleOCR-VL-1.6, 再让 --model 指向该目录。
#        (镜像约 10GB, 模型约 2GB, 两者彻底解耦, 换模型不用重建镜像)
#
#  构建(一般不用手动 build, compose 里已写 build 配置):
#    GPU: docker build --build-arg BASE_IMAGE=vllm/vllm-openai:v0.29.0-cu129 \
#           -t paddleocr-vl-vllm:gpu .
#    CPU: docker build --build-arg BASE_IMAGE=vllm/vllm-openai-cpu:latest-x86_64 \
#           -t paddleocr-vl-vllm:cpu .
#
#  运行(推荐用 compose):
#    GPU: docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
#    CPU: docker compose up -d --build
#
#  手动 docker run 示例(GPU):
#    docker run -d --name paddleocr-vl-vllm --gpus all --ipc=host -p 8080:8080 \
#      -v "$PWD/checkpoints/PaddleOCR-VL-1.6:/models/PaddleOCR-VL-1.6:ro" \
#      paddleocr-vl-vllm:gpu \
#      --model /models/PaddleOCR-VL-1.6 \
#      --served-model-name PaddleOCR-VL-1.6-0.9B PaddleOCR-VL-0.9B \
#      --trust-remote-code --host 0.0.0.0 --port 8080 \
#      --dtype float16 --gpu-memory-utilization 0.90 \
#      --max-num-batched-tokens 16384 --max-model-len 32768 \
#      --no-enable-prefix-caching --mm-processor-cache-gb 0
#
#  验证:
#    curl http://127.0.0.1:8080/health      # 就绪返回 200(未就绪时连接被拒)
#    curl http://127.0.0.1:8080/v1/models   # 看有哪些模型别名
#
#  客户端(PaddleOCR SDK, 在宿主机或别的容器里跑):
#    paddleocr doc_parser -i 某.pdf --pipeline_version v1.6 \
#      --vl_rec_backend vllm-server --vl_rec_server_url http://127.0.0.1:8080/v1
#
#  ---------------- 关键坑(改动前先看这几条) ----------------
#
#  ① vLLM 版本
#     PaddleOCR-VL 从 vLLM 0.11.1 起才支持, 1.6 建议用较新版本。这里默认钉
#     v0.29.0 的 **CUDA 12.9 变体**(原因见 ②)。要锁版本就把 BASE_IMAGE 换成
#     带具体 tag 的镜像, 例如
#       vllm/vllm-openai:v0.29.0-cu129 / vllm/vllm-openai-cpu:v0.29.0-x86_64
#     ⚠ CPU 镜像的 tag **必须带 -x86_64 后缀**(或 arm64), 否则拉不到。
#
#  ② CUDA 版本 / cuda-compat ← 本项目曾在此翻车两次, 务必看清
#
#     (a) 大版本要看得懂: vLLM 官方同一版本同时提供 CUDA 12.x 与 13.x 两种变体
#           vllm/vllm-openai:v0.29.0          -> CUDA 13.0, 需要驱动 >= 580
#           vllm/vllm-openai:v0.29.0-cu129    -> CUDA 12.9, 驱动 >= 525 即可
#         查镜像自带哪个 CUDA:
#           docker run --rm --entrypoint env <镜像> | grep CUDA_VERSION
#         NVIDIA 规则(**只有 12.x 有 minor version compatibility**):
#           CUDA 12.x 应用 -> 驱动 >= 525 可跑(能跨 12.y 小版本)
#           CUDA 13.x 应用 -> 驱动必须 >= 580
#
#     (b) 真正坑人的是 cuda-compat, 不是 CUDA 版本本身:
#         官方镜像里带了 /usr/local/cuda/compat(更高版本的驱动库, 如 580.95)。
#         NVIDIA container runtime 启动容器时会**自动**把这个目录挂进库路径,
#         它既不看 VLLM_ENABLE_CUDA_COMPATIBILITY(vLLM 自己的开关, 默认 0),
#         也不受 NVIDIA_REQUIRE_CUDA 里 GeForce 不匹配的影响。
#         结果: 容器里实际用的是 compat 的 libcuda, 等于在 GeForce 上强行前向
#         兼容 -> 必然报
#           Error 804: forward compatibility was attempted on non supported HW
#         此时**换镜像 CUDA 版本(13.0 -> 12.9)完全无效**, 两次翻车都是它。
#         判别: 容器内 `nvidia-smi` 的 "CUDA Version" 若显示 compat 库的版本
#               (如 13.0), 而不是宿主机驱动支持的上限(535 -> 12.2), 就是中招。
#         修法: 构建时删掉该目录(见下面的 rm -rf), 让容器回落到宿主机驱动的
#               libcuda, 剩余部分交给 CUDA 12.x minor version compatibility。
#               实测 RTX 3080 + 535 驱动 + cu129: 删掉后 torch 正常识别 GPU。
#         (另一条路是升级宿主机驱动 >=580, 需重启, QEMU/云主机未必允许)
#
#     (c) 删掉 compat 之后还有第二道坎: 运行期 JIT 编译的 kernel。
#         镜像里的 nvcc 是 CUDA 12.9, 默认吐出 PTX ISA 8.8; 而驱动 535 的
#         PTX JIT 只认到 PTX ISA 8.1(CUDA 12.2)。minor version compatibility
#         明确**不覆盖 PTX JIT**, 于是运行期现编的 kernel 一加载就报
#           RuntimeError: ... failed with error code device kernel image is invalid
#         (CUDA error 209; 本机典型触发点是 flashinfer 的 top-k/top-p 采样 kernel)
#         修法(都写在 docker-compose.gpu.yml 的 environment 里):
#           TORCH_CUDA_ARCH_LIST=8.6    # 只针对本卡出 SASS, 不走 PTX JIT
#           VLLM_USE_FLASHINFER_SAMPLER=0   # 该采样 kernel 是 JIT 的, 直接关掉
#
#  ③ 精度(dtype): 模型权重是 bfloat16, 但 bf16 需要 Ampere(sm_80)及以上。
#     图灵/20 系卡(sm_75, 例如 RTX 2080 Ti / T4)不支持 bf16, 一旦用 auto/bf16
#     vLLM 启动就报
#        Bfloat16 is only supported on GPUs with compute capability of at least 8.0
#     所以 compose 里 GPU 版 --dtype 默认 float16; A100/30 系/40 系/H100 想要
#     原生 bf16, 设 .env 的 VLLM_DTYPE=bfloat16 即可。
#
#  ④ 端口变量名: 用 SERVE_PORT。**不要**用 VLLM_PORT —— 那是 vLLM 自己的
#     内部 RPC 端口环境变量(仅 api_server_count>1 时用), 含义不是 HTTP 端口。
#
#  ⑤ PaddleOCR 客户端报 "The model PaddleOCR-VL-0.9B does not exist." 时,
#     说明客户端发来的 model 名和服务端对不上 —— 用 --served-model-name
#     多挂几个别名即可(下面已挂 1.6 和 0.9B 两个)。
# ============================================================

ARG BASE_IMAGE=vllm/vllm-openai:v0.29.0-cu129
FROM ${BASE_IMAGE}

# FROM 之前声明的 ARG 在阶段内要重新声明才能读到
ARG BASE_IMAGE

LABEL org.opencontainers.image.title="paddleocr-vl-vllm" \
      org.opencontainers.image.description="PaddleOCR-VL-1.6 文档解析模型的 vLLM 推理服务" \
      org.opencontainers.image.base.name="${BASE_IMAGE}"

# SERVE_PORT = 服务监听端口(HEALTHCHECK 与 compose 都用它; 故意不用 VLLM_PORT,
# 那是 vLLM 自己的内部 RPC 端口变量)
ENV PYTHONUNBUFFERED=1 \
    SERVE_PORT=8080

# 只补一个 curl 给 HEALTHCHECK 用; 基础镜像自带就不动它(避免无谓的 apt 网络操作)
# 同时删掉基础镜像自带的 cuda-compat 驱动库 —— 见文件头 ②(b): 它会被
# NVIDIA container runtime 自动加载, 在 GeForce 上强行前向兼容, 导致启动即报
#   Error 804: forward compatibility was attempted on non supported HW
# 删掉后容器用宿主机驱动的 libcuda(CUDA 12.2), 由 CUDA 12.x minor version
# compatibility 兜住 vLLM/CUDA 12.9 的运行时。
# ⚠ 不要"为了保险"再把它装回来, 也不要设 VLLM_ENABLE_CUDA_COMPATIBILITY=1。
RUN set -eux; \
    rm -rf /usr/local/cuda/compat; \
    if ! command -v curl >/dev/null 2>&1; then \
      apt-get update; \
      apt-get install -y --no-install-recommends curl; \
      rm -rf /var/lib/apt/lists/*; \
    fi; \
    mkdir -p /models; \
    test ! -e /usr/local/cuda/compat

EXPOSE 8080

# 就绪探针: vLLM 的 /health 要等引擎真正起来才返回 200, 启动期(curl 失败)由
# start-period 覆盖 —— 首次启动要加载权重 + torch.compile 编译图, 给足 15 分钟。
# 用 shell 形式是为了能展开容器运行时的 ${SERVE_PORT}。
HEALTHCHECK --interval=30s --timeout=10s --start-period=900s --retries=3 \
  CMD curl -fsS "http://127.0.0.1:${SERVE_PORT:-8080}/health" || exit 1

# 不覆盖基础镜像的 ENTRYPOINT(它就是 vLLM 的 serve 入口),
# 启动参数由 compose 的 command / docker run 的尾参传入(见文件头示例)。
