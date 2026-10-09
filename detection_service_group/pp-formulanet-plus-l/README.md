# PP-FormulaNet_plus-L 公式识别服务（Docker 部署）

把 [PP-FormulaNet_plus-L](https://www.modelscope.cn/models/PaddlePaddle/PP-FormulaNet_plus-L)
包装成一个支持**并发 + 批量**的 HTTP 推理服务。

模型权重（约 700MB）**不进镜像**，通过 volume 从宿主机挂载进容器。

---

## 1. 目录结构

```
pp-formulanet-plus-l/
├── Dockerfile                     # 镜像（CUDA runtime 基础镜像 + PaddleOCR 3.0.3，支持 CPU/GPU 变体）
├── docker-compose.yaml            # 单机部署（挂载模型；不含 GPU 直通，没 GPU 的机器也能直接起）
├── docker-compose.gpu.yaml        # GPU 覆盖文件：有 GPU 时叠加，启用 GPU 直通
├── docker-up.sh                   # 一键脚本：自动探测 nvidia-smi 后决定是否叠加 GPU 文件
├── docker-compose.cuda12.yaml     # 已废弃（保留占位，见文件内说明）
├── .env.example                   # 默认环境变量模板，和 pp-doclayout-plus-l 保持一致
├── requirements.txt
├── .dockerignore
├── app/
│   ├── config.py                  # 全部配置走环境变量
│   ├── engine.py                  # predictor 池：并发/排队/统计
│   └── main.py                    # FastAPI 接口
├── deploy/k8s/
│   └── pp-formulanet-plus-l.yaml  # Deployment + Service + HPA
├── scripts/
│   ├── download_model.sh          # 从 ModelScope 拉模型（可选）
│   ├── check_model.py             # 校验模型目录是否完整
│   ├── smoke_test_api.py          # 离线冒烟测试（不需要 GPU / Paddle）
│   └── client_demo.py             # 调用/压测示例
└── models/                        # ← 本地模型放这里（已 gitignore）
    └── PP-FormulaNet_plus-L/
```

## 2. 准备模型（本地已有可跳过）

容器内需要的是**推理模型目录**，目录里必须有这三个文件：

```
models/PP-FormulaNet_plus-L/
├── inference.json          # 1.6 MB   模型结构
├── inference.pdiparams     # 694 MB   权重
├── inference.yml           # 2.2 MB   预处理/后处理配置
├── config.json             # 3.8 MB   可选
└── README.md               # 可选
```

校验一下：

```bash
python scripts/check_model.py ./models/PP-FormulaNet_plus-L
```

> 首次运行前确认模型文件对容器进程可读（`chmod -R a+r models/PP-FormulaNet_plus-L`），
> 容器以非 root 用户（uid 1000）运行。

### 不装 Paddle 也能先验证服务接线

`scripts/smoke_test_api.py` 用一个假的 `paddleocr` 桩模块把服务真跑起来，
覆盖探针、三种推理接口、参数校验、并发限流（503）、指标、优雅退出共 24 项检查：

```bash
pip install fastapi uvicorn httpx python-multipart numpy opencv-python
python scripts/smoke_test_api.py   # 期望：24 通过 / 0 失败，退出码 0
```

## 3. 快速开始（Docker Compose）

前置条件：

- Docker Engine ≥ 20.10、Compose V2
- **GPU 可选**：有 NVIDIA GPU 且装了 nvidia-container-toolkit 时自动用 GPU；否则自动用 CPU
  （验证 GPU 可用：`docker run --rm --gpus all nvidia/cuda:12.6.1-runtime-ubuntu22.04 nvidia-smi`）

启动：

```bash
cp .env.example .env
# 1) 把模型放到 ./models/PP-FormulaNet_plus-L（或改 .env 中的 LOCAL_MODEL_DIR / MODEL_DIR）
# 2) 构建并启动。Compose 会先修正宿主机模型目录的读取权限，
#    再启动服务；有 GPU 时自动用 GPU，没有就自动用 CPU
docker compose up -d --build

# 3) 看日志（首次加载模型约 1~3 分钟）
docker compose logs -f

# 4) 健康检查
curl http://127.0.0.1:${SERVICE_PORT:-9001}/health
curl http://127.0.0.1:${SERVICE_PORT:-9001}/v1/info
```

> 注意：GPU 直通配置放在 `docker-compose.gpu.yaml` 里。
> 有 GPU 的机器请叠加它，否则容器看不到显卡（应用会回退到 CPU，不会报错）：
>
> ```bash
> docker compose -f docker-compose.yaml -f docker-compose.gpu.yaml up -d --build
> ```
>
> Compose 会先运行一次性 `model-permissions` 服务，以 root 身份对模型 bind mount 设置读取/遍历权限，然后才启动应用；应用仍以只读方式挂载模型。此步骤发生在 `up` 的容器启动阶段（镜像构建本身不包含模型目录）。
>
> ```bash
> chmod +x docker-up.sh
> ./docker-up.sh                 # 有 GPU 就叠加 docker-compose.gpu.yaml，没有就纯 CPU
> FORCE_CPU=1 ./docker-up.sh     # 强制按 CPU 部署
> ```

浏览器打开 <http://127.0.0.1:9001/docs> 有交互式 API 文档。

### 设备自动检测（GPU / CPU）

`DEVICE` 默认 `auto`，容器启动时自动探测 Paddle 可见的 CUDA 设备数：

| 检测结果 | 实际使用 |
| --- | --- |
| 有可用 CUDA 设备 | `gpu:0` |
| 无 CUDA（纯 CPU 机器 / 无 NVIDIA 驱动 / 容器未挂载 GPU / 装了 CPU 版 Paddle） | `cpu` |

- 决策原因会写进启动日志（`设备选择：...`），也会从 `/v1/info` 的
  `config.device` / `config.device_auto` / `config.device_note` 看到。
- 显式写 `DEVICE=gpu:0` 但没检测到 GPU 时，**默认自动回退 CPU**；
  想让它直接报错而不悄悄降级，设 `DEVICE_STRICT=true`。
- 并发默认值随设备自适应：GPU 时 `PREDICTOR_POOL_SIZE=2`；CPU 时 1~2，
  `CPU_THREADS` 取 `核数 / 池大小`。显式设置这两个变量则不被覆盖。
- `USE_TENSORRT=true` 在 CPU 模式下会被自动关闭。

### 基础镜像怎么选

镜像默认基于 `nvidia/cuda:12.6.1-runtime-ubuntu22.04`（宿主机驱动 ≥ 550.54.14），
Paddle 由 pip 安装。基础镜像和 Paddle 变体都可用构建参数切换：

| 目标 | 构建参数 |
| --- | --- |
| 默认（GPU 机器；无 GPU 时也能跑 CPU） | `BASE_IMAGE=nvidia/cuda:12.6.1-runtime-ubuntu22.04`、`PADDLE_VARIANT=gpu` |
| 纯 CPU 机器（镜像更小） | `BASE_IMAGE=ubuntu:22.04`、`PADDLE_VARIANT=cpu` |

```bash
# 默认构建
docker compose build

# CPU 精简镜像
docker build --build-arg BASE_IMAGE=ubuntu:22.04 --build-arg PADDLE_VARIANT=cpu \
  -t pp-formulanet-plus-l:cpu .
```

> `PADDLE_VARIANT=cpu` 装的是 `paddlepaddle`（CPU 版），此时只能用 CPU；
> 默认的 `gpu` 变体（`paddlepaddle-gpu`）在无 GPU 环境同样能正常做 CPU 推理，
> 所以不确定目标机器时保持默认即可。

> 也可直接改 `.env`：`DOCKER_BASE_IMAGE=ubuntu:22.04`、`PADDLE_VARIANT=cpu`，
> 然后照旧 `docker compose up -d --build`。K8s 清单把 `image` 换成 CPU 镜像即可。

### 不使用 compose 的裸 docker 命令

```bash
docker build -t pp-formulanet-plus-l:3.0.0 .

docker run -d --name pp-formulanet-plus-l \
  --gpus '"device=0"' \
  --shm-size=2g \
  -p 9001:9001 \
  -v /绝对路径/PP-FormulaNet_plus-L:/models/PP-FormulaNet_plus-L:ro \
  -e PREDICTOR_POOL_SIZE=2 \
  -e MODEL_DIR=/models/PP-FormulaNet_plus-L \
  -e DEVICE=auto \
  pp-formulanet-plus-l:3.0.0
```

> 无 GPU 机器去掉 `--gpus` 参数即可，其余不用改：`DEVICE=auto` 会自动选 CPU。

### 国内构建加速

```bash
docker compose build --build-arg PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
# 基础镜像拉不动时，换成国内镜像源的同名镜像即可（路径保持一致）：
docker build --build-arg BASE_IMAGE=<国内源>/nvidia/cuda:12.6.1-runtime-ubuntu22.04 \
  -t pp-formulanet-plus-l:3.0.0 .
```

## 4. API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/v1/formula/recognition` | 上传图片文件（多张） |
| POST | `/v1/formula/recognition/base64` | base64 图片（多张） |
| POST | `/v1/formula/recognition/url` | 图片 URL（多张） |
| GET | `/health` `/ready` | 存活 / 就绪探针 |
| GET | `/v1/info` `/v1/stats` `/metrics` | 配置、并发统计、Prometheus 指标 |

请求：

```bash
# 单张
curl -X POST http://127.0.0.1:9001/v1/formula/recognition \
  -F "files=@formula.png"

# 一次多张（服务端会按 batch_size 批推理）
curl -X POST http://127.0.0.1:9001/v1/formula/recognition \
  -F "files=@f1.png" -F "files=@f2.png" -F "files=@f3.png" \
  -F "batch_size=3"

# base64
curl -X POST http://127.0.0.1:9001/v1/formula/recognition/base64 \
  -H "Content-Type: application/json" \
  -d '{"images":["'"$(base64 -w0 formula.png)"'"],"batch_size":1}'
```

响应（只返回 LaTeX，原始结果可通过 `include_raw=true` 打开）：

```json
{
  "count": 1,
  "batch_size": 1,
  "elapsed_ms": 1832.4,
  "per_image_ms": 1832.4,
  "results": [
    {
      "index": 0,
      "filename": "formula.png",
      "rec_formula": "\\zeta_{0}(\\nu)=-\\frac{\\nu\\varrho^{-2\\nu}}{\\pi}\\int_{\\mu}^{\\infty}d\\omega\\cdots",
      "error": null
    }
  ]
}
```

Python 调用与压测：

```bash
pip install httpx
python scripts/client_demo.py --url http://127.0.0.1:9001 --image ./formula.png --concurrency 4
```

### PDF 论文怎么用（重要）

**服务只收图片，不接受直接上传 PDF**（会返回 400）。原因：PP-FormulaNet 是「单公式」识别模型，
不会自己在页面上找公式；PDF 丢给 PaddleOCR 只会「一页识别出一个 LaTeX」，对论文页面没有意义。

所以 PDF 要在客户端先渲染/裁剪成图片，样例脚本 `test_pp_formulanet_service.py` 已实现：

```bash
pip install requests pymupdf
python test_pp_formulanet_service.py paper.pdf                       # 每页整页识别
python test_pp_formulanet_service.py paper.pdf --page 3              # 指定页
python test_pp_formulanet_service.py paper.pdf --page 3 --rect 72,300,520,360   # 指定公式区域
python test_pp_formulanet_service.py paper.pdf --auto                # 自动取页面内嵌图片区域
```

想要「整页自动定位所有公式」，需要带版面检测的 `FormulaRecognitionPipeline`，
那要额外挂 layout 等模型；当前部署只挂了公式识别模型。

## 5. 并发是怎么做的

```
                 gunicorn/uvicorn(1 进程, 异步)
                            │
     ┌──────────────────────┼──────────────────────┐
     │  准入信号量 pool_size + MAX_QUEUE            │  ← 超出直接 503 + Retry-After
     └──────────────────────┼──────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              │   PredictorPool (N 个)     │
              │  predictor#0  推理中 ────┐ │
              │  predictor#1  空闲       │ │  ← Paddle predictor 非线程安全，
              │  predictor#2  推理中 ────┤ │     所以一个请求独占一个实例
              └──────────────────────────┘ │
                                           ▼
                              线程池执行（不阻塞事件循环）
```

关键点：

1. **Paddle Inference 的 predictor 不是线程安全的**，所以不是简单加锁串行化，而是
   建 N 个 `FormulaRecognition` 实例构成池（`PREDICTOR_POOL_SIZE`），实现真正并行。
2. **准入限流**：池满后请求进入排队；排队超过 `QUEUE_TIMEOUT` 秒返回 `503`，
   并带 `Retry-After` 头，避免请求无限堆积拖垮服务。
3. **单请求多图**走 `batch_size` 批推理，比循环调用快（尤其 GPU 上）；
   `MAX_BATCH_SIZE` 控制单请求图片数上限。
4. **不阻塞事件循环**：推理通过线程池执行，异步接口在高并发下仍能正常响应探针。
5. **可观测**：`/v1/stats` 看实时 in-flight/排队数，`/metrics` 接 Prometheus。

### 并发参数怎么给

| 场景 | 建议值 |
| --- | --- |
| 单张 T4 / 显存 16GB | `PREDICTOR_POOL_SIZE=2`、`CPU_THREADS=4` |
| 单张 A10/3090/4090，追求吞吐 | `PREDICTOR_POOL_SIZE=3~4`、`MAX_BATCH_SIZE=8` |
| 多张 GPU（如 gpu:0,1） | `DEVICE=gpu:0,1` + `PREDICTOR_POOL_SIZE` 适当调大 |
| 纯 CPU | 什么都不用设（`DEVICE=auto` 自动选 cpu，池大小也按核数自动给） |

开销参考（官方数据，Tesla T4，仅模型推理耗时）：PP-FormulaNet_plus-L 单张约 **1.47s**，
模型约 698MB。GPU 上并发不会线性提速（算力争抢），
`PREDICTOR_POOL_SIZE` 建议**不要超过 GPU 数的 2~4 倍**，边压测边看 `/v1/stats` 与 `nvidia-smi` 调参。

## 6. 环境变量

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `MODEL_DIR` | `/models/PP-FormulaNet_plus-L` | 容器内模型目录（挂载点） |
| `DEVICE` | `auto` | `auto`（自动检测，无 GPU 用 CPU）/ `cpu` / `gpu:0` / `gpu:0,1`；显式写 gpu 但无 GPU 时自动回退 cpu |
| `DEVICE_STRICT` | `false` | `true` 时不自动回退：指定 `gpu:0` 但没 GPU 就直接报错 |
| `PREDICTOR_POOL_SIZE` | 按设备自动（GPU:2，CPU:1~2） | 并行推理路数（**并发核心参数**），显式设置则覆盖 |
| `MAX_QUEUE` | `64` | 排队上限，超出 503 |
| `QUEUE_TIMEOUT` | `30` | 排队超时（秒） |
| `DEFAULT_BATCH_SIZE` / `MAX_BATCH_SIZE` | `1` / `8` | 批大小与单请求图片数上限 |
| `CPU_THREADS` | 按设备自动（CPU: 核数/池大小，GPU: 4） | 单个 predictor 的 CPU 线程数 |
| `ENABLE_MKLDNN` | `true` | CPU 推理启用 MKL-DNN 加速 |
| `USE_TENSORRT` / `PRECISION` | `false` / `fp32` | TensorRT 子图加速（首次会编译引擎，耗时长） |
| `WARMUP` | `true` | 启动时用合成图预热 |
| `MAX_IMAGE_MB` | `20` | 单图体积上限 |
| `WEB_CONCURRENCY` | `1` | uvicorn worker 数，**GPU 场景保持 1**（多进程会成倍占显存） |
| `TEMP_DIR` | `/dev/shm/pp-formulanet` | 临时图片目录，内存盘更快 |
| `LOG_LEVEL` | `INFO` | 日志级别 |

## 7. Kubernetes 部署

```bash
# 1) 先把模型放到每个目标节点的 /data/models/PP-FormulaNet_plus-L
# 2) 修改 yaml 里的 hostPath、nodeSelector 标签、镜像地址
kubectl apply -f deploy/k8s/pp-formulanet-plus-l.yaml

kubectl -n formula-ocr get pod -w
kubectl -n formula-ocr port-forward svc/pp-formulanet-plus-l 9001:9001
curl http://127.0.0.1:9001/health
```

清单要点：

- `startupProbe` 最长给 15 分钟，**模型加载期间不会被误杀**；`readinessProbe` 打 `/ready`，
  模型真正就绪后才接入流量。
- `maxUnavailable: 0` 滚动更新，避免服务中断。
- 每副本 `nvidia.com/gpu: 1`，HPA `maxReplicas` 别超过集群 GPU 总数。
- 多节点共享模型时，把 `hostPath` 换成 RWX/ROX 的 PVC。

## 8. 常见问题

**Q1：`/health` 一直返回 `"model_ready": false`**
模型还在加载。看日志 `docker compose logs -f`，正常会有
`predictor #0 加载完成`、`预热完成`、`服务就绪`。

**Q2：报错 `AssertionError` / `paddle` 不支持 CUDA**
若日志里有 `设备选择：DEVICE=auto：未检测到可用 GPU，使用 cpu`，说明这台机器/容器没有可用 GPU，
服务已经自动改用 CPU（这不是错误）。要排查 GPU 为何不可用：

```bash
# 宿主机能否看到显卡
docker run --rm --gpus all nvidia/cuda:12.6.1-runtime-ubuntu22.04 nvidia-smi
# 容器内的 Paddle 是否支持 CUDA
docker compose exec pp-formulanet-plus-l python3 -c "import paddle; print(paddle.is_compiled_with_cuda())"
```

若为 `False`，说明装成了 CPU 版 Paddle（`PADDLE_VARIANT=cpu`），重新用默认的 `gpu` 变体构建即可。

**Q2b：想让“检测不到 GPU”变成硬失败？**
设 `DEVICE=gpu:0` + `DEVICE_STRICT=true`，此时不会自动回退 CPU，而是直接报错。

**Q3：启动时尝试联网下载模型 / 卡在 HuggingFace**
镜像里已设置 `PADDLE_PDX_MODEL_SOURCE=BOS`、`PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True`、
`HF_HUB_OFFLINE=1`。若仍报缺少文件，说明 `MODEL_DIR` 指向的目录不完整，
用 `python scripts/check_model.py` 检查。

**Q4：容器里 `Permission denied` 读模型**
容器以 uid 1000 运行，宿主机模型文件需可读：
`sudo chmod -R a+rX models/PP-FormulaNet_plus-L`。
临时绕过：`docker compose run --user root ...` 或给 compose 加 `user: root`。

**Q5：`ImportError: libGL.so.1`**
镜像已安装 `libgl1` 等图形库。若你换了基础镜像，需补装
`libgl1 libglib2.0-0 libsm6 libxext6 libxrender1`。

**Q6：并发上不去 / `/v1/stats` 里 `idle` 长期为 0**
GPU 已打满，属正常现象。此时应减少单请求图片数、或加卡扩容（K8s 加副本）。
若 GPU 利用率低而 `waiting` 高，说明 `PREDICTOR_POOL_SIZE` 太小，可以调大。

**Q7：返回 503 `服务繁忙`**
触发了排队保护，看 `rejected_total` 指标。缓解方式：调大 `PREDICTOR_POOL_SIZE` / `MAX_QUEUE`，
或客户端做指数退避重试（响应的 `Retry-After` 头可用）。

**Q8：想要更高精度之外的吞吐（TensorRT）**
设 `USE_TENSORRT=true` + `PRECISION=fp16`。注意首次构建引擎很慢（可能十几分钟），
且需要容器内 TensorRT 正常（官方基础镜像已带 8.6）。建议先在测试环境验证 BLEU 是否可接受。

---

### 验证过的环境

- 基础镜像：`nvidia/cuda:12.6.1-runtime-ubuntu22.04`
  （纯 CPU 机器可换 `ubuntu:22.04` + `PADDLE_VARIANT=cpu`）
- PaddlePaddle 3.0.0（`paddlepaddle-gpu` / `paddlepaddle`）+ PaddleOCR 3.0.3
  （模型卡 FAQ 明确说明公式识别推理**强依赖 Paddle 3.0 正式版**，版本要对齐）
- Python 3.8+（基础镜像自带）
