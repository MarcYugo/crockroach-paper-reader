# SLANet_plus 表格结构识别服务（Docker 部署）

把 [SLANet_plus](https://www.modelscope.cn/models/PaddlePaddle/SLANet_plus)
包装成一个支持**并发 + 批量**的 HTTP 推理服务。

- 输入：**裁剪好的表格区域图片**
- 输出：表格的 HTML 骨架（`<table><tr><td colspan="4"></td>...</table>`）与结构置信度
- 模型权重很小（约 8MB）**不进镜像**，通过 volume 从宿主机挂载进容器
- 端口默认 **9002**（避免与 pp-formulanet-plus-l 的 9001 冲突）

> 注意：SLANet_plus 只识别“结构”，输出 HTML 里 **cell 文字为空**。
> 需要 cell 文字时，请叠加文本检测/识别模型，或由上游 `router_service`
> 把 OCR 结果回填进 HTML 骨架。

---

## 1. 目录结构

```
slanet_plus_service/
├── Dockerfile                     # 镜像（CUDA runtime 基础镜像 + PaddleOCR 3.0.3，支持 CPU/GPU 变体）
├── docker-compose.yaml            # 单机部署（挂载模型；不含 GPU 直通，没 GPU 的机器也能直接起）
├── docker-compose.gpu.yaml        # GPU 覆盖文件：有 GPU 时叠加，启用 GPU 直通
├── docker-up.sh                   # 一键脚本：自动探测 nvidia-smi 后决定是否叠加 GPU 文件
├── docker-compose.cuda12.yaml     # 已废弃（保留占位，见文件内说明）
├── download.sh                    # 从 ModelScope 拉模型（最简一行命令）
├── .env.example                   # 默认环境变量模板（复制成 .env 使用）
├── requirements.txt
├── .dockerignore
├── app/
│   ├── config.py                  # 全部配置走环境变量
│   ├── engine.py                  # predictor 池：并发/排队/统计
│   └── main.py                    # FastAPI 接口
├── deploy/k8s/
│   └── slanet-plus.yaml           # Deployment + Service + HPA
├── scripts/
│   ├── download_model.sh          # 从 ModelScope 拉模型 + 校验
│   ├── check_model.py             # 校验模型目录是否完整、model_name 是否匹配
│   ├── smoke_test_api.py          # 离线冒烟测试（不需要 GPU / Paddle）
│   └── client_demo.py             # 调用/压测示例
├── test_slanet_service.py         # PDF → 表格区域 → HTML 的端到端样例
└── models/                        # ← 本地模型放这里（已 gitignore）
    └── SLANet_plus/
```

## 2. 准备模型（本地已有可跳过）

容器内需要的是**推理模型目录**，目录里必须有这三个文件：

```
models/SLANet_plus/
├── inference.json          # 308 KB   模型结构
├── inference.pdiparams     # 7.3 MB   权重
├── inference.yml           # 2.0 KB   预处理/后处理配置
├── config.json             # 4.8 KB   可选（含 Global.model_name）
└── README.md               # 可选
```

下载：

```bash
# 方式一：根目录的一行命令
modelscope download --model "PaddlePaddle/SLANet_plus" --local_dir "./models/SLANet_plus"

# 方式二：脚本（会自动补装 modelscope 并校验）
bash scripts/download_model.sh ./models/SLANet_plus
```

校验一下：

```bash
python scripts/check_model.py ./models/SLANet_plus
# 期望输出：[OK] 必需文件齐全 / [OK] model_name 一致: SLANet_plus
```

> 首次运行前确认模型文件对容器进程可读（`chmod -R a+r models/SLANet_plus`），
> 容器以非 root 用户（uid 1000）运行。

### 不装 Paddle 也能先验证服务接线

`scripts/smoke_test_api.py` 用一个假的 `paddleocr` 桩模块把服务真跑起来，
覆盖探针、三种推理接口、参数校验（含 PDF 拦截）、并发限流（503）、指标、
优雅退出共 30 项检查：

```bash
pip install fastapi uvicorn httpx python-multipart numpy opencv-python
python scripts/smoke_test_api.py   # 期望：30 通过 / 0 失败，退出码 0
```

## 3. 快速开始（Docker Compose）

前置条件：

- Docker Engine ≥ 20.10、Compose V2
- **GPU 可选**：默认按 CPU 构建/运行（SLANet_plus 很小，CPU 也够快）
  - 有 NVIDIA GPU 且装了 nvidia-container-toolkit 时想用 GPU，见下面「基础镜像怎么选」
  - 验证 GPU 可用：`docker run --rm --gpus all nvidia/cuda:12.6.1-runtime-ubuntu22.04 nvidia-smi`

启动：

```bash
cp .env.example .env
# 1) 把模型放到 ./models/SLANet_plus（或改 .env 中的 LOCAL_MODEL_DIR / MODEL_DIR）
# 2) 构建并启动
docker compose up -d --build

# 3) 看日志（首次加载模型约 10~60 秒）
docker compose logs -f

# 4) 健康检查
curl http://127.0.0.1:${SERVICE_PORT:-9002}/health
curl http://127.0.0.1:${SERVICE_PORT:-9002}/v1/info
```

浏览器打开 <http://127.0.0.1:9002/docs> 有交互式 API 文档。

### 设备自动检测（GPU / CPU）

`DEVICE` 默认 `auto`，容器启动时自动探测 Paddle 可见的 CUDA 设备数：

| 检测结果 | 实际使用 |
| --- | --- |
| 有可用 CUDA 设备 | `gpu:0` |
| 无 CUDA（纯 CPU 机器 / 容器未挂载 GPU / 装了 CPU 版 Paddle） | `cpu` |

> `DEVICE=auto` 只选择设备，不会把 GPU 版 Paddle 切换成 CPU 版。GPU 版
> Paddle 导入时需要系统提供 `libcuda.so.1`；纯 CPU 环境或容器未挂载 NVIDIA
> 驱动时，必须用 `PADDLE_VARIANT=cpu` 重新构建镜像，否则启动会报
> `Can not import paddle core` / `libcuda.so.1: cannot open shared object file`。

- 决策原因会写进启动日志（`设备选择：...`），也会从 `/v1/info` 的
  `config.device` / `config.device_auto` / `config.device_note` 看到。
- 显式写 `DEVICE=gpu:0` 但没检测到 GPU 时，**默认自动回退 CPU**；
  想让它直接报错而不悄悄降级，设 `DEVICE_STRICT=true`。
- 并发默认值随设备自适应：GPU 时 `PREDICTOR_POOL_SIZE=2`；
  CPU 时按核数给 1~4，`CPU_THREADS` 取 `核数 / 池大小`。显式设置则不被覆盖。
- `USE_TENSORRT=true` 在 CPU 模式下会被自动关闭。

### 基础镜像怎么选

默认使用 `ubuntu:22.04` 和 CPU 版 Paddle。GPU 部署应使用
`nvidia/cuda:12.6.1-runtime-ubuntu22.04`（宿主机驱动 ≥ 550.54.14）并安装 GPU 版 Paddle。
`PADDLE_VARIANT` 决定安装哪一版。

`.env.example` 里默认是 **CPU 精简镜像**（`DOCKER_BASE_IMAGE=ubuntu:22.04`、
`PADDLE_VARIANT=cpu`），因为 SLANet_plus 很小、CPU 也够快。要用 GPU：

```bash
# 1) 改 .env（或构建时用环境变量覆盖）：
#      DOCKER_BASE_IMAGE=nvidia/cuda:12.6.1-runtime-ubuntu22.04
#      PADDLE_VARIANT=gpu
# 2) 叠加 GPU 直通文件：
docker compose -f docker-compose.yaml -f docker-compose.gpu.yaml up -d --build
# 或者用脚本，它会自动判断：
chmod +x docker-up.sh
./docker-up.sh                 # 有 GPU 就叠加 docker-compose.gpu.yaml，没有就纯 CPU
FORCE_CPU=1 ./docker-up.sh     # 强制按 CPU 部署
```

> ⚠️ 注意：`PADDLE_VARIANT=cpu` 时装的是 CPU 版 Paddle，
> 即使叠加了 `docker-compose.gpu.yaml` 也用不到显卡。要用 GPU 必须同时改成 `gpu`。

### 国内构建加速

```bash
# pip 源加速（默认 pypi.org）：
docker compose build --build-arg PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
```

## 4. API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/health` | 存活探针（不依赖模型） |
| GET | `/ready` | 就绪探针（模型加载完成才 200） |
| GET | `/v1/info` | 服务与模型配置信息 |
| GET | `/v1/stats` | 并发/吞吐统计 |
| GET | `/metrics` | Prometheus 文本格式指标 |
| POST | `/v1/table/structure/recognition` | 上传图片文件（多张） |
| POST | `/v1/table/structure/recognition/base64` | base64 图片（多张） |
| POST | `/v1/table/structure/recognition/url` | URL 图片（多张） |

请求示例：

```bash
# 单张
curl -F "files=@table.png" http://127.0.0.1:9002/v1/table/structure/recognition

# 一次多张（服务端会按 batch_size 批推理）
curl -F "files=@a.png" -F "files=@b.png" -F "batch_size=2" \
     http://127.0.0.1:9002/v1/table/structure/recognition

# base64
curl -X POST http://127.0.0.1:9002/v1/table/structure/recognition/base64 \
     -H "Content-Type: application/json" \
     -d '{"images":["<base64>"],"batch_size":1}'
```

响应（每个输入图片一条 `results` 记录）：

```json
{
  "count": 1,
  "batch_size": 1,
  "elapsed_ms": 12.34,
  "per_image_ms": 12.34,
  "results": [
    {
      "index": 0,
      "filename": "table.png",
      "html": "<html><body><table><tr><td colspan=\"4\"></td></tr></table></body></html>",
      "structure_score": 0.99948,
      "num_cells": 13,
      "error": null
    }
  ]
}
```

- `html`：表格结构（HTML 骨架，cell 文字为空）。
- `structure_score`：结构置信度（逐 cell 分数会取均值）。
- `num_cells`：识别到的 cell 数量。
- 传 `include_raw=true` 时，额外返回结构 token 数组 `structure`、cell 坐标 `cells`
  与 PaddleX 原始结果 `raw`（用于调试/回填）。

### PDF 论文怎么用（重要）

服务**只接收图片**，收到 PDF 会直接返回 400。请把表格区域渲染成 PNG 再上传：

```bash
pip install requests pymupdf
python test_slanet_service.py paper.pdf --page 3 --rect 72,300,520,420
```

## 5. 并发是怎么做的

Paddle Inference 的 predictor **不是线程安全**的，所以服务用一个 predictor 池来支撑并发：

```
                 ┌──────────────── admission semaphore (pool_size + max_queue) ────────────────┐
  HTTP 请求  →  │  排队(可超时)  →  [predictor#0] [predictor#1] ... [predictor#N-1]  →  返回结果  │
                 └─────────────────────────────────────────────────────────────────────────────┘
                      排队超过 QUEUE_TIMEOUT → 503 + Retry-After
```

1. **进程内 N 路并行**：启动时加载 `PREDICTOR_POOL_SIZE` 个 `TableStructureRecognition`
   实例，请求进来时独占一个、用完归还，N 路真正并行。
2. **准入信号量**：池外再套一层 `pool_size + max_queue` 的并发上限，
   排队超过 `QUEUE_TIMEOUT`（默认 30s）直接 503，避免请求无限堆积把服务拖死。
3. **批量推理**：单请求可带多张图，服务端按 `batch_size` 合并推理，吞吐更高。
4. **多进程（可选）**：`WEB_CONCURRENCY` 用 uvicorn 多 worker。
   GPU 场景建议保持 1（多进程会成倍占显存），靠池在进程内并发；
   CPU 场景可用多 worker，但要相应下调每个 worker 的 `PREDICTOR_POOL_SIZE`，
   否则会超卖 CPU（总并行数 ≈ `WEB_CONCURRENCY × PREDICTOR_POOL_SIZE`）。
5. **可观测**：`/v1/stats` 与 `/metrics` 暴露 `inflight` / `waiting` / `idle` /
   `rejected_total` 等，方便判断该加副本还是加池。

### 并发参数怎么给

| 场景 | 建议 |
| --- | --- |
| 单机 CPU（8 核） | `PREDICTOR_POOL_SIZE=2`、`CPU_THREADS=4`、`MAX_QUEUE=64` |
| 单机 CPU（16+ 核） | `PREDICTOR_POOL_SIZE=4`、`CPU_THREADS=4` |
| 单卡 GPU | `PREDICTOR_POOL_SIZE=2`、`CPU_THREADS=4`、`WEB_CONCURRENCY=1` |
| 高并发入口 | 前面加副本（K8s HPA），单副本池别开太大 |

## 6. 环境变量

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `MODEL_NAME` | `SLANet_plus` | 必须与模型 `inference.yml` 里的 `Global.model_name` 一致 |
| `MODEL_DIR` | `/models/SLANet_plus` | 容器内模型目录 |
| `LOCAL_MODEL_DIR` | `./models/SLANet_plus` | 宿主机模型目录（compose 用） |
| `DEVICE` | `auto` | `auto` / `cpu` / `gpu:0` / `gpu:0,1` |
| `DEVICE_STRICT` | `false` | true 时 GPU 不可用直接报错，不静默降级 |
| `PREDICTOR_POOL_SIZE` | 按设备自动 | 并行 predictor 数 |
| `CPU_THREADS` | 按设备自动 | 单 predictor 的 CPU 线程数 |
| `ENABLE_MKLDNN` | `true` | CPU 启用 MKL-DNN 加速 |
| `USE_TENSORRT` | `false` | GPU 启用 TensorRT 子图加速 |
| `PRECISION` | `fp32` | TensorRT 精度 `fp32`/`fp16` |
| `WARMUP` | `true` | 启动预热，避免首请求抖动 |
| `MAX_QUEUE` | `64` | 允许排队的请求数 |
| `QUEUE_TIMEOUT` | `30` | 排队超时（秒） |
| `DEFAULT_BATCH_SIZE` | `1` | 默认批大小 |
| `MAX_BATCH_SIZE` | `8` | 单请求最大图片数 |
| `MAX_IMAGE_MB` | `20` | 单张图片体积上限 |
| `TEMP_DIR` | `/dev/shm/slanet-plus` | 临时文件目录（tmpfs） |
| `WEB_CONCURRENCY` | `1` | uvicorn worker 数 |
| `SERVICE_PORT` | `9002` | 宿主机端口 |
| `LOG_LEVEL` | `INFO` | 日志级别 |

## 7. Kubernetes 部署

```bash
# 1) 先把模型放到每个目标节点的 /data/models/SLANet_plus
# 2) 修改 yaml 里的 hostPath、镜像地址（默认 CPU 部署，2 副本 + HPA）
kubectl apply -f deploy/k8s/slanet-plus.yaml
kubectl -n formula-ocr port-forward svc/slanet-plus 9002:9002
curl http://127.0.0.1:9002/health
```

清单默认：2 副本、`DEVICE=cpu`、每副本 `PREDICTOR_POOL_SIZE=2`、HPA 2→8。
要用 GPU 或 PVC，见 `deploy/k8s/slanet-plus.yaml` 底部注释。

> Namespace `formula-ocr` 与 pp-formulanet-plus-l 共用；若已存在会报 AlreadyExists，可忽略。

## 8. 常见问题

**Q：`/health` 一直是 `loading`？**
A：模型还在加载，或模型目录没挂对。看日志里的 `模型目录不存在` / `缺少必需文件`；
本地先跑 `python scripts/check_model.py ./models/SLANet_plus`。

**Q：启动报 `Model name mismatch, please input the correct model dir.`？**
A：`MODEL_NAME` 与模型 `inference.yml` 的 `Global.model_name` 不一致，改成 `SLANet_plus`。

**Q：有 GPU 但日志显示用了 cpu？**
A：① `.env` 里 `PADDLE_VARIANT=cpu` 装的是 CPU 版 Paddle；② 没叠加
`docker-compose.gpu.yaml`，容器看不到显卡。逐项检查：

```bash
nvidia-smi                                            # 宿主机能否看到显卡
docker run --rm --gpus all nvidia/cuda:12.6.1-runtime-ubuntu22.04 nvidia-smi
docker compose exec slanet-plus curl -s http://127.0.0.1:9002/v1/info | grep device
```

**Q：返回的 HTML 里 `<td>` 没有文字？**
A：SLANet_plus 只做结构识别，cell 文字由文本识别模型负责，需在客户端/上游回填。

**Q：并发一高就 503？**
A：这是准入保护的正常行为。调大 `MAX_QUEUE` / `QUEUE_TIMEOUT`，
或调大 `PREDICTOR_POOL_SIZE`，或增加副本。

**Q：Windows 上 `docker compose up` 报 `/dev/shm` 挂载错误？**
A：`docker-compose.yaml` 里的 `- /dev/shm:/dev/shm` 是为 Linux 准备的。
Windows/macOS 的 Docker Desktop 请把该行删掉（`shm_size` 已足够提供共享内存），
或设 `LOCAL_TEMP_DIR` 指向一个有效目录。

### 验证过的环境

- PaddleOCR 3.0.3 / paddlex 3.0.3 / Paddle 3.0.0
- Python 3.10（镜像内 venv）
- Docker Engine 27.x、Compose V2
- 模型：`PaddlePaddle/SLANet_plus`（inference 模型，约 8MB）
