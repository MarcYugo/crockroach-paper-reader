# router_service —— 公式/表格识别流水线路由服务

把三个模型服务串成一条**生产者-消费者流水线**，对外只暴露一个接口：

```
                ┌─────────────────────────── router_service :9003 ───────────────────────────┐
 client ──────▶ │ ① /v1/router/predict                                                       │
 (PDF 渲染图)   │      │                                                                     │
                │      ▼                                                                     │
                │  ② yolov13-formula-detection :9000   （生产者：检测框 + 类别 + 框内图像）  │
                │      │                                                                     │
                │      ▼  按 class_name 分流                                                  │
                │  ③ ┌───────────────┐        ┌───────────────┐                              │
                │    │ 公式缓存池     │        │ 表格缓存池     │  ← 缓冲池（自带背压）        │
                │    └───────┬───────┘        └───────┬───────┘                              │
                │            │ workers（并行）          │ workers（并行）                     │
                │            ▼                        ▼                                    │
                │  ④ pp-formulanet-plus-l :9001   slanet-plus :9002  （消费者）              │
                │            │（返回 LaTeX）           │（返回表格 HTML）                     │
                │            └──────────┬─────────────┘                                      │
                │                       ▼                                                    │
                │  ⑤ 把识别结果追加回 yolov13 的 JSON，返回总数据                              │
                └───────────────────────────────────────────────────────────────────────────┘
```

三个关键特性对应 [instruction.md](./instruction.md) 的需求：

| 需求 | 实现 |
| --- | --- |
| 接收请求 → 分发 → 分类 → 再分发 → 合并返回 | 上面的 ①～⑤；分类见 `app/grouping.py`，合并见 `app/merge.py` |
| 缓存池 + 流水线，检测是生产者、识别服务是消费者 | `app/pools.py`（`BufferPool`）+ `app/pipeline.py`（worker 协程） |
| 同时接收 2 个客户端请求，用 session id 区分结果 | `MAX_CONCURRENT_REQUESTS` 请求准入 + `app/sessions.py` 按 `session_id` 归档 |

`router_service` 本身**不做推理**（不需要 torch / paddle / GPU），只做转发、排队与合并。

---

## 1. 目录结构

| 文件 | 说明 |
| --- | --- |
| `Dockerfile` | 服务镜像（python:3.11-slim + FastAPI，很小、启动秒级） |
| `docker-compose.yaml` | **单独部署**：下游三个服务在别处跑时用这个（下游地址在 `.env` 里配） |
| `.env.example` | 单独部署的默认环境变量模板 |
| `app/main.py` | FastAPI 接口（预测、会话、缓存池状态、指标） |
| `app/pipeline.py` | 生产者-消费者流水线：投递、批量取件、失败兜底、统计 |
| `app/pools.py` | 公式缓存池 / 表格缓存池（容量、批量窗口、背压、指标） |
| `app/grouping.py` | 检测框分类：公式 / 表格 / 图片 |
| `app/merge.py` | 把识别结果追加回 yolov13 的响应 |
| `app/sessions.py` | 会话存储（多客户端隔离、结果回查、TTL 清理） |
| `app/clients.py` | 下游服务的异步 HTTP 客户端（连接池复用） |
| `app/config.py` | 全部配置走环境变量 |
| `scripts/smoke_test_router.py` | 离线冒烟测试：桩下游 + 真服务，51 项断言（含流水线加速验证） |
| `requirements.txt` | Python 依赖 |

仓库根目录还提供了把**四个服务一起**拉起来的编排文件：

| 文件 | 说明 |
| --- | --- |
| `../docker-compose.yml` | 四服务同网络（CPU 也能跑） |
| `../docker-compose.gpu.yml` | GPU 覆盖文件（已 include 基础文件，可单独 `-f` 使用） |
| `../.env.example` | 全栈部署的默认变量 |

---

## 2. 快速开始

### 2.1 单独部署（推荐先跑这个验证接线）

前提：`yolov13_formula_detection_service`(:9000)、`pp-formulanet-plus-l`(:9001)、
`slanet_plus_service`(:9002) 已经在宿主机或别的地方跑起来了。

```bash
cd router_service
cp .env.example .env
docker compose up -d --build
curl http://127.0.0.1:9003/v1/info
```

> ⚠️ **下游地址必须改对，否则连接被拒。**
> `.env.example` 里的 `127.0.0.1` 是**容器内**的地址，指的是容器自己，不是宿主机。
> 下游跑在宿主机上（容器里只有 router）时，把 `.env` 改成宿主地址：
>
> ```bash
> DETECTION_URL=http://host.docker.internal:9000   # Docker Desktop（Win/macOS）
> FORMULA_URL=http://host.docker.internal:9001
> TABLE_URL=http://host.docker.internal:9002
> ```
>
> 下游在别的机器上就写那台机器的 IP（`http://192.168.1.50:9000`）。
> `host.docker.internal` 已通过 `extra_hosts: host-gateway` 配好，Linux 上也能解析。
> 只有本服务以 `network_mode: host` 运行、或不走容器直接跑 app 时，`127.0.0.1` 才指向真正的本机服务。

### 2.2 全栈一键部署（含三个模型服务）

```bash
cd ..                        # 回到 formula_table_service_group 根目录
cp .env.example .env
docker compose up -d --build

# 有 NVIDIA GPU 时使用 GPU 覆盖文件（已 include 基础文件，两种写法等价）
docker compose -f docker-compose.gpu.yml up -d --build

docker compose ps
curl http://127.0.0.1:9003/ready
```

> 三个模型服务首次加载权重需要几分钟（pp-formulanet 约 1~3 分钟），
> 这期间 `/v1/router/predict` 会返回 502、`/ready` 返回 503，属正常现象。
> 用 `docker compose logs -f` 观察加载进度。

### 2.3 不使用 compose

```bash
docker build -t router-service:1.0.0 .

docker run -d --name router-service -p 9003:9003 \
  -e DETECTION_URL=http://host.docker.internal:9000 \
  -e FORMULA_URL=http://host.docker.internal:9001 \
  -e TABLE_URL=http://host.docker.internal:9002 \
  --add-host=host.docker.internal:host-gateway \
  router-service:1.0.0
```

### 2.4 本地直接跑（开发调试）

```bash
pip install -r requirements.txt
# 下游地址用环境变量指定
set DETECTION_URL=http://127.0.0.1:9000
set FORMULA_URL=http://127.0.0.1:9001
set TABLE_URL=http://127.0.0.1:9002
python -m uvicorn app.main:app --host 0.0.0.0 --port 9003 --reload
```

Swagger 文档：<http://127.0.0.1:9003/docs>

---

## 3. 接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/health` | 自身存活 + 三个下游状态（不阻塞） |
| GET | `/ready` | 就绪探针；下游任一不可用返回 503 |
| GET | `/v1/info` | 服务信息与全部配置 |
| GET | `/v1/stats` | 请求准入 / 流水线 / 会话 汇总 |
| GET | `/v1/pools` | 公式池、表格池的积压与吞吐 |
| GET | `/metrics` | Prometheus 文本指标 |
| POST | `/v1/router/predict` | **主接口**：JSON + base64 图片 |
| POST | `/v1/router/predict/upload` | multipart 文件上传（等价通道） |
| GET | `/v1/router/sessions` | 列出所有会话 |
| GET | `/v1/router/session/{session_id}` | 单个会话状态（请求数、in_flight、失败数） |
| GET | `/v1/router/session/{session_id}/results` | 该会话最近若干次请求的合并结果 |
| DELETE | `/v1/router/session/{session_id}` | 清除会话（释放结果缓存） |

### POST /v1/router/predict

```bash
curl -s -X POST http://127.0.0.1:9003/v1/router/predict \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "client-42",
    "keep_crops": true,
    "images": [
      {"name": "page_0001.png", "page_index": 0,
       "pdf_width": 612.0, "pdf_height": 792.0,
       "data": "data:image/png;base64,iVBORw0KGgo..."}
    ],
    "params": {"conf": 0.25, "imgsz": 640}
  }'
```

* `session_id`：**可选**。同一个 PDF 的多页/多次请求请复用同一个值，这样结果会归到同一个会话；
  不传则服务自动生成并在响应里回传。
* `params.return_crops`：由服务强制为 `true`（识别必须拿到框内图像），传了也会被忽略。
* `params` 里其余字段全部可省略，省略时用服务默认值（环境变量 `DETECT_*`）。

### 响应：保留 yolov13 元数据并分组检测框

检测服务的其它字段（`status` / `model` / `classes` / `images` / `summary` / `timing` …）保持不变。
router 将原始 `detections` 拆分为 `recognized_detections` 和 `unrecognized_detections`：

```json
{
  "status": "ok",
  "session_id": "client-42",
  "recognized_detections": [
    {
      "page_index": 0, "class_id": 1, "category_id": 2, "class_name": "DisplayedFormulaLine",
      "confidence": 0.93,
      "bbox": [577.0, 657.0, 689.0, 683.0],
      "bbox_pdf": [277.0, 315.4, 330.7, 327.8],
      "bbox_norm": [0.4965, 0.4061, 0.0878, 0.0158],
      "crop": {"format": "png", "encoding": "base64", "data": "iVBORw0KGgo..."},

      "formula": {
        "latex": "\\zeta_{0}(\\nu)=-\\frac{\\nu\\varrho^{-2\\nu}}{\\pi}\\int_{\\mu}^{\\infty}d\\omega",
        "error": null,
        "elapsed_ms": 823.4,
        "wait_ms": 12.0
      }
    },
    {
      "page_index": 1, "class_name": "Table", "confidence": 0.88,
      "crop": {"format": "png", "encoding": "base64", "data": "iVBORw0KGgo..."},

      "table": {
        "html": "<html><body><table><tr><td></td><td></td></tr></table></body></html>",
        "structure_score": 0.97,
        "num_cells": 12,
        "error": null,
        "elapsed_ms": 210.5,
        "wait_ms": 8.1
      }
    }
  ],
  "unrecognized_detections": [
    {
      "page_index": 1, "class_name": "Figure", "confidence": 0.91,
      "crop": {"format": "png", "encoding": "base64", "data": "iVBORw0KGgo..."}
    },
    {
      "page_index": 1, "class_name": "FormulaNumber", "confidence": 0.88,
      "crop": {"format": "png", "encoding": "base64", "data": "iVBORw0KGgo..."}
    }
  ],

  "router": {
    "request_id": "req_9f3c1a20b7d4",
    "session_id": "client-42",
    "counts": {"formula": 1, "table": 1, "figure": 1, "other": 1, "total_detections": 4},
    "recognition": {"formula_ok": 1, "table_ok": 1, "failed": 0},
    "timing": {"detection_ms": 4820.1, "recognition_ms": 1180.6, "total_ms": 6003.2},
    "pipeline": "detection producer -> formula/table buffer pools -> recognition consumers"
  }
}
```

* `recognized_detections` 包含命中 `FORMULA_CLASSES` 的公式框及命中 `TABLE_CLASSES` 的表格框；`formula` / `table` 只出现在对应框上。
* `unrecognized_detections` 包含 Figure 及未配置为公式/表格识别的类别（例如不在 `FORMULA_CLASSES` 中的公式类）；这些框不会调用识别服务，也不会附加 `formula` / `table` 结果块。
* router 响应不再提供合并的 `detections` 数组；需要遍历全部检测框的客户端可拼接上述两个数组。
* `router.counts.total_detections` 始终等于两个数组长度之和。
* `wait_ms` = 该框在缓存池里排队等待的时间；`elapsed_ms` = 所在批次识别耗时（批内均摊）。
* 某一个框识别失败**不会**让整个请求失败：该框的 `*.error` 有值，其余照常返回。
* `keep_crops: false` 时所有 `crop` 字段都被移除，用于只要文字结果的场景（响应体小很多）。

### POST /v1/router/predict/upload

表单字段与 yolov13 的 `/predict/upload` 对齐：
`files`（可重复）/ `file`、`session_id`、`keep_crops`、`conf`、`iou`、`imgsz`、`max_det`、`batch`、
`dedup`、`dedup_thr`、`classes`、`crop_format`、`crop_quality`、`crop_padding`、`page_indices`、`pdf_sizes`。

```bash
curl -s -X POST http://127.0.0.1:9003/v1/router/predict/upload \
  -F "files=@page_0001.png" -F "files=@page_0002.png" \
  -F "session_id=client-42" -F "pdf_sizes=612x792,612x792"
```

> 该通道会把上传的图片转成 base64 后走同一条流水线（保证两条入口行为一致）。

---

## 4. 流水线细节

### 4.1 缓存池与背压

| 池 | 内容 | 默认消费者 |
| --- | --- | --- |
| 公式缓存池 | 公式框的框内图像（base64） | `FORMULA_WORKERS=2` 个 worker，每次最多取 `FORMULA_BATCH_SIZE=4` 条 |
| 表格缓存池 | 表格框的框内图像（base64） | `TABLE_WORKERS=1` 个 worker，每次最多取 `TABLE_BATCH_SIZE=1` 条 |

* worker 先阻塞等第一条任务，再在 `BATCH_WINDOW_MS`（默认 20ms）内尽量凑满一批，
  然后**一次调用**下游的 `/base64` 批量接口 —— 减少往返、提高 GPU 利用率。
* `POOL_OVERFLOW=block`（默认）：池满时投递方等待空位，形成背压，**不丢数据**；
  `drop`：池满时该条任务立即判失败（适合宁可降级也不愿排队拖慢的场景，表现为 `*.error` 有值）。
* 两个池属于**同一个进程**的所有请求：客户端 A 的公式任务和客户端 B 的公式任务会一起凑批，
  由 `session_id` 区分归属；这就是「持续流水」而不是「一请求一模型」。
* 看积压与吞吐：`GET /v1/pools`。

### 4.2 多客户端与 session

* `MAX_CONCURRENT_REQUESTS`（默认 **2**）控制**同时在处理**的请求数，
  第 3 个请求会在网关排队，最多等 `QUEUE_TIMEOUT` 秒，超时返回 `503`（提示服务繁忙）。
* 每个请求都归属一个 `session_id`，池里的任务与归档的结果都带这个标记，互不串扰。
* 想批量跑多个 PDF：直接用不同的 `session_id` 并发发请求即可（流水线会自然重叠）。

### 4.3 超时与失败语义

| 情况 | 表现 |
| --- | --- |
| 检测服务不可用 / 模型未就绪 | `502`，`detail` 带下游错误（未就绪时是 503 的 detail） |
| 某个框识别失败 | 该框 `formula.error` / `table.error` 有值，`router.recognition.failed` +1，整体仍是 `200` |
| 识别超过 `REQUEST_TIMEOUT` | 记录等待告警并继续等待，未完成的框不会被提前标记失败；HTTP 请求会保持等待直到识别完成 |
| 公式服务返回 predictor 忙碌 503 | Router 保留当前批次并按退避间隔重试，直到公式服务接受并完成该批次 |
| 并发超过上限且排队超时 | `503` 服务繁忙 |

---

## 5. 环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `DETECTION_URL` | `http://127.0.0.1:9000` | yolov13 检测服务地址 |
| `FORMULA_URL` | `http://127.0.0.1:9001` | pp-formulanet-plus-l 地址 |
| `TABLE_URL` | `http://127.0.0.1:9002` | slanet_plus 地址 |
| `SERVICE_PORT` | `9003` | 容器内监听端口 |
| `SERVICE_HOST` | `0.0.0.0` | 监听地址 |
| `LOG_LEVEL` | `INFO` | 日志级别 |
| `MAX_CONCURRENT_REQUESTS` | `2` | 同时处理的请求数上限 |
| `MAX_QUEUE` | `64` | 排队容量上限，排满后新请求直接 503 |
| `QUEUE_TIMEOUT` | `30` | 排队等待准入的秒数，超时 503 |
| `REQUEST_TIMEOUT` | `300` | 等待识别结果多久后开始记录告警（秒）；告警后继续等待，不会丢弃未完成的框 |
| `SESSION_TTL` | `1800` | 会话（含结果缓存）保留秒数，0=不清理 |
| `FORMULA_WORKERS` | `2` | 公式识别并发 worker 数 |
| `TABLE_WORKERS` | `1` | 表格识别并发 worker 数 |
| `FORMULA_BATCH_SIZE` | `4` | 公式 worker 每次取件上限（下游上限 8） |
| `TABLE_BATCH_SIZE` | `1` | 表格 worker 每次取件上限；当前避免 SLANet/PaddleX 批量预测的 bbox 数量不匹配 |
| `BATCH_WINDOW_MS` | `20` | 凑批等待窗口（毫秒） |
| `POOL_MAX_SIZE` | `512` | 每个缓存池的容量 |
| `POOL_OVERFLOW` | `block` | `block`=背压不丢数据；`drop`=池满判失败 |
| `HTTP_CONNECT_TIMEOUT` | `10` | 连接下游超时（秒） |
| `HTTP_READ_TIMEOUT` | `300` | 读取下游响应超时（秒） |
| `HTTP_MAX_CONNECTIONS` | `32` | HTTP 连接池上限 |
| `FORCE_RETURN_CROPS` | `true` | 强制让检测服务返回框内图像（识别必需） |
| `DETECT_CONF` / `DETECT_IOU` / `DETECT_IMGSZ` | `0.25` / `0.7` / `640` | 检测默认参数 |
| `DETECT_MAX_DET` / `DETECT_BATCH` | `300` / `8` | 检测默认参数 |
| `DETECT_DEDUP` / `DETECT_DEDUP_THR` | `true` / `0.8` | 检测默认参数 |
| `DETECT_CROP_FORMAT` / `DETECT_CROP_QUALITY` / `DETECT_CROP_PADDING` | `png` / `90` / `0` | 框内图像参数 |
| `FORMULA_CLASSES` | `InlineFormula,DisplayedFormulaLine,FormulaNumber,DisplayedFormulaBlock` | 判为公式的类别 |
| `TABLE_CLASSES` | `Table` | 判为表格的类别 |
| `FIGURE_CLASSES` | `Figure` | 判为图片的类别（不识别） |

`FORMULA_CLASSES` / `TABLE_CLASSES` / `FIGURE_CLASSES` 也接受 `category_id` / `class_id` 数字，
方便换成别的训练集。

---

## 6. 测试

### 6.1 离线冒烟测试（不需要 GPU / 不需要下游）

脚本会用桩服务模拟三个下游，然后把 router service 真正跑起来跑完 51 项断言
（探针、分流计数、结果合并、批量取件、会话隔离、2 客户端并发、multipart 通道、参数透传、流水线加速）。

```bash
pip install fastapi uvicorn httpx python-multipart
python scripts/smoke_test_router.py       # 期望：51 通过 / 0 失败，退出码 0
```

其中「流水线加速验证」用固定耗时的桩下游做对比：3 个公式框 + 2 个表格框，
串行需要 `3*0.3 + 2*0.3 = 1.5s`，实测只要约 `0.4s`（批量推理 + 两条流水线并行）。

### 6.2 联调真实部署

```bash
python scripts/smoke_test_router.py --url http://127.0.0.1:9003
```

### 6.3 整组部署的端到端测试（仓库根目录脚本）

`../test_group_service.py` 直接打四个真实服务（9000 / 9001 / 9002 / 9003），
把 PDF 渲染图送进完整流水线，校验探针、合并结构、分类计数、会话归档、上传通道与双客户端并发：

```bash
cd ..                                            # 到仓库根目录
python test_group_service.py                     # 默认地址 + 自带 latex_sample.pdf
python test_group_service.py --json report.json --out result.json
python test_group_service.py --url http://192.168.1.50:9003 --wait 300   # 远程部署 / 首启等模型加载
python test_group_service.py --skip-direct --skip-concurrency            # 只测 router 主流程
```

结尾会打印「总计 N 项：通过 / 失败 / 告警 / 跳过」并列出失败项，有失败时退出码为 1。

### 6.4 手工压一下流水线

```bash
# 同一时刻两个客户端
curl -s -X POST http://127.0.0.1:9003/v1/router/predict -H "Content-Type: application/json" \
  -d '{"session_id":"A","images":[{"data":"<base64>"}]}' &
curl -s -X POST http://127.0.0.1:9003/v1/router/predict -H "Content-Type: application/json" \
  -d '{"session_id":"B","images":[{"data":"<base64>"}]}' &
wait

curl -s http://127.0.0.1:9003/v1/pools | python -m json.tool     # 看两个池的积压/吞吐
curl -s http://127.0.0.1:9003/v1/router/sessions | python -m json.tool
```

---

## 7. 故障排查

| 现象 | 原因 / 处理 |
| --- | --- |
| 报 `Connection refused` / `网络错误`，三个下游全挂 | 下游 URL 还是容器内的 `127.0.0.1`。改成 `http://host.docker.internal:<端口>` 或下游机器 IP |
| `/ready` 返回 503 | 三个下游没就绪；`/health` 的 `downstream` 字段会指出是哪一个 |
| `predict` 返回 502 且 detail 提到 503 | 检测模型还在加载（首次启动几分钟） |
| `*.error` 是「检测结果未包含框内图像」 | 下游检测关闭了 `return_crops`；本服务默认强制开启，检查是否被改过 |
| `503 服务繁忙` | 并发已到 `MAX_CONCURRENT_REQUESTS`；调大该值或调大 `QUEUE_TIMEOUT` |
| 公式识别很慢 | 调大 `FORMULA_WORKERS` / `FORMULA_BATCH_SIZE`；或在 pp-formulanet 侧调大 `PREDICTOR_POOL_SIZE` |
| 池里积压持续增长 | 消费不过来；`GET /v1/pools` 看 `size`/`in_flight`，必要时加 `POOL_MAX_SIZE` 与 worker 数 |
| 内存占用高 | 大 PDF 的 base64 会占用较多内存；可 `keep_crops: false`、降低 `DETECT_CROP_QUALITY`、调小 `SESSION_TTL` |
