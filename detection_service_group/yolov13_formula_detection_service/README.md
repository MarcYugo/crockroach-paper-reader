# YOLOv13 公式检测服务

把仓库根目录的 [inference.py](./inference.py) 的推理能力封装成 HTTP 服务：**输入 PDF 渲染后的页面图片，输出检测框、类别、置信度，以及框内区域的裁剪图像（base64）**，全部以 JSON 返回。

- 推理后处理（坐标换算、类别过滤、同类框去重）直接 `import` 复用 `inference.py`，输出字段与命令行脚本保持一致。
- 检测类别：`InlineFormula` / `DisplayedFormulaLine` / `FormulaNumber` / `DisplayedFormulaBlock` / `Table` / `Figure`（`category_id = class_id + 1`）。

## 目录结构

| 文件 | 说明 |
| --- | --- |
| `Dockerfile` | 服务镜像（Python 3.11 + torch 2.4.0 + FastAPI） |
| `docker-compose.yml` | GPU 部署编排（NVIDIA Container Toolkit） |
| `docker-compose.cpu.yml` | 纯 CPU 部署编排 |
| `app.py` | FastAPI 接口（`/health`、`/info`、`/predict`、`/predict/upload`） |
| `detector.py` | 模型加载、批推理、后处理与框内图像裁剪 |
| `schemas.py` | 请求/响应模型、参数合并与校验 |
| `config.py` | 环境变量解析 |
| `requirements.txt` | 服务 Python 依赖 |
| `test_detection_service.py` | 测试脚本：输入 PDF，调用 `/predict` 并校验返回的 JSON |
| `.env` | 部署变量（`TORCH_INDEX_URL` / `PORT` / `INSTALL_FLASH_ATTN`），由 docker compose 自动读取 |
| `models/yolov13_arxivformula_sft/yolov13s_ep5_bs32_lr_0.01.pt` | 默认权重（Compose 以只读文件挂载，可替换） |

> 构建上下文是**当前仓库根目录**（镜像里需要 `inference.py` 与 `yolov13/` 源码），两个 compose 文件均已用 `context: .` 配好。

## 快速开始

```bash
# GPU（需要宿主机 NVIDIA 驱动 + nvidia-container-toolkit）
docker compose up -d --build

# 纯 CPU
docker compose -f docker-compose.cpu.yml up -d --build

curl -s http://127.0.0.1:8000/info
```

不使用 compose：

```bash
# 在仓库根目录运行，构建上下文包含 Dockerfile、inference.py 与 yolov13/
docker build -f Dockerfile -t yolov13-formula-detection:latest .
docker run -d --name formula-detection --gpus all -p 8000:8000 \
  -e DEVICE=0 -e CONF=0.25 -e IMGSZ=640 \
  yolov13-formula-detection:latest
```

### 部署变量 .env

同目录的 [`.env`](./.env) 由 docker compose 自动读取，无需改 compose 文件即可调整构建与端口：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `TORCH_INDEX_URL` | `https://download.pytorch.org/whl/cu121` | 构建镜像时的 PyTorch 源（`cu121` / `cu118` / `cpu`）。仅 `docker-compose.yml` 使用；`docker-compose.cpu.yml` 固定用 CPU 源。改用 CPU 源时 `INSTALL_FLASH_ATTN` 必须为 `0` |
| `PORT` | `8000` | 服务端口，同时作为容器内监听端口（`SERVICE_PORT`）与宿主机映射端口。改后请求地址随之变化，如 `PORT=9000` → `http://127.0.0.1:9000/docs`。直接用 `docker run` 时可用 `-e PORT=9000` 或 `-e SERVICE_PORT=9000` |
| `INSTALL_FLASH_ATTN` | `0` | `1` = 安装 `wheels/` 中的 FlashAttention-2（需 cu121 + Ampere 及以上 GPU）。仅 `docker-compose.yml` 使用；为 `0` 时自动回退 PyTorch SDPA，功能不受影响 |

改完变量后需要**重建**才会生效（`TORCH_INDEX_URL` / `INSTALL_FLASH_ATTN` 属于构建参数）：

```bash
docker compose up -d --build          # 仅改 PORT 时 docker compose up -d 即可
```

在别的目录用 `-f` 调用时，需显式指定变量文件：

```bash
docker compose --env-file yolov13_formula_detection_service/.env \
  -f yolov13_formula_detection_service/docker-compose.yml up -d --build
```

Swagger 文档：`http://127.0.0.1:8000/docs`

## 测试

服务启动后，用测试脚本跑一遍「PDF → JSON」：

```bash
pip install pymupdf requests pillow          # 仅本地需要，或用服务镜像的依赖
python test_detection_service.py                                  # 默认用 latex_sample.pdf，1 页
python test_detection_service.py --pdf paper.pdf --max-pages 4     # 多页
python test_detection_service.py --url http://127.0.0.1:9000       # 端口以 .env 的 PORT 为准
python test_detection_service.py --out result.json                # 顺便保存返回的 JSON
```

脚本会渲染 PDF 页面 → 调用 `/predict` → 打印每页检测数与分类统计，并校验：状态为 `ok`、`bbox` 在图像范围内、
`category_id == class_id + 1`、`crop` 能被 base64 解码且尺寸与 `crop.rect` 一致。通过则输出 `TEST PASSED`（退出码 0）。

## 接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/health` | 健康检查；模型未就绪返回 503 |
| GET | `/info` | 类别表、权重路径、设备、默认参数与请求限制 |
| POST | `/predict` | JSON 请求（图片为 base64） |
| POST | `/predict/upload` | multipart 文件上传 |

### POST /predict（JSON + base64）

```bash
curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "images": [
      {
        "name": "page_0001.png",
        "page_index": 0,
        "pdf_width": 612.0,
        "pdf_height": 792.0,
        "data": "data:image/png;base64,iVBORw0KGgo..."
      }
    ],
    "params": {"conf": 0.25, "imgsz": 640, "crop_format": "png"}
  }'
```

- `images[].data`：图片 base64，可带 `data:image/png;base64,` 前缀。
- `images[].page_index`：0 基页码，缺省按数组顺序；`pdf_width/pdf_height`（pt）可选，传入后 `bbox_pdf` 才是 PDF 坐标。
- `params` 全部可省略，省略时使用服务默认值（环境变量）。

### POST /predict/upload（multipart）

```bash
curl -s -X POST http://127.0.0.1:8000/predict/upload \
  -F "files=@page_0001.png" \
  -F "files=@page_0002.png" \
  -F "classes=InlineFormula,DisplayedFormulaLine" \
  -F "crop_format=jpg" -F "crop_padding=2" \
  -F "pdf_sizes=612x792,612x792"
```

表单字段：`files`（可重复，或用 `file` 传单张）、`conf`、`iou`、`imgsz`、`max_det`、`batch`、`dedup`、`dedup_thr`、`classes`、`return_crops`、`crop_format`、`crop_quality`、`crop_padding`、`page_indices`、`pdf_sizes`。

### 响应结构

```json
{
  "status": "ok",
  "model": {"weights": "...best.pt", "imgsz": 640, "conf": 0.25, "iou": 0.7, "max_det": 300, "batch": 8, "device": "cuda:0"},
  "classes": [{"id": 0, "category_id": 1, "name": "InlineFormula"}],
  "images": [
    {"image_index": 0, "page_index": 0, "page_number": 1, "file_name": "page_0001.png",
     "width": 1275, "height": 1650, "num_detections": 19, "num_duplicates_removed": 6}
  ],
  "detections": [
    {
      "page_index": 0,
      "page_number": 1,
      "class_id": 1,
      "category_id": 2,
      "class_name": "DisplayedFormulaLine",
      "confidence": 0.93,
      "bbox": [577.0, 657.0, 689.0, 683.0],
      "bbox_pdf": [277.0, 315.4, 330.7, 327.8],
      "bbox_norm": [0.4965, 0.4061, 0.0878, 0.0158],
      "crop": {
        "format": "png",
        "encoding": "base64",
        "width": 112,
        "height": 26,
        "bytes": 1832,
        "rect": [577, 657, 689, 683],
        "data": "iVBORw0KGgo..."
      }
    }
  ],
  "dedup": {"enabled": true, "overlap_threshold": 0.8, "removed_count": 6, "removed": []},
  "summary": {"num_images": 1, "total_detections": 19, "duplicates_removed": 6,
              "per_class": {"DisplayedFormulaLine": 10, "FormulaNumber": 8, "DisplayedFormulaBlock": 1}},
  "timing": {"inference_ms": 4779.5, "total_ms": 4925.4}
}
```

字段语义：

- `bbox`：渲染图像素坐标 `[x1, y1, x2, y2]`，左上原点 —— PDF 阅读器画框直接用这个。
- `bbox_pdf`：PDF 点坐标；未传 `pdf_width/pdf_height` 时与 `bbox` 相同。
- `bbox_norm`：归一化 `xywh`（与 YOLO 标签一致）。
- `crop.rect`：实际裁剪区域（含 `crop_padding` 外扩，已裁剪到图像范围内），`crop.data` 为其 base64 图像。
- `dedup.removed`：被同类去重去掉的较小框（附 `overlap_ratio`、`kept_bbox` 等），用于核查。
- `detections` 按阅读顺序（页 → 上 → 左）排序。

## 环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `MODEL_PATH` | `models/yolov13_arxivformula_sft/yolov13s_ep5_bs32_lr_0.01.pt` | 权重路径 |
| `DEVICE` | 空（自动） | `""` 自动、`0` 第一块 GPU、`cpu` |
| `IMGSZ` | `640` | 推理输入尺寸（小字号行内公式漏检时可试 960/1280） |
| `CONF` | `0.25` | 置信度阈值 |
| `IOU` | `0.7` | NMS IoU 阈值 |
| `MAX_DET` | `300` | 每张图最多保留的框数 |
| `BATCH` | `8` | 每批送入模型的图片数 |
| `DEDUP` | `true` | 同类框去重（公式框检查） |
| `DEDUP_THR` | `0.8` | 去重阈值，`1.0` 只去除完全重合的框 |
| `RETURN_CROPS` | `true` | 默认是否返回框内区域图像 |
| `CROP_FORMAT` | `png` | `png` / `jpg` |
| `CROP_QUALITY` | `90` | `jpg` 质量 |
| `CROP_PADDING` | `0` | 框外扩像素数 |
| `WARMUP` | `true` | 启动时预热一次，避免首个请求超时 |
| `MAX_IMAGES` | `50` | 单次请求图片数上限 |
| `MAX_PIXELS` | `40000000` | 单张图片像素上限（约 40MP） |
| `MAX_UPLOAD_MB` | `30` | 单张图片字节上限 |
| `SERVICE_PORT` | `8000` | 容器内监听端口；未设置时回退到 `PORT` |
| `LOG_LEVEL` | `info` | 日志级别 |

请求级参数（`/predict` 的 `params`、`/predict/upload` 的表单字段）优先级高于环境变量。

## 常见问题

- **模型加载失败**：服务仍会启动，但 `/predict` 返回 503、`/info.error` 会给出原因；常见原因是 `MODEL_PATH` 不存在。
- **权重 `PermissionError`**：检测服务以 UID 1000 的 `appuser` 运行。挂载目录及其子目录必须允许该用户遍历，所有 `.pt` 权重文件必须允许读取。进入 `yolov13_formula_detection_service` 目录后，在宿主机执行：
  ```bash
  find models -type d -exec chmod a+rx {} +
  find models -type f -name '*.pt' -exec chmod a+r {} +
  ```
  这只调整目录的读/遍历权限及 `.pt` 文件的读权限，不改变所有权；挂载仍为只读。若权重是通过其他路径挂载，请对宿主机对应的模型目录执行同样操作。修改后重建容器无需重新构建镜像，执行 `docker compose up -d --force-recreate`。
- **想用 FlashAttention-2 提速**：把 [`.env`](./.env) 里的 `INSTALL_FLASH_ATTN` 改成 `1` 后重建即可（或 `docker compose build --build-arg INSTALL_FLASH_ATTN=1`；仅 Ampere 及以上 GPU 有意义），会安装 `wheels/` 里的 `flash_attn-2.8.3+cu12torch2.4...cp311...whl`；默认不装，模型会自动回退到 PyTorch 的 `scaled_dot_product_attention`。
- **GPU 不生效**：确认宿主机有 NVIDIA 驱动与 `nvidia-container-toolkit`，`docker run` 需要 `--gpus all`，compose 已配好 `deploy.resources.reservations.devices`。
- **替换权重**：compose 以只读方式将 `./models/yolov13_arxivformula_sft/yolov13s_ep5_bs32_lr_0.01.pt` 挂载到容器内；替换该文件后执行 `docker compose up -d --force-recreate` 以重新挂载。确保 Docker 宿主机可读取此文件。
- **镜像较大**：CUDA 版 torch 及其 CUDA 运行库约 7~8 GB；纯 CPU 版明显更小，可用 `docker-compose.cpu.yml`。

## 本地（非容器）运行

```bash
pip install -r yolov13_formula_detection_service/requirements.txt
cd yolov13_formula_detection_service
python app.py        # 读取上面的环境变量，默认监听 0.0.0.0:8000
```
