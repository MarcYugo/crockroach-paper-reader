"""detection-service-group 客户端：公式/表格/图片检测服务组（`formula_table_service_group`）。

服务组是一条**生产者-消费者流水线**（`router_service` :9003 是唯一入口）：

    yolov13 公式检测（生产者） → 公式缓存池 / 表格缓存池 → pp-formulanet-plus-l /
    slanet_plus（消费者），最后把结果并回检测 JSON 返回。

本模块只做**客户端**，对接 `POST /v1/router/predict`（JSON + base64 页面图）：

  1. 服务组只吃**图片**、不吃 PDF —— 先用 PyMuPDF 把页面渲染成 PNG
     (`render_page_png`)，以 data URI 塞进 `images[*].data`；同一项里带上
     `pdf_width` / `pdf_height`(pt)，服务组回包里的 `bbox_pdf` 才是 PDF 坐标
     （`bbox_norm` 是归一化值，两种情况下都按渲染图算，见第 3 点）。
  2. 回包里提取公式框和图片框：`recognized_detections[*]` 中带 `formula` 的条目提供
     LaTeX；`unrecognized_detections[*]` 中的 Figure 条目提供图片裁图。
  3. `bbox_norm` 是 `[x中心, y中心, 宽, 高]` 的**归一化值**（YOLO 标签口径，除以页面
     渲染尺寸；渲染图与 PDF 同比例，所以乘「页面显示宽高」即可还原像素矩形 —— 前端
     `r2-math.js` 就是这么定位覆盖层的）。
  4. Figure 裁图通过 `extract_figure_detections` 提供给解析器；公式框裁图用于 KaTeX
     渲染失败时兜底。图片裁图是显示内容，解析流程会强制要求服务组返回 crop。

对外契约（`pdf_parser/options.py::status()` 与 `pdf_parser/backend_detection_service_group.py`
在用，别改坏）：`DEFAULT_URL` / `DEFAULT_TIMEOUT` / `PROBE_TIMEOUT` / `check_server` /
`render_page_png` / `collect_images` / `predict` / `extract_formula_boxes` /
`extract_figure_detections`。

⚠️ 失败一律抛 `DetectionGroupUnavailable`；调用方（解析后处理）只加 warning，
**绝不因为服务组挂了让整篇解析失败**。
"""
from __future__ import annotations

import base64
import json
import socket
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

# 默认地址与 `formula_table_service_group/router_service` 的端口保持一致
# （见其 docker-compose.yaml：SERVICE_PORT=9003）
DEFAULT_URL = "http://127.0.0.1:9003/v1"
# 一次 predict 的读取超时：检测 + 公式/表格识别全在这一次请求里完成，
# 页数多时流水线要跑一会儿（router 侧 REQUEST_TIMEOUT 默认 300s）
DEFAULT_TIMEOUT = 600.0
# 就绪探测的超时：面板拉状态时不能卡住
PROBE_TIMEOUT = 2.0
# 仅当 config.json 或 parser.dpi 不存在时使用；正常解析读取配置值。
_FALLBACK_DPI = 150


class DetectionGroupUnavailable(RuntimeError):
    """detection-service-group（检测服务组）不可用，或预测请求失败/超时。"""


def configured_dpi(config_file: str | Path | None = None) -> int:
    """读取 config.json 的 parser.dpi；配置文件缺失时使用兼容回退值。"""
    path = (Path(config_file) if config_file is not None
            else Path(__file__).resolve().parents[1] / "config.json")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return _FALLBACK_DPI
    except (OSError, json.JSONDecodeError) as exc:
        raise DetectionGroupUnavailable(f"无法读取公式渲染 DPI 配置 {path}：{exc}") from exc

    parser = raw.get("parser") if isinstance(raw, dict) else None
    value = parser.get("dpi") if isinstance(parser, dict) else None
    if value is None:
        return _FALLBACK_DPI
    try:
        return max(48, min(400, int(value)))
    except (TypeError, ValueError) as exc:
        raise DetectionGroupUnavailable(
            f"config.json 中 parser.dpi 不是有效整数：{value!r}") from exc


def normalize_url(url: str | None = None) -> str:
    """补全默认地址，并去掉结尾多余的 '/'。"""
    u = (url or "").strip() or DEFAULT_URL
    return u.rstrip("/")


def _root(url: str) -> str:
    """服务根地址（`/v1` 之外）。`/health`、`/ready`、`/metrics` 挂在根路径上。"""
    return url[:-3] if url.endswith("/v1") else url


def _get_json(url: str, timeout: float) -> dict:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data if isinstance(data, dict) else {}


def check_server(url: str | None = None,
                 timeout: float = PROBE_TIMEOUT) -> tuple[bool, str]:
    """探测服务组就绪状态。返回 `(是否就绪, 说明文本)`。

    `GET /health` 由 router 返回自身状态与三个下游的状态；`status == "ok"` 是
    流水线已就绪（下游是否可用另看 `downstream`，这里如实拼进说明文本，
    方便面板区分「router 没起」与「某个模型服务没起」）。
    """
    base = normalize_url(url)
    health_url = _root(base) + "/health"
    try:
        data = _get_json(health_url, timeout)
    except Exception as exc:  # 连接失败/超时/非 JSON 都算不可用
        return False, f"无法连接 detection-service-group {health_url}：{exc}"

    status = str(data.get("status") or "").lower()
    down = data.get("downstream") or {}
    problems = [f"{name}={info.get('status')}"
                for name, info in down.items()
                if isinstance(info, dict) and str(info.get("status")) != "ok"]
    if status != "ok":
        return False, (f"服务组尚未就绪（{health_url} 返回 status={status or '未知'}）"
                       + (f"；下游：{'、'.join(problems)}" if problems else ""))
    if problems:
        return True, (f"服务组已就绪，但下游有异常：{'、'.join(problems)}"
                      "（Figure 图片或公式识别可能缺失或降级）")
    return True, "服务组已就绪（检测 / 公式 / 表格三个下游均正常）"


# =====================================================================
#  页面渲染（服务组只吃图片，PDF → PNG 在本地做）
# =====================================================================
def render_page_png(page, dpi: int | None = None) -> bytes:
    """PyMuPDF 页面 → PNG 字节（整页渲染，作为检测/识别的输入）。

    dpi 越大越清晰、上传与推理越慢；96~200 是比较稳的区间。**不需要 Pillow**：
    `pix.tobytes("png")` 直接给出 PNG。
    """
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:  # pragma: no cover - 项目本就依赖 PyMuPDF
        raise DetectionGroupUnavailable(f"缺少 PyMuPDF：pip install PyMuPDF ({exc})") from exc

    use_dpi = configured_dpi() if dpi is None else max(48, min(400, int(dpi)))
    zoom = max(0.5, float(use_dpi) / 72.0)
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    return pix.tobytes("png")


def collect_images(pdf_path, page_nos, opts: dict | None = None, *,
                   dpi: int | None = None) -> list[dict]:
    """把 `page_nos`（**1 基**页码）渲染成 `/v1/router/predict` 要的 `images` 数组。

    每项：`{name, page_index(0 基), pdf_width, pdf_height, data(data URI)}`。
    只渲染需要的页是主要的省时手段（没有公式空位的页根本不请求，见调用方）。
    """
    import fitz  # PyMuPDF

    options = opts or {}
    if dpi is not None:
        raw_dpi = dpi
    elif options.get("dpi") is not None:
        raw_dpi = options["dpi"]
    else:
        raw_dpi = configured_dpi()
    use_dpi = max(48, min(400, int(raw_dpi)))
    picks = sorted({int(p) for p in page_nos})
    out: list[dict] = []
    doc = fitz.open(str(pdf_path))
    try:
        for pno in picks:
            if pno < 1 or pno > doc.page_count:
                continue
            page = doc.load_page(pno - 1)
            if page.rotation % 360 != 0:
                page.set_rotation(0)       # 与版式解析同一坐标系
            png = render_page_png(page, use_dpi)
            out.append({
                "name": f"page_{pno:04d}.png",
                "page_index": pno - 1,     # 0 基：与服务组回包口径一致
                "pdf_width": float(page.rect.width),
                "pdf_height": float(page.rect.height),
                "data": "data:image/png;base64," + base64.b64encode(png).decode("ascii"),
            })
    finally:
        doc.close()
    return out


# =====================================================================
#  预测 / 结果提取
# =====================================================================
def predict(images: list[dict], opts: dict | None = None, *,
            url: str | None = None, timeout: float | None = None,
            conf: float | None = None,
            keep_crops: bool | None = None) -> dict:
    """`POST /v1/router/predict`：检测 → 分流 → 识别 → 合并，返回原始响应 dict。

    `images` 由 `collect_images()` 构造；`keep_crops`（默认取 `detection_service_group_keep_crops`，
    再默认 True）决定回包里带不带**框内裁图**——前端在 KaTeX 渲染失败时用它兜底
    （见 `frontend/js/r2-math.js` 的 `buildFormulaBoxes`）。只要预算够就开着；
    关掉只丢兜底能力，不影响公式框与 LaTeX。
    任何失败（连不上 / 非 2xx / 非 JSON）都抛 `DetectionGroupUnavailable`。
    """
    options = opts or {}
    base = normalize_url(url or options.get("detection_service_group_url"))
    use_timeout = float(timeout if timeout is not None
                        else options.get("detection_service_group_timeout")
                        or DEFAULT_TIMEOUT)
    # 先做一个快速的 TCP 连接探测：服务没起/端口不通时立刻失败 —— 否则
    # urlopen 会一直等到读取超时（默认 300s），解析就被拖住了。
    parts = urlsplit(base)
    try:
        socket.create_connection((parts.hostname or "127.0.0.1",
                                  parts.port or 80),
                                 timeout=min(5.0, use_timeout)).close()
    except OSError as exc:
        raise DetectionGroupUnavailable(f"无法连接服务组 {base}：{exc}") from exc
    if keep_crops is None:
        keep_crops = options.get("detection_service_group_keep_crops")
        if keep_crops is None:
            keep_crops = True
    payload: dict = {"images": list(images), "keep_crops": bool(keep_crops)}
    if conf is None:
        conf = options.get("detection_service_group_conf")
    if conf is not None:
        try:
            payload["params"] = {"conf": max(0.01, min(0.99, float(conf)))}
        except (TypeError, ValueError):
            pass
    req = urllib.request.Request(
        base + "/router/predict",
        data=json.dumps(payload).encode("utf-8"), method="POST")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=max(5.0, use_timeout)) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:300]
        raise DetectionGroupUnavailable(
            f"服务组请求失败(HTTP {exc.code})：{detail}") from exc
    except Exception as exc:
        raise DetectionGroupUnavailable(f"服务组请求失败：{exc}") from exc
    if not isinstance(data, dict):
        raise DetectionGroupUnavailable(f"服务组响应不是 JSON 对象：{str(data)[:200]}")
    return data


def _crop_data_uri(crop) -> str:
    """服务组的 `crop`（`{format, encoding, data}`）→ data URI；拿不到返回空串。"""
    if not isinstance(crop, dict):
        return ""
    data = str(crop.get("data") or "").strip()
    if not data:
        return ""
    if data.startswith("data:"):          # 已是 data URI：原样用
        return data
    fmt = str(crop.get("format") or "png").strip().lower()
    mime = "image/jpeg" if fmt in ("jpg", "jpeg") else f"image/{fmt or 'png'}"
    return f"data:{mime};base64,{data}"


def extract_formula_boxes(resp: dict) -> dict[int, list[dict]]:
    """从服务组回包里拆出**公式框**：`{page_index(0 基): [框, …]}`。

    每个框（即 `pages[].formula_boxes` 的落盘格式）：

        {"bbox_norm": [x中心, y中心, 宽, 高],   # 归一化(YOLO 口径)，前端按页面显示尺寸还原
         "latex": "\\frac{…}",                 # pp-formulanet-plus-l 的识别结果（可能为空）
         "crop": "data:image/png;base64,…",    # 框内裁图：前端渲染失败时的兜底（keep_crops=false 时没有）
         "score": 0.93,                        # 检测置信度（可空）
         "class_name": "DisplayedFormulaLine"} # 检测类别（InlineFormula / …）

    只要 **latex 或 crop 有一个**就保留（识别失败的框只留裁图，前端直接贴图兜底）；
    两个都没有的条目跳过；表格（`table`）与图片类条目也跳过 —— 本次只落地公式。
    """
    out: dict[int, list[dict]] = {}
    for det in (resp or {}).get("recognized_detections") or []:
        if not isinstance(det, dict):
            continue
        form = det.get("formula")
        if not isinstance(form, dict):
            continue                       # 表格 / 图片类：没有 formula 块
        latex = str(form.get("latex") or "").strip()
        crop_uri = _crop_data_uri(det.get("crop"))
        if not latex and not crop_uri:
            continue                       # 既没 LaTeX 也没裁图：这条没法落地
        norm = det.get("bbox_norm")
        if not isinstance(norm, (list, tuple)) or len(norm) != 4:
            continue
        try:
            box = [float(v) for v in norm]
        except (TypeError, ValueError):
            continue
        try:
            page_index = int(det.get("page_index"))
        except (TypeError, ValueError):
            continue
        item: dict = {"bbox_norm": [round(v, 6) for v in box]}
        if latex:
            item["latex"] = latex
        if crop_uri:
            item["crop"] = crop_uri
        score = det.get("confidence")
        if isinstance(score, (int, float)):
            item["score"] = round(float(score), 4)
        if det.get("class_name"):
            item["class_name"] = str(det["class_name"])
        if form.get("error"):
            item["error"] = str(form["error"])
        out.setdefault(page_index, []).append(item)
    return out


def extract_figure_detections(resp: dict, image_labels) -> list[dict]:
    """从服务组响应的两个检测分组中提取图片类检测框。"""
    labels = {str(label).strip().casefold() for label in (image_labels or ())}
    out = []
    for key in ("recognized_detections", "unrecognized_detections"):
        for det in (resp or {}).get(key) or []:
            if not isinstance(det, dict):
                continue
            if str(det.get("class_name") or "").strip().casefold() in labels:
                out.append(det)
    return out
