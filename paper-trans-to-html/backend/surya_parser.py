"""Surya 2 推理服务客户端 · 版面分析 / 整页 OCR / 插图裁剪

参考 `doc_parse_service/test_surya_service.py`：把 Surya 2 当成**外部推理服务**
接入(OpenAI 兼容接口, llama-server / vllm)，本进程只做客户端：

  1) `SURYA_INFERENCE_URL` / `SURYA_INFERENCE_BACKEND` 必须在 `import surya`
     之前写进环境变量，否则 surya 会尝试自己拉起后端(见 SuryaClient.__init__)；
  2) `RecognitionPredictor` 做「整页 OCR」：每页一次 VLM 调用，返回按阅读顺序
     排列的 block(每个 block 含 label + html，表格是 <table>、公式是 <math>)。
     ⚠️ Surya 2 的 `<math>` 标签里装的是 **LaTeX 源码**(不是 MathML)，例如
     `<math display="block">\ell_t(W)=... \quad (1)</math>`；公式块的标签是
     `Equation`(raw_label `Equation-Block`)。取源码交给前端 MathJax 排版，
     **不要**把 `<math>` 当 MathML 塞进 DOM(那会把 `\ell_t(W)` 当纯文本显示)：
  3) 图片类 block(Picture/Figure/Diagram/ChemicalBlock)按 polygon 从页面图像
     裁剪成 PNG。注意：这些标签同时在模型的 SKIP_OCR_LABELS 里，一定是
     `skipped=True / html=""`，所以**绝不能用 skipped / 空 html 当过滤条件**，
     只能靠 label + 裁剪区域判断；
  4) 这里给出的坐标是**渲染图像素**，由 `pdf_parser` 负责换算成 PDF pt 后再落盘，
     以保证与前端阅读器的版式坐标系一致。

依赖(可选)：`surya-ocr>=0.22`(客户端) + `Pillow`；缺失时由 `pdf_parser`
自动回退到纯 PyMuPDF 解析。
"""
from __future__ import annotations

import json
import os
import re
import urllib.request
from pathlib import Path
from typing import Any, Optional

# 默认地址与 docker-compose.yml 里的 SURYA_PORT 保持一致
DEFAULT_URL = "http://127.0.0.1:8060/v1"
DEFAULT_BACKEND = "llamacpp"
MODEL_ALIAS = "datalab-to/surya-ocr-2"
PROBE_TIMEOUT = 1.5          # auto 模式探测服务的超时(秒)
CHECK_TIMEOUT = 10.0         # 显式检查服务的超时(秒)

# 版面标签里属于「图片类」的 canonical 名称(见 surya.layout.label.LAYOUT_PRED_RELABEL):
#   Image -> Picture, Complex-Block/Figure -> Figure, Diagram, Chemical-Block -> ChemicalBlock
DEFAULT_IMAGE_LABELS: tuple[str, ...] = ("Picture", "Figure", "Diagram", "ChemicalBlock")

# 这些标签的 html 有额外价值(表格/公式/列表)，默认整段保留进 doc.json
RICH_LABELS: tuple[str, ...] = ("Table", "Equation", "EquationBlock", "Formula",
                               "Math", "ListGroup")

# 版面标签 -> 前端可用的粗分类(仅作为附加信息，不影响渲染)
_KIND_BY_LABEL = {
    "PageHeader": "header",
    "PageFooter": "footer",
    "SectionHeader": "heading",
    "Caption": "caption",
    "Footnote": "footnote",
    "ListGroup": "list",
    "Table": "table",
    "Equation": "formula",        # Surya 2 实际用的标签(raw_label "Equation-Block")
    "EquationBlock": "formula",   # 兼容旧版/别名
    "Formula": "formula",
    "Math": "formula",
    "Text": "text",
}

# 规范化查表：标签去掉非字母数字后再比对，能吃下 "Equation-Block"/"Section-Header"
# 这类带连字符的 raw_label 写法。
_KIND_NORM = {re.sub(r"[^a-z0-9]", "", k.lower()): v
              for k, v in _KIND_BY_LABEL.items()}


class SuryaUnavailable(RuntimeError):
    """Surya 服务不可用，或本机缺少 surya-ocr / Pillow 依赖。"""


# =====================================================================
#  标签 / 文本小工具
# =====================================================================
def resolve_image_labels(spec: Any = None) -> set[str]:
    """None -> 默认图片类标签；'Picture,Figure' / ['Picture'] -> 集合"""
    if not spec:
        return set(DEFAULT_IMAGE_LABELS)
    if isinstance(spec, str):
        return {s.strip() for s in spec.split(",") if s.strip()}
    return {str(s).strip() for s in spec if str(s).strip()}


def label_kind(label: str | None) -> str:
    """版面标签 -> 粗分类(text/heading/table/formula/...)，未知按 text 处理。

    先精确匹配 canonical 标签，再退化为「去掉非字母数字后」的模糊匹配，
    这样 `Equation` 与 `Equation-Block` 都能识别成 formula。
    """
    raw = label or ""
    if raw in _KIND_BY_LABEL:
        return _KIND_BY_LABEL[raw]
    return _KIND_NORM.get(re.sub(r"[^a-z0-9]", "", raw.lower()), "text")


def safe_name(label: str | None) -> str:
    """标签转成可用作文件名/键名的片段。"""
    return re.sub(r"[^\w.-]+", "_", label or "block").strip("_") or "block"


def normalize_url(url: str | None) -> str:
    """补全默认地址，并去掉结尾多余的 '/'。"""
    u = (url or "").strip() or DEFAULT_URL
    return u.rstrip("/")


def bbox_from_polygon(polygon) -> Optional[list[int]]:
    """polygon(渲染像素) -> 轴对齐 bbox [x0, y0, x1, y1]；无法解析时返回 None"""
    pts: list[tuple[float, float]] = []
    for p in polygon or []:
        try:
            pts.append((float(p[0]), float(p[1])))
        except (TypeError, IndexError, ValueError):
            continue
    if not pts:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]


def html_to_text(html: str | None) -> str:
    """把 block 的 html 粗略转成纯文本(保留段落/表格单元格/换行)。"""
    if not html:
        return ""
    h = html
    h = re.sub(r"</(p|div|tr|li|h[1-6])>", "\n", h, flags=re.I)
    h = re.sub(r"<br\s*/?>", "\n", h, flags=re.I)
    h = re.sub(r"</(td|th)>", " | ", h, flags=re.I)
    h = re.sub(r"<[^>]+>", "", h)
    h = (h.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
          .replace("&quot;", '"').replace("&#39;", "'").replace("&nbsp;", " "))
    h = re.sub(r"[ \t]+\n", "\n", h)
    h = re.sub(r"\n{2,}", "\n", h)
    return h.strip()


_MATH_TAG_RE = re.compile(r"<math\b[^>]*>(.*?)</math>", re.S | re.I)


def latex_from_html(html: str | None) -> str:
    """从 Surya 的公式 html 里取出 **LaTeX 源码**。

    ⚠️ Surya 2 的 `<math>` 里装的是 LaTeX（不是 MathML），例如：
        `<math display="block">\\ell_t(W) = \\text{CE}(...) \\quad (1)</math>`
        → `\\ell_t(W) = \\text{CE}(...) \\quad (1)`
    没有 `<math>` 包裹时退化为「去掉所有标签」的文本（公式块通常整段就是 LaTeX）。
    """
    if not html:
        return ""
    mt = _MATH_TAG_RE.search(html)
    body = mt.group(1) if mt else html
    body = re.sub(r"<[^>]+>", "", body)
    body = (body.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
            .replace("&quot;", '"').replace("&#39;", "'").replace("&nbsp;", " "))
    body = re.sub(r"[ \t]+", " ", body)
    return re.sub(r"\n{2,}", "\n", body).strip()


# =====================================================================
#  服务探测
# =====================================================================
def check_server(url: str | None = None, timeout: float = CHECK_TIMEOUT
                 ) -> tuple[bool, list[str], str]:
    """探测服务端 /v1/models。返回 (是否就绪, 模型名列表, 说明文本)"""
    base = normalize_url(url)
    models_url = base + "/models" if base.endswith("/v1") else base
    try:
        req = urllib.request.Request(models_url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:  # 连接失败/超时/非 JSON 都算不可用
        return False, [], f"无法连接 Surya 推理服务 {models_url}：{exc}"

    names = [m.get("id") for m in (data.get("data") or []) if isinstance(m, dict)]
    if names and MODEL_ALIAS not in names:
        return True, names, (f"服务就绪，但模型列表未见 {MODEL_ALIAS}：{names}；"
                             "请确认服务端用 --alias datalab-to/surya-ocr-2 启动。")
    return True, names, f"服务就绪，模型：{names or ['(未上报模型名)']}"


def available(url: str | None = None, timeout: float = PROBE_TIMEOUT) -> tuple[bool, str]:
    """轻量探测(供 auto 模式使用)，只关心「能不能连上」。"""
    ok, _models, msg = check_server(url, timeout=timeout)
    return ok, msg


# =====================================================================
#  页面渲染 / 插图裁剪
# =====================================================================
def render_page_image(page, dpi: int = 192):
    """PyMuPDF 页面 → PIL 图像(整页渲染，作为 VLM 的输入)。

    dpi 越大识别越准、显存/耗时越高；96~192 是比较稳的区间。
    """
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:  # pragma: no cover - 项目本就依赖 PyMuPDF
        raise SuryaUnavailable(f"缺少 PyMuPDF：pip install PyMuPDF ({exc})") from exc
    try:
        from PIL import Image  # noqa: F401  仅用于确认 Pillow 可用
    except ImportError as exc:
        raise SuryaUnavailable(f"Surya 解析需要 Pillow：pip install Pillow ({exc})") from exc

    zoom = max(0.5, float(dpi) / 72.0)
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    return pix.pil_image()


def crop_block(image, bbox: Optional[list] = None, polygon=None, pad: int = 4):
    """按 block 的 bbox / polygon(渲染图像素)裁剪，返回 (bbox_px, crop)。

    区域退化(宽高为 0)时 crop 为 None。
    """
    box = bbox or bbox_from_polygon(polygon)
    if not box:
        return None, None
    x0 = max(0, int(box[0]) - pad)
    y0 = max(0, int(box[1]) - pad)
    x1 = min(int(image.size[0]), int(box[2]) + pad)
    y1 = min(int(image.size[1]), int(box[3]) + pad)
    bbox_px = [x0, y0, x1, y1]
    if x1 <= x0 or y1 <= y0:
        return bbox_px, None
    return bbox_px, image.crop((x0, y0, x1, y1))


def crop_filename(pno: int, idx: int, ext: str = "png") -> str:
    """插图文件名 —— 必须匹配 app.py 的 /api/doc/{id}/img/{fname} 路由正则。"""
    return f"p{pno:03d}_i{idx}.{ext}"


def save_crop(crop, images_dir: Path | str, fname: str) -> str:
    """插图落盘(默认 PNG)，返回文件名。"""
    d = Path(images_dir)
    d.mkdir(parents=True, exist_ok=True)
    crop.save(d / fname)
    return fname


# =====================================================================
#  block 结果整理
# =====================================================================
def blocks_from_result(page_res) -> list[dict]:
    """把 Surya 的 PageOCRResult.blocks 转成 dict 列表(按 reading_order 排序)。

    坐标为渲染图像素；`text` 由 html 粗转而来，供无文本层的扫描件兜底使用。
    """
    out: list[dict] = []
    for b in (getattr(page_res, "blocks", None) or []):
        polygon = getattr(b, "polygon", None) or []
        html = getattr(b, "html", "") or ""
        conf = getattr(b, "confidence", None)
        label = getattr(b, "label", None) or "Text"
        out.append({
            "label": label,
            "kind": label_kind(label),
            "raw_label": getattr(b, "raw_label", None),
            "reading_order": getattr(b, "reading_order", None),
            "confidence": round(float(conf), 4) if conf is not None else None,
            "skipped": bool(getattr(b, "skipped", False)),
            "error": bool(getattr(b, "error", False)),
            "polygon": [[float(p[0]), float(p[1])] for p in polygon] if polygon else [],
            "bbox": bbox_from_polygon(polygon),
            "html": html,
            "text": html_to_text(html),
        })
    out.sort(key=lambda blk: blk["reading_order"] if blk["reading_order"] is not None else 0)
    return out


# =====================================================================
#  客户端
# =====================================================================
class SuryaClient:
    """Surya 2 推理服务客户端(整页 OCR)。

    只连接外部服务，**不会**自动拉起后端；用完调用 `close()` 释放引用。
    """

    def __init__(self, url: str | None = None, backend: str | None = None):
        self.url = normalize_url(url or os.environ.get("SURYA_INFERENCE_URL"))
        self.backend = (backend or os.environ.get("SURYA_INFERENCE_BACKEND")
                        or DEFAULT_BACKEND).strip().lower() or DEFAULT_BACKEND
        # 关键：必须在 import surya 之前设好，否则 surya 会尝试自行拉起后端
        os.environ["SURYA_INFERENCE_URL"] = self.url
        os.environ["SURYA_INFERENCE_BACKEND"] = self.backend
        self._manager = None
        self._predictor = None

    # ---------- 内部 ----------
    def _ensure(self) -> None:
        if self._predictor is not None:
            return
        try:
            from surya.inference import SuryaInferenceManager
            from surya.recognition import RecognitionPredictor
        except ImportError as exc:
            raise SuryaUnavailable(
                "本机缺少 surya-ocr(需 >= 0.22)：pip install -U surya-ocr "
                f"({exc})") from exc
        try:
            self._manager = SuryaInferenceManager(method=self.backend)
            self._predictor = RecognitionPredictor(self._manager)
        except Exception as exc:
            raise SuryaUnavailable(
                f"初始化 Surya 客户端失败(服务 {self.url})：{exc}") from exc

    # ---------- 对外 ----------
    def check(self, timeout: float = CHECK_TIMEOUT) -> tuple[bool, str]:
        """连接前体检：确认服务在跑、模型名对得上。"""
        ok, _models, msg = check_server(self.url, timeout=timeout)
        return ok, msg

    def ocr(self, images: list) -> list:
        """整页 OCR：images 为 PIL 图像列表，返回 PageOCRResult 列表。"""
        if not images:
            return []
        self._ensure()
        try:
            return list(self._predictor(images))
        except Exception as exc:
            raise SuryaUnavailable(f"Surya 推理失败(服务 {self.url})：{exc}") from exc

    def ocr_page(self, image) -> list[dict]:
        """单页 OCR，直接返回整理好的 block dict 列表(坐标为渲染像素)。"""
        results = self.ocr([image])
        return blocks_from_result(results[0]) if results else []

    def close(self) -> None:
        """释放客户端。

        这里连接的是**外部服务**，不负责停止服务端进程(参照
        test_surya_service.py 的客户端用法)；若环境变量来自本地自动拉起，
        surya 自己会管理其生命周期。
        """
        self._predictor = None
        self._manager = None
