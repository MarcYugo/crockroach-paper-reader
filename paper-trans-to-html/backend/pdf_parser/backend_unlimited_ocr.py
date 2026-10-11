r"""Unlimited-OCR 解析后端：版式来源 = Unlimited-OCR 整页 OCR 的版面项。

**独立实现**：本模块不引用 `backend_surya` / `surya_parser` 的任何代码 —— 两条整页
OCR 链（Surya 与 Unlimited-OCR）的解析、排版估算与前端渲染各自维护、互不影响；
共享的只有项目级通用件（客户端 `../unlimited_ocr.py`、`entry` / `converter` 的
装配约定）。改名 / 调整一边的排版策略时，另一边不会被牵连。

与「PyMuPDF + detection-service-group」组合的区别（本后端存在的意义）：

  * **完全使用 Unlimited-OCR 的数据**：文字 / 公式（LaTeX）/ 表格（HTML）/ 插图框 /
    阅读顺序全部来自模型回包（见 `../unlimited_ocr.py`）；不读 PyMuPDF 的文字层、
    行框与字体，也不调用 detection-service-group。服务不可用时**直接报错**
    （`unlimited_ocr.UnlimitedOcrUnavailable`，不会悄悄回退 PyMuPDF，
    免得用户以为看到了 OCR 的效果）。
  * 页面 `texts[]` 是**块级**数据，字段：

        {"id": "p1b2", "kind": "heading"|"text"|"reference"|"caption"|"footnote"
                |"formula"|"table"|"header"|"footer",
         "utype": "title",                  # Unlimited-OCR 原始类型（排障 / 统计）
         "label": "",                       # 统一为空（前端不会把它当表格标题渲染）
         "x","y","w","h": 86.6,             # 绝对坐标（PDF pt）
         "size","line_h": 10.0,             # 字号 / 行高估算（pt）
         "align": "justify",                # 对齐（正文类＝justify）
         "text": "1 Introduction",          # 块文本（行内数学已摘成 math 区间）
         "lines": [{"runs": [{"t": "1 Introduction"}]}],
         "math": [{"a": 3, "b": 12, "latex": "…"}],   # text 里的数学区间
         "latex": "…",                      # 仅公式块：KaTeX 源码
         "html": "<table>…"}                # 仅表格块：原始表格 HTML

  * 前端走**它自己那套**块级流式渲染（`frontend/js/reader2.js::buildUnlimitedItem`，
    `DOC.parser === "unlimited_ocr"` 时启用）：块按 bbox 绝对定位、块内文字正常换行；
    正文类两端对齐、标题加粗放大、图注/脚注/页眉页脚小一号；行内公式按 `math` 的
    精确字符区间就地换 KaTeX；独立公式（kind="formula" + latex）KaTeX 块渲染；
    表格（kind="table" + html）白名单渲染、不参与句子选中。
  * 坐标换算：模型回包 bbox 是 **0–1000 归一化坐标**（x、y 各自除以页面宽高再乘
    1000；见 `../unlimited_ocr.py` 模块注释）—— 这里按 `归一化值 / 1000 × 页面 pt
    边长` 换成 PDF 点；页面 `w/h` 由渲染图像素 × 72/dpi 得到，全页只有一把尺子。
  * 行内公式：模型把 LaTeX 直接写进正文（`\(…\)` / `$$…$$` / 少见单 `$…$`）——
    解析时把定界符摘掉、字符区间记进 `math`，前端在原位换成 KaTeX。
  * 公式块（`equation`）：文本形如 `\[ … \]` —— 剥掉定界符存进 `latex`。
  * 表格：原样存进 `html`（前端白名单渲染）；另用本地 `_table_text` 生成纯文本，
    作为可翻译 / 可复制的文本。
  * 插图（`image` / `chart`）：由客户端按 bbox 裁剪落盘到 `images_dir`，写进
    `pages[].images`（`source="unlimited_ocr"`），前端沿用图片层渲染。
  * 字号：模型不给字号 —— 本模块自带一套估算（`_document_body_size` ×
    `_block_metrics`，从块几何 + 字符数反解），全 PDF 统一正文基准。

统计（`parser_info`）：`unlimited_ocr_url` / `unlimited_ocr_model` /
`unlimited_ocr_dpi` / `unlimited_ocr_ms` / `unlimited_ocr_blocks` /
`unlimited_ocr_images` / `unlimited_ocr_pdf_pages` / `unlimited_ocr_types`
（模型原始类型直方图）+ 各渲染 kind 的条数（`unlimited_ocr_kind_*`）。

服务端镜像见仓库根 `unlimited_ocr_service/`；客户端见 `../unlimited_ocr.py`。
"""
from __future__ import annotations

import hashlib
import html as _html_mod
import math
import re
import statistics
from pathlib import Path
from typing import Optional

from .. import unlimited_ocr

__all__ = ["_parse_with_unlimited_ocr"]

# =====================================================================
#  排版估算（本模块自带的一套：块几何 + 字符数 → 字号/行高）
# =====================================================================
# 页基准字号（pt）的兜底：拿不到任何块估算值时用
_FALLBACK_BODY_SIZE = 10.0
# 字号反解系数：`h·w = k · chars · s²`（k = 行距倍率 × 平均字宽/em）。
# 实测校准（latex_sample.pdf，正文 10pt）：OCR 框紧贴字形，k≈0.55 反解最准；
# 换用 0.725 那档（给“检测框向外膨胀”的版面校准用）会把字号估小 ~20%。
_SIZE_K = 0.55
# 标题级别 → 相对页正文的字号倍率（模型不给字号，只能按排版惯例估）
_HEADING_MUL = {1: 1.45, 2: 1.22, 3: 1.08, 4: 1.02, 5: 1.0, 6: 1.0}
# 正文类 kind：两端对齐（align="justify"）、参与列间隙恢复（见 _restore_column_gaps）
_JUSTIFY_KINDS = ("text", "reference", "caption", "footnote")


def _estimate_size_pt(block_w: float, block_h: float, chars: int) -> float:
    """单块字号粗估：`h = nlines × 1.2s`、`nlines = chars × w_em·s / w`，解出 s。"""
    chars = max(8, int(chars))
    if block_w <= 0 or block_h <= 0:
        return 0.0
    return math.sqrt(block_h * block_w / (_SIZE_K * chars))


def _document_body_size(blocks: list[dict]) -> float:
    """全 PDF 正文基准字号（pt）：跨所有页面取中位数，避免不同页差异。

    只统计 `text` 类块（`reference` / 图注 / 页脚等小字号块除外，免得把基准拉低）。
    """
    ests = []
    for b in blocks:
        if b["kind"] not in ("text",) or len(b["text"]) < 60:
            continue
        est = _estimate_size_pt(b["w"], b["h"], len(b["text"]))
        if 5.0 <= est <= 22.0:
            ests.append(est)
    if not ests:
        return _FALLBACK_BODY_SIZE
    return min(20.0, max(6.0, statistics.median(ests)))


def _block_metrics(kind: str, level: Optional[int], w: float, h: float,
                   text: str, body: float) -> tuple[float, float]:
    """块的（字号, 行高）估算，单位 pt（按类型相对页基准缩放）。

    字号只按类型给：标题按级别放大、图注/脚注/页眉页脚/参考文献小一号、正文用页
    基准字号；块宽高只决定换行与容器位置，不二次缩放（同一份 PDF 全篇字号才统一）。
    行高按类型给固定倍率（正文 1.42、标题 1.25、图注 1.32、表格 1.3）。
    """
    if kind == "heading":
        size = body * _HEADING_MUL.get(level or 2, 1.12)
        lh_mul = 1.25
    elif kind == "caption":
        size = body * 0.92
        lh_mul = 1.32
    elif kind == "reference":
        # 参考文献：比正文小一号
        size = body * 0.92
        lh_mul = 1.32
    elif kind in ("footnote", "header", "footer"):
        size = body * 0.88
        lh_mul = 1.32
    elif kind == "table":
        size = body * 0.95
        lh_mul = 1.30
    elif kind == "formula":
        size = body
        lh_mul = 1.30
    else:                        # text
        size = body
        lh_mul = 1.42
    size = min(28.0, max(5.5, size))
    return round(size, 2), round(size * lh_mul, 2)


# 列间隙恢复（双栏页面：OCR 框的膨胀/漂移把列间空当挤没，两端对齐后两栏会贴到一起）
_MIN_COL_GAP = 10.0        # pt：相邻列之间应保留的空隙（LaTeX 默认 \columnsep）
_COL_EDGE_TOL = 1.5        # pt：框允许的轻微重叠（负间隙）容差


def _restore_column_gaps(items: list[dict]) -> None:
    """把双栏页被挤没的列间空隙恢复到 _MIN_COL_GAP（就地修改）。

    「垂直重叠、水平相邻且间隙 < _MIN_COL_GAP」的正文类块对，两侧各让一半；
    每个块至多修一次。要求**至少一侧是多行块**（单行×单行的紧邻更可能是检测器
    把一行拆成两块，不碰）；标题/公式/表格/页眉页脚不参与。
    """
    cand = [it for it in items if it.get("kind") in _JUSTIFY_KINDS]
    if len(cand) < 2:
        return

    def _multiline(it: dict) -> bool:
        lh = float(it.get("line_h") or 0)
        return lh > 0 and it.get("h", 0) >= lh * 2.0

    used = set()
    for a in cand:
        if a["id"] in used:
            continue
        for b in cand:
            if b is a or b["id"] in used or b["x"] <= a["x"]:
                continue
            gap = b["x"] - (a["x"] + a["w"])
            if not (-_COL_EDGE_TOL <= gap < _MIN_COL_GAP):
                continue
            overlap = (min(a["y"] + a["h"], b["y"] + b["h"])
                       - max(a["y"], b["y"]))
            if overlap <= 0 or overlap < min(a["h"], b["h"]) * 0.5:
                continue
            if not (_multiline(a) or _multiline(b)):
                continue
            d = (_MIN_COL_GAP - gap) / 2.0
            if a["w"] - d < 24 or b["w"] - d < 24:
                continue        # 防御：别把块挤没了
            a["w"] = round(a["w"] - d, 3)
            b["x"] = round(b["x"] + d, 3)
            b["w"] = round(b["w"] - d, 3)
            # lines[0] 是块宽/位置的镜像（前端主要用 item 的 x/w，保持一致）
            if a.get("lines"):
                a["lines"][0]["w"] = a["w"]
            if b.get("lines"):
                b["lines"][0]["x"] = b["x"]
                b["lines"][0]["w"] = b["w"]
            used.add(a["id"])
            used.add(b["id"])
            break


def _pdf_meta(pdf_path: Path) -> tuple[str, list[dict], int]:
    """PDF 标题 / 自带书签 / 页数（只读元数据，不参与版式渲染）。"""
    try:
        import fitz
    except ImportError:
        return "", [], 0
    doc = fitz.open(str(pdf_path))
    try:
        title = str((doc.metadata or {}).get("title") or "").strip()
        toc: list[dict] = []
        try:
            raw = doc.get_toc(simple=True) or []
        except Exception:
            raw = []
        for row in raw:
            try:
                level, name, page = int(row[0]), str(row[1] or "").strip(), int(row[2])
            except Exception:
                continue
            if name:
                toc.append({"level": max(1, min(level, 4)),
                            "title": name[:200], "page": max(1, page)})
        return title, toc, int(doc.page_count)
    finally:
        doc.close()


_NUM_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)")


def _heading_level(text: str) -> int:
    """标题级别：按编号（`1.2.3` → 3）判定，认不出按 2。"""
    m = _NUM_RE.match(text or "")
    if m:
        return max(1, min(4, m.group(1).count(".") + 1))
    return 2


# =====================================================================
#  表格 HTML → 纯文本（本模块自带；表格的展示用原始 html，前端白名单渲染）
# =====================================================================
def _strip_html(s: str) -> str:
    return " ".join(_html_mod.unescape(re.sub(r"<[^>]+>", "", s or "")).split())


def _table_text(html: str) -> str:
    """表格 HTML → 可读纯文本：一行一个 `<tr>`，单元格用 `" | "` 分隔。

    模型给的表格 HTML 结构简单（`<table><tr><td>…`）；万一没有 `<tr>`，
    退化为整段剥标签的文本。
    """
    rows = re.findall(r"<tr\b[^>]*>(.*?)</tr>", html or "", re.I | re.S)
    out: list[str] = []
    for row in rows:
        cells = re.findall(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", row, re.I | re.S)
        line = " | ".join(_strip_html(c) for c in cells).strip()
        if line:
            out.append(line)
    if not out:
        return _strip_html(html)
    return "\n".join(out)

# 模型原始类型 → 前端渲染 kind（认不出的类型：有文本按普通文本，无文本丢弃）
_KIND_BY_TYPE = {
    "title": "heading",
    "text": "text",
    "paragraph": "text",
    "content": "text",
    "abstract": "text",
    "aside_text": "text",
    "ref_text": "reference",              # 参考文献：比正文小一号
    "reference": "reference",
    "footnote": "footnote",
    "page_footnote": "footnote",
    "endnote": "footnote",
    "equation": "formula",
    "equation_block": "formula",
    "formula": "formula",
    "table": "table",
    "table_caption": "caption",
    "table_title": "caption",
    "image_caption": "caption",
    "figure_caption": "caption",
    "figure_title": "caption",
    "page_number": "footer",
    "page_header": "header",
    "header": "header",
    "footer": "footer",
    "image": "figure",
    "figure": "figure",
    "chart": "figure",
    "diagram": "figure",
}

#：公式块文本的定界符（按优先级匹配；`$$` 必须排在 `$` 前面）
_LATEX_DELIMS = (("\\[", "\\]"), ("\\(", "\\)"), ("$$", "$$"), ("$", "$"))

# 单 `$…$` 的保守判据（与前端 `r2-math.js::_splitInlineMath` 同口径）：
# 长度上限、不能带首尾空白、必须含 \ _ ^ { } 之一（挡货币符号）
_DOLLAR_MAX = 120
_MATH_HINT_RE = re.compile(r"[\\_^{}]")


def _latex_block(raw: str) -> str:
    """公式块文本（`\\[ … \\]` / `$$…$$`）→ 纯 LaTeX；没有定界符就原样返回去空格。"""
    s = (raw or "").strip()
    for left, right in _LATEX_DELIMS:
        if (len(s) > len(left) + len(right)
                and s.startswith(left) and s.endswith(right)):
            return s[len(left):-len(right)].strip()
    return s


def _record_math(out: list[str], ranges: list[dict], inner: str) -> None:
    """把一段内联 LaTeX（`inner`）追加进字符流并记下区间（首尾空白不入区间）。"""
    core = inner.strip()
    if not core:
        return
    start = len(out)
    out.extend(core)
    ranges.append({"a": start, "b": start + len(core), "latex": core})


def _text_with_math(raw: str) -> tuple[str, list[dict], list[dict]]:
    """块文本 →（text, runs, math 区间）。

    把 `\\(…\\)` / `\\[…\\]` / `$$…$$`（以及保守判据下的单 `$…$`）摘成数学区间，
    `text` 里不再含 LaTeX 定界符；runs 覆盖整段文本（单 run，前端 styleRun 直接用）。
    """
    out: list[str] = []
    ranges: list[dict] = []
    i, n = 0, len(raw or "")
    while i < n:
        ch = raw[i]
        if ch == "\\" and i + 1 < n and raw[i + 1] in "([":
            close = "\\)" if raw[i + 1] == "(" else "\\]"
            j = raw.find(close, i + 2)
            if j >= 0:
                _record_math(out, ranges, raw[i + 2:j])
                i = j + 2
                continue
        elif ch == "$":
            if raw.startswith("$$", i):
                j = raw.find("$$", i + 2)
                if j >= 0:
                    _record_math(out, ranges, raw[i + 2:j])
                    i = j + 2
                    continue
            else:
                k = raw.find("$", i + 1)
                inner = raw[i + 1:k] if k >= 0 else ""
                if (k >= 0 and "\n" not in inner and len(inner) <= _DOLLAR_MAX
                        and inner.strip() == inner and inner.strip()
                        and _MATH_HINT_RE.search(inner)):
                    _record_math(out, ranges, inner)
                    i = k + 1
                    continue
        out.append(ch)
        i += 1
    text = "".join(out)
    runs = [{"t": text}] if text else []
    return text, runs, ranges


def _figure_image(it: dict, images_dir: Optional[Path], scale: float) -> Optional[dict]:
    """图片类项 → `pages[].images` 条目（pt 坐标；图片由客户端已落盘）。"""
    info = it.get("image")
    if not isinstance(info, dict) or images_dir is None:
        return None
    fname = str(info.get("file") or "").strip()
    bounds = info.get("bbox") or it.get("bbox")
    if not fname or not isinstance(bounds, (list, tuple)) or len(bounds) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(v) for v in bounds)
    except (TypeError, ValueError):
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    try:
        digest = hashlib.sha1((Path(images_dir) / fname).read_bytes()).hexdigest()[:12]
    except OSError:
        return None
    typ = str(it.get("type") or "").strip().lower()
    return {
        "x": round(x0 * scale, 3), "y": round(y0 * scale, 3),
        "w": round((x1 - x0) * scale, 3), "h": round((y1 - y0) * scale, 3),
        "file": fname, "source": "unlimited_ocr",
        "class_name": "Chart" if typ == "chart" else "Figure",
        "cache_key": digest,
    }


# =====================================================================
#  入口：Unlimited-OCR 版面数据 → doc.json 的 pages 结构
# =====================================================================
def _parse_with_unlimited_ocr(pdf_path: Path, images_dir: Optional[Path],
                              opts: dict) -> dict:
    """整份 PDF 走 Unlimited-OCR 整页 OCR，转成阅读器的版式数据。

    失败一律抛 `unlimited_ocr.UnlimitedOcrUnavailable`（由 `entry.parse_pdf` 向上抛给
    转换接口，页面上会显示具体原因）—— **不回退 PyMuPDF**，见模块文档。
    """
    raw = unlimited_ocr.ocr_pdf(pdf_path, opts, images_dir=images_dir)
    scale = 72.0 / float(raw.get("dpi") or 150)
    pdf_title, toc, pdf_pages = _pdf_meta(pdf_path)

    pages_out: list[dict] = []
    kind_counts: dict[str, int] = {}
    total_images = 0
    document_blocks: list[dict] = []
    for pg in raw.get("pages") or []:
        pw = float(pg.get("w") or 0) * scale           # 页面 pt 宽
        ph = float(pg.get("h") or 0) * scale           # 页面 pt 高
        mids: list[dict] = []
        images: list[dict] = []
        for order, it in enumerate(pg.get("items") or []):
            typ = str(it.get("type") or "").strip().lower()
            kind = _KIND_BY_TYPE.get(typ)
            if kind is None and str(it.get("text") or "").strip():
                kind = "text"                          # 认不出的类型：有文本按文本
            if kind is None:
                continue
            box = it.get("bbox")
            if not (isinstance(box, (list, tuple)) and len(box) == 4):
                continue
            if kind == "figure":
                image = _figure_image(it, images_dir, scale)
                if image is None:
                    continue        # 裁图缺失/过小：没有可显示的东西
                images.append(image)
                total_images += 1
                kind_counts["figure"] = kind_counts.get("figure", 0) + 1
                continue
            try:
                nx0, ny0, nx1, ny1 = (float(v) for v in box)
            except (TypeError, ValueError):
                continue
            # 归一化（0–1000）→ PDF 点
            x0, x1 = nx0 / 1000.0 * pw, nx1 / 1000.0 * pw
            y0, y1 = ny0 / 1000.0 * ph, ny1 / 1000.0 * ph
            if x1 - x0 <= 0.5 or y1 - y0 <= 0.5:
                continue

            raw_text = str(it.get("text") or "")
            item: dict = {
                "id": f"p{pg.get('page')}b{order}",
                "kind": kind,
                "utype": typ,               # 模型原始类型（排障 / 统计）
                "unlimited": True,          # 数据侧标记：本块来自 Unlimited-OCR
                "label": "",                # 统一为空（前端不会拿它当表格标题）
                "x": round(x0, 3), "y": round(y0, 3),
                "w": round(x1 - x0, 3), "h": round(y1 - y0, 3),
                "text": "", "math": [], "html": "", "lines": [],
            }
            item["align"] = "justify" if kind in _JUSTIFY_KINDS else "left"

            if kind == "formula":
                latex = _latex_block(raw_text)
                if not latex:
                    continue            # 拿不到 LaTeX 的公式块：没有可渲染的内容
                item["latex"] = latex
            elif kind == "table":
                if "<table" in raw_text.lower():
                    # 表格 HTML 原样下发（前端白名单渲染）；纯文本用本地 _table_text
                    # 生成（可翻译 / 可复制）—— 不引用任何其它后端模块
                    item["html"] = raw_text
                    text = _table_text(raw_text)
                    runs = [{"t": text}] if text else []
                    math = []
                else:
                    # 模型没给表格结构（个别页会给 Markdown 文本）：按普通段落处理
                    text, runs, math = _text_with_math(raw_text)
                if not text:
                    continue
                item["text"], item["math"] = text, math
                item["lines"] = [{"x": item["x"], "y": item["y"],
                                  "w": item["w"], "h": item["h"], "runs": runs}]
            else:
                text, runs, math = _text_with_math(raw_text)
                if not text:
                    continue            # 空文本块：没有可显示 / 可选中的内容
                item["text"], item["math"] = text, math
                item["lines"] = [{"x": item["x"], "y": item["y"],
                                  "w": item["w"], "h": item["h"], "runs": runs}]
                if kind == "heading":
                    item["level"] = _heading_level(text)

            mids.append(item)
            kind_counts[kind] = kind_counts.get(kind, 0) + 1
            document_blocks.append({"kind": kind, "w": item["w"],
                                    "h": item["h"], "text": item["text"]})

        page = {
            "w": round(pw, 3),
            "h": round(ph, 3),
            "texts": mids,
            "images": images,
            "text": "\n\n".join(t["text"] for t in mids if t.get("text")),
        }
        pages_out.append(page)

    # 全 PDF 统一正文基准；caption、table、footnote 等仅按类型相对缩放。
    body = _document_body_size(document_blocks)
    for page in pages_out:
        for item in page["texts"]:
            size, line_h = _block_metrics(item["kind"], item.get("level"),
                                          item["w"], item["h"], item["text"], body)
            item["size"] = size
            item["line_h"] = line_h
            if item.get("lines"):
                item["lines"][0]["s"] = size

    # 列间隙恢复：双栏页面里 OCR 框的膨胀/漂移会把两列之间的空当挤没（两端对齐后
    # 两栏文字贴到一起）—— 把间隙修回 _MIN_COL_GAP（见本函数说明）。
    for page in pages_out:
        _restore_column_gaps(page["texts"])

    stats = dict(raw.get("stats") or {})
    return {
        "num_pages": len(pages_out),
        "page_w": pages_out[0]["w"] if pages_out else 0,
        "page_h": pages_out[0]["h"] if pages_out else 0,
        "pdf_title": pdf_title,
        "pages": pages_out,
        "toc": toc,                    # PDF 自带书签（没有就是空列表）
        "parser": "unlimited_ocr",
        # 公式怎么落地的：unlimited_ocr = 公式来自 OCR 回包的 LaTeX（行内区间 +
        # 独立公式块），与 "surya"（Surya 的 <math>）、"space"（PyMuPDF 擦成空格）、
        # "boxes"（服务组公式框覆盖层）并列。
        "formula_action": "unlimited_ocr",
        "parser_info": {
            "unlimited_ocr_url": raw.get("url"),
            "unlimited_ocr_model": raw.get("model"),
            "unlimited_ocr_dpi": raw.get("dpi"),
            "unlimited_ocr_ms": stats.get("unlimited_ocr_ms", 0),
            "unlimited_ocr_blocks": sum(len(p["texts"]) + len(p["images"])
                                        for p in pages_out),
            "unlimited_ocr_pdf_pages": pdf_pages,
            "unlimited_ocr_images": total_images,
            "unlimited_ocr_types": stats.get("unlimited_ocr_types") or {},
            **{f"unlimited_ocr_kind_{k}": v for k, v in sorted(kind_counts.items())},
        },
    }
