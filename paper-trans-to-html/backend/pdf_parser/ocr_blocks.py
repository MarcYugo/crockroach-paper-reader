"""PaddleOCR-VL 整页 OCR 文本 → 版式块(含「文字层是否不可见」判定)。

见包文档 `backend/pdf_parser/__init__.py`。
"""

from __future__ import annotations

import re
import textwrap
import fitz  # PyMuPDF

from .images import (_covered_ratio, _invisible_text_rects, _merge_rects)
from .text_layer import (_join_lines, _text_line_rects, _union_line_rect)


# =====================================================================
#  后端三 · PaddleOCR-VL 推理服务(整页 OCR)，版式/插图由 PyMuPDF 补齐
# =====================================================================
_OCR_MARGIN = 24.0          # 合成行框时的页面边距(pt)


def _text_layer_invisible(page, texts: list[dict]) -> bool:
    """判断本地文字层是否基本不可见(扫描件的隐形 OCR 文字层)。

    扫描件常带一层「不可见文字」(render mode 3 / alpha=0)用于检索复制，
    看得见的笔画其实在底图里。这种情况下应以 OCR 文本为准，别把旧文字层当正文。
    """
    hidden = _merge_rects(_invisible_text_rects(page))
    if not hidden:
        return False
    rects = _text_line_rects(texts)
    if not rects:
        return False
    ratios = [_covered_ratio(r, hidden) for r in rects]
    return (sum(ratios) / len(ratios)) >= 0.5


def _drop_lines_in_rects(texts: list[dict], rects: list[fitz.Rect]) -> list[dict]:
    """删掉中心点落在给定额形区域内的文字行(这些区域改由公式渲染显示)。"""
    if not rects:
        return texts
    out: list[dict] = []
    for blk in texts:
        old = blk.get("lines") or []
        lines = [ln for ln in old
                 if not any(r.contains(fitz.Point(ln["x"] + ln["w"] / 2.0,
                                                  ln["y"] + ln["h"] / 2.0))
                            for r in rects)]
        if not lines:
            continue
        if len(lines) != len(old):
            x0, y0, x1, y1 = _union_line_rect(lines)
            blk = dict(blk, x=x0, y=y0, w=x1 - x0, h=y1 - y0,
                       lines=lines, text=_join_lines(lines))
        out.append(blk)
    return out


def _synth_ocr_blocks(text: str, page, pno: int) -> list[dict]:
    """纯文本兜底：把整页 OCR 文本按「段 → 行」合成文字块(近似版式)。

    这是 PaddleOCR-VL 的**常规**路径：模型输出的是整页 Markdown/HTML，没有坐标，
    所以把文本贴着页边距自上而下排，按空行分段，段内每行一行框，行高按可用高度/
    总行数自适应。坐标单位是 PDF pt，与 PyMuPDF 后端一致，前端渲染器无需改动。

    段落整段都是数学时(模型常用 `$$…$$` / `\\[…\\]` 包住独立公式)抽成块的
    `latex` 字段交给前端排版，避免公式源码当正文显示。
    """
    paras = [p for p in re.split(r"\n\s*\n", (text or "").strip()) if p.strip()]
    para_lines = [[ln.strip() for ln in p.splitlines() if ln.strip()] for p in paras]
    para_lines = [ls for ls in para_lines if ls]

    if not para_lines:
        return []

    w = max(72.0, page.rect.width - 2 * _OCR_MARGIN)
    avail = max(72.0, page.rect.height - 2 * _OCR_MARGIN)
    total = sum(len(ls) for ls in para_lines)
    lh = min(28.0, max(9.0, avail / max(1, total)))
    size = max(6.0, min(14.0, round(lh * 0.78, 1)))
    gap = min(lh * 0.6, 12.0)

    blocks: list[dict] = []
    y = _OCR_MARGIN
    for i, ls in enumerate(para_lines):
        lines = [{
            "x": _OCR_MARGIN, "y": y + j * lh, "w": w, "h": lh,
            "runs": [{"t": t, "s": size, "fam": "serif", "b": False,
                      "i": False, "up": False, "c": "#000000"}],
        } for j, t in enumerate(ls)]
        body = _join_lines(lines)
        if not body:
            continue
        blk: dict = {
            "id": f"p{pno}b{i}",
            "x": _OCR_MARGIN, "y": y, "w": w, "h": lh * len(ls),
            "text": body, "lines": lines,
        }
        # 整段基本被数学定界符包住 → 当成独立公式(前端 KaTeX 渲染)
        cand, ratio = extract_latex(body)
        if cand and ratio >= 0.6:
            blk["kind"] = "formula"
            blk["latex"] = cand
        elif looks_like_latex(body):        # 无定界符的裸 LaTeX
            blk["kind"] = "formula"
            blk["latex"] = body
        blocks.append(blk)
        y += lh * len(ls) + gap
    return blocks


# 可选的**结构化**输出格式：逐块一行的「<标签> [x0, y0, x1, y1]<文本>」，
# 坐标为 **0~1000 的归一化值**(每轴独立，与渲染 DPI 无关)，由调用方换算成 PDF pt。
# PaddleOCR-VL 默认输出整页 Markdown/HTML(没有坐标)，走上面的 `_synth_ocr_blocks`；
# 这段解析是给「会吐坐标」的 OCR 输出留的兼容路径(命中就能还原真实版式)。
_OCR_BLOCK_RE = re.compile(
    r"^\s*([A-Za-z_][\w-]*)\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,"
    r"\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]\s*(.*)$")

# 视为插图的标签(其余按文字处理)
_OCR_IMAGE_LABELS = {"image", "figure", "picture", "diagram", "chart", "chemical"}

# 标签 -> 前端可用的粗分类(仅作附加信息，不影响渲染)
_OCR_KIND = {
    "title": "heading", "section_header": "heading", "heading": "heading",
    "caption": "caption", "table": "table", "formula": "formula",
    "equation": "formula", "list": "list", "list_item": "list",
    "page_number": "footer", "page_footer": "footer", "page_header": "header",
}

# 公式类标签：整块就是数学
_OCR_MATH_LABELS = {"formula", "equation", "math", "equation_block",
                    "display_formula", "formula_block"}

# 数学定界符：\[...\] / \(...\) / $$...$$ / $...$（非贪婪、可跨行）
_OCR_MATH_SEG = re.compile(
    r"\\\[(.+?)\\\]|\\\((.+?)\\\)|\$\$(.+?)\$\$|"
    r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)", re.S)


def extract_latex(text: str) -> tuple[str, float]:
    """从 OCR 文本里抽出 LaTeX 片段，返回 (拼接的 LaTeX, 数学字符占比)。

    模型用 `\\( … \\)`(行内) / `\\[ … \\]`(独立) / `$$…$$` 包住公式；占比用于判断
    “整块是不是公式”：占比高 → 整块交给 MathJax 排版；占比低(公式只是行内提及) → 保持
    文字，避免把正常段落里的一个符号误当整块公式。
    """
    src = text or ""
    total = len(src.strip())
    if not total:
        return "", 0.0
    segs: list[str] = []
    used = 0
    for mt in _OCR_MATH_SEG.finditer(src):
        used += len(mt.group(0))
        seg = next((g for g in mt.groups() if g is not None), "")
        if seg.strip():
            segs.append(seg.strip())
    return "\n".join(segs), (used / total)


# “看起来就是公式”的强特征命令(用于无定界符的裸 LaTeX)。只收「正文里几乎不会出现」
# 的命令，避免把普通句子误判成公式（`\alpha`/`\beta` 这类单个希腊字母不算）。
_STRONG_MATH_RE = re.compile(
    r"\\(?:frac|dfrac|tfrac|sum|prod|int|iint|oint|sqrt|lim|nabla|partial|"
    r"infty|cdot|times|leq|geq|neq|approx|equiv|ell|mathcal|mathbb|mathbf|"
    r"mathrm|operatorname|left|right|quad|qquad|begin\{)")
_LATEX_CMD_RE = re.compile(r"\\[A-Za-z]{2,}")


def looks_like_latex(text: str) -> bool:
    """无定界符时判断整块是不是 LaTeX 公式（保守策略，避免误伤正文）。

    必须含一个“强数学命令”(\\frac/\\sum/\\int/\\sqrt/…)，再满足其一：
      * 至少 2 个 LaTeX 命令；或
      * 只有 1 个，但同时出现数学符号(下标/上标/等号/花括号)——如 `E = \\sum_{i=1}^{n} x_i`。
    另外限制长度 ≤ 500 字符：整段正文不会这么短还全被命令包住。
    """
    body = (text or "").strip()
    if not body or len(body) > 500:
        return False
    if not _STRONG_MATH_RE.search(body):
        return False
    if len(_LATEX_CMD_RE.findall(body)) >= 2:
        return True
    return bool(re.search(r"[=^_{}]", body))


def parse_ocr_blocks(text: str) -> list[dict]:
    """把「<标签> [x0,y0,x1,y1]<文本>」这种结构化 OCR 输出解析成块。

    坐标为 **0~1000 的归一化值**(每轴独立)，调用方换算成 PDF pt。不匹配的行(续行/纯文本)
    接到上一个块上；整体不含坐标格式时返回空列表，调用方回退到“整页纯文本”的处理
    (PaddleOCR-VL 的整页 Markdown 就走这条路)。
    """
    blocks: list[dict] = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        mt = _OCR_BLOCK_RE.match(line)
        if not mt:
            if blocks:
                blocks[-1]["text"] = (blocks[-1]["text"] + " " + line).strip()
            continue
        try:
            x0, y0, x1, y1 = (float(mt.group(i)) for i in range(2, 6))
        except ValueError:
            continue
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
        blocks.append({"label": mt.group(1).strip().lower(),
                       "bbox": [x0, y0, x1, y1], "text": mt.group(6).strip()})
    return blocks


def _ocr_block_lines(body: str, rect: fitz.Rect) -> list[dict]:
    """在 block 的矩形内还原行框。

    模型给的是“整段一个框”：文本里带换行就按换行切；只有一整行时，用
    「行数 ≈ 盒高/行高、行高 ≈ 1.35×字号」与「行宽×行数 ≈ 文本长×字号」联立估计
    字号与行数，再按字数近似折行，让行框尽量贴满盒子。
    """
    parts = [p.strip() for p in (body or "").splitlines() if p.strip()]
    if not parts:
        return []
    w = max(1.0, rect.width)
    h = max(1.0, rect.height)
    if len(parts) == 1:
        n_chars = max(1, len(parts[0]))
        size0 = max(5.0, min(30.0, (h * w / (0.65 * n_chars)) ** 0.5))
        n_lines = max(1, int(round(h / (1.35 * size0))))
        if n_lines > 1:
            per = max(8, int(round(n_chars / n_lines)))
            parts = textwrap.wrap(parts[0], per) or [parts[0]]
    lh = h / len(parts)
    size = max(5.5, min(28.0, round(lh * 0.78, 1)))
    return [{
        "x": rect.x0, "y": rect.y0 + i * lh, "w": w, "h": lh,
        "runs": [{"t": t, "s": size, "fam": "serif", "b": False,
                  "i": False, "up": False, "c": "#000000"}],
    } for i, t in enumerate(parts)]
