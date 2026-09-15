"""文字层：字体族判定 + 行(lines)/行内样式(runs)抽取 + 行框并集。

见包文档 `backend/pdf_parser/__init__.py`。
"""

from __future__ import annotations

import re
import fitz  # PyMuPDF
from typing import Optional


_SERIF = ("times", "georgia", "garamond", "palatino", "book", "caslon",
          "minion", "charter", "utopia", "song", "serif", "latinmodern", "notoserif")
_MONO = ("courier", "consolas", "menlo", "mono", "sfmono", "dejavusansmono")
_SANS = ("helvetica", "arial", "verdana", "tahoma", "trebuchet", "gill",
         "futura", "univers", "myriad", "calibri", "roboto", "inter", "notosans", "sans")

def _family(font: str) -> str:
    name = re.sub(r"[^a-z0-9 ]", "", (font or "").split("+")[-1].lower())
    if any(k in name for k in _MONO):
        return "mono"
    if "symbol" in name or "song" in name:
        return "serif"
    if any(k in name for k in _SERIF):
        return "serif"
    if any(k in name for k in _SANS):
        return "sans"
    return "serif"


def _join_lines(lines: list) -> str:
    """把块内多行拼成整段纯文本(处理断行连字符/换行空格)，用于翻译/zh_all。"""
    out = ""
    for ln in lines:
        t = "".join(r["t"] for r in ln["runs"]).strip()
        if not t:
            continue
        if out.endswith("-"):
            out = out[:-1] + t
        elif out:
            out += " " + t
        else:
            out = t
    return out


# =====================================================================
#  文字
# =====================================================================
def _line_baseline(spans: list) -> Optional[float]:
    """本行「正文基线」：按**字符数加权**取 span 基线(origin.y)的众数。

    一行里正文的字数一定多于上下标，所以众数就是正文基线；并列时取字号大的
    那个（上标必然比它所属的正文小）。唯一用途是校验 PyMuPDF 报的「上标」
    是不是真抬升了，见 `_is_super`。
    """
    w: dict[float, tuple[int, float]] = {}
    for sp in spans:
        org = sp.get("origin")
        if not org:
            continue
        y = round(org[1], 1)
        n, sz = w.get(y, (0, 0.0))
        w[y] = (n + max(1, len((sp.get("text") or "").strip())),
                max(sz, sp.get("size", 0)))
    if not w:
        return None
    return max(w.items(), key=lambda kv: (kv[1][0], kv[1][1]))[0]


def _is_super(sp, fl: int, base: Optional[float]) -> bool:
    """这个 span 是不是**真的**上标。

    ⚠️ PyMuPDF 的 bit0(上标) 不能单独用：它是「基线比前一个 span 抬高」的启发式，
    TeX 把公式的上下标排回正文基线时会**误判**。实测本样本真论文 7 页：bit0 标了
    173 个字符，其中 **157 个(91%) 是「正文字号 + 正好压在本行基线上」的正文**
    （例：` is the value vector of formu-`、`is the IC array and`、
    `, where each column (stock) can`）。
    前端 `styleRun` 见到 up 就 `vertical-align:super; font-size × 0.7`，于是
    「公式后面的正文」被缩掉 30% —— 这就是「正文公式后字号变小」的根因。
    真上标实测抬升 3.6~5.2pt，阈值 0.2×字号(≈1.4pt) 安全边际很大；假上标抬升 0.00。
    注：下标(抬升为负)这里一律不算——bit0 本来也只管上标。
    """
    if not (fl & 1):
        return False
    org = sp.get("origin")
    if not org or base is None:
        return True                     # 拿不到几何信息时，只能信 bit0
    size = sp.get("size", 10) or 10
    return (base - org[1]) > 0.2 * size


def _parse_text_blocks(page) -> list[dict]:
    blocks: list[dict] = []
    raw = page.get_text("dict")
    for blk in raw.get("blocks", []):
        if blk.get("type") != 0:
            continue  # 图片块在这里忽略，交给专门的图片管线
        lines = []
        for ln in blk.get("lines", []):
            sps = ln.get("spans", [])
            base = _line_baseline(sps)
            runs = []
            for sp in sps:
                t = sp.get("text", "")
                if not t:
                    continue
                fl = sp.get("flags", 0)
                runs.append({
                    "t": t,
                    "s": round(sp.get("size", 10), 1),
                    # 原始字体名：只服务于「公式候选区」判定(见 `_font_math_index`)，
                    # 出页面前由 `_strip_font_keys` 剥掉 —— 前端不认识它，
                    # doc.json 也没必要为它变大。`fam` 仍是粗分类。
                    "f": sp.get("font", ""),
                    "fam": _family(sp.get("font", "")),
                    "b": bool(fl & 16),   # 粗体
                    "i": bool(fl & 2),    # 斜体
                    "up": _is_super(sp, fl, base),   # 上标（已校验基线，见函数注释）
                    "c": "#%06x" % (sp.get("color", 0) & 0xFFFFFF),
                })
            if not runs:
                continue
            lx0, ly0, lx1, ly1 = ln["bbox"]
            lines.append({"x": lx0, "y": ly0, "w": lx1 - lx0, "h": ly1 - ly0, "runs": runs})
        if not lines:
            continue
        x0, y0, x1, y1 = blk["bbox"]
        text = _join_lines(lines)
        if not text:
            continue
        blocks.append({
            "id": f"p{page.number}b{len(blocks)}",
            "x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0,
            "text": text, "lines": lines,
        })
    return blocks


def _text_line_rects(texts: list[dict]) -> list[fitz.Rect]:
    rects = []
    for blk in texts:
        for ln in blk.get("lines", []):
            rects.append(fitz.Rect(ln["x"], ln["y"], ln["x"] + ln["w"], ln["y"] + ln["h"]))
    return rects

def _union_line_rect(lines: list[dict]) -> tuple[float, float, float, float]:
    x0 = min(l["x"] for l in lines)
    y0 = min(l["y"] for l in lines)
    x1 = max(l["x"] + l["w"] for l in lines)
    y1 = max(l["y"] + l["h"] for l in lines)
    return x0, y0, x1, y1
