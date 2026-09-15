"""公式定位 · 底层：字体名判据(CM/AMS/Symbol) + 按字体候选区取区间。

见包文档 `backend/pdf_parser/__init__.py`。
"""

from __future__ import annotations

import re
from typing import Optional

from .math_text import (_has_glyphs, _math_ish, _tighten)


# ---------------------------------------------------------------------
#  字体名 → 公式候选区(独立于 OCR 的定位信号)
#
#  电子版 PDF 里数学排版用的是**独立的数学字体族**(Computer Modern / AMS /
#  Symbol)，与正文截然分开。实测真论文：正文 100% 是 `NimbusRomNo9L-*`，公式字形
#  100% 是 `CMR/CMMI/CMSY/CMEX`。于是「哪些字形是公式」可以**不依赖 OCR** 先判出来，
#  而且粒度是 **span 级** —— 远比 `_math_ish`(「整行像不像正文」)精确：像
#  `t,s = closet+h,s − closet,s` 这种带 `close` 标识符的公式不会再被判成正文。
#
#  它与 OCR 是**互补**而不是替代：
#    * 字体候选区负责**定位**，OCR 从「找位置」退化为「验证 + 补漏」；
#    * OCR 的 LaTeX 负责**内容**，并补上字体判据的漏检 —— 公式里的标识符/标点
#      常用正文字体排(实测 `AR` 与句末 `,` 都是 `NimbusRomNo9L-Regu`)。
#
#  两条判据都用不上时(扫描件没有字体名 / Word 导出数学与正文同体)整条退回 OCR
#  锚点法(见 `_display_job` / `_ocr_line_replacements`)。
# ---------------------------------------------------------------------
# 「强」数学字体：只在数学排版里出现，命中即可判为公式字形
_MATH_FONT_STRONG = (
    "cmmi", "cmsy", "cmex", "cmmib", "cmbsy", "cmbex",
    "msam", "msbm", "eufm", "eurm", "symbol", "mtmi", "mtsy", "mtex",
    "euclid", "rsfs", "stmary", "stmry", "wasy",
    "lmmi", "lmsy", "lmex", "lmmib",
)
# 「弱」数学字体：数学(正体数字/`log`/`exp`)与正文都可能用 —— 只有在它**不是本页
# 主字体**时才算公式字形(LaTeX 默认模板整篇都是 CM 系，此时弱判据会整页误判)。
_MATH_FONT_WEAK = ("cmr", "cmb", "cmss", "cmtt", "cmu",
                   "latinmodern", "lmr", "lmss", "lmtt",
                   "stix", "xits", "cambria math", "campmath",
                   "newtxmath", "mathpazo", "txfonts")
# 「符号专用」字体：数学排版里的大符号/运算符(∑ ∫ √ ± 大括号)才用它们。
# 用来当「整行摘除」的**证据** —— 表格里的文字也会被数学字体排(实测 `Top-k`/
# `avgIC`/`h = 1` 都是 `CMMI9`/`CMR9`)，但它们几乎不会用到这一档字体。
_SYMBOL_FONT = ("cmsy", "cmex", "msam", "msbm", "wasy", "stmary", "stmry",
                "eufm", "euex", "symbol")
# 一行「几乎全是数学字形」的下限(残余部分还要再不像正文，才能整行摘掉)
_LINE_MATH_PURE = 0.9


def _font_base(font: str) -> str:
    """字体名归一：去子集前缀(`ABCDEF+`)与结尾字号(`CMMI10`→`cmmi`)，只留小写字母。"""
    name = (font or "").split("+")[-1].strip().lower()
    return re.sub(r"[\d\s]+$", "", re.sub(r"[^a-z0-9\s]", "", name))


def _font_math_level(font: str, weak_on: bool) -> int:
    """0=非数学字体 1=弱数学字体 2=强数学字体。"""
    base = _font_base(font)
    if not base:
        return 0
    if any(base.startswith(p) for p in _MATH_FONT_STRONG):
        return 2
    if weak_on and any(base.startswith(p) for p in _MATH_FONT_WEAK):
        return 1
    return 0


def _weak_math_enabled(local_texts: list[dict]) -> bool:
    """本页能不能用「弱数学字体」判据。

    主字体本身就是弱数学字体(整篇 CM 系，如 LaTeX 默认模板)时不能 —— 否则整页
    正文都会被当成公式字形。
    """
    counts: dict[str, int] = {}
    for blk in local_texts:
        for ln in blk.get("lines") or []:
            for r in ln.get("runs") or []:
                f = r.get("f") or ""
                counts[f] = counts.get(f, 0) + len(r.get("t") or "")
    if not counts:
        return False
    top = max(counts.items(), key=lambda kv: kv[1])[0]
    base = _font_base(top)
    return not any(base.startswith(p) for p in _MATH_FONT_WEAK)


def _line_math_spans(line: dict, weak_on: bool
                     ) -> tuple[list[tuple[int, int]], int]:
    """一行的数学字形区间 `[(a, b), …]`(行内**原串**下标，与 `_apply_replacements` 一致)。

    相邻的数学 run 会合并成一段，**纯空白的数学 run 也算在内**：数学排版里的间距
    字形(如 `CMSY10` 的空格)会把 `′` 和 `×` 隔开，不合并就断成两段。
    """
    runs = line.get("runs") or []
    bounds: list[tuple[int, int]] = []
    levels: list[int] = []
    pos = 0
    for r in runs:
        t = r.get("t") or ""
        bounds.append((pos, pos + len(t)))
        levels.append(_font_math_level(r.get("f") or "", weak_on) if t else 0)
        pos += len(t)
    spans: list[tuple[int, int]] = []
    math_chars = 0
    for i, lvl in enumerate(levels):
        a, b = bounds[i]
        if not lvl or b <= a:
            continue
        math_chars += b - a
        if spans and a <= spans[-1][1]:
            spans[-1] = (spans[-1][0], b)
        else:
            spans.append((a, b))
    return spans, math_chars


def _font_math_index(metas: list[dict], local_texts: list[dict]
                     ) -> tuple[dict[int, list[tuple[int, int]]], dict]:
    """整页扫一遍，给出「每个行下标 → 该行的数学字形区间」。返回 (索引, 统计)。

    只登记**含真字形**的区间：纯空白/纯标点的数学 run(如某些图例里的空格字形)
    不算公式，否则会把正常的行也标成候选。
    """
    weak_on = _weak_math_enabled(local_texts)
    index: dict[int, list[tuple[int, int]]] = {}
    strong_lines = weak_lines = chars = 0
    for m in metas:
        ln = local_texts[m["bi"]]["lines"][m["li"]]
        spans, math_chars = _line_math_spans(ln, weak_on)
        spans = [(a, b) for a, b in spans if _has_glyphs(m["raw"][a:b])]
        if not spans:
            continue
        index[m["idx"]] = spans
        chars += math_chars
        if any(_font_math_level(r.get("f") or "", weak_on) == 2
               for r in (ln.get("runs") or [])):
            strong_lines += 1
        else:
            weak_lines += 1
    return index, {"font_lines": len(index), "font_strong_lines": strong_lines,
                   "font_weak_lines": weak_lines, "font_chars": chars,
                   "font_weak_pages": 1 if weak_on else 0}


def _mask_out(raw: str, spans) -> str:
    """把数学字形区间从原串里抹掉(其余字符原样保留)——用于「剩下的是不是正文」。"""
    out = list(raw or "")
    for a, b in spans or []:
        for i in range(max(0, a), min(len(out), b)):
            out[i] = " "
    return "".join(out)


def _has_symbol_font(line: dict) -> bool:
    """这行用没用「符号专用」数学字体(见 `_SYMBOL_FONT`)。"""
    for r in line.get("runs") or []:
        if not (r.get("t") or "").strip():
            continue
        base = _font_base(r.get("f") or "")
        if any(base.startswith(p) for p in _SYMBOL_FONT):
            return True
    return False


def _font_math_line(raw: str, spans, covered: int) -> bool:
    """这一行能不能**整行**当公式摘掉：数学字形以外只剩空白/标点。

    只有这样的行才算是「纯粹的一条公式」(独立公式的行几乎都是)。混着正文单词的行
    必须交给行内路径去「切两侧」(`_inline_job`) —— 实测 `where a^{(i)}` 这类行有
    一半字符是数学字体，按 50% 占比就整行摘会把 `where` 一起删掉；表格里的
    `20100101 to 20170831.` 同理。
    """
    if not spans:
        return False
    rest = _mask_out(raw, spans)
    if not re.search(r"[0-9A-Za-z\u3400-\u4dbf\u4e00-\u9fff]", rest):
        return True                        # 残余只有空白/标点 → 纯公式行
    return (covered >= _LINE_MATH_PURE * max(1, len((raw or "").strip()))
            and _math_ish(rest)
            and bool(re.search(r"[=+^_∑∫√()\[\]/]", rest)))


def _widen_with_cand(a: int, b: int, spans) -> list[tuple[int, int]]:
    """把 `[a, b)` 与字体候选区取**并集**，返回按优先级排好的区间表(先并集后原样)。

    字体判据会漏掉公式里用正文字体排的那截(`AR`)，而 OCR 锚点算出来的区间又可能
    停在标点上 —— 两者取并集通常正好补齐。调用方按序试并用 `_math_glyphs` 校验，
    并集一旦吃进正文就自动退回原区间，所以这个"扩张"是**安全**的。
    """
    if b <= a or not spans:
        return [(a, b)]
    hit = [(ca, cb) for ca, cb in spans if min(b, cb) > max(a, ca)]
    if not hit:
        return [(a, b)]
    wide = (min(a, min(ca for ca, _ in hit)), max(b, max(cb for _, cb in hit)))
    return [(a, b)] if wide == (a, b) else [wide, (a, b)]


def _retighten(raw: str, a: int, b: int) -> tuple[int, int]:
    """扩张之后把区间再收紧一次：开头空白/标点留给正文，结尾空白也留给渲染层。

    扩张(`_widen_with_cand` / `_widen_cut`)是往公式字形上贴，可能多带一个空格进去
    —— 不收紧的话块的 x/w 会变、KaTeX 覆盖层会整体左移(实测差 4.7pt)。把
    `_tighten` 的两条规则重新套一遍即可；`AR` 这种字母开头的扩张不受影响。
    """
    a2 = _tighten(raw, a, b)[0]
    return a2, _tighten(raw, a2, b)[1]


def _widen_cut(pos: int, spans, *, left: bool) -> int:
    """把切点吸附到与之相接的字体候选区边界上(没有相接的候选就原样返回)。

    `left=True` 只允许**左移**(公式在左行尾部，切点要往前挪才能把用正文字体排的
    标识符 `AR` 包进去)；`left=False` 只允许**右移**。只往一个方向挪是关键：
    否则切点会被拉到候选区里，把 `AR` 这种「候选区左侧」的字形漏在外面。
    """
    best: Optional[int] = None
    for ca, cb in spans or []:
        if left:
            if ca <= pos <= cb or 0 <= ca - pos <= 6:
                best = ca if best is None else min(best, ca)
        elif ca <= pos <= cb or 0 <= pos - cb <= 6:
            best = cb if best is None else max(best, cb)
    if best is None:
        return pos
    return min(pos, best) if left else max(pos, best)
