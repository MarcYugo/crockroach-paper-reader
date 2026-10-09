"""公式定位 · 底层：字体名判据(CM/AMS/Symbol) + 按字体候选区取区间 + 字形擦除。

`_blank_math_spans` 是 PyMuPDF 后端专用的落地方式：判出来的公式字形换成**等长空格**
(与 `pymupdf_extract.py --drop-formula` 一致)，而不是像 OCR 后端那样补 LaTeX。它**只
干一件事**：把数学字体的字形就地换成等长空格 —— 行框、字符数、前后文位置都不变；
**不记框、不记锚点**(见 `_blank_math_spans`)。

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
# 同一行里两段空格相隔多少个字符以内还算同一条公式(判据见 `_group_math_pieces`)
_MERGE_GAP_CHARS = 4
# CM 系「文字」字体：公式里的数字、运算符、`log`/`exp` 用它，但纯 pdflatex 的正文
# 也是它 —— 与 `pymupdf_extract.py` 的 CM_FONTS 同一条。算不算公式要按**占比**判
# (见 `_cm_text_math_on`)，不能跟弱数学字体共用开关。
_CM_TEXT_RE = re.compile(r"cm(r|bx|ti|ss|tt|sl)\d*", re.IGNORECASE)


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


def _cm_text_math_on(texts: list[dict], threshold: float = 0.5) -> bool:
    """CM 系**文字**字体(CMR/CMBX/CMTI…)到底算不算公式 —— 与 `pymupdf_extract.py`
    的 `build_math_filter` 同一口径：CM 系字符占比 < threshold 就算公式字体。

    为什么得单独判：`_weak_math_enabled` 的开关是「整页主字体是不是弱数学字体」，
    而实测 `samples/latex_sample.pdf` 的正文是 `LMRoman10-Regular`(65.7%，它**本身**
    就在弱字体表里)、公式的数字与运算符是 `CMR10`/`CMR7`(11.1%)。共用一个开关时
    “正文是弱字体 → 弱判据整体关掉”，公式里的 `=`/`+`/数字就全留在正文里(实测文本
    里剩下 `=1`、`=  1 +  2 +  + (9)`)。纯 pdflatex 文档(正文就是 CMR)时占比 ≥ 50%，
    依旧不把 CM 当公式，不会把正文整段擦掉。
    """
    counts: dict[str, int] = {}
    for blk in texts or []:
        for ln in blk.get("lines") or []:
            for r in ln.get("runs") or []:
                f = r.get("f") or ""
                counts[f] = counts.get(f, 0) + len(r.get("t") or "")
    total = sum(counts.values())
    if not total:
        return False
    cm = sum(n for f, n in counts.items() if _CM_TEXT_RE.match(_font_base(f)))
    return cm / total < threshold


def _math_font_flags(texts: list[dict]) -> tuple[bool, bool]:
    """本页字体判据的两个开关 `(cm_on, weak_on)` —— 两条都是**页级**判据。

    必须在**同一份 run 集合**上算一次、多处复用：擦除(`_blank_math_spans`)与
    「相邻 run 之间补空格」(`text_layer._insert_gap_spaces`) 用的是同一套
    `_blank_math_level` 判据 —— 补空格若切在公式内部，一处空格区会被断成两处，
    空位统计与「哪些页有公式」的判据都会偏(见 `_group_math_pieces`)。
    """
    return _cm_text_math_on(texts), _weak_math_enabled(texts)


def _accumulate_stats(acc: Optional[dict], add: dict) -> None:
    """把单页统计累加进整份文档的统计(`acc` 为 None 时忽略)。

    原来住在 `math_block.py`(已随 OCR 公式注入层一起删除)；它服务的其实是
    `_blank_math_spans` 的统计汇总，于是搬到这里与擦除逻辑同处。
    """
    if acc is None:
        return
    for k, v in (add or {}).items():
        acc[k] = acc.get(k, 0) + v


def _blank_math_level(font: str, cm_on: bool, weak_on: bool) -> int:
    """擦除用的字体分级：0=正文 1=弱(CM 系按占比判、其余按主字体判) 2=强数学字体。"""
    if _font_math_level(font, weak_on):
        return _font_math_level(font, weak_on)
    if cm_on and _CM_TEXT_RE.match(_font_base(font)):
        return 1
    return 0


# 正文空格的「字宽 / 字号」经验值：Georgia、Times 这类正文衬线字约 0.25em。
# 空位有几个空格**不再由公式的字形数决定**，而是由它的原宽反推
# (`round(宽 / 0.25em)`)—— 空格只是文本里的占位符，真正决定排版的是行级 `gaps`
# 下发的原宽(前端按它把空位撑到正好，见 `_line_gap_widths`)。
_SPACE_EM = 0.25


def _kept_width(geo: list[tuple], erased: set) -> tuple[float, bool]:
    """**保留正文**在原稿里占的总宽(pt)；第二个值表示每一段都算得出来。

    逐 run 用 bbox 宽(`x1 - x0`)。补间距造出来的空格 run(`text_layer._space_run`)
    没有几何 —— 它填的就是前后两个 run 之间那段空隙，按那两个 run 的间隙计。
    """
    total, ok, n = 0.0, True, len(geo)
    for i, (x0, x1) in enumerate(geo):
        if i in erased:
            continue
        if x0 is not None and x1 is not None:
            total += max(0.0, x1 - x0)
            continue
        left = next((geo[k][1] for k in range(i - 1, -1, -1) if geo[k][1] is not None), None)
        right = next((geo[k][0] for k in range(i + 1, n) if geo[k][0] is not None), None)
        if left is not None and right is not None and right > left:
            total += right - left
        else:
            ok = False
    return total, ok


def _line_gap_widths(ln: dict, geo: list[tuple], gaps: list[dict],
                     erased: set) -> list[Optional[float]]:
    """一条行里每条空位的**原宽(pt)** —— 「用行框宽度反推」。

    原理：行框(`ln.w` = 原稿这一行排到哪)里**不被保留正文占**的那部分宽度，就是这些
    空位应该占的总宽：

        total = 行框宽 − Σ(保留 run 的原宽)          （保留宽度见 `_kept_width`）
        w_i   = total × 比例_i

    比例用原稿几何定 —— 逐条的「间隙」= 首个被擦 run 的左边 → 它后面那个 run 的左边
    (公式在行尾时退到行右边界)；几何缺失的段落用已知段落的均值，全缺就等分。
    单空位行直接拿到 `total`(最准)；多空位行按各自间隙分(比按空格数平摊准得多 ——
    实测同一行里两段公式的宽度能差好几倍)。

    拿不到几何/行框宽时返回 `None`(那时前端**也不做任何反推**，空位就不定宽)。
    """
    n = len(geo)
    line_w = float(ln.get("w") or 0.0)
    keep_w, keep_ok = _kept_width(geo, erased)
    if not gaps or not keep_ok or line_w <= 0:
        return [None] * len(gaps)
    total = line_w - keep_w
    if total <= 0:
        return [None] * len(gaps)
    line_x1 = float(ln.get("x") or 0.0) + line_w
    est: list[Optional[float]] = []
    for g in gaps:
        x0 = geo[g["i0"]][0] if g["i0"] < n else None
        nxt = next((geo[k][0] for k in range(g["i1"] + 1, n) if geo[k][0] is not None), None)
        x1 = nxt if nxt is not None else line_x1
        est.append(max(0.0, x1 - x0) if (x0 is not None and x1 is not None) else None)
    known = [e for e in est if e is not None]
    avg = (sum(known) / len(known)) if known else (total / len(gaps))
    est = [avg if e is None else e for e in est]
    s = sum(est)
    if s <= 0:
        est = [1.0] * len(gaps)                      # 几何退化(全零宽) → 等分
        s = float(len(gaps))
    return [round(total * e / s, 3) for e in est]


def _split_count(n: int, widths: list[Optional[float]]) -> list[int]:
    """把 `n` 个空格分给一处空位覆盖的几个 run；返回每段写几个。

    权重用各 run **自己的原宽**(几何) —— 不是字形数；每个 run 至少留 1 个字符
    (否则那个 run 会变成空串)。取整零头补/扣到最宽的那段上，总和正好 `n`。
    """
    k = len(widths)
    if k <= 1:
        return [max(1, n)]
    n = max(n, k)                                   # 每段至少 1 个
    ws = [max(0.0, w or 0.0) for w in widths]
    s = sum(ws) or k
    out = [max(1, int(round(n * (w or 0.0) / s))) if s else 1 for w in ws]
    diff = n - sum(out)
    while diff:
        i = out.index(max(out))
        if diff > 0:
            out[i] += diff
            diff = 0
        else:
            take = min(-diff, out[i] - 1)
            if take <= 0:
                break
            out[i] -= take
            diff += take
    return out


def _blank_math_spans(texts: list[dict]) -> tuple[list[dict], dict]:
    """把数学字体的 run **就地**换成等长空格；返回 `(空格位列表, 统计)`。

    这是 `pymupdf_extract.py` 里 `--drop-formula` 的那套落地方式(PyMuPDF 没有公式
    语义，任何 `open()`/`get_text()` 开关都认不出公式，只能按字体名判)：公式在文字层里
    只是一堆按字体分开的字形，常乱序、上下标被拆到同一 y 的多行里，原样留着就是乱码，
    还会顺着 `block.text` / `page.text` 污染句子切分、翻译与复制。判据既然已经在手
    (`_font_math_level` + `_weak_math_enabled`)，就把这些字形**换成等长空格**：
    行框、字符数、前后文位置全都不变，正文照旧可切分/可翻译。

    与「拿到真 LaTeX 再把字形从数据里删掉」的落地方式是**互斥的两种**：这里没有
    LaTeX，只把字形擦成空格，块结构一个不动。

    ⚠️ **空格只是占位符，原宽才是排版依据**：

      * 返回的每个空格位只记 `(块下标, 行下标, 行内起, 行内止, 空格数)` —— 下游用它
        挑「哪些页有公式」（送 detection-service-group 检测，见 `backend_detection_service_group.py`）
        与排障；
      * 挂在行上的 `gaps`(`[[行内起, 空格数, 原宽 pt], …]`，**进 doc.json**)是给
        **前端排版**用的。原宽由 `_line_gap_widths` **用行框宽度反推**——这是
        解析侧的**唯一**实现，前端不再自己反推(只按原宽撑开)；
        空格的字宽 ≠ 原公式宽度，不把宽度补回空位，公式**后面的正文就会整体左移**
        (实测一行差 60px，行尾也收不到原行框的右边缘)；
      * 空格**个数由原宽反推**(`round(宽 / 0.25em)`)，**不再等于公式的字形数** ——
        字形数只说“这里原来有 20 个小符号”，跟这段该占多宽没关系。公式空位被正文字体
        的标识符切开时(见 `_group_math_pieces`)，这几个空格按各自 run 的原宽分配。

    框 / 覆盖层一概没有(公式框链路已删除)。整行都被擦空的行/块由调用方
    `backend_pymupdf._blank_math_and_prune` 清理掉(行被删时 `gaps` 跟着消失)。

    空格位 = 一段**字符相邻**的空格区域：同一个 run 里连着擦的、或紧挨着的两个数学
    run 擦出来的空格都算一处；中间夹了非数学字符就断成两处。

    统计里的 `formula_spans` / `formula_chars` 是**被擦掉的 run 数与字形数**(含 `+`/`=`
    这类只由标点组成的 run，与脚本同口径)，`formula_spaces` 是**实际写下的空格数**
    (由原宽反推，与字形数不再相等)，`formula_lines` 是涉及的行数。

    字体判据有两条，各管一档(见 `_blank_math_level`)：强数学字体一律擦；CM 系文字
    字体(公式里的数字与运算符)按**占比**判(`_cm_text_math_on`)；其余弱数学字体
    (LM/STIX/Cambria Math…)仍按整页主字体判(`_weak_math_enabled`)，避免把正文擦掉。

    ⚠️ 必须在本页文字行**最终确定之后**调用(即去重影之后)：被判为「图里的字」而删掉的
    行不该再被算进统计。调用时 run 上要带 `f`(字体名)，用完由 `_strip_font_keys` 剥掉。
    统计键与 `_font_math_index` 同风格。
    """
    cm_on, weak_on = _math_font_flags(texts)
    segs: list[dict] = []                  # 一处「字符相邻的空格区」一条
    spans = math_chars = math_spaces = 0
    lines: set[tuple[int, int]] = set()
    for bi, blk in enumerate(texts or []):
        for li, ln in enumerate(blk.get("lines") or []):
            runs = ln.get("runs") or []
            geo = [(r.get("x0"), r.get("x1")) for r in runs]   # 原稿几何(临时键)
            # ---- ① 先只**标记**公式 run：定出本行的空位(字符相邻的连着算一处) ----
            # 擦除**不挑内容** —— 与 `pymupdf_extract.py` 一致：字体是数学字体就抹掉。
            # `+`/`=` 这类只由标点组成的 run 也是公式的一部分(`_has_glyphs(" +")` 为
            # 假，但它在公式里就是个运算符)；按它过滤会把这半个公式留在正文里(实测
            # 文本里剩下 `Summation formula:\n\n+    +       + (9)`)。
            gaps: list[dict] = []
            cur: Optional[dict] = None
            off = 0
            for ri, r in enumerate(runs):
                t = r.get("t") or ""
                a, off = off, off + len(t)
                if not t:
                    continue
                if not _blank_math_level(r.get("f") or "", cm_on, weak_on):
                    cur = None          # 中间夹了非公式字符 → 空位到此为止
                    continue
                spans += 1
                math_chars += len(t)
                lines.add((bi, li))
                if cur is not None and cur["boff"] == a:
                    cur["boff"] = off  # 与上一段在字符上紧挨着 → 还是同一处空位
                    cur["i1"] = ri
                else:
                    cur = {"bi": bi, "li": li, "i0": ri, "i1": ri, "boff": off}
                    gaps.append(cur)
            if not gaps:
                continue
            # ---- ② 原宽：**行框宽度反推**（解析侧的唯一实现，前端只按它撑开） ----
            erased = {k for g in gaps for k in range(g["i0"], g["i1"] + 1)}
            widths = _line_gap_widths(ln, geo, gaps, erased)
            # ---- ③ 空格数由**宽度**定，并写回被擦的 run ----
            # 一处空位可能盖住好几个 run(公式被正文字体的标识符切开)：空格按**各 run
            # 自己的原宽**分给它们，让每段都还占着原来的位置。
            for g, w in zip(gaps, widths):
                idx = list(range(g["i0"], g["i1"] + 1))
                size = max((runs[k].get("s") or 10.0) for k in idx)
                n = (max(1, int(round(w / (_SPACE_EM * size)))) if w else len(idx))
                for k, cnt in zip(idx, _split_count(n, [
                        (geo[k][1] - geo[k][0]
                         if (geo[k][0] is not None and geo[k][1] is not None) else None)
                        for k in idx])):
                    runs[k]["t"] = " " * cnt
                g["n"] = n
                math_spaces += n
            # ---- ④ 字符区间必须按**改写后**的长度算(空格数变了，后面 run 的下标会移) ----
            starts: list[int] = []
            p = 0
            for r in runs:
                starts.append(p)
                p += len(r.get("t") or "")
            for g in gaps:
                g.pop("boff", None)
                g["a"] = starts[g["i0"]]
                g["b"] = starts[g["i1"]] + len(runs[g["i1"]].get("t") or "")
                segs.append(g)
            # 行级 `gaps` 进 doc.json：`[行内起, 空格数, 原宽 pt]` —— 前端按原宽把
            # 这段空格撑到正好(见函数文档)；原宽缺失时写 `None`，前端就不定宽。
            ln["gaps"] = [[g["a"], g["n"], (round(w, 3) if w else None)]
                          for g, w in zip(gaps, widths)]
    return segs, {"formula_spans": spans, "formula_chars": math_chars,
                  "formula_spaces": math_spaces,
                  "formula_lines": len(lines),
                  "formula_weak_pages": 1 if weak_on else 0,
                  "formula_cm_pages": 1 if cm_on else 0}


def _group_math_pieces(texts: list[dict], segs: list[dict]) -> list[dict]:
    """把同一行里**挨得够近**的几段空格并成一条公式；返回分组后的空格位。

    一条公式在 PDF 里常被**正文字体**排的标识符切成好几段：
    `x = (-b ± √(b²-4ac)) / 2a` 里的 `x`、数字用的是正文字体(判据不算数学字形)，
    于是擦出来是「一截空格 + x + 一截空格 + …」。不合并的话，空位统计会虚高
    （实测样本 184 段 → 60 条公式）、送检测的页也无法按整条公式记账。

    合并判据(没有几何可用，只能看字面)：同一行，且两段之间的原文
      * 不含空白 —— 有空白就说明中间隔着一个完整的词，那是两条公式或正文；
      * 长度 ≤ `_MERGE_GAP_CHARS`；
      * 不含 ≥3 个连续字母 —— 挡住 `and` / `where` 这类连词把两条公式粘成一条。

    跨行的合并**不做**：一条公式跨两行要靠行框几何判断，那套几何判据早已随本地
    公式框一起删了；实测跨行的独立公式只占少数，先不猜。

    分组结果在组级带 `a`/`b`/`n`(整体的外框区间与空格数)；逐段的原文区间留在
    `parts` 里（目前没有下游使用 —— 公式覆盖层按 detection-service-group 的公式框定位，
    不再按段）。
    """
    groups: list[dict] = []
    for s in segs:
        cur = groups[-1] if groups else None
        if cur is not None and cur["bi"] == s["bi"] and cur["li"] == s["li"] \
                and s["a"] >= cur["b"]:
            line = (texts[s["bi"]].get("lines") or [])[s["li"]]
            gap = _line_text(line)[cur["b"]:s["a"]]
            if (len(gap) <= _MERGE_GAP_CHARS and not re.search(r"\s", gap)
                    and not re.search(r"[A-Za-z]{3,}", gap)):
                cur["b"] = s["b"]
                cur["n"] += s["n"]
                cur["parts"].append(s)
                continue
        groups.append({"bi": s["bi"], "li": s["li"], "a": s["a"], "b": s["b"],
                       "n": s["n"], "parts": [s]})
    return groups


def _line_text(line: dict) -> str:
    """一行擦除后的原文(各部分拼接) —— 用来取两段空格之间的"夹心"内容。"""
    return "".join(r.get("t") or "" for r in line.get("runs") or [])


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
