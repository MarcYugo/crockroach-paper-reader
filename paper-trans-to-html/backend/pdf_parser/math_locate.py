"""公式定位：锚点匹配 + 视觉行/栏判定 + 候选行(含按字体的 display 定位)。

见包文档 `backend/pdf_parser/__init__.py`。
"""

from __future__ import annotations

import re
from typing import Optional

from .math_font import (_font_math_line, _has_symbol_font)
from .math_text import (_ANCHOR_LENS, _ANCHOR_MIN_TOTAL, _ANCHOR_RANK_GAP,
                        _ANCHOR_RANK_MIN, _ANCHOR_SPAN_MAX, _SPAN_MAX_LINES,
                        _all_pos, _bag_ratio, _lines_text, _math_ish,
                        _meta_at)
from .text_layer import (_union_line_rect)


def _job_glyphs(metas: list[dict], job: Optional[dict]) -> str:
    """定位结果 `job` 落点上「会被公式覆盖的那堆原字形」拼成的文本。"""
    if not job:
        return ""
    if job.get("kind") == "repl":
        m = metas[job["mi"]]
        return m["raw"][job["a"]:job["b"]]
    if job.get("kind") == "span":
        return " ".join([job["lm"]["raw"][job["a"]:]]
                        + [m["raw"] for m in job["mids"]]
                        + [job["rm"]["raw"][:job["b"]]])
    return _lines_text(job.get("lines"))          # 独立公式：lines 是行对象


def _job_rank(metas: list[dict], latex: str, job: Optional[dict]) -> float:
    """锚点歧义打分：这个落点的字形像不像**这条** LaTeX。"""
    return _bag_ratio(latex, _job_glyphs(metas, job))


def _find_anchor_pair(page_norm: str, left: str, right: str,
                      validate=None, rank=None) -> Optional[tuple[int, int]]:
    """在整页规整文本里给「左锚点(取尾巴) … 右锚点(取脑袋)」找**唯一**的自洽组合。

    返回 `(左锚点末字符下标, 右锚点首字符下标)`；找不到或有多种组合 → None。
    锚点长度由长到短试：长锚点更稳，但 OCR 一旦误读(如 gplearn→gpleam)、或正文里有
    连字(ﬁ)，整条就找不到；短锚点容易命中却可能撞车 —— 所以取第一个「只给出一种
    组合」的长度，命中即止；两个锚点在整页里的间距还得小得像一条公式(见 _ANCHOR_SPAN_MAX)。

    `validate(左锚点末下标, 右锚点首下标) -> bool` 由调用方传：把「锚点拼得上、但落点
    不是同一处」的候选筛掉。论文里经常有两句一模一样的话(实测页 2 有两句
    "can be calculated as")，光看锚点分不开，除非再看「两个锚点中间是不是公式字形」。

    `rank(左锚点末下标, 右锚点首下标) -> float`(可选)是**歧义消解**：validate 之后还剩
    多条候选时给它打分，取分最高、且与次高拉开差距的那条。实测页 2 有**两条** `\\[…\\]`
    各自都能配到两处("can be calculated as" 出现两次)，只看"唯一性"会把两条一起废掉；
    这时唯一好用的判据就是「这个落点的字形像不像**这条** LaTeX」。给不出分就照旧放弃。
    """
    for L in _ANCHOR_LENS:
        ln, rn = left[-L:], right[:L]
        if len(ln) + len(rn) < _ANCHOR_MIN_TOTAL:
            continue
        lhits = _all_pos(page_norm, ln)
        if not lhits:
            continue
        pairs = [(j + len(ln) - 1, k)
                 for j in lhits for k in _all_pos(page_norm, rn)
                 if j + len(ln) <= k and k - (j + len(ln)) <= _ANCHOR_SPAN_MAX]
        if validate is not None:
            pairs = [p for p in pairs if validate(*p)]
        if len(pairs) == 1:
            return pairs[0]
        if len(pairs) > 1 and rank is not None:
            scored = sorted(((rank(*p), p) for p in pairs), key=lambda x: -x[0])
            if (scored[0][0] >= _ANCHOR_RANK_MIN
                    and scored[0][0] - scored[1][0] >= _ANCHOR_RANK_GAP):
                return scored[0][1]
    return None


def _display_line_cands(metas: list[dict], local_texts: list[dict],
                        y_lo: float, y_hi: float, x_lo: float, x_hi: float,
                        skip: set) -> list[dict]:
    """纵向区间 `[y_lo, y_hi]` 里「像公式字形」的本地行(纯筛选，不改数据)。"""
    cand: list[dict] = []
    for m in metas:
        if m["idx"] in skip:
            continue
        ln = local_texts[m["bi"]]["lines"][m["li"]]
        cy = ln["y"] + ln["h"] / 2.0
        if not (y_lo - 2.0 <= cy <= y_hi + 2.0):
            continue
        if ln["x"] + ln["w"] < x_lo - _ANCHOR_X_PAD or ln["x"] > x_hi + _ANCHOR_X_PAD:
            continue
        if not _math_ish(m["raw"]) or not re.search(r"[=+^_∑∫√()\[\]/]", m["raw"]):
            continue
        cand.append(m)
    return cand


# 锚点行的左右容许外扩(pt)：公式比正文略宽是常态，但隔着一栏就不该算了
_ANCHOR_X_PAD = 24.0
# 同一「视觉行」的判定：两行的纵向重叠 ≥ 较矮那行的这个比例。
# 上下标会被 PyMuPDF 排成独立的行、bbox 与正文行上下错开几个 pt(实测 `where a^{(i)}`
# 在 y 633.69，同行右半截 `t,n) is the value vector` 在 y 636.53)，但两者重叠很多；
# 相邻的下一行正文则完全不重叠 —— 这个门槛把两类分得很干净。
_ROW_OVERLAP = 0.40
# 同一视觉行内「算同一栏」的最大空隙：词间空格 ~2.5pt、上下标块之间 ~10pt，栏间距 ~28pt。
_COL_GAP = 12.0
# 「补行」时对**额外行**要求的数学字形占比(比 `_LINE_MATH_PURE` 松)。
# 位置已经由种子行(纯公式行)钉死了，额外行只需要「字形以数学字体为主」作旁证。
_ROW_MATH_RATIO = 0.5
# 补行最多迭代几轮(公式被切成几块)；一轮不动就停。也防止沿 x 无休止地爬。
_ROW_EXTEND_PASS = 4


def _row_span(metas: list[dict], local_texts: list[dict],
              line: dict) -> tuple[float, float]:
    """锚点行所在「视觉行 ∩ 所在栏」的 x 范围。

    ⚠️ **不能只用锚点行自己的 x/w**：独立公式排在同一栏里、可以远超锚点句。实测
    `where a^{(i)} = …` 只是那一行的第一个块(x 324.96~367.62)，同一行还有三个块一直
    排到栏右 558.00；公式主体(376.56~496.44)与贴右的编号(546.38~558.00)全在锚点块
    之外 —— 按锚点块取窗口会把整条公式的右半截排除在候选之外，公式块只剩左半截宽
    (54.5pt / 真值 119.9pt)，前端据此把 KaTeX 缩到 0.6 下限、右侧原字形碎片留在
    文字层按正文 15px 画着，就是「公式尺寸不对 + 与碎片重影」。

    取法：先按纵向重叠圈出**同一视觉行**的所有行，再按 x 相邻(间隙 ≤ `_COL_GAP`)
    聚类，取与锚点行横向重叠最多的那一段 —— 另一栏与锚点行隔着一整条栏缝，天然
    被排除在外。
    """
    y0, y1 = line["y"], line["y"] + line["h"]
    x0, x1 = line["x"], line["x"] + line["w"]
    segs: list[tuple[float, float]] = []
    for m in metas:
        ln = local_texts[m["bi"]]["lines"][m["li"]]
        a, b = ln["y"], ln["y"] + ln["h"]
        if min(b, y1) - max(a, y0) < _ROW_OVERLAP * min(b - a, y1 - y0):
            continue                        # 不同视觉行(上一行 / 下一行)
        segs.append((ln["x"], ln["x"] + ln["w"]))
    if not segs:
        return x0, x1
    segs.sort()
    best: Optional[tuple[float, tuple[float, float]]] = None
    lo, hi = segs[0]
    for a, b in segs[1:] + [(float("inf"), float("inf"))]:
        if a - hi > _COL_GAP:               # 栏缝 → 当前这段结束，结算一次
            ov = min(hi, x1) - max(lo, x0)
            if ov > 0 and (best is None or ov > best[0]):
                best = (ov, (lo, hi))
            lo, hi = a, b
        else:
            hi = max(hi, b)
    if best is None:                        # 锚点行自己就是一段(单块成行、左右没邻居)
        return x0, x1
    return min(best[1][0], x0), max(best[1][1], x1)


def _x_window(metas: list[dict], local_texts: list[dict], lb: dict,
              la: dict) -> tuple[float, float]:
    """左右锚点行给出的 x 扫描窗口：两个锚点**整行(同栏)**范围的并集 ± `_ANCHOR_X_PAD`。

    **唯一实现** —— `_math_rows_between`(字体候选路径)与 `_display_job`(启发式兜底)
    都用它，不要再各写一份容差(历史上两处的 x 容差不一样，同一个公式两条路得到的结果
    不同)。
    """
    lx0, lx1 = _row_span(metas, local_texts, lb)
    ax0, ax1 = _row_span(metas, local_texts, la)
    return min(lx0, ax0) - _ANCHOR_X_PAD, max(lx1, ax1) + _ANCHOR_X_PAD


def _row_of(m: dict, local_texts: list[dict]) -> dict:
    """metas 条目 → 它对应的行对象。"""
    return local_texts[m["bi"]]["lines"][m["li"]]


def _extend_rows(metas: list[dict], local_texts: list[dict], rows: list[dict],
                 *, y_lo: float, y_hi: float, skip: set, anchor: set,
                 font_cands: Optional[dict]) -> list[dict]:
    """把「与已选公式行同一视觉行、横向相接、且字形以数学字体为主」的行补进来。

    **为什么需要它**：一条独立公式常被 PyMuPDF 切成好几个块，而后续块里往往**混着
    正文衬线体**排的字形(实测 `step\_num` 的下划线、下标 `model` 是 NimbusRomNo9L) →
    过不了 `_font_math_line` 的 0.9 行纯度门槛(实测该行 `covered/len=0.85`)，于是整条
    公式只摘到左半截，右半截留在文字层按正文字号画着 —— 就是「公式尺寸不对 + 重影」。
    这类行在几何上与已选中的纯公式行**同一视觉行、横向相接**，位置已被种子行钉死，
    所以对它们只要一个更松的旁证：字形以数学字体为主(≥ `_ROW_MATH_RATIO`)。

    安全阀(三重，缺一不可)：
      · 必须先有种子行(经正常判据选出的公式行) → 不会无中生有；
      · **纵向**按**种子**行的并集带判定，所以只能在同一视觉行里补，
        不会爬到公式的下一行去(链式爬升会把别的公式的字形也吞掉)；
      · **横向**要与已选集**相接**(间隙 ≤ `_COL_GAP`) → 不跳栏、不跳隔开的另一条公式；
      · 必须有**字体证据**(`font_cands` 里有这一行)，扫描件不适用。

    纵向用种子带(固定)、横向用增长中的并集 —— 前者防上下漂移，后者允许公式被切成
    3 块以上时逐块接上去(`_ROW_EXTEND_PASS` 轮封顶)。
    """
    if not rows or not font_cands:
        return rows
    seeds = [_row_of(m, local_texts) for m in rows]
    sy0 = min(l["y"] for l in seeds)
    sy1 = max(l["y"] + l["h"] for l in seeds)
    picked = list(rows)
    taken = {m["idx"] for m in rows} | set(skip) | set(anchor)
    for _ in range(_ROW_EXTEND_PASS):
        x0 = min(_row_of(m, local_texts)["x"] for m in picked)
        x1 = max(_row_of(m, local_texts)["x"] + _row_of(m, local_texts)["w"]
                 for m in picked)
        grew = False
        for m in metas:
            if m["idx"] in taken:
                continue
            ln = _row_of(m, local_texts)
            cy = ln["y"] + ln["h"] / 2.0
            if not (y_lo - 2.0 <= cy <= y_hi + 2.0):
                continue                    # 不在公式的纵向区间里
            a, b = ln["y"], ln["y"] + ln["h"]
            if min(b, sy1) - max(a, sy0) < _ROW_OVERLAP * min(b - a, sy1 - sy0):
                continue                    # 不与种子行同一视觉行
            if max(ln["x"] - x1, x0 - (ln["x"] + ln["w"])) > _COL_GAP:
                continue                    # 与已选集不相接(隔栏 / 隔得远)
            spans = font_cands.get(m["idx"])
            if not spans:
                continue                    # 没有字体证据的行不补
            covered = sum(bb - aa for aa, bb in spans)
            if covered < _ROW_MATH_RATIO * max(1, len((m["raw"] or "").strip())):
                continue                    # 字形不算「以数学字体为主」
            picked.append(m)
            taken.add(m["idx"])
            grew = True
        if not grew:
            break
    return picked


def _math_rows_between(metas: list[dict], local_texts: list[dict], le: int,
                       rs: int, *, font_cands: Optional[dict] = None,
                       skip: Optional[set] = None,
                       window: str = "gap") -> list[dict]:
    """两个锚点之间「整行都是公式字形」的那几行(返回 metas 条目，按阅读顺序排)。

    **公式行的识别只有这一个实现** —— 独立公式(`_display_job*`)与跨块行内公式
    (`_inline_job`)共用它，判据只有一处、要调也只调这里。

    两件事和以前不同：
      * 定位一律按**纵向区间**，不看 `metas` 顺序。双栏论文里 PyMuPDF 的块顺序与
        阅读顺序常常不一致(实测页 2 的 `a^{(i)}_{i,s}`：左右锚点相邻，公式行却排到
        右锚点之后) —— 按顺序取"中间那些行"会一行都取不到，整条公式只能放弃。
      * 判据分两档：有 `font_cands` 就用**字体名**(准，且不受"像不像正文"干扰)，
        没有(扫描件 / 数学与正文同体)才退回 `_math_ish` 的启发式。

    `window`：
      * `"gap"`  —— y 取「左锚点行底 ~ 右锚点行顶」。独立公式整条夹在两行正文之间；
      * `"band"` —— y 取两个锚点行的**并集带**。行内公式的上下标会被 PyMuPDF 拆成
        独立行，它们与本行正文是同一个纵向带，按 `"gap"` 取会一行都取不到。

    x 窗口按「锚点行所在的**整行**(同栏)」取(见 `_row_span`)，**不是**锚点行自己的
    x/w —— 独立公式可以远超锚点句(锚点句只写到栏中间时，公式的后半截会被整条丢掉)。
    """
    skip = set() if skip is None else skip
    mb, _k1 = _meta_at(metas, le)
    ma, _k2 = _meta_at(metas, rs)
    if mb is None or ma is None:
        return []
    lb = local_texts[mb["bi"]]["lines"][mb["li"]]
    la = local_texts[ma["bi"]]["lines"][ma["li"]]
    if window == "band":
        y_lo = min(lb["y"], la["y"])
        y_hi = max(lb["y"] + lb["h"], la["y"] + la["h"])
    else:
        y_lo, y_hi = lb["y"] + lb["h"], la["y"]
        if y_hi < y_lo:
            return []
    x_lo, x_hi = _x_window(metas, local_texts, lb, la)
    anchor = (mb["idx"], ma["idx"])
    rows: list[dict] = []
    for m in metas:
        if m["idx"] in skip or m["idx"] in anchor:
            continue
        ln = local_texts[m["bi"]]["lines"][m["li"]]
        cy = ln["y"] + ln["h"] / 2.0
        if not (y_lo - 2.0 <= cy <= y_hi + 2.0):
            continue
        if ln["x"] + ln["w"] < x_lo or ln["x"] > x_hi:
            continue
        spans = (font_cands or {}).get(m["idx"])
        if spans:
            covered = sum(b - a for a, b in spans)
            if not _font_math_line(m["raw"], spans, covered):
                continue          # 混着正文的行不整行摘(半条公式留给行内路径)
        elif not (_math_ish(m["raw"])
                  and re.search(r"[=+^_∑∫√()\[\]/]", m["raw"])):
            continue
        rows.append(m)
    # 行纯度门槛会把「公式里混着正文衬线体字形」的那几个块挡在外面(独立公式常被切成
    # 多块) → 在种子行的同一视觉行里把横向相接的那些行补回来(见 `_extend_rows`)。
    rows = _extend_rows(metas, local_texts, rows,
                        y_lo=y_lo, y_hi=y_hi, skip=skip, anchor=set(anchor),
                        font_cands=font_cands)
    rows.sort(key=lambda m: (round(local_texts[m["bi"]]["lines"][m["li"]]["y"], 1),
                             round(local_texts[m["bi"]]["lines"][m["li"]]["x"], 1)))
    return rows


def _display_job(metas: list[dict], local_texts: list[dict], le: int, rs: int,
                 page, skip: set) -> Optional[dict]:
    """一对锚点之间是不是「隔着一条独立公式」；是就返回要摘成公式块的那几行。

    公式夹在「前一句的行底」与「后一句的行顶」之间，按**纵向区间**找它，不用文档
    顺序 —— 真实论文里块顺序很乱(一页近百块)，顺序法很不可靠。
    """
    mb, _k1 = _meta_at(metas, le)
    ma, _k2 = _meta_at(metas, rs)
    if mb is None or ma is None:
        return None
    lb = local_texts[mb["bi"]]["lines"][mb["li"]]
    la = local_texts[ma["bi"]]["lines"][ma["li"]]
    y_lo, y_hi = lb["y"] + lb["h"], la["y"]
    if not 1.0 < (y_hi - y_lo) <= 0.35 * page.rect.height:
        return None
    cand = _display_line_cands(metas, local_texts, y_lo, y_hi,
                              *_x_window(metas, local_texts, lb, la), skip)
    if not 1 <= len(cand) <= 6:
        return None
    lines = [local_texts[m["bi"]]["lines"][m["li"]] for m in cand]
    x0, y0, x1, y1 = _union_line_rect(lines)
    if x1 - x0 <= 0 or y1 - y0 <= 0:
        return None
    return {"cand": cand, "lines": lines, "rect": (x0, y0, x1, y1)}
    for L in _ANCHOR_LENS:
        ln, rn = left[-L:], right[:L]
        if len(ln) < 4 or len(rn) < 4:
            continue
        lhits = _all_pos(page_norm, ln)
        if not lhits:
            continue
        pairs = [(j + len(ln) - 1, k)
                 for j in lhits for k in _all_pos(page_norm, rn)
                 if j + len(ln) <= k and k - (j + len(ln)) <= _ANCHOR_SPAN_MAX]
        if len(pairs) == 1:
            return pairs[0]
    return None

def _display_job_font(metas: list[dict], local_texts: list[dict], le: int,
                      rs: int, page, skip: set,
                      font_cands: dict) -> Optional[dict]:
    """用**字体候选**找「一对锚点之间夹着的那条独立公式」。

    与 `_display_job` 唯一的区别是候选行的判据：不再猜「像不像正文」，而是看
    「这行的字形是不是数学字体」。这正是记忆里坑 #8 的解法 —— `close` 这类标识符
    会让 `_math_ish` 把公式判成正文，而字体判据不受影响。
    """
    if not font_cands:
        return None
    mb, _k1 = _meta_at(metas, le)
    ma, _k2 = _meta_at(metas, rs)
    if mb is None or ma is None:
        return None
    lb = local_texts[mb["bi"]]["lines"][mb["li"]]
    la = local_texts[ma["bi"]]["lines"][ma["li"]]
    if not 1.0 < (la["y"] - (lb["y"] + lb["h"])) <= 0.35 * page.rect.height:
        return None
    cand = _math_rows_between(metas, local_texts, le, rs,
                              font_cands=font_cands, skip=skip, window="gap")
    if not 1 <= len(cand) <= _SPAN_MAX_LINES:
        return None
    # 整行摘除是**高风险操作**(删错了就是丢正文)，所以只在「证据充分」时才认：
    #   · 候选 ≥ 2 行 —— 公式被切成上下标多行是常态；或
    #   · 组里出现 `CMSY`/`CMEX` 这类**符号专用**字体(大 ∑ / ∫ / 大括号 / 运算符)。
    # 表格里的文字也会被数学字体排(`Top-k`/`avgIC`/`h = 1` 实测是 `CMMI9`/`CMR9`)，
    # 但它们是「单行 + 无符号字体」—— 正好被这条挡在外面，交给锚点法去判。
    if len(cand) < 2 and not any(
            _has_symbol_font(local_texts[m["bi"]]["lines"][m["li"]])
            for m in cand):
        return None
    lines = [local_texts[m["bi"]]["lines"][m["li"]] for m in cand]
    x0, y0, x1, y1 = _union_line_rect(lines)
    if x1 - x0 <= 0 or y1 - y0 <= 0:
        return None
    return {"cand": cand, "lines": lines, "rect": (x0, y0, x1, y1),
            "from": "font"}
