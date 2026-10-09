"""文字层：字体族判定 + 行(lines)/行文本(run)抽取 + 行框并集 + 相邻 run 补间距。

见包文档 `backend/pdf_parser/__init__.py`。

⚠️ 本层输出的 run **一行只有一个**(见 `_merge_line_runs`)：块内只留
「块 `text` + 每行几何 + 每行文本」三层，行内不再有碎片。为此在两处做了补偿：

  * `_insert_gap_spaces` —— PyMuPDF 的相邻 span 之间**有肉眼可见的空隙、却没有
    空格字符**时(表格把几个单元格并进同一行最常见)补一个空格，否则合并后会粘成
    `Baseline0.85`。判据用 span 的 `bbox`/`origin`，且**公式字形两侧一律不补**：
    补进去的空格会把一处空格区切成两段，空位统计与「哪些页有公式」的判据都会偏
    (见 `math_font._math_font_flags` 与 `_group_math_pieces`)。
  * 合并取「本行字符数最多的那个 run」的样式(与前端 `bodyRunOf` 同口径)，于是
    行内上标/小字号的碎片不会再让整行字号忽大忽小("公式后正文变小"的老问题)。

补间距必须在 `math_font._blank_math_spans` **之前**跑：它改变字符数，而公式空格位、
块的 canonical 偏移、译文缓存键全都按字符数算 —— 必须同源。
"""

from __future__ import annotations

import re
import fitz  # PyMuPDF
from typing import Optional

from .math_font import _blank_math_level, _math_font_flags


# 相邻 span 的水平空隙超过 `0.25 × 字号` 就认为「这里本该有一个空格」(字距/连字
# 造成的空隙远小于它：实测同一词内的 span 间隙 ≤ 0.1×字号)。
_GAP_SPACE_MIN = 0.25
# 基线差超过 `0.5 × 字号` 就不是「同一行的连续文字」(上下标被拆成同 y 的两个 span)
_GAP_BASE_TOL = 0.5
# run 上只在本层内部使用的临时几何键：补间距(`_insert_gap_spaces`)要靠它，
# 擦除公式时还要靠它 + 行框宽度**反推空位原宽**(`math_font._line_gap_widths` → 行的
# `gaps`)；出页面前由 `_drop_gap_geom`(及兜底的 `backend_pymupdf._strip_font_keys`)摘掉。
_GAP_KEYS = ("x0", "x1", "oy")
# **行**上的临时几何键：逐字符的左边界 x（长度 = 行字符数 + 1，最后一个是最右边界）。
# 只有 `backend._local_page` 要（`sentences.cut_blocks_by_sentence` 按句切块时，
# 句子端点落在行内部就得把行拆成两段，宽度必须落到**字符边界**上），出页面前由
# `_merge_line_runs` / `_drop_gap_geom` 摘掉 —— 不进 doc.json。
_CHAR_X = "_cx"


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
# =====================================================================
#  相邻 run 补间距 + 每行合并成单个 run
# =====================================================================
def _is_math_run(run: dict, cm_on: bool, weak_on: bool) -> bool:
    """这个 run 会不会被当公式字形擦掉(判据见 `math_font._blank_math_level`)。"""
    return _blank_math_level(run.get("f") or "", cm_on, weak_on) > 0


def _span_text(sp: dict) -> str:
    """span 的文本：`dict` 抽法给 `text`，`rawdict` 抽法只给逐字符的 `chars`。

    两种抽法逐 span 逐字一致（实测 688/688 个 span 相同），所以用哪一种都不影响
    下游；`rawdict` 多给的 `chars` 里带**逐字符 bbox**，那是拆行时唯一的精确几何
    来源(见 `_CHAR_X`)。
    """
    t = sp.get("text")
    if t is not None:
        return t
    return "".join((c.get("c") or "") for c in sp.get("chars") or [])


def _span_char_x(sp: dict, text: str) -> Optional[tuple[list[float], float]]:
    """一个 span 的**逐字符左边界** + 末字符右边界；取不到就返回 None。

    优先用 `rawdict` 的 `chars[*].bbox`（精确）；字符数对不上/没有 chars（`dict`
    抽法）时退回按 span 框的**线性插值** —— 误差上限就是几个字符宽，而且只有
    「句子端点正好落在这个 span 里」时才会看出来。
    """
    n = len(text)
    if not n:
        return None
    chars = sp.get("chars") or []
    if len(chars) == n:
        lefts = [float((c.get("bbox") or (0.0,))[0]) for c in chars]
        box = chars[-1].get("bbox") or (0.0, 0.0, 0.0, 0.0)
        return lefts, float(box[2])
    bx = sp.get("bbox")
    if not bx:
        return None
    x0, x1 = float(bx[0]), float(bx[2])
    return [x0 + (x1 - x0) * i / n for i in range(n)], x1


def _space_run(src: dict) -> dict:
    """照 `src` 的样式造一个空格 run(补间距用)。`f` 留空 → 不会被当公式字形。"""
    return {"t": " ", "s": src.get("s", 10.0), "f": "",
            "fam": src.get("fam", "serif"), "b": False, "i": False,
            "up": False, "c": src.get("c", "#000000")}


def _needs_gap_space(prev: dict, nxt: dict, cm_on: bool, weak_on: bool) -> bool:
    """`prev` 与 `nxt` 之间要不要补一个空格(口径见 `_insert_gap_spaces`)。"""
    if any(k not in prev for k in _GAP_KEYS) or any(k not in nxt for k in _GAP_KEYS):
        return False                     # 补进去的空格 run 自己没有几何 → 不再补
    if (prev.get("t") or "")[-1:].isspace() or (nxt.get("t") or "")[:1].isspace():
        return False                     # 原文已经有空白，别补成两个
    if _is_math_run(prev, cm_on, weak_on) or _is_math_run(nxt, cm_on, weak_on):
        return False                     # 公式内部不补：会把一处空格区切成两段
    size = max(prev.get("s") or 10.0, nxt.get("s") or 10.0)
    if abs((prev.get("oy") or 0.0) - (nxt.get("oy") or 0.0)) > _GAP_BASE_TOL * size:
        return False                     # 基线不同 → 上下标/另一行，不是连续文字
    return (nxt["x0"] - prev["x1"]) > _GAP_SPACE_MIN * size


def _gap_space_x(cx: list[float], pos: int, prev: dict) -> float:
    """补出来的那个空格该落在哪（逐字符几何，见 `_CHAR_X`）。

    它就是那段空隙的左端 —— `prev` 的右边界(`x1`)，夹在前后两个字符的左边界之间。
    只有「拆行点正好落在这个空格上」时才会用到它（差半个空格宽，肉眼看不出来）。
    """
    left = cx[pos - 1] if 0 < pos <= len(cx) - 1 else None
    right = cx[pos] if 0 <= pos < len(cx) else None
    x = prev.get("x1")
    x = float(x) if x is not None else (
        (left + right) / 2.0 if left is not None and right is not None else
        left if left is not None else right)
    if x is None:
        return 0.0
    if left is not None:
        x = max(x, left)
    if right is not None:
        x = min(x, right)
    return x


def _insert_gap_spaces(blocks: list[dict], cm_on: bool, weak_on: bool) -> int:
    """相邻 span 之间「有空隙、没空格」的位置补一个空格；返回补了几处。

    为什么需要：PyMuPDF 是按 **span** 切字的，同一视觉行里的两截文字只要字体/字号/颜色
    不同就是两个 span，**中间的空隙不体现在任何字符上**。表格最常见：`Method` /
    `Accuracy` 并进同一行，两个 span 之间隔了 20pt，字符流里却是 `MethodAccuracy`。
    合并成单个 run 后这种粘连会直接显示出来，所以必须在合并**之前**把空隙变成空格。

    ⚠️ 改变字符数 ⇒ 必须在 `_blank_math_spans`(算空格位) **之前**调用：公式的
    `at{a,b}`、块的 canonical、译文缓存键都按字符数算，两处口径必须同源。

    补出来的空格没有几何，所以顺带在逐字符几何(`_CHAR_X`)里插一个位置(见
    `_gap_space_x`)—— 否则后面按句切块时，空格之后的字符会整体错开一格。
    """
    added = 0
    for blk in blocks or []:
        for ln in blk.get("lines") or []:
            runs = ln.get("runs") or []
            out: list[dict] = []
            ins: list[tuple[int, float]] = []      # (字符下标, 那个空格的左边界)
            pos = 0
            for r in runs:
                if out and _needs_gap_space(out[-1], r, cm_on, weak_on):
                    cx = ln.get(_CHAR_X)
                    if cx:
                        ins.append((pos, _gap_space_x(cx, pos, out[-1])))
                    out.append(_space_run(out[-1]))
                    added += 1
                    pos += 1
                out.append(r)
                pos += len(r.get("t") or "")
            ln["runs"] = out
            cx = ln.get(_CHAR_X)
            if ins and cx is not None:
                for p, x in reversed(ins):              # 从后往前插，前面的下标不挪
                    cx.insert(p, x)
    return added


def _drop_gap_geom(blocks: list[dict]) -> None:
    """摘掉临时几何键：run 上的 `_GAP_KEYS` 与行上的逐字符几何(`_CHAR_X`)。

    都是内部辅助，不进 doc.json（`_merge_line_runs` 会顺带调一次）。
    """
    for blk in blocks or []:
        for ln in blk.get("lines") or []:
            ln.pop(_CHAR_X, None)
            for r in ln.get("runs") or []:
                for k in _GAP_KEYS:
                    r.pop(k, None)


def _merge_line_runs(blocks: list[dict]) -> list[dict]:
    """每行的多个 run 合并成**一个**(行几何一个不动)；原地改并返回 `blocks`。

    口径有三条硬约束：

      * **字符数守恒** —— 只是 `"".join(...)`，不 strip、不补空格：下游(公式空格位、
        块 canonical、行内公式的 `at{a,b}`)全都按字符数寻址。
      * **样式取本行字符数最多的那个 run**(与前端 `bodyRunOf` 同口径)：公式里的
        上标/下标是独立的小字号 span，取「第一个 run」会让整行字号随排版顺序乱跳。
      * `up` 一律置 false —— 整行不该被当成上标(前端会 `font-size × 0.7` +
        `vertical-align:super`)。

    代价(已知并接受)：行内的粗体/斜体/上标与颜色被抹平成行级样式；真正的上标
    (`x²` 的 `2`)会与正文同号显示。换来的是一行一个文本节点，前端渲染与按行
    覆盖（公式框、行内公式）都退化成「一行一段文字」。

    ⚠️ 必须在 `_blank_math_spans` **之后**调用：公式识别需要逐字形的字体名(`f`)。
    行上的逐字符几何(`_CHAR_X`)也在这里摘掉 —— 它只服务于「按句切块」(`sentences`)。
    """
    for blk in blocks or []:
        for ln in blk.get("lines") or []:
            ln.pop(_CHAR_X, None)
            runs = ln.get("runs") or []
            if len(runs) > 1:
                main = max(runs, key=lambda r: len(r.get("t") or ""))
                ln["runs"] = [{
                    "t": "".join(r.get("t") or "" for r in runs),
                    "s": main.get("s", 10.0),
                    "fam": main.get("fam", "serif"),
                    "b": bool(main.get("b")),
                    "i": bool(main.get("i")),
                    "up": False,
                    "c": main.get("c", "#000000"),
                }]
    _drop_gap_geom(blocks)
    return blocks


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
        w[y] = (n + max(1, len(_span_text(sp).strip())),
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


def _row_top(spans: list, fallback: float) -> float:
    """本行「文字顶」：取**可见字符最多的那个 span 自己** bbox 的顶。

    ⚠️ PyMuPDF 给的行 bbox 是**该行所有 span 的并集**。行内公式(分式/求和/积分)
    的字形上下都伸出正文，于是含公式的行顶比正文文字顶高出约 7pt —— 实测本样本：

      「conjugates.」            并集顶 484.91  基线 492.94  并集 ascent  8.03
      「For sequences, the … ∑」 并集顶 490.61  基线 506.05  并集 ascent 15.44
                                 （该行正文 span 顶 = 498.02）

    顶到顶只差 5.70pt，正文却有 10pt 高 → 必然叠字；而 PDF 自身的基线间距是规整的
    (同一段内实测 11.95~13.11pt)，错的只是「拿哪个顶当顶」。前端是按 `y` 当**文字顶**
    绝对定位的(`.ln{position:absolute;line-height:1}` + 字号取 span 字号，见
    `reader2.js` 的 `buildTextItem`/`styleRun`)，所以这里必须给「正文顶」：
    改后含公式的行回到自己的行位，同一视觉行的左右片段也对齐到同一个 y。
    """
    best = None
    best_n = 0
    for sp in spans:
        n = len(_span_text(sp).strip())
        if n == 0 or n <= best_n:
            continue
        best, best_n = sp, n
    if best is None or "bbox" not in best:
        return fallback
    return best["bbox"][1]


def _parse_text_blocks(page, align_row_top: bool = False,
                       gap_spaces: bool = False,
                       char_x: bool = False) -> list[dict]:
    """抽一页文字块；`align_row_top` 时(只有 PyMuPDF 后端这么调)：

      行的 `y` 由「整行并集 bbox 的顶」改成「行内正文 span 的顶」(见 `_row_top`)。

    不这么做的话，行内公式(分式/上下标伸出)会把整行并集框的顶撑高，而前端是按 `y` 当
    **文字顶**绝对定位的 —— 含公式的行会与同一视觉行的其他片段错位、叠字。默认不开，
    旧 OCR 后端(Surya)的合成行因此逐字不变 —— 它们的行是 OCR 合成出来的、框高均匀，
    本来就没有「行内公式把行顶撑高」这个问题。

    `gap_spaces` 时(PyMuPDF 后端)额外把「有视觉空隙、却没有空格字符」的相邻 span
    之间补一个空格(见 `_insert_gap_spaces`)。因为是**页级**字体判据，它必须等整页
    span 都抽完才算得出来，所以是「先带临时几何收完，再回头补」，几何键随后摘掉；
    补完要重算块 `text`(字符数变了)。

    `char_x` 时(PyMuPDF 后端)用 `page.get_text("rawdict")` 抽，并给每行挂一个内部键
    `_CHAR_X`(逐字符左边界，长度 = 行字符数 + 1)。`sentences.cut_blocks_by_sentence`
    要按句把块重切、句子端点落在行内部时得把行**拆成两段** —— 拆点必须落在字符
    边界上，宽度得靠它 (`_span_char_x`)，拿不到就退回线性插值。

    ⚠️ `gap_spaces=True` 时**不在这里摘** `_GAP_KEYS`：紧接着的 `_blank_math_spans`
    要用它 + 行框宽度反推每个公式空位的原宽(`math_font._line_gap_widths`，结果随
    行级 `gaps` 下发前端)。调用方在擦除之后
    会走 `_merge_line_runs`，那里统一摘掉(`_CHAR_X` 也一起)；`_strip_font_keys` 再兜一道。
    """
    blocks: list[dict] = []
    raw = page.get_text("rawdict" if char_x else "dict")
    for blk in raw.get("blocks", []):
        if blk.get("type") != 0:
            continue  # 图片块在这里忽略，交给专门的图片管线
        lines = []
        for ln in blk.get("lines", []):
            sps = ln.get("spans", [])
            base = _line_baseline(sps)
            runs = []
            cx: list[float] = []                   # 逐字符左边界(见 `_CHAR_X`)
            cx_end = None                          # 末字符右边界
            cx_ok = bool(char_x)
            for sp in sps:
                t = _span_text(sp)
                if not t:
                    continue
                fl = sp.get("flags", 0)
                run = {
                    "t": t,
                    "s": round(sp.get("size", 10), 1),
                    # 原始字体名：只服务于「公式候选区」判定(见 `_font_math_index`)与
                    # 擦除(`_blank_math_spans`)；出页面前由 `_strip_font_keys` 剥掉 ——
                    # 前端不认识它，doc.json 也没必要为它变大。`fam` 仍是粗分类。
                    "f": sp.get("font", ""),
                    "fam": _family(sp.get("font", "")),
                    "b": bool(fl & 16),   # 粗体
                    "i": bool(fl & 2),    # 斜体
                    "up": _is_super(sp, fl, base),   # 上标（已校验基线，见函数注释）
                    "c": "#%06x" % (sp.get("color", 0) & 0xFFFFFF),
                }
                if gap_spaces:
                    # 补间距要看的几何，用完由 `_drop_gap_geom` 摘掉
                    bx = sp.get("bbox") or (0.0, 0.0, 0.0, 0.0)
                    org = sp.get("origin")
                    run.update({"x0": float(bx[0]), "x1": float(bx[2]),
                                "oy": float(org[1]) if org else float(bx[1])})
                if cx_ok:
                    got = _span_char_x(sp, t)
                    if got is None:
                        cx_ok = False              # 有一截拿不到几何 → 整行不给(会错位)
                    else:
                        cx.extend(got[0])
                        cx_end = got[1]
                runs.append(run)
            if not runs:
                continue
            lx0, ly0, lx1, ly1 = ln["bbox"]
            if align_row_top:
                # 行顶对齐正文 span(见 `_row_top`)。`h` 仍取到**并集底边**：块的上下边界
                # 不因此移动，只有「文字画在哪」变准。
                ly0 = _row_top(sps, ly0)
            line = {"x": lx0, "y": ly0, "w": lx1 - lx0, "h": ly1 - ly0, "runs": runs}
            if cx_ok and cx_end is not None and len(cx) == sum(len(r["t"]) for r in runs):
                line[_CHAR_X] = cx + [float(cx_end)]
            lines.append(line)
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
    if gap_spaces and blocks:
        # 页级字体判据要先看到整页 span 才算得准 → 收完再回头补间距（见函数文档）
        cm_on, weak_on = _math_font_flags(blocks)
        if _insert_gap_spaces(blocks, cm_on, weak_on):
            for blk in blocks:
                blk["text"] = _join_lines(blk["lines"])   # 字符数变了，块文本要重算
        # ⚠️ 这里**不摘** `_GAP_KEYS` —— `_blank_math_spans` 要用它记下空位原宽
        # （见函数文档）；摘除交给 `_merge_line_runs` / `_strip_font_keys`。
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
