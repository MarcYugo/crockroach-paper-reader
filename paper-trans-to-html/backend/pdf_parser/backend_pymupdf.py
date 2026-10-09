"""后端一 · PyMuPDF 本地抽取(兜底)。

见包文档 `backend/pdf_parser/__init__.py`。

公式：PyMuPDF 没有数学语义，用 `pymupdf_extract.py` 那套办法落地 —— 按**字体名**
判出公式字形，再换成**空格**(`_blank_math_spans`；空格个数由「行框宽度反推出来的
空位原宽」决定，见 `math_font._line_gap_widths`)，被擦空的行/块清掉。
见 `_blank_math_and_prune`。
"""

from __future__ import annotations

import fitz  # PyMuPDF
from pathlib import Path
from typing import Optional

from .math_font import (_SPACE_EM, _accumulate_stats, _blank_math_spans,
                        _group_math_pieces)
from .options import (DEFAULT_OPTIONS)
from .sentences import cut_blocks_by_sentence
from .text_layer import (_CHAR_X, _GAP_KEYS, _join_lines, _merge_line_runs,
                         _parse_text_blocks, _union_line_rect)


# =====================================================================
#  后端一 · PyMuPDF 本地抽取(兜底)
# =====================================================================
def _strip_font_keys(texts: list[dict]) -> None:
    """出页面前剥掉 runs 里的内部辅助键：原始字体名(`f`)与补间距/量空位宽的几何。

    `f` 只服务于公式字体的判据(`_font_math_index` 的候选区判定与 `_blank_math_spans`
    的擦除)；`_GAP_KEYS`(`x0`/`x1`/`oy`)只服务于「相邻 span 补空格」与「用行框宽度
    反推空位原宽」；行上的 `_CHAR_X`(逐字符几何)只服务于「按句子切块」——
    前端不认识它们，doc.json 也没必要为它们变大，于是解析完就丢掉。
    （正常路径上 `_merge_line_runs` 已经摘过，这里再兜一道，任何分支都不会漏。）
    """
    for blk in texts or []:
        for ln in blk.get("lines") or []:
            ln.pop(_CHAR_X, None)
            for r in ln.get("runs") or []:
                r.pop("f", None)
                for k in _GAP_KEYS:
                    r.pop(k, None)


def _blank_math_and_prune(texts: list[dict]) -> tuple[list[dict], list[dict], dict]:
    r"""公式字形 → 等长空格，并把被擦空的行/块清掉；返回 `(文字块, 空格位, 统计)`。

    `_blank_math_spans`(见 `math_font.py`)负责擦除本身，这里补上两件事：

      * **擦完之后的重建** ——
        - 被擦空的行/块不留下：整行是空格时它已无内容(`_join_lines` 会当成空)，留着
          只会变成一个不可见的空壳块(占着前端 hover/选中区、doc.json 白变大)。
        - 块框一律按**留下来的行**重算：行的 `y` 现在已经对齐正文 span(见
          `_parse_text_blocks` 的 `align_row_top`)，而 `blk` 的框还是 PyMuPDF 的并集框
          —— 不重算的话块顶会比第一行文字高出 3~7pt(第 2 页实测 40 个块里有 15 对
          矩形因此互相叠着，前端的 hover 框也会高出正文一截)。
        - `kind="formula"` 且带 `latex` 的块(**带 LaTeX 的公式块**)**跳过**：它按设计
          就只有几何(`runs` 恒为空)，按「行里有没有文字」清理会把它整块删掉。
        - **被删掉那几段的横向位置并回正文** —— 一条公式常被拆成好几条 line(记号字号
          不同)，只有与正文同基线的那条留下；其余几段整行是空格、跟着上面这一步走了，
          它们占的位置就成了两段文本之间的**空洞**(选不中、高亮不到)。调用
          `_absorb_orphan_gaps` 把这些无主续段并进左边邻居的**行尾空位**，
          统计写进 `formula_absorbed_lines`。
      * **空格位**（`slots`）—— 一条公式一条（同一行里挨得够近的几段空格会先并成
        一组，见 `math_font._group_math_pieces`）：

            {"at": {"id": 块 id, "a": 起, "b": 止} | None,   # 整组的外框区间
             "spaces": 空格数,
             "whole_line": 该行是否只剩这段空格}

        `at` 是那段空格在**清理后**结构里的字符区间；整行被「擦空即删」删掉时
        `at` 为 `None`（那条公式在正文里已经没有位置，但这一页照样要送检测 ——
        见 `backend_detection_service_group.py`）。

    `slots` 是**内部字段**：`entry.parse_pdf` 在返回前会把它从结果里摘掉
    (`pop_private`)，它不进 doc.json —— detection-service-group 后处理用它挑「哪些页
    有公式」（见 `backend_detection_service_group.py`）。

    ⚠️ 要求 runs 上带 `f`(字体名，擦除判据)；它是内部辅助键，出页面前由
    `_strip_font_keys` 剥掉。
    """
    segs, stats = _blank_math_spans(texts)
    # ① 先只定「哪些行被擦空要删」(判断，不建结构)：续段吸收要用它，而且它会改写
    # 行文本(只有一种改写：在行首/行尾补空格)，所以必须排在所有算偏移的步骤之前
    # (`_group_math_pieces` / `line_at` 都是按**改完之后**的文本算的)。
    keep_lines: dict[int, list[int]] = {}  # 原块下标 → 该块保留下来的行下标
    for bi, blk in enumerate(texts):
        if blk.get("kind") == "formula" and blk.get("latex"):
            continue                       # 公式块没有字形，不参与清理
        keep = [li for li, ln in enumerate(blk.get("lines") or [])
                if _join_lines([ln])]
        if keep:
            keep_lines[bi] = keep

    # ② 「被擦空即删」的行/块把它们占的横向位置一起带走了 —— 正文块的行框只画到公式的
    # 第一段为止，屏幕上于是留下一段既没有字符、也进不了任何句子区间的**空洞**
    # (两段正文中间那段点不着、也高亮不到的空白)。把它并进相邻正文的空位里。
    changed, stats["formula_absorbed_lines"] = _absorb_orphan_gaps(texts, keep_lines)
    for bi, li, pos, n in changed:          # 补了空格 → 该行原有空位的偏移跟着挪
        for s in segs:
            if s["bi"] == bi and s["li"] == li:
                if s["a"] >= pos:
                    s["a"] += n
                if s["b"] > pos:
                    s["b"] += n

    groups = _group_math_pieces(texts, segs)      # 同一行里挨得近的碎片并成一条公式

    kept: list[dict] = []
    # (原块下标, 原行下标) → (块 id, 该行在**清理后**结构里的块内 canonical 起址)
    line_at: dict[tuple[int, int], tuple[str, int]] = {}
    for bi, blk in enumerate(texts):
        if blk.get("kind") == "formula" and blk.get("latex"):
            kept.append(blk)               # 公式块没有字形，不参与清理
            continue
        src = blk.get("lines") or []
        keep_idx = keep_lines.get(bi) or []
        if not keep_idx:
            continue
        g = 0
        for li in keep_idx:
            line_at[(bi, li)] = (blk.get("id") or "", g)
            # 行文本**原样**(不 strip)拼接、行间一个 '\n' —— 与前端
            # `buildTextItem` 里 `item.canonical` 的口径一致
            g += sum(len(r.get("t") or "") for r in (src[li].get("runs") or [])) + 1
        lines = [src[li] for li in keep_idx]
        x0, y0, x1, y1 = _union_line_rect(lines)
        blk = dict(blk)
        blk.update({"lines": lines, "x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0,
                    "text": _join_lines(lines)})   # 字形变了，块文本要跟着重算
        kept.append(blk)

    # 「被擦空即删」的行/块把它们占的横向位置一起带走了 —— 正文块的行框只画到公式的
    # 第一段为止，屏幕上于是留下一段既没有字符、也进不了任何句子区间的**空洞**
    # (两个块中间那段点不着、也高亮不到的空白)。并进相邻正文的空位见上面第②步。
    slots: list[dict] = []
    for g in groups:
        line = (texts[g["bi"]].get("lines") or [])[g["li"]]
        # 该行还剩不剩别的文字（擦除后仍留在行里的部分）
        whole = not "".join(r.get("t") or ""
                            for r in line.get("runs") or []).strip()
        at = line_at.get((g["bi"], g["li"]))
        slots.append({
            "at": ({"id": at[0], "a": at[1] + g["a"], "b": at[1] + g["b"]}
                     if at else None),
            "spaces": g["n"],
            "whole_line": whole,
        })
    return kept, slots, stats


# 「无主续段」与正文的接边容差(pt)，三个：
#   * `_CHAIN_TOL` —— 续段与续段之间串链时的横向接边容差：同一条公式的相邻两段之间可能
#     就隔一个 thin space（实测 `lim_{h→0}` 与 `(f(x+h)−f(x))/h` 差 3.36pt），而不同公式之间
#     隔着正文（≥ 8pt），所以 4pt 能把一条公式认全、又不会把相邻两条公式粘起来。
#   * `_GAP_JOIN_TOL` —— 纵向接边容差：PyMuPDF 的行框按字形边缘算，前后两段横向实测是
#     0.01pt；纵向则常见 1pt 上下的错位（字号不同的行，顶对齐之后行框并不重叠，实测
#     `√∑ⁿ` 与 `i=1 v²` 差 1.03pt）。
#   * `_HOST_TOL` —— 续段离宿主行框边缘多远以内还算「紧跟在它旁边」。实测 0~4pt；
#     而跨栏/跨视觉行的巧合最近也有 40pt 以上，所以 8pt 足够分辨。
_CHAIN_TOL = 4.0
_GAP_JOIN_TOL = 2.0
_HOST_TOL = 8.0


def _line_rect(ln: dict) -> list[float]:
    """行的矩形 `[x0, x1, y0, y1]`(PDF 坐标 pt)。"""
    x = float(ln.get("x") or 0.0)
    y = float(ln.get("y") or 0.0)
    return [x, x + float(ln.get("w") or 0.0), y, y + float(ln.get("h") or 0.0)]


def _band_overlap(a: list[float], b: list[float]) -> float:
    """两行的**纵向**交集高度；为负表示分开多少 pt(见 `_GAP_JOIN_TOL` 的容差)。"""
    return min(a[3], b[3]) - max(a[2], b[2])


def _joinable(a: list[float], b: list[float]) -> bool:
    """两段算不算同一视觉行上**紧跟着**的内容：横向接着(或叠着) + 纵向接着(或叠着)。"""
    return (b[0] <= a[1] + _CHAIN_TOL and a[0] <= b[1] + _CHAIN_TOL
            and _band_overlap(a, b) > -_GAP_JOIN_TOL)


def _tail_gap(ln: dict) -> Optional[list]:
    """行的**行尾空位**（`gaps` 里正好挨着行尾的那一条）；没有就返回 None。

    行尾空位是「公式擦掉之后留在行末的那段空格」，也是这里唯一能加宽的落点 ——
    空位没宽度(旧数据)或公式不在行末时返回 None，调用方就什么都不做。
    """
    n = sum(len(r.get("t") or "") for r in ln.get("runs") or [])
    gaps = ln.get("gaps") or []
    if not n or not gaps:
        return None
    tail = gaps[-1]
    if len(tail) < 3 or tail[0] + tail[1] != n or not tail[2]:
        return None
    return tail


def _head_gap(ln: dict) -> Optional[list]:
    """行的**行首空位**（`gaps` 里从第 0 个字符开始的那一条）；没有就返回 None。

    公式把一行**顶头**排时（双栏页里很常见），左边没有行尾空位可以加宽，只能把这段
    空白并进右边那条行的行首空位（整条行左移 + 空位加宽，见 `_absorb_orphan_gaps`）。
    """
    gaps = ln.get("gaps") or []
    if not gaps:
        return None
    head = gaps[0]
    if len(head) < 3 or head[0] != 0 or not head[2]:
        return None
    return head


def _space_count(runs: list[dict], w: float) -> int:
    """把宽度 `w`(pt) 折成**几个空格**（口径与 `_blank_math_spans` 一致：0.25em 一个字宽）。"""
    size = max((float(r.get("s") or 10.0) for r in runs), default=10.0)
    return max(1, int(round(w / (_SPACE_EM * size))))


def _absorb_orphan_gaps(texts: list[dict], keep_lines: dict[int, list[int]]
                        ) -> tuple[list[tuple[int, int, int, int]], int]:
    r"""把「被擦空而删掉的行/块」并进相邻正文的**空位**；返回 `(补空格的位置, 并掉的行数)`。

    第一个返回值 `[(块下标, 行下标, 插入位置, 空格数), …]` —— 只有宿主那一侧**本来没有
    空位**时才需要新建（那得往文本里补空格），调用方要拿它把该行已有空位、以及 `segs`
    (公式空位用的字符区间)的偏移一起补正（见 `_blank_math_and_prune` 的第②步）。

    症状（2026-09 修「两段公式之间的空白选不住」）：屏幕上会出现一段**空洞** ——
    `the formula ▊` 与 `, where the discriminant` 之间空着一截，既没有字符、也进不了
    任何句子区间：点不着、拖不到、高亮也不覆盖（看着像排版坏了）。双栏页里更明显：
    `H(X) = −Σᵢ …` 顶头起一行时，左右两段正文之间白着 100px。

    根因：一条行内公式常被 PyMuPDF 拆成**好几条 line** —— 记号的字号/基线不一样
    （实测 `x = −b ± √(b²−4ac)/2a`：`√` raise 成一条、被开方数一条、分母一条）。
    只有与正文同基线的那条留在正文块里，其余几条整行都是空格 → 跟着「擦空即删」走了
    （见 `_blank_math_and_prune` 的 `keep_idx`）。正文块的行框仍然只画到它自己那条
    line 的右边缘（`ln.w` 是 PyMuPDF 给的行框），被删掉那几段**本来占的横向位置**就凭空
    少了 —— 实测 p1b45 行尾空位 40.496pt，而右侧正文起点还差 32pt(48px)才接上。

    改法：把这些「无主」的行按几何**接边**串成一条条续段（`_joinable`），再按续段在
    视觉行上的位置挑一种落点，把那段宽度补到相邻正文的**空位**上：

      * **续段两边都是正文**（行内公式）→ 补进**左邻行的行尾空位**，行框 `ln.w` 同样加；
      * **续段顶头起一行**（左边没有正文）→ 补进**右邻行的行首空位**：那条行整体左移
        extra、行首空位加同样多（文本右端仍在原位）；
      * 宿主那一侧**没有现成空位**时，就把空位**新建**出来 —— 往行尾/行首补几个空格，
        并按 `[起, 空格数, 原宽]` 记进行的 `gaps`，前端照旧把它当一个空位 span 渲染。
        实测：`represents the in-` 这一行行首没有空位，就这样把它左边那半条公式补回。

      * **空位是空格 span**：它的宽度(前端按 `data-gw` 撑开)就是底色/选中高亮/前景的
        范围 —— 补上它，那段空白就整块变成可点、可选中、可高亮的一整段。
      * **行框必须与空位同步改**：`ln.w`/`ln.x` 是前端 `fitLineEl` 的 `boxW`(行框宽)。
        只加空位不改行框，多出来的宽会被当成「排不下」而触发整行 `scaleX` 压缩。

    ⚠️ 补空格是**唯一**会改文本的地方，而且只补在**行首/行尾**（不切进句子里）：
    句子个数与顺序不变（笔记/高亮都按句子下标存，见前端 `setAnnoRanges`）、
    `sentenceText`/`selFragText` 本来就 trim + 压缩空白，所以复制/翻译的文本不受影响；
    代价只是那一块的译文缓存键变了(要重译一次)。**因此本函数必须排在任何算偏移的步骤
    之前**（`_group_math_pieces` / `line_at`）。

    四步判据（全都是**几何**的，不猜内容）：

      1. **串链**：横向接着/叠着 + 纵向接着/叠着的续段算同一条公式的几段，先并成一条链
         （`√` 之后跟着被开方数、上下标各自成行）。链的两端就是这段空白要补到哪。
      2. **找宿主**：宿主必须是「行框边缘**紧贴**在链那一侧(`_HOST_TOL` 以内)、且与本链
         纵向相交」的那条保留行（左邻取最靠右的、右邻取最靠左的）。
         ⚠️ 只要求「紧贴」这一条，是因为**跨栏**（双栏页里左栏行尾 x 与右栏某段 x
         相距 40pt 以上）与**跨视觉行**的巧合都靠它挡掉；实测里这类假宿主正是两栏之间
         的公式碎片，吸进去会把左栏的行尾空位一路画到右栏去。
      3. **封顶**：同一视觉行上挡在这段空白那一侧的**最近的**行框给这段空白封顶 ——
         空位只补到它的边缘为止（实测 `between vectors, while the norm ▊ gives the
         Euclidean length.` 就补到 `gives` 那行之前），绝不盖到别的正文上。
         那一侧没有正文的（独立成行的公式、整行都是公式的行）不动：那不是「两段正文之间
         的空白」，补宽只会得到一个横跨半页的高亮块。
      4. **一条链只补一处**：先认左邻的行尾空位，成了就不再往右邻补 —— 否则同一段空白
         会被两条行各画一遍(底色叠成深一块)。

    ⚠️ `_band_overlap` 一律用**宿主自己那条视觉行**去比，不用链的纵向包络：一条公式的
    几段可能上下拉开十几 pt（分子/分母），用包络会把隔壁行的正文也算成"同一行"。
    """
    hosts: list[tuple[list[float], dict, int, int]] = []   # (矩形, 行, 块下标, 行下标)
    orphans: list[list[float]] = []
    for bi, blk in enumerate(texts):
        if blk.get("kind") == "formula" and blk.get("latex"):
            continue                           # 公式块没有字形，也没有行
        keep = set(keep_lines.get(bi) or ())
        for li, ln in enumerate(blk.get("lines") or []):
            if li in keep:
                hosts.append((_line_rect(ln), ln, bi, li))
            else:
                orphans.append(_line_rect(ln))   # 无主续段：整行被擦空 → 要删的行
    if not hosts or not orphans:
        return [], 0

    # ---- ① 无主续段串成链：横向相接/相叠 + 纵向接着/叠着 ----
    chains: list[list[float]] = []
    for r in sorted(orphans, key=lambda t: t[0]):
        for c in chains:
            if _joinable(c, r):
                c[0], c[1] = min(c[0], r[0]), max(c[1], r[1])
                c[2], c[3] = min(c[2], r[2]), max(c[3], r[3])
                break
        else:
            chains.append(list(r))
    merged = True                              # 链与链之间也可能接上 → 并到不再变
    while merged:
        merged, out = False, []
        for c in chains:
            for d in out:
                if _joinable(c, d):
                    d[0], d[1] = min(d[0], c[0]), max(d[1], c[1])
                    d[2], d[3] = min(d[2], c[2]), max(d[3], c[3])
                    merged = True
                    break
            else:
                out.append(c)
        chains = out

    # ---- ② 落点一：链左边有正文 → 补进那条行的**行尾空位** ----
    grow: dict[int, float] = {}                # id(行对象) → 行尾要补的宽度(pt)
    grow_head: dict[int, float] = {}           # id(行对象) → 行首要补的宽度(pt)
    where: dict[int, tuple[int, int]] = {}     # id(行对象) → (块下标, 行下标)
    absorbed: set[int] = set()                 # 已补过的链下标（一条链只补一处）
    for ci, c in enumerate(chains):
        best: Optional[tuple[float, float, dict, int, int]] = None
        for r, ln, bi, li in hosts:
            if r[1] > c[0] + _GAP_JOIN_TOL or c[0] - r[1] > _HOST_TOL:
                continue                       # 宿主必须**紧接**在链的左边
            ov = _band_overlap(r, c)
            if ov <= 0:
                continue                       # 不同视觉行
            if best is None or (ov, r[1]) > (best[0], best[1]):
                best = (ov, r[1], ln, bi, li)
        if best is None:
            continue                           # 整行都是公式(独立成行的公式) → 不动
        _ov, hx1, ln, bi, li = best
        # ---- 封顶：同一视觉行上、排在宿主**正文之后**的那条**最近的**行框 ----
        # 行尾空位最多补到它的起点为止。比的是宿主的**正文**边界（行框右边缘减去行尾
        # 空位宽）而不是行框边界 —— 行框互相压着的那种行（实测 p1b48/p1b49 叠着 4.9pt）
        # 起点落在宿主行框之内，它不是"右边的正文"，却也不能被盖住。
        hr = _line_rect(ln)
        tail = _tail_gap(ln)
        content_end = hx1 - (float(tail[2]) if tail else 0.0)
        cap: Optional[float] = None
        for r2, l2, _b2, _l2 in hosts:
            if l2 is ln or _band_overlap(r2, hr) <= 0:
                continue                       # 宿主自己 / 不在同一视觉行
            if r2[0] <= content_end + _GAP_JOIN_TOL:
                continue                       # 在宿主正文之前 → 不算右边的正文
            cap = r2[0] if cap is None else min(cap, r2[0])
        if cap is None or c[0] >= cap - _GAP_JOIN_TOL:
            continue                           # 右边没有正文 → 不是「两段正文之间的空白」
        extra = min(c[1], cap) - hx1
        if extra > 0.5:
            key = id(ln)
            grow[key] = max(grow.get(key, 0.0), extra)
            where[key] = (bi, li)
            absorbed.add(ci)

    # ---- ③ 落点二：链在**行首**（左边没有正文）→ 补进右邻行的**行首空位** ----
    # 双栏页里很常见：公式起头的一行（`H(X) = −Σᵢ p(xᵢ) log p(xᵢ) quanti-`）左边
    # **根本没有行尾空位可补** —— 能补的地方是它右边那条行的行首空位：把那条行
    # 整体**向左挪** extra、同时把行首空位加宽 extra（文本右端仍在原位）。
    for ci, c in enumerate(chains):
        if ci in absorbed:
            continue                           # 这条链已经在②里补进左邻的行尾空位了
        best2: Optional[tuple[float, dict, int, int]] = None
        for r, ln, bi, li in hosts:
            if r[0] < c[1] - _GAP_JOIN_TOL or r[0] - c[1] > _HOST_TOL:
                continue                       # 宿主必须**紧接**在链的右边
            if _band_overlap(r, c) <= 0:
                continue                       # 不同视觉行
            if best2 is None or r[0] < best2[0]:
                best2 = (r[0], ln, bi, li)
        if best2 is None:
            continue
        _hx0, ln, bi, li = best2
        # 左侧封顶：同一视觉行上、排在宿主**正文之前**的那条**最近的**行框的右边缘
        # （左栏正文的右边缘 —— 空白只补到它右边为止，不会画到栏间空档上）。
        hr = _line_rect(ln)
        head = _head_gap(ln)
        content_start = hr[0] + (float(head[2]) if head else 0.0)
        left: Optional[float] = None
        for r2, l2, _b2, _l2 in hosts:
            if l2 is ln or _band_overlap(r2, hr) <= 0:
                continue                       # 宿主自己 / 不在同一视觉行
            if r2[1] >= content_start - _GAP_JOIN_TOL:
                continue                       # 在宿主正文之后 → 不算左边的正文
            left = r2[1] if left is None else max(left, r2[1])
        start = c[0] if left is None else max(c[0], left)
        extra = hr[0] - start
        if extra > 0.5:
            key = id(ln)
            grow_head[key] = max(grow_head.get(key, 0.0), extra)
            where[key] = (bi, li)

    # ---- ④ 落到几何上（那一侧没有现成空位时，补几个空格把空位建出来） ----
    by_key = {id(ln): ln for _r, ln, _bi, _li in hosts}
    changed: list[tuple[int, int, int, int]] = []      # 补过空格的行
    done = 0
    for key, extra in grow.items():
        ln = by_key[key]
        bi, li = where[key]
        runs = ln.get("runs") or []
        if not runs:
            continue
        done += 1
        tail = _tail_gap(ln)
        if tail is not None:
            tail[2] = round(float(tail[2]) + extra, 3)
        else:
            pos = sum(len(r.get("t") or "") for r in runs)     # 行尾
            n = _space_count(runs, extra)
            runs[-1]["t"] = (runs[-1].get("t") or "") + " " * n
            ln.setdefault("gaps", []).append([pos, n, round(extra, 3)])
            changed.append((bi, li, pos, n))
        ln["w"] = round(float(ln.get("w") or 0.0) + extra, 3)
    for key, extra in grow_head.items():
        ln = by_key[key]
        bi, li = where[key]
        runs = ln.get("runs") or []
        if not runs:
            continue
        done += 1
        head = _head_gap(ln)
        if head is not None:
            head[2] = round(float(head[2]) + extra, 3)
        else:
            n = _space_count(runs, extra)
            runs[0]["t"] = " " * n + (runs[0].get("t") or "")
            gaps = ln.setdefault("gaps", [])
            for g in gaps:                     # 行首插了空格 → 原有空位偏移整体后移
                g[0] += n
            gaps.insert(0, [0, n, round(extra, 3)])
            changed.append((bi, li, 0, n))
        ln["x"] = round(float(ln.get("x") or 0.0) - extra, 3)
        ln["w"] = round(float(ln.get("w") or 0.0) + extra, 3)
    return changed, done


def _local_page(page, doc, images_dir, pno: int, idx0: int = 0,
                opts: Optional[dict] = None,
                mstats: Optional[dict] = None) -> tuple[dict, int]:
    """纯 PyMuPDF 抽一页：文字块和版式。图片由 detection-service-group 提供。

    `mstats` 非空时把本页「公式→空格」的统计累加进去(供 parser_info 汇报)。
    """
    opts = opts or DEFAULT_OPTIONS
    # align_row_top：行顶对齐正文 span 顶(行内公式会把「整行并集框」的顶撑高，
    # 前端按 y 当文字顶定位 → 不对齐就会叠字，见 `text_layer._row_top`)
    # gap_spaces：相邻 span 有视觉空隙却没有空格字符时补一个空格(见 text_layer)
    # char_x：带出逐字符几何，拆行(拆点落到字符边界)靠它，见 `sentences` 模块
    texts = _parse_text_blocks(page, align_row_top=True, gap_spaces=True, char_x=True)
    # Figure 插图由 detection-service-group 检测并裁剪；本地只保留文字/几何版式。
    images = []
    # ★ 每个块 = 恰好一句(见 `sentences.cut_blocks_by_sentence`)。
    # ⚠️ 必须排在下面那一步**之前**：擦除/空位都按当时的块/行结构算字符偏移，
    # 而重切只改「分组」不改「字符」 —— 排在前面就不需要任何偏移重映射。
    texts, sstat = cut_blocks_by_sentence(texts)
    # 公式字形 → 等长空格(与 `pymupdf_extract.py --drop-formula` 同一套做法)。
    texts, slots, stats = _blank_math_and_prune(texts)
    _accumulate_stats(mstats, stats)
    _accumulate_stats(mstats, sstat)
    texts = _merge_line_runs(texts)      # 一律一行一个 run(见 text_layer 模块文档)
    _strip_font_keys(texts)
    page_out = {
        "w": page.rect.width,
        "h": page.rect.height,
        "texts": texts,
        "images": images,
        "text": "\n\n".join(t["text"] for t in texts if t.get("text")),
    }
    if slots:
        # 公式空位的**内部**字段：detection-service-group 后处理用它挑「哪些页有公式」
        # (`backend_detection_service_group.py`)；出 `parse_pdf` 前由 `entry.pop_private`
        # 摘掉 —— 不进 doc.json。
        page_out["_formula_slots"] = slots
    return page_out, idx0 + len(images)


def _pdf_outline(doc) -> list[dict]:
    """PDF 自带书签(目录) → `[{level, title, page}]`；没有书签就是空列表。

    直接拿来当阅读器侧栏目录：比让 LLM 猜更准、也不用花钱。`page` 从 1 开始。
    """
    try:
        raw = doc.get_toc(simple=True) or []
    except Exception:
        return []
    out: list[dict] = []
    for row in raw:
        try:
            level, title, page = int(row[0]), str(row[1] or "").strip(), int(row[2])
        except Exception:
            continue
        if not title:
            continue
        out.append({"level": max(1, min(level, 4)),
                    "title": title[:200],
                    "page": max(1, page)})
    return out


def _parse_with_pymupdf(pdf_path: Path, images_dir: Optional[Path], opts: dict) -> dict:
    """本地 PyMuPDF 解析整份 PDF(不需要任何外部服务)。

    公式不进 `latex`(本地判不出 LaTeX)，而是按字体名从字形里擦除成等长空格 ——
    屏幕上那里先显示为一段空白，随后由 detection-service-group 后处理
    (`backend_detection_service_group.py`)把「公式框 + LaTeX」记进
    `pages[].formula_boxes`，前端据此在页面上覆盖 KaTeX。统计在 `parser_info`。
    """
    doc = fitz.open(str(pdf_path))
    try:
        if images_dir is not None:
            images_dir.mkdir(parents=True, exist_ok=True)

        pages_out = []
        ghost_total = 0
        mstats: dict = {}          # 公式→空格 的整份统计(见 `_blank_math_spans`)
        for pno in range(doc.page_count):
            page = doc.load_page(pno)
            # 归一化旋转，统一坐标系(文字/点阵图/矢量区域一致)
            if page.rotation % 360 != 0:
                page.set_rotation(0)
            page_out, _idx = _local_page(page, doc, images_dir, pno, opts=opts,
                                         mstats=mstats)
            ghost_total += page_out.get("ghost_text_removed", 0)
            pages_out.append(page_out)

        meta_title = (doc.metadata or {}).get("title") or ""
        return {
            "num_pages": len(pages_out),
            "page_w": pages_out[0]["w"] if pages_out else 0,
            "page_h": pages_out[0]["h"] if pages_out else 0,
            "pdf_title": meta_title,
            "pages": pages_out,
            "toc": _pdf_outline(doc),          # PDF 自带书签(没有就是空列表)
            "parser": "pymupdf",
            "formula_action": "space",         # 公式字形已换成等长空格(与 pymupdf_extract.py 一致)
            "parser_info": {
                "ghost_text_removed": ghost_total,   # 为去重影剔除的“图内文字”行数
                **mstats,                            # formula_spans / chars / lines …
            },
        }
    finally:
        doc.close()
