"""公式注入：唯一的「公式块」构造出口 + 摘行实现 + `_inject_ocr_math` 总调度。

见包文档 `backend/pdf_parser/__init__.py`。
"""

from __future__ import annotations

import re
import fitz  # PyMuPDF
from pathlib import Path
from typing import Optional

from .images import (_carve_images)
from .math_font import (_font_math_index)
from .math_inline import (_ocr_line_replacements)
from .math_locate import (_display_job, _display_job_font, _find_anchor_pair,
                          _job_rank)
from .math_text import (_STYLE_KEYS, _anchor_of, _apply_replacements,
                        _body_run, _norm_keep, _ocr_math_lines, _page_index,
                        _same_style)
from .text_layer import (_join_lines, _union_line_rect)


# ---------------------------------------------------------------------
#  用 OCR 的 LaTeX 补公式(行内 + 独立)，**本地版式完全不动**
#
#  有可见文字层的电子版 PDF，几何信息(PyMuPDF 精确行框/字体)比 OCR 可靠，但**公式**
#  在 PDF 里是一堆碎片化字形(常乱序、上下标被拆成同一 y 的多行)，抽出来就是乱码。
#  PaddleOCR-VL 输出的整页文本里带正确的 LaTeX 定界符，正好补这一块：
#
#    * 行内公式：拿公式**前后的普通文字**当锚点，在本地行里定位，把两个锚点之间那段
#      字形换成 `$latex$` —— 前端 `renderInlineMath()` 会就地换成 KaTeX，行框不动；
#    * 独立公式：拿公式**前一句的结尾**和**后一句的开头**当锚点，中间那几行字形就是
#      这条公式，摘出来合成一个 `kind="formula"` 块(坐标取这几行的并集) —— 前端按
#      display 模式覆盖渲染，位置就是原来那几行的位置。
#
#  宁可不变，不可切错：任何一处锚点对不上、或有歧义(多行都能匹配)，这一条就**整条放弃**，
#  保持原字形不动。可用 `paddle_ocr_math=false` 整体关掉。
# ---------------------------------------------------------------------

def _slice_runs(runs: list[dict], start: int, end: int) -> list[dict]:
    """按**原始字符下标**切 runs，保留 `[start, end)`；样式(粗体/斜体/字号/颜色)跟着走。"""
    chars: list[str] = []
    styles: list[dict] = []
    for r in runs:
        t = r.get("t") or ""
        chars.extend(t)
        styles.extend([r] * len(t))
    out: list[dict] = []
    for i in range(max(0, start), min(len(chars), end)):
        r, ch = styles[i], chars[i]
        if out and _same_style(out[-1], r):
            out[-1]["t"] += ch
        else:
            out.append({"t": ch, **{k: r.get(k) for k in _STYLE_KEYS}})
    return out


def _prefix_width(line: dict, n_chars: int) -> float:
    """估算行内**前 n 个字符**占多宽(pt)：按 runs 的「字数×字号」比例，再用行宽标定。

    只用于把被切掉脑袋的那一行往右挪回来 / 给公式块起点定 x，不需要像素级精确。
    """
    runs = line.get("runs") or []
    w = float(line.get("w") or 0.0)
    total = sum(len(r.get("t") or "") * (r.get("s") or 10.0) for r in runs)
    if total <= 0 or w <= 0 or n_chars <= 0:
        return 0.0
    scale, acc, n = w / total, 0.0, 0
    for r in runs:
        take = max(0, min(len(r.get("t") or ""), n_chars - n))
        acc += take * (r.get("s") or 10.0)
        n += take
        if n >= n_chars:
            break
    return max(0.0, min(w, acc * scale))


def _retighten_block(blk: dict) -> None:
    """行被增删/裁切之后重算块的包围盒；行空了就标记丢弃(调用方统一清理)。"""
    lines = blk.get("lines") or []
    if not lines:
        blk["_drop"] = True
        return
    x0, y0, x1, y1 = _union_line_rect(lines)
    blk.update(x=x0, y=y0, w=x1 - x0, h=y1 - y0)


# ---------------------------------------------------------------------
#  公式块：唯一的构造出口 + 唯一的「摘行」实现
#
#  公式有三种形态(一行内 / 跨块行内 / 独立成段)，历史上各写了一套落地代码：
#    · 行内同块 → `_apply_replacements` 把字形**删掉**换成 `\(latex\)`(干净)；
#    · 跨块行内 → 切两侧 + 摘中间行，但把**原字形复制进新公式块**的 `lines`；
#    · 独立公式 → 同上。
#  后两种只靠前端 `.blk.has-math .ln{visibility:hidden}` 遮住字形：KaTeX 没渲染成
#  就整片露出乱码，字形还顺着 `block.text` / `page.text` 污染句子切分、翻译与复制。
#  现在一律走「字形从数据里删掉、公式块只留几何 + latex」，三种形态行为一致。
# ---------------------------------------------------------------------
# 公式编号行(如 "(1)")：贴栏右排、与公式主体隔一大段空白 —— 算「主体宽度」时要排除，
# 否则块框被编号撑宽、宽度适配永远轮不到缩小。前端 `r2-math.js::_TAG_LINE_RE` 有同一条
# 规则；但公式块现在不带行文本，所以判定在这里做、随块下发(`lines[*].tag`)。
# ⚠️ 两处规则必须保持一致。
_TAG_LINE_RE = re.compile(r"^[（(]\s*\d+\s*[a-z]?\s*[)）]$")


def _line_geom(ln: dict) -> dict:
    """公式块里的一行：**只留几何 + 行字号(+编号标记)，不带任何字形文本**。

    公式块里的字形是 PyMuPDF 对数学排版的碎片化产物：上下标被拆成独立行，CMSY/CMEX
    的符号还会因为字体没有 `/ToUnicode` 被读成 `n`/`o`/`′` 这类错字。它们既没有阅读
    价值，又会顺着 `block.text` / `page.text` 污染句子切分、翻译与复制。公式的**内容**
    一律由 `latex` 提供，字形只剩「位置 + 字号」一个用途(前端据此摆覆盖层、按原文主体
    宽度缩放)。
    """
    runs = ln.get("runs") or []
    txt = "".join(r.get("t") or "" for r in runs).strip()
    sizes = [float(r.get("s") or 0.0) for r in runs if (r.get("t") or "").strip()]
    return {"x": round(float(ln.get("x") or 0.0), 3),
            "y": round(float(ln.get("y") or 0.0), 3),
            "w": round(float(ln.get("w") or 0.0), 3),
            "h": round(float(ln.get("h") or 0.0), 3),
            "s": round(max(sizes) if sizes else 0.0, 2),
            "tag": bool(_TAG_LINE_RE.match(txt)),
            "runs": []}


def _detach_lines(local_texts: list[dict], lines: list[dict]) -> None:
    """把这几行从各自所属的块里摘掉，并重算块框(块被摘空 → 标记 `_drop`)。

    **公式落地时"删字形"的唯一实现**。原来这段在两处各写了一遍(独立公式一处、跨块
    行内一处)，行为还不一样(一处重算了 `text`、一处没有)，是"同类公式表现不一致"的
    来源之一 —— 现在统一走这里，要改摘行规则只改这一个函数。
    """
    remove = {id(ln) for ln in lines or []}
    if not remove:
        return
    for blk in local_texts:
        if not any(id(ln) in remove for ln in (blk.get("lines") or [])):
            continue
        blk["lines"] = [ln for ln in blk["lines"] if id(ln) not in remove]
        if blk["lines"]:
            _retighten_block(blk)
            blk["text"] = _join_lines(blk["lines"])
        else:
            blk["_drop"] = True


def _formula_block(pid: str, latex: str, lines: list[dict], *,
                   label: str = "formula") -> dict:
    """合成公式块 —— **所有公式块的唯一构造出口**，字段契约固定：

        id / kind="formula" / label / latex / x,y,w,h
        text  = ""                        公式块没有「可读文本」(内容在 latex 里)
        lines = [{x,y,w,h,s,tag}]         只有几何与行字号，`runs` 恒为空

    于是「公式覆盖掉原来的字形」成了**数据层的事实**，不再依赖前端的 CSS 隐藏：
    前端只需按 `latex` 渲染，`lines` 只用来摆覆盖层与按原文主体宽度缩放。
    """
    geom = [_line_geom(ln) for ln in lines]
    if not geom:
        raise ValueError("formula block needs at least one line")
    x0 = min(g["x"] for g in geom)
    y0 = min(g["y"] for g in geom)
    x1 = max(g["x"] + g["w"] for g in geom)
    y1 = max(g["y"] + g["h"] for g in geom)
    return {"id": pid, "kind": "formula", "label": label, "latex": latex,
            "x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0,
            "text": "", "lines": geom}


def _materialize_inline_span(local_texts: list[dict], sp: dict, *, pno: int,
                             slot: int, page=None, images=None,
                             images_dir=None, img_idx: int = 0) -> int:
    """把一条「跨块行内公式」真正落地，返回新的图片序号。

    公式被 PyMuPDF 切成了几块/几行时，行内替换表达不了，于是改成：
      * 左行：切掉公式字形那截尾巴(行宽跟着缩)；
      * 右行：切掉公式字形那截脑袋(整行右移，把切掉的宽度补回来)；
      * 中间那几行：整体摘掉(它们本身就是公式字形)；
      * 在这些字形的并集位置上合成一个 `formula` 块，交给前端 KaTeX 覆盖渲染。
    只有**确认是公式字形**的片段才会被删，混着正文的片段在 `_ocr_line_replacements`
    里就已经被挡掉了。
    """
    lm, rm = sp["lm"], sp["rm"]
    a, b, latex = sp["a"], sp["b"], sp["latex"]
    lline, rline = sp["lline"], sp["rline"]
    mid_lines = sp["mlines"]

    rects: list[fitz.Rect] = []
    glyph_lines: list[dict] = []          # 公式块的内容行(两侧切下来的那截也算)
    if sp["l_has"]:                       # 公式从这一行的中间开始
        wl = _prefix_width(lline, a)
        tail = _slice_runs(lline["runs"], a, len(lm["raw"]))
        tail_w = max(1.0, lline["x"] + lline["w"] - (lline["x"] + wl))
        rects.append(fitz.Rect(lline["x"] + wl, lline["y"],
                               lline["x"] + lline["w"], lline["y"] + lline["h"]))
        lline["runs"] = _slice_runs(lline["runs"], 0, a)
        lline["w"] = max(1.0, wl)
        glyph_lines.append({"x": lline["x"] + wl, "y": lline["y"],
                            "w": tail_w, "h": lline["h"], "runs": tail})
    if sp["r_has"]:                       # 公式在这一行的中间结束
        wr = _prefix_width(rline, b)
        head = _slice_runs(rline["runs"], 0, b)
        rects.append(fitz.Rect(rline["x"], rline["y"],
                               rline["x"] + wr, rline["y"] + rline["h"]))
        rline["runs"] = _slice_runs(rline["runs"], b, len(rm["raw"]))
        rline["x"] = rline["x"] + wr
        rline["w"] = max(1.0, rline["w"] - wr)
        glyph_lines.append({"x": rline["x"] - wr, "y": rline["y"],
                            "w": max(1.0, wr), "h": rline["h"], "runs": head})
    for ln in mid_lines:
        rects.append(fitz.Rect(ln["x"], ln["y"], ln["x"] + ln["w"], ln["y"] + ln["h"]))
    glyph_lines += mid_lines
    # 内容行全空(两侧都切没了、中间也没行) → 没有可渲染的东西，索性不动数据
    glyph_lines = [ln for ln in glyph_lines if ln.get("runs")]
    if not rects or not glyph_lines:
        return img_idx
    hole = fitz.Rect(min(r.x0 for r in rects), min(r.y0 for r in rects),
                     max(r.x1 for r in rects), max(r.y1 for r in rects))
    # 公式可能正画在某张「烘焙图」里 → 先挖个洞，免得和图里的公式重影
    if images is not None and page is not None:
        img_idx = _carve_images(images, page, images_dir, pno, hole, img_idx)

    # 侧边被整行吃掉(公式就贴在行首/行尾) → 那一行也当成中间行整行摘掉
    gone = [ln for ln, runs in ((lline, lline["runs"]), (rline, rline["runs"]))
            if not runs]
    _detach_lines(local_texts, mid_lines + gone)
    for ln in (lline, rline):             # 被切过的两行所在块也要重算框
        for blk in local_texts:
            if any(ln is x for x in (blk.get("lines") or [])):
                _retighten_block(blk)
                blk["text"] = _join_lines(blk["lines"])
                break

    # 按阅读顺序排一下：两侧切下来的那截会插在中间行前后，排完顺序才对
    glyph_lines.sort(key=lambda ln: (round(ln["y"], 1), ln["x"]))
    local_texts.append(_formula_block(f"p{pno}s{slot}", latex, glyph_lines))
    return img_idx

def _inject_ocr_math(local_texts: list[dict], ocr_text: str, *,
                     pno: int, page, images: Optional[list] = None,
                     images_dir: Optional[Path] = None,
                     img_idx: int = 0) -> tuple[dict, int]:
    """保留本地版式，用 OCR 的 LaTeX 补公式。返回 (统计, 新图片序号)。

    与「字体候选区」的分工(见本节顶部说明)：
      ① 先按**字体名**判出每个字形「是不是公式」—— 这是独立于 OCR 的定位信号；
      ② OCR 的 LaTeX 仍是内容来源，并负责**验证 / 补漏**：
         · 独立公式：优先按字体候选挑行(`_display_job_font`)，挑不到才用「像不像
           正文」的启发式兜底(`_display_job`)；
         · 行内公式：锚点算出的区间与字体候选取并集(`_widen_with_cand`)，并集不合格
           就退回原区间。
      ③ 统计回报「候选行数 / 被用上 / 没被用上」，定位结果可查(parser_info 的
         math_font_*)。

    三条路径(行内替换 / 跨块摘行 / 独立公式)最后都落到同一套「落地」规则上：
    `_detach_lines` 摘字形 + `_formula_block` 合成只带几何与 `latex` 的公式块。
    """
    stats = {"inline": 0, "display": 0, "anchor_missed": 0}
    ocr_lines = _ocr_math_lines(ocr_text)
    if not ocr_lines or not local_texts:
        return stats, img_idx
    metas, page_norm = _page_index(local_texts)
    if not metas or not page_norm:
        return stats, img_idx

    # 字体候选区必须在任何改动 `local_texts` 之前算：后面的就地替换会重建 runs
    # (新 runs 不带字体名)，算晚了就什么都没有。
    font_cands, font_stats = _font_math_index(metas, local_texts)
    stats.update(font_stats)
    font_hit: set[int] = set()                # 被 OCR 公式用上的字体候选行

    used: set[int] = set()                    # 已被公式占用的本地行(metas 下标)
    display_lines: set[int] = set()           # 已被独立公式预定的行
    display_jobs: list[tuple[list[dict], str]] = []
    span_jobs: list[dict] = []                # 跨块的行内公式(循环结束再落地)

    for oi, ol in enumerate(ocr_lines):
        segs = ol["segs"]
        if not segs:
            continue                               # 这一行没有公式
        has_text = any(_norm_keep(s) for k, s in enumerate(segs) if k % 2 == 0)

        if not has_text:
            # ---------- 整行就是独立公式：用前后句子的首尾当锚点定位 ----------
            latex = next((s["latex"] for s in segs if isinstance(s, dict)), "")
            if not latex:
                continue
            before = _anchor_of(ocr_lines, oi, -1)
            after = _anchor_of(ocr_lines, oi, +1)
            if not before or not after:
                continue

            seen: dict[tuple[int, int], Optional[dict]] = {}

            def _job_for(le: int, rs: int) -> Optional[dict]:
                """这一对锚点算不算「前后两句夹着一条独立公式」；算就给出要摘的行。"""
                if (le, rs) in seen:
                    return seen[(le, rs)]
                # ① 先按字体名挑公式行(可靠)；② 挑不到才用「像不像正文」的启发式兜底
                #    —— 扫描件 / 数学与正文同体的 PDF 只有这条路可走。
                job = _display_job_font(metas, local_texts, le, rs, page,
                                        used | display_lines, font_cands)
                if job is None:
                    job = _display_job(metas, local_texts, le, rs, page,
                                       used | display_lines)
                seen[(le, rs)] = job
                return job

            # 与行内公式同一套判定：锚点由长到短试 + 只剩**一个**有效候选。
            # （原来要求 16 字符锚点在全页“只出现一次”，一撞车就放弃 —— 实测页 2
            #   两句一模一样的 "can be calculated as" 让两条 `\[…\]` 都没进来。）
            pair = _find_anchor_pair(page_norm, before, after,
                                     validate=lambda le, rs: _job_for(le, rs) is not None,
                                     rank=lambda le, rs: _job_rank(
                                         metas, latex, _job_for(le, rs)))
            if pair is None:
                stats["anchor_missed"] = stats.get("anchor_missed", 0) + 1
                continue
            job = _job_for(*pair)
            x0, y0, x1, y1 = job["rect"]
            # 这块公式可能正画在某张「烘焙图」里 → 先挖个洞，免得和图里的公式重影
            if images is not None:
                img_idx = _carve_images(images, page, images_dir, pno,
                                        fitz.Rect(x0, y0, x1, y1), img_idx)
            display_jobs.append((job["lines"], latex))
            display_lines.update(m["idx"] for m in job["cand"])
            font_hit.update(m["idx"] for m in job["cand"]
                            if m["idx"] in font_cands)
            continue

        # ---------- 正文里的行内公式：整页定位锚点，就地替换那一段字形 ----------
        res = _ocr_line_replacements(metas, page_norm, segs, ol["plain"],
                                     ocr_lines=ocr_lines, oi=oi,
                                     font_cands=font_cands,
                                     local_texts=local_texts)
        repls, spans = res["repls"], res["spans"]
        stats["anchor_missed"] = (stats.get("anchor_missed", 0)
                                   + res.get("unplaced", 0))
        if not repls and not spans:
            continue
        touched = set(repls) | {i for sp in spans
                                for i in range(sp["lm"]["idx"], sp["rm"]["idx"] + 1)}
        if not touched or (touched & (used | display_lines)):
            continue
        new_by_line: dict[int, list[dict]] = {}
        for mi, rl in repls.items():
            m = metas[mi]
            runs = local_texts[m["bi"]]["lines"][m["li"]]["runs"]
            new_runs = _apply_replacements(runs, m["raw"], rl, _body_run(runs))
            if new_runs is None:
                new_by_line = {}
                break
            new_by_line[mi] = new_runs
        if repls and not new_by_line:
            continue                     # 行内替换失败 → 这条 OCR 行不动(宁可不变)
        for mi, new_runs in new_by_line.items():
            m = metas[mi]
            local_texts[m["bi"]]["lines"][m["li"]]["runs"] = new_runs
            used.add(mi)
            if mi in font_cands:
                font_hit.add(mi)
        if new_by_line:
            stats["inline"] += 1
        # 跨块的公式只登记、不当场改数据 —— 删行会让 metas 失准，等循环结束统一落地。
        # 这里就把三条**行对象**抓住：后面按 index 再查会错位(前面的 span 可能已经删过行)。
        for sp in spans:
            sp = dict(sp)
            sp["lline"] = local_texts[sp["lm"]["bi"]]["lines"][sp["lm"]["li"]]
            sp["rline"] = local_texts[sp["rm"]["bi"]]["lines"][sp["rm"]["li"]]
            sp["mlines"] = [local_texts[m["bi"]]["lines"][m["li"]]
                             for m in sp["mids"]]
            span_jobs.append(sp)
            used.update(range(sp["lm"]["idx"], sp["rm"]["idx"] + 1))
            font_hit.update(i for i in range(sp["lm"]["idx"],
                                             sp["rm"]["idx"] + 1)
                            if i in font_cands)

    # 独立公式：把中间那几行字形摘出来，合成一个 formula 块(坐标取这几行的并集)，
    # 位置就是原来那几行的位置 → 前端按 display 模式覆盖渲染，版式不变。
    for lines, latex in display_jobs:
        _detach_lines(local_texts, lines)
        local_texts.append(
            _formula_block(f"p{pno}m{stats['display']}", latex, lines))
        stats["display"] += 1

    # 跨块的行内公式：两侧各切一刀、中间几行摘掉，合成 formula 块让 KaTeX 覆盖渲染
    for slot, sp in enumerate(span_jobs):
        img_idx = _materialize_inline_span(local_texts, sp, pno=pno, slot=slot,
                                           page=page, images=images,
                                           images_dir=images_dir, img_idx=img_idx)
        stats["inline_split"] = stats.get("inline_split", 0) + 1


    if any(b.get("_drop") for b in local_texts):
        local_texts[:] = [b for b in local_texts if not b.get("_drop")]
    for blk in local_texts:
        blk.pop("_drop", None)
        if blk.get("lines") and blk.get("kind") != "formula":
            blk["text"] = _join_lines(blk["lines"])   # 行内容变了 → 重算 text
    # 定位质量可观测：字体判据标了多少行像公式，其中多少行被 OCR 的公式用上了；
    # 剩下的那部分既是潜在漏检(OCR 没给出 LaTeX)，也可能是字体误检。
    stats["font_used"] = len(font_hit)
    stats["font_missed"] = max(0, len(font_cands) - len(font_hit))
    # 收尾体检：公式全部落地之后，正文里**还剩多少数学字形没被覆盖** —— 直接回答
    # 「为什么这里没盖住」。口径天然准确：重建过的 runs 不带字体名(见
    # `_apply_replacements` / `_slice_runs` 只复制 `_STYLE_KEYS`)，公式块没有 runs，
    # 所以再扫一遍还认得出数学字体的行，就是货真价实的残留。
    left_metas, _ = _page_index(local_texts)
    _, left_stats = _font_math_index(left_metas, local_texts)
    stats["leftover_lines"] = left_stats["font_lines"]
    stats["leftover_chars"] = left_stats["font_chars"]
    return stats, img_idx

def _accumulate_stats(acc: Optional[dict], add: dict) -> None:
    """把单页统计累加进整份文档的统计(`acc` 为 None 时忽略)。"""
    if acc is None:
        return
    for k, v in (add or {}).items():
        acc[k] = acc.get(k, 0) + v
