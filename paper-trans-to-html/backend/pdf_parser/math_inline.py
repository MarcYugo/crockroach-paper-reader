"""行内公式落地：把两个锚点之间的字形换成 `\(latex\)`。

见包文档 `backend/pdf_parser/__init__.py`。
"""

from __future__ import annotations

from typing import Optional

from .math_font import (_retighten, _widen_cut, _widen_with_cand)
from .math_locate import (_find_anchor_pair, _job_rank, _math_rows_between)
from .math_text import (_SPAN_MAX_LINES, _anchor_of, _has_glyphs,
                        _math_glyphs, _meta_at, _norm_keep, _tighten)


def _ocr_line_replacements(metas: list[dict], page_norm: str,
                           segs: list, plain: str = "",
                           ocr_lines: Optional[list] = None,
                           oi: int = 0,
                           font_cands: Optional[dict] = None,
                           local_texts: Optional[list] = None) -> dict:
    """把一条 OCR 行的公式翻译成「要对本地文本做的改动」。

    返回 `{"repls": {行下标: [(a, b, latex), …]}, "spans": [跨块公式, …]}`：

    * `repls` —— 公式字形和锚点同在一行：把本地 raw[a:b] 就地换成 `\\(latex\\)`；
    * `spans` —— 公式被 PyMuPDF **切成了好几块/好几行**(论文里的行内公式极常见，
      上下标还会单独成一行)：行内替换表达不了，交给
      `_materialize_inline_span` 去「切两侧 + 摘中间 + 合成覆盖块」。

    `font_cands` 是 `_font_math_index` 给出的「每行数学字形区间」：有它就把 OCR 锚点
    算出来的区间**吸附**到真实的公式字形上(见 `_widen_with_cand`)，OCR 退化为
    「验证 + 补漏」；没有(扫描件等)就完全按锚点走。

    ⚠️ **逐条公式独立判定**：某一条定位不到只丢它自己，不牵连同一条 OCR 行里的别的
    公式 —— OCR 是按整段输出的，一条失败就整条放弃会把整段的公式全废掉。
    """
    repls: dict[int, list] = {}
    spans: list[dict] = []
    unplaced = 0                      # 这一行里「有 LaTeX、但没定位上」的公式条数
    for k in range(1, len(segs), 2):
        seg = segs[k]
        if not isinstance(seg, dict):
            continue
        latex = (seg.get("latex") or "").strip()
        if not latex:
            continue
        left_norm = _norm_keep(segs[k - 1])
        right_norm = _norm_keep(segs[k + 1]) if k + 1 < len(segs) else ""
        # 行首/行尾的公式：本行里没有那一侧的锚点 → 借相邻 OCR 行的文字顶上。
        # ⚠️ **只在这两种情形回退**：夹在两条公式中间的空片段不能借 —— 那会拿相邻
        # 行的字去凑锚点，把整段中间的字形(甚至别人的公式)都当成自己的字形吞掉。
        if not left_norm and k == 1:
            left_norm = _norm_keep(_anchor_of(ocr_lines, oi, -1))
        if not right_norm and k + 2 >= len(segs):
            right_norm = _norm_keep(_anchor_of(ocr_lines, oi, +1))
        if not left_norm or not right_norm:
            unplaced += 1
            continue
        # 锚点组合可能不止一种(页面上两句一样的话)：交给 `_inline_job` 按「中间是不是
        # 公式字形」把错的候选筛掉，而不是要求锚点在全页唯一。
        judged: dict[tuple[int, int], Optional[dict]] = {}

        def _job_for(le: int, rs: int) -> Optional[dict]:
            key = (le, rs)
            if key not in judged:
                judged[key] = _inline_job(metas, le, rs, latex, plain,
                                          font_cands, local_texts)
            return judged[key]

        pair = _find_anchor_pair(page_norm, left_norm, right_norm,
                                 validate=lambda le, rs: _job_for(le, rs) is not None,
                                 rank=lambda le, rs: _job_rank(metas, latex,
                                                               _job_for(le, rs)))
        if pair is None:
            unplaced += 1
            continue
        job = _job_for(*pair)
        if job is None:
            unplaced += 1
            continue
        if job["kind"] == "repl":
            repls.setdefault(job["mi"], []).append((job["a"], job["b"], latex))
        else:
            spans.append({**job, "latex": latex})
    return {"repls": repls, "spans": spans, "unplaced": unplaced}


def _inline_job(metas: list[dict], le: int, rs: int, latex: str,
                plain: str, cands: Optional[dict] = None,
                local_texts: Optional[list] = None) -> Optional[dict]:
    """一条行内公式该怎么落：同城替换 还是 跨块合成。定位不到就 None。

    返回 `{"kind": "repl", "mi", "a", "b"}`(把该行 raw[a:b] 换成 `\(latex\)`)
    或 `{"kind": "span", "lm", "a", "rm", "b", "mids", "l_has", "r_has"}`
    (公式横跨多行，交给 `_materialize_inline_span` 切行 + 合成覆盖块)。

    `cands`(可选)是 `_font_math_index` 给出的「每行数学字形区间」：锚点算出来的区间
    先与它取并集再校验(见 `_widen_with_cand`)，并集不合格就退回原区间 —— 既补上
    字体判据漏掉的(`AR`)，也修正锚点算偏的边界，而且**不会因此变松**：仍然要过
    `_math_glyphs`。

    `local_texts`(可选)供「中间行取不到时的纵向兜底」用(见下)；不传就只按顺序取。
    """
    cands = cands or {}
    lm, kl = _meta_at(metas, le)
    rm, kr = _meta_at(metas, rs)
    if lm is None or rm is None:
        return None

    if lm["idx"] == rm["idx"]:                    # 公式和两个锚点都在同一行
        a, b = _tighten(lm["raw"], lm["ridx"][kl] + 1, rm["ridx"][kr])
        for a2, b2 in _widen_with_cand(a, b, cands.get(lm["idx"])):
            a2, b2 = _retighten(lm["raw"], a2, b2)
            if b2 > a2 and _math_glyphs(lm["raw"][a2:b2], latex, plain):
                return {"kind": "repl", "mi": lm["idx"], "a": a2, "b": b2}
        return None

    # ---------- 跨行：公式字形散在好几行/好几个块上 ----------
    a = _tighten(lm["raw"], lm["ridx"][kl] + 1, len(lm["raw"]))[0]
    b = rm["ridx"][kr]
    # 左右两个切点吸附到字体候选区的边界上：公式里用正文字体排的那截(`AR`)只有这里
    # 才能补回来。吸附过头会被下面的 `_math_glyphs` 挡掉(混进正文就不认)。
    # ⚠️ 左侧照旧要收紧(空白/标点留给正文)，**右侧不要**：`b` 后面的那个空格一般是
    # 正文字号(实测 `\frac{1}{K}` 那条，整块里唯一的 10pt run 就是这个尾空格)，
    # 收掉它会让前端的 `formulaBaseSize`(取块内最大 run 字号)从 10 掉到 7 → 公式
    # 被渲染小 20%。保持与改动前一致。
    a = _retighten(lm["raw"],
                   _widen_cut(a, cands.get(lm["idx"]), left=True),
                   len(lm["raw"]))[0]
    b = _widen_cut(b, cands.get(rm["idx"]), left=False)
    # 中间那几行 = 这条公式自己的字形行。先按 `metas` 顺序取(便宜)；**取不到就换
    # 纵向区间再取一次** —— 双栏论文里 PyMuPDF 的块顺序与阅读顺序常常不一致，公式行
    # 可能被排到右锚点之后(实测页 2 的 `a^{(i)}_{i,s}`：两个锚点在顺序表里正好相邻)，
    # 顺序法会判成「中间没有行」而整条放弃 → 碎片留在正文里。
    mids = [metas[i] for i in range(lm["idx"] + 1, rm["idx"])]
    if not mids and local_texts is not None:
        mids = _math_rows_between(metas, local_texts, le, rs,
                                  font_cands=cands, window="band")
    if len(mids) > _SPAN_MAX_LINES:
        return None
    left_tail, right_head = lm["raw"][a:], rm["raw"][:b]
    l_has, r_has = _has_glyphs(left_tail), _has_glyphs(right_head)
    pieces = ([left_tail] if l_has else []) + ([right_head] if r_has else [])
    pieces += [m["raw"] for m in mids]
    if not pieces:
        return None                           # 两侧都没字形、中间也没行 → 无从谈起
    if not all(_math_glyphs(p, latex, plain) for p in pieces):
        return None                           # 要删的里面混着正文 → 定位错了
    if not l_has and not mids:
        # 公式只在右行的行首(如脚注 `\(^7\)`) → 还是行内替换最稳
        if r_has:
            a0, b0 = _tighten(rm["raw"], 0, b)
            if b0 > a0:
                return {"kind": "repl", "mi": rm["idx"], "a": a0, "b": b0}
        return None
    if not r_has and not mids:
        # 公式只在左行的行尾
        a0, b0 = _tighten(lm["raw"], a, len(lm["raw"]))
        if b0 > a0:
            return {"kind": "repl", "mi": lm["idx"], "a": a0, "b": b0}
        return None
    # 两侧都有字形、中间一行也没有 —— **相邻两行之间**断开的公式(实测页 2 的
    # `a^{(i)}_{i,s}`：`a^{(i)}` 是上一行的尾、`_{i,s}` 是下一行的头)。
    # 从前这里直接放弃("收益不抵风险")，结果这类公式**一个字都不覆盖**、原字形
    # 留在页面上 —— 而这恰恰是「有的公式盖住了、有的没盖住」最常见的来源。
    # 落到 `_materialize_inline_span` 的 mids=[] 分支即可：两侧各切一刀、合成
    # 一个覆盖块；要删的两截都已经过了 `_math_glyphs` 校验。
    return {"kind": "span", "lm": lm, "a": a, "rm": rm, "b": b,
            "mids": mids, "l_has": l_has, "r_has": r_has}
