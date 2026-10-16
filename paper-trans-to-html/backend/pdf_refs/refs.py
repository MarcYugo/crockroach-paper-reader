"""引用跳转：把 PDF 链接或编号引用（正文引用标记 → 参考文献条目）整理成前端用的映射。

解析入口按后端分发策略：PyMuPDF 与 Unlimited-OCR 共用此处的通用实现；Surya
单独处理其整页 OCR 块，允许一块中有多条参考文献，并同时扫描编号引用与可用内部跳转。

首选数据：`pages[].links`（见 `pdf_parser/links.py`，解析时采集，重解析后自动更新）。
通用实现优先按 PDF 内部跳转映射；没有可用映射时，再按正文令牌与条目做回退匹配：
数字令牌（`[n]`）关联编号条目；作者-年份令牌（方括号的 `[Smith et al., 2020]`
与圆括号的 `(Smith et al., 2020)` 两种写法）按“年份 + 姓氏/实词”关联无编号条目
—— 校验从严、宁缺勿错（对不上就不标）。

通用实现的步骤（Surya 的条目化策略独立实现）：

  1. **找参考文献区**：全文档**最后一个** “References / Bibliography / 参考文献”
     标题块；没有标题时兜底看 Unlimited-OCR 路线特有的 `kind="reference"` 块。
     区里再遇到 Appendix/Acknowledgments 标题就截断（别把附录算进来）。
  2. **条目化**：编号制（`[1]` / `1.` / `1)`）按编号起新条目、不带编号的块并进上一条
     （PyMuPDF 路线把长条目切成“一句一块”，不并起来卡片就只剩半句）；无编号
     （作者-年份制）按“含年份”起新条目、其余块并进上一条。
  3. **优先映射 `goto` 链接**：
     * 目标点（`to`）落在哪个条目块（几何就近，≤28pt）→ 得到**答案条目**；
     * 链接矩形 ∩ 源侧块 → 在该块文本里找**能对上答案条目**的引用令牌
       （编号制按编号；作者-年份按“年份 + 姓氏/实词”匹配，方括号/圆括号写法都认）；
     * 同一块里多个候选时按几何就近取（只需近似位置，两种后端口径都够用）；
     * 对不上就丢（**宁缺勿错**）。
     **数字令牌会展开成它能对上的全部条目**：`[10-12]` 即使只被 hyperref 指到
     「12」一条链接，卡片里也按 10/11/12 三条列出（列表里没有对应编号的逐个跳过）；
     `[3,5]` 一条令牌挂多条链接时，同一个 `(块, 起, 止)` 会归并成一个标记。
  4. 如果没有映射出任何内部跳转，则扫描正文令牌回退关联：数字令牌（`[n]`）只关联
     编号条目；作者-年份令牌按“年份 + 姓氏/实词”关联无编号条目。

输出结构（`GET /api/doc/{id}/refs` 的响应体）：

    {
      "ok": True,
      "scan": "ok" | "no-links-data",   # no-links-data = 还是“采集链接之前”的旧数据
      "section_page": 参考文献区起始页下标（找不到为 None）,
      "entries": [{"i": 0, "n": 1, "page": 4, "blocks": ["p4b2", …],
                   "text": "[1] J. Smith, …", "uri": "https://…"}],
      "links":   [{"page": 0, "block": "p0b5", "start": 12, "end": 16,
                   "token": "[1]", "refs": [0], "nums": [1]}],
      "counts":  {"entries": 42, "links": 57, "skipped": 3},
    }

⚠️ `start/end` 是**块 canonical 文本**里的字符区间：canonical = 每行文本 + "\n"
依次拼接，与前端 `reader2.js` 的 canonical 完全同口径（PyMuPDF 路线取全部行；
surya/unlimited_ocr 只有一条“伪行”，取 `lines[0]` —— 两种口径都在 `_canonical()`
实现）。所以锚点只存区间、不存 DOM；选句/高亮/缩放导致块重建后，前端按同区间
把 `.cite` 标记重打一遍即可。本模块**只读 doc，不写任何东西**（每次请求现算）。
"""
from __future__ import annotations

import logging
import re

log = logging.getLogger("uvicorn.error")

__all__ = ["build_refs"]

# 参考文献区标题：允许 “7 References” / “REFERENCES” / “Bibliography.” 等写法
_RE_HEADING = re.compile(
    r"(?i)^\s*(?:[0-9]+[.\s]*)?(references|bibliography|works\s+cited|"
    r"literature\s+cited|参考文献)\s*[:.]?\s*$")
# 参考文献区之后的下一个大节标题 → 截断（附录/致谢里也有“条目”样式的文本）
_RE_SECTION_END = re.compile(
    r"(?i)^\s*(?:[A-Z]\.|[0-9]+[.\s]*)?(appendix|appendices|supplementary|"
    r"acknowledg(?:e)?ments?)\b")
# 条目开头（编号制）：`[12]` / `(12)` / `12.` / `12)`
_RE_NUMBER_START = re.compile(r"^\s*(?:\[(\d{1,3})\]|\((\d{1,3})\)|(\d{1,3})\s*[.)])\s*")
# 正文里的编号式引用令牌：`[12]` / `[3, 5]` / `[10–12]`
_RE_CITE_TOKEN = re.compile(r"\[\s*([\d\s,;，、\-–—]{1,40}?)\s*\]")
# 作者-年份式引用令牌：`(Smith et al., 2020)`
_RE_YEAR_TOKEN = re.compile(r"\(([^()\n]{2,180}?)\)")
_RE_YEAR = re.compile(r"(?:19|20)\d{2}")
_RE_WORD = re.compile(r"[A-Za-z]{4,}")

# 作者-年份匹配时忽略的高频词（避免“(this is, 2020)”之类误命中）
_AY_STOP = {"et", "al", "the", "and", "for", "with", "from", "this", "that",
            "proc", "proceedings", "conference", "journal", "vol", "pp", "ed",
            "eds", "press", "university", "arxiv", "https", "http", "www"}

# 作者-年份令牌（方括号写法 `[Smith et al., 2020]`：LaTeX 默认 \cite 渲染出来的样子）
_RE_AY_BRACKET = re.compile(r"\[([^\[\]\n]{2,150}?)\]")
# 条目开头的方括号标签（`[Smith et al., 2020] 正文…`）→ 取出姓氏词用于令牌匹配
_RE_AY_LABEL = re.compile(r"^\s*\[([^\[\]\n]{1,120}?),\s*(?:19|20)\d{2}[a-z]?\]")
# 标签词匹配的忽略词：短姓氏（`Ke`/`Wu` 等 2 字母）不设长度门槛，介词/缩写继续忽略
_AY_LABEL_STOP = _AY_STOP | {"de", "van", "von", "der", "la", "jr", "sr", "ii", "iii"}


# =====================================================================
#  canonical 文本（必须与 frontend/js/reader2.js 完全同口径）
# =====================================================================
def _canonical(t: dict, ocr_style: bool) -> str:
    """块 canonical 文本 = 每行“run 文本拼接”+ '\n'，行序按数据顺序。

    PyMuPDF 路线：前端对 `t.lines` 逐行建 `.ln`，全部行都要拼；
    surya / unlimited_ocr：前端只取 `lines[0]` 这条“伪行”，这里也只看第一条。
    """
    lines = t.get("lines") or []
    if ocr_style:
        lines = lines[:1]
    parts = ["".join(str(r.get("t") or "") for r in (ln.get("runs") or []))
             for ln in lines]
    if not parts:                    # 异常/旧数据：退回块 text（尽量可用）
        return str(t.get("text") or "")
    return "".join(p + "\n" for p in parts)


def _line_segments(t: dict, ocr_style: bool):
    """[(行首 offset, 行尾 offset, 行), …]（与 `_canonical` 的拼接对应）。
    行末那个 '\n' 不属于任何行段（区间取 `off <= b` 时视为落在该行）。"""
    lines = t.get("lines") or []
    if ocr_style:
        lines = lines[:1]
    segs = []
    pos = 0
    for ln in lines:
        n = sum(len(str(r.get("t") or "")) for r in (ln.get("runs") or []))
        segs.append((pos, pos + n, ln))
        pos += n + 1
    return segs


# =====================================================================
#  几何小工具（PDF 坐标，pt）
# =====================================================================
def _bbox(t: dict):
    x = float(t.get("x") or 0.0)
    y = float(t.get("y") or 0.0)
    return (x, y, x + float(t.get("w") or 0.0), y + float(t.get("h") or 0.0))


def _overlap(a, b) -> float:
    ox = min(a[2], b[2]) - max(a[0], b[0])
    oy = min(a[3], b[3]) - max(a[1], b[1])
    return ox * oy if ox > 0 and oy > 0 else 0.0


def _estimate_xy(t: dict, ocr_style: bool, off: int):
    """令牌起点的**近似**页面位置（只用于“同一块里多个候选挑一个”的就近比较）。

    PyMuPDF 路线用真实行几何（x/w 按行内字符比例插值）；OCR 路线的“伪行”
    实际是整段（行内还会折行），横向只能按块宽比例估 —— 够用即可，
    反正选完还要过“令牌内容对不对得上答案条目”的校验。
    """
    bx, by, bx1, by1 = _bbox(t)
    for a, b, ln in _line_segments(t, ocr_style):
        if a <= off <= b:
            lx = float(ln.get("x") if ln.get("x") is not None else bx)
            ly = float(ln.get("y") if ln.get("y") is not None else by)
            lw = float(ln.get("w") or (bx1 - bx) or 1.0)
            lh = float(ln.get("h") or t.get("line_h") or (t.get("size") or 10) * 1.4)
            frac = (off - a) / max(1, b - a)
            return lx + frac * lw, ly + lh / 2
    text_len = max(1, len(_canonical(t, ocr_style)))
    frac = off / text_len
    return bx + frac * (bx1 - bx), by + (by1 - by) / 2


# =====================================================================
#  参考文献区与条目
# =====================================================================
def _find_references_start(pages):
    """全文档**最后一个** “References/Bibliography/参考文献” 标题块 (页, 块)。"""
    best = None
    for pi, pg in enumerate(pages):
        for bi, t in enumerate(pg.get("texts") or []):
            txt = str(t.get("text") or "").strip()
            if txt and len(txt) <= 40 and _RE_HEADING.match(txt):
                best = (pi, bi)
    return best


def _region_blocks(pages, start):
    """参考文献区里的块 `(页, 块下标, 块数据)`：标题块之后 → 下个大节标题之前。"""
    if start is None:
        # 兜底：unlimited_ocr 的 kind="reference"（后端把 ref_text 映射成它）
        out = []
        for pi, pg in enumerate(pages):
            for bi, t in enumerate(pg.get("texts") or []):
                if str(t.get("kind") or "") == "reference":
                    out.append((pi, bi, t))
        return out
    out = []
    sp, sb = start
    for pi in range(sp, len(pages)):
        for bi, t in enumerate(pages[pi].get("texts") or []):
            if pi == sp and bi <= sb:
                continue
            txt = str(t.get("text") or "").strip()
            if txt and len(txt) <= 40 and _RE_SECTION_END.match(txt):
                return out
            out.append((pi, bi, t))
    return out


def _clean(txt) -> str:
    return " ".join(str(txt or "").split())


def _build_entries(region):
    """条目化 → `(entries, block2entry)`（块 id → 条目下标）。

    * 编号制（区里 ≥3 块以 `[n]`/`n.` 开头；短列表 ≤4 块时 ≥2 块即可）：
      编号块起新条目，不带编号的块并进上一条 —— PyMuPDF 路线“一句一块”把长条目
      切碎时，这样卡片才是完整一条；碎片（页码等，<4 字符）直接丢。
    * 无编号（作者-年份制）：含年份的块起新条目，其余块并进上一条。
    """
    sample = region[:12]
    votes = sum(1 for _, _, t in sample
                if _RE_NUMBER_START.match(_clean(t.get("text"))))
    # 短列表（≤4 块）放宽到“≥2 个编号块”：两三条的参考文献列表并不罕见，
    # 按作者-年份处理会让 `[n]` 令牌全都对不上。
    numbered = votes >= (2 if len(sample) <= 4 else 3)
    entries: list[dict] = []
    by_block: dict[str, int] = {}

    def _new(pi, t, txt, n):
        bid = str(t.get("id") or f"p{pi}b{len(entries)}")
        entries.append({"i": len(entries), "n": n, "page": pi,
                        "blocks": [bid], "text": txt[:1500], "uri": ""})
        by_block[bid] = len(entries) - 1

    def _append(pi, t, txt):
        if not entries:
            return
        e = entries[-1]
        bid = str(t.get("id") or f"p{pi}b{len(entries)}")
        if bid not in e["blocks"]:
            e["blocks"].append(bid)
        if len(e["text"]) < 1500:
            e["text"] = (e["text"] + " " + txt)[:1500]
        by_block[bid] = e["i"]

    for pi, bi, t in region:
        txt = _clean(t.get("text"))
        if not txt or txt.isdigit() or len(txt) < 4:
            continue                                  # 页码/页眉碎片
        m = _RE_NUMBER_START.match(txt)
        if numbered:
            if m:
                _new(pi, t, txt, int(next(g for g in m.groups() if g)))
            elif entries:
                _append(pi, t, txt)
        else:
            if _RE_YEAR.search(txt) or not entries:
                _new(pi, t, txt, None)
            else:
                _append(pi, t, txt)
    return entries, by_block


# =====================================================================
#  链接映射
# =====================================================================
def _expand_nums(inner: str) -> set:
    """`"3, 5-7"` → {3, 5, 6, 7}（范围限制在 ≤40 内展开，防呆）。"""
    nums: set = set()
    for m in re.finditer(r"(\d{1,3})\s*(?:[-–—]\s*(\d{1,3}))?", inner):
        a = int(m.group(1))
        nums.add(a)
        if m.group(2):
            b = int(m.group(2))
            if a < b <= a + 40:
                nums.update(range(a, b + 1))
    return nums


def _entry_block_at(page, to, by_block):
    """目标点 → 条目块：先找包含点的条目块（±3pt），否则取最近条目块（≤28pt）。"""
    x, y = float(to[0]), float(to[1])
    best = None
    best_d = 1e9
    for t in page.get("texts") or []:
        if str(t.get("id")) not in by_block:
            continue
        x0, y0, x1, y1 = _bbox(t)
        if x0 - 3 <= x <= x1 + 3 and y0 - 3 <= y <= y1 + 3:
            return t
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        d = abs(cx - x) + abs(cy - y)
        if d < best_d:
            best_d, best = d, t
    return best if best_d <= 28 else None


def _ay_entry_profile(entry) -> tuple:
    """作者-年份条目画像 `(年份集, 实词集, 标签词集)` —— 令牌匹配的三件套。

    * 年份集：条目文本里出现的全部年份；
    * 实词集：≥4 字母、去掉高频词后的词（不含 2-3 字母的短姓氏）；
    * 标签词集：条目开头 `[Smith et al., 2020]` 标签里的词（2 字母也行 ——
      `[Wu, …, 1998]`、`[Ke et al., 2017]` 这类短姓氏靠它才匹配得上）。
    """
    etext = str(entry.get("text") or "")
    years = set(_RE_YEAR.findall(etext))
    words = {w for w in (x.lower() for x in _RE_WORD.findall(etext))
             if w not in _AY_STOP}
    label_words: set = set()
    m = _RE_AY_LABEL.match(etext)
    if m:
        label_words = {w for w in re.findall(r"[A-Za-z]{2,}", m.group(1).lower())
                       if w not in _AY_LABEL_STOP}
    return years, words, label_words


def _ay_token_match(inner: str, years: set, words: set, label_words: set) -> bool:
    """令牌内文能否对上条目：按 `;` 分段，某一段里要**同时**出现条目的年份和
    条目的实词（或标签姓氏词）。分段校验避免 `[A, 2020; B, 2020]` 这种多引用
    令牌把年份和别家的姓氏串到一块。"""
    for seg in re.split(r"[;；]", inner):
        if not any(y in years for y in _RE_YEAR.findall(seg)):
            continue
        low = seg.lower()
        if words and any(w in low for w in words):
            return True
        if label_words:
            seg_words = {w for w in re.findall(r"[a-z]{2,}", low)
                         if w not in _AY_LABEL_STOP}
            if seg_words & label_words:
                return True
    return False


def _cite_candidates(t, ocr_style, entry):
    """块文本里能对上 `entry` 的引用令牌候选 `[(起, 止, 原文), …]`。

    编号制条目 → 找包含该编号的 `[…]` 令牌（支持 `[3,5]`/`[10-12]`）；
    作者-年份制条目 → 找“年份对得上、且含条目里某个实词/姓氏”的 `[…]`/`(…)` 令牌。
    """
    canon = _canonical(t, ocr_style)
    out = []
    if entry.get("n"):
        want = int(entry["n"])
        for m in _RE_CITE_TOKEN.finditer(canon):
            if want in _expand_nums(m.group(1)):
                out.append((m.start(), m.end(), m.group(0)))
        return out
    years, words, label_words = _ay_entry_profile(entry)
    if not years:
        return out
    # 方括号写法优先：圆括号里套方括号时（`(see [Smith et al., 2020])`）
    # 标更精确的里层。
    for pattern in (_RE_AY_BRACKET, _RE_YEAR_TOKEN):
        got = [(m.start(), m.end(), m.group(0))
               for m in pattern.finditer(canon)
               if _ay_token_match(m.group(1), years, words, label_words)]
        if got:
            return got
    return out


def _pick_candidate(t, ocr_style, cands, rect):
    """多个候选 → 按“估算位置 vs 链接矩形中心”的就近原则挑一个。"""
    if len(cands) == 1:
        return cands[0]
    rx = (rect[0] + rect[2]) / 2
    ry = (rect[1] + rect[3]) / 2
    best, best_d = None, 1e9
    for c in cands:
        x, y = _estimate_xy(t, ocr_style, c[0])
        d = abs(x - rx) + 2 * abs(y - ry)      # 纵向更敏感（换行错位比横向误差更常见）
        if d < best_d:
            best_d, best = d, c
    return best


def _source_blocks(pg, rect):
    """源页上与该矩形有重叠的块，按重叠面积从大到小（最多 6 个）。"""
    hits = []
    for t in pg.get("texts") or []:
        area = _overlap(rect, _bbox(t))
        if area > 0:
            hits.append((area, t))
    hits.sort(key=lambda z: -z[0])
    return [t for _, t in hits[:6]]


def _map_links(pages, ocr_style, entries, by_block, *,
               exclude_source_blocks: set[str] | None = None):
    """逐条 `goto` 链接做“目标 → 条目、源 → 令牌”的映射；对不上就跳过。"""
    merged: dict = {}
    skipped = 0
    num2entry: dict = {}                  # 编号 → 条目下标（同号取第一条）
    for e in entries:
        if e.get("n") and e["n"] not in num2entry:
            num2entry[e["n"]] = e["i"]
    for pi, pg in enumerate(pages):
        for lk in pg.get("links") or []:
            if lk.get("kind") != "goto":
                continue
            tp = lk.get("page")
            rect = lk.get("rect")
            to = lk.get("to")
            if (not isinstance(tp, int) or not (0 <= tp < len(pages))
                    or not rect or not to):
                skipped += 1
                continue
            tgt = _entry_block_at(pages[tp], to, by_block)
            if tgt is None:
                skipped += 1
                continue
            eidx = by_block[str(tgt.get("id"))]
            entry = entries[eidx]
            chosen = None
            for t in _source_blocks(pg, rect):
                bid = str(t.get("id"))
                if (bid in by_block or bid in (exclude_source_blocks or set())):
                    # 参考文献区自身的跳转不算引用
                    continue
                cands = _cite_candidates(t, ocr_style, entry)
                if cands:
                    s, e, tok = _pick_candidate(t, ocr_style, cands, rect)
                    chosen = (bid, s, e, tok)
                    break
            if chosen is None:
                skipped += 1
                continue
            bid, s, e, tok = chosen
            rec = merged.get((bid, s, e))
            if rec is None:
                rec = {"page": pi, "block": bid, "start": s, "end": e,
                       "token": tok, "refs": [], "nums": []}
                merged[(bid, s, e)] = rec
            # 数字令牌（`[10-12]` / `[3,5]`）**展开成它能对上的全部条目**：
            # hyperref 往往整段只做一条链接（比如只指到 12），但读者要看到范围里
            # 的三条；列表里没有对应编号的（检测漏了某条等）逐个跳过。
            if entry.get("n"):
                nums = sorted(_expand_nums(tok))
                idxs = [num2entry[nn] for nn in nums if nn in num2entry]
                nums = [nn for nn in nums if nn in num2entry]   # 只留真对上条目的编号
                if eidx not in idxs:
                    idxs.append(eidx)     # 兜底：链接指到的那条必须在
            else:
                nums, idxs = [], [eidx]   # 作者-年份：令牌即条目
            for i2 in idxs:
                if i2 not in rec["refs"]:
                    rec["refs"].append(i2)
            for nn in nums:
                if nn not in rec["nums"]:
                    rec["nums"].append(nn)
    for rec in merged.values():               # 汇总后统一排序（可能多链接贡献）
        rec["nums"] = sorted(set(rec["nums"]))
        rec["refs"] = sorted(set(rec["refs"]),
                             key=lambda i: (entries[i].get("n") or 0, i))
    links = sorted(merged.values(), key=lambda r: (r["page"], r["block"], r["start"]))
    return links, skipped


def _map_numeric_citations(pages, ocr_style, entries, by_block):
    """没有可用 PDF 内部跳转时，按正文方括号编号与条目编号建立回退映射。"""
    num2entry: dict[int, int] = {}
    for entry in entries:
        n = entry.get("n")
        if n is not None and n not in num2entry:
            num2entry[n] = entry["i"]
    if not num2entry:
        return []

    merged = {}
    for pi, pg in enumerate(pages):
        for t in pg.get("texts") or []:
            bid = str(t.get("id") or "")
            if not bid or bid in by_block:
                continue
            canon = _canonical(t, ocr_style)
            for match in _RE_CITE_TOKEN.finditer(canon):
                nums = sorted(n for n in _expand_nums(match.group(1))
                              if n in num2entry)
                if not nums:
                    continue
                refs = sorted({num2entry[n] for n in nums},
                              key=lambda i: (entries[i].get("n") or 0, i))
                merged[(bid, match.start(), match.end())] = {
                    "page": pi, "block": bid, "start": match.start(),
                    "end": match.end(), "token": match.group(0),
                    "refs": refs, "nums": nums,
                }
    return sorted(merged.values(),
                  key=lambda r: (r["page"], r["block"], r["start"]))


def _scan_author_year_citations(pages, ocr_style, entries, skip_blocks: set) -> list:
    """没有可用 PDF 内部跳转时，按正文作者-年份令牌与无编号条目做文本匹配。

    令牌两种写法都扫：方括号（`[Smith et al., 2020]`，LaTeX `\\cite` 的默认渲染）
    与圆括号（`(Smith et al., 2020)`）；匹配校验与链接路径共用
    （见 `_ay_entry_profile` / `_ay_token_match`）。同一块里与已收令牌重叠的
    一律跳过 —— `(see [Smith et al., 2020])` 只标里面那个。
    """
    profiles = []
    for entry in entries:
        if entry.get("n"):
            continue                          # 编号制条目走数字令牌的回退
        years, words, label_words = _ay_entry_profile(entry)
        if years:
            profiles.append((entry["i"], years, words, label_words))
    if not profiles:
        return []

    merged: dict = {}
    for pi, pg in enumerate(pages):
        for t in pg.get("texts") or []:
            bid = str(t.get("id") or "")
            if not bid or bid in skip_blocks:
                continue
            canon = _canonical(t, ocr_style)
            taken: list = []

            def collect(match):
                refs = [i for i, years, words, label_words in profiles
                        if _ay_token_match(match.group(1), years, words, label_words)]
                if not refs:
                    return
                if any(not (match.end() <= s or match.start() >= e)
                       for s, e in taken):
                    return
                taken.append((match.start(), match.end()))
                merged[(bid, match.start(), match.end())] = {
                    "page": pi, "block": bid, "start": match.start(),
                    "end": match.end(), "token": match.group(0),
                    "refs": sorted(refs), "nums": [],
                }

            for match in _RE_AY_BRACKET.finditer(canon):
                collect(match)
            for match in _RE_YEAR_TOKEN.finditer(canon):
                collect(match)
    return sorted(merged.values(),
                  key=lambda r: (r["page"], r["block"], r["start"]))


def _merge_links(*groups) -> list:
    """按 `(块, 起, 止)` 归并多路映射结果（`refs`/`nums` 去重、排序）。"""
    merged: dict = {}
    for group in groups:
        for link in group:
            key = (link["block"], link["start"], link["end"])
            current = merged.get(key)
            if current is None:
                current = dict(link, refs=[], nums=[])
                merged[key] = current
            current["refs"].extend(link.get("refs") or [])
            current["nums"].extend(link.get("nums") or [])
    for link in merged.values():
        link["refs"] = sorted(set(link["refs"]))
        link["nums"] = sorted(set(link["nums"]))
    return sorted(merged.values(),
                  key=lambda link: (link["page"], link["block"], link["start"]))


def _attach_uris(pages, entries, by_block):
    """把条目上的外链（LINK_URI）挂到条目：与条目块重叠面积最大者获胜。"""
    if not entries:
        return
    for pg in pages:
        for lk in pg.get("links") or []:
            if lk.get("kind") != "uri":
                continue
            rect = lk.get("rect")
            uri = str(lk.get("uri") or "")
            if not rect or not uri:
                continue
            best, best_area = None, 0.0
            for t in pg.get("texts") or []:
                eidx = by_block.get(str(t.get("id")))
                if eidx is None or entries[eidx]["uri"]:
                    continue
                area = _overlap(rect, _bbox(t))
                if area > best_area:
                    best_area, best = area, eidx
            if best is not None:
                entries[best]["uri"] = uri


# =====================================================================
#  入口
# =====================================================================
def _build_standard_refs(doc: dict) -> dict:
    """Shared reference-link pipeline for PyMuPDF and Unlimited-OCR."""
    pages = doc.get("pages") or []
    ocr_style = str(doc.get("parser") or "") in ("surya", "unlimited_ocr")
    info = {"ok": True, "scan": "ok", "section_page": None,
            "entries": [], "links": [], "counts": {}}
    if not isinstance((doc.get("parser_info") or {}).get("links_total"), int):
        info["scan"] = "no-links-data"        # 这篇还是“采集链接之前”的旧数据
    start = _find_references_start(pages)
    region = _region_blocks(pages, start)
    if not region:
        info["counts"] = {"entries": 0, "links": 0, "skipped": 0}
        return info
    entries, by_block = _build_entries(region)
    if start is not None:
        info["section_page"] = start[0]
    elif entries:
        info["section_page"] = entries[0]["page"]
    links, skipped = _map_links(pages, ocr_style, entries, by_block)
    if not links:
        links = _merge_links(
            _map_numeric_citations(pages, ocr_style, entries, by_block),
            _scan_author_year_citations(pages, ocr_style, entries, set(by_block)),
        )
    _attach_uris(pages, entries, by_block)
    info["entries"] = entries
    info["links"] = links
    info["counts"] = {"entries": len(entries), "links": len(links), "skipped": skipped}
    return info


def build_refs(doc: dict) -> dict:
    """Dispatch reference extraction to the strategy for this document parser."""
    parser = str(doc.get("parser") or "")
    if parser == "surya":
        from .refs_surya import build_surya_refs
        return build_surya_refs(doc)
    if parser == "unlimited_ocr":
        from .refs_unlimited_ocr import build_unlimited_ocr_refs
        return build_unlimited_ocr_refs(doc)
    from .refs_pymupdf import build_pymupdf_refs
    return build_pymupdf_refs(doc)
