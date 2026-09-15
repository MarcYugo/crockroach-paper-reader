"""公式文本工具：LaTeX 定界符解析、锚点规整化、样式、字符索引。

见包文档 `backend/pdf_parser/__init__.py`。
"""

from __future__ import annotations

import re
from typing import Optional


_NORM_CHAR_RE = re.compile(r"[0-9A-Za-z\u3400-\u4dbf\u4e00-\u9fff\ufb00-\ufb06]")
# 「算字形」的字符类(见 `_has_glyphs` / `_tighten`)。**比 `_NORM_CHAR_RE` 宽**：
# 后者是"锚点匹配用的规整字母表"(只留拉丁/数字/汉字，OCR 那边的 `\sigma` 也只是
# 字母序列)，这里要回答的是"这截原字形算不算有内容"，必须认得出 σ/α/×/−/∑/′/⟨ 这些
# **只出现在公式里**的字形。两者用途不同，故意不合并。
_GLYPH_CLS = (
    "0-9A-Za-z\u3400-\u4dbf\u4e00-\u9fff\ufb00-\ufb06"    # 拉丁 / 汉字 / 连字
    "\u0370-\u03ff\u1d00-\u1d7f\u1d400-\u1d7ff"            # 希腊字母 / 数学字母
    "\u2032-\u2037"                                        # ′ ″ ‴ ‵ ‶ ‷
    "\u2070-\u209f"                                        # 上标 / 下标
    "\u2100-\u214f"                                        # ℘ ℝ ℓ 等
    "\u2190-\u21ff\u2300-\u23ff"                           # 箭头 / ⌈ ⌉ ⟨ ⟩
    "\u2200-\u22ff\u27c0-\u27ef\u2980-\u2aff"              # 数学算子 / 数学括号
)
_GLYPH_RE = re.compile("[" + _GLYPH_CLS + "]")
# `_tighten` 用来把开头的"空白 + 标点"整段往后让，所以要的是**反类**
_NON_GLYPH_PREFIX_RE = re.compile("[^" + _GLYPH_CLS + "]*")
# 连字(ligature)展开表。PDF 里 "ﬁnance" 取到的是 U+FB01 一个字符，OCR 给的是
# "finance" 七个字母 —— 不展开就永远对不上锚点(实测 "Sharpe ratio … In ﬁnance …"
# 与 "stratiﬁed backtests" 两条 OCR 行都因此整条报废)。前端 `_normText` 里有同一张表，
# 两边口径必须一致，否则注入会把公式摆到错的地方。
_LIG_EXPAND = {"\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi",
               "\ufb04": "ffl", "\ufb05": "st", "\ufb06": "st"}
# 独立公式：\[…\] / $$…$$(可跨行)
_DISPLAY_MATH_RE = re.compile(r"\\\[(.+?)\\\]|\$\$(.+?)\$\$", re.S)
# 行内公式：独立公式占位符 / \(…\) / $…$
_LINE_MATH_TOKEN_RE = re.compile(r"\x00(\d+)\x00|\\\((.+?)\\\)|\$([^$\n]{1,200}?)\$")
_STYLE_KEYS = ("s", "fam", "b", "i", "up", "c")
# 行内公式锚点的候选长度(规整字符数)：由长到短试，取第一个**唯一**命中的那一档
_ANCHOR_LENS = (48, 32, 20, 12, 8)
# 左右锚点加起来至少要这么多规整字符，否则没有区分度。**不能要求两边各自都够长**：
# 公式前面常常只有一个短词(`and \( \sigma_p \) is the annualized …`)，卡"每边 ≥4"
# 会让这类公式永远配不上锚点、碎片留在正文里(实测页 5 的 `σ_p` 就是这样丢的)。
_ANCHOR_MIN_TOTAL = 8
# 两个锚点之间(也就是公式字形)在整页规整文本里最多能隔多少字符：超过就是锚点对错了
_ANCHOR_SPAN_MAX = 200
# 多条锚点候选都说得通时，用「落点字形 vs LaTeX」的相似度打分挑一条：最低分与
# 相对次高名的最低差距。两个阈值都踩过：页 2 两条 `\[…\]` 的正确落点得分
# 1.00 / 0.92，错误落点 0.39 / 0.23。
_ANCHOR_RANK_MIN = 0.35
_ANCHOR_RANK_GAP = 0.15
# 一条跨块公式最多允许吞掉几行本地文字行(再多就当成锚点定位错了)
_SPAN_MAX_LINES = 12


def _norm_keep(s: str) -> str:
    """匹配用规整：只留字母/数字/汉字并转小写(空白与标点全丢掉)；连字先展开。"""
    return "".join(_LIG_EXPAND.get(c, c)
                   for c in _NORM_CHAR_RE.findall(s or "")).lower()


def _norm_keep_idx(s: str) -> tuple[str, list[int]]:
    """同 `_norm_keep`，另外给出每个规整字符在原串里的下标(用来回切原始文本)。

    ⚠️ 连字会展开成 2 个字符(如 "ﬁ"→"fi")，这两个字符的下标**相同**(都指向那个
    连字)，所以返回的 `idx` 是**非严格递增**的；调用方只用它做区间切分，不受影响。
    """
    chars: list[str] = []
    idx: list[int] = []
    for m in _NORM_CHAR_RE.finditer(s or ""):
        for ch in _LIG_EXPAND.get(m.group(0), m.group(0)):
            chars.append(ch.lower())
            idx.append(m.start())
    return "".join(chars), idx


def _looks_math_body(body: str) -> bool:
    """单 `$…$` 的保守判定(货币符号也写成 $)：首尾不接空格且含 \\ _ ^ { } 之一。"""
    s = body or ""
    if not s or s[:1].isspace() or s[-1:].isspace():
        return False
    return bool(re.search(r"[\\_^{}]", s))


def _same_style(a: dict, b: dict) -> bool:
    return all(a.get(k) == b.get(k) for k in _STYLE_KEYS)


def _body_run(runs: list[dict]) -> dict:
    """行内公式用哪个样式渲染：取字符最多的那个 run(正文字体)，没有就取第一个。"""
    if runs:
        return max(runs, key=lambda r: len(r.get("t") or ""))
    return {"t": "", "s": 10.0, "fam": "serif", "b": False, "i": False,
            "up": False, "c": "#000000"}


def _split_line_math(line: str, keep: list[str]) -> Optional[list]:
    """把一行切成 [文字, {latex}, 文字, …]（保证首尾都是文字段，公式都在奇数位）。"""
    segs: list = []
    last = 0
    hit = False
    for m in _LINE_MATH_TOKEN_RE.finditer(line):
        if m.group(1) is not None:                 # 独立公式占位符
            latex = keep[int(m.group(1))]
        elif m.group(2) is not None:               # \(…\)
            latex = m.group(2).strip()
        else:                                      # $…$
            if not _looks_math_body(m.group(3)):
                continue                           # 像货币符号 → 当普通文字
            latex = (m.group(3) or "").strip()
        if not latex:
            continue
        if m.start() > last:
            segs.append(line[last:m.start()])
        segs.append({"latex": latex})
        last = m.end()
        hit = True
    if not hit:
        return None
    if last < len(line):
        segs.append(line[last:])
    if not segs or not any(isinstance(s, dict) for s in segs):
        return None
    if isinstance(segs[0], dict):
        segs.insert(0, "")
    if isinstance(segs[-1], dict):
        segs.append("")
    return segs


def _ocr_math_lines(ocr_text: str) -> list[dict]:
    """把整页 OCR 文本切成「行」，并标出每行的公式。

    返回 `[{raw, segs, plain}]`：`segs` 是 `[文字, {"latex":…}, 文字, …]` 交替的片段表
    (公式都在奇数位)，该行没有公式时为 None；`plain` 是该行去掉公式后的规整文字。
    独立公式常跨行，所以先整体抽成占位符再按行切。

    ⚠️ **没有公式的行也要返回** —— 独立公式靠前后句子的文字定位，把普通行丢掉就找不到
    锚点了(只有一个 `$$…$$` 段落的页面就是这种情况)。
    """
    src = (ocr_text or "").strip()
    if not src:
        return []
    keep: list[str] = []

    def _stash(m):
        body = next((g for g in m.groups() if g is not None), "")
        keep.append(body.strip())
        return f"\x00{len(keep) - 1}\x00"

    src = _DISPLAY_MATH_RE.sub(_stash, src)
    out: list[dict] = []
    for raw_line in src.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        segs = _split_line_math(line, keep)
        if segs:
            texts = [_norm_keep(s) for k, s in enumerate(segs) if k % 2 == 0]
            plain = "".join(texts)
            plains = [t for t in texts if t]
        else:
            plain = _norm_keep(line)
            plains = [plain] if plain else []
        out.append({"raw": line, "segs": segs, "plain": plain,
                    "plains": plains})
    return out


def _gap_ok(gap: str, latex: str, plain: str = "") -> bool:
    """两段锚点之间那截是不是「这条公式的字形」：短，且不像成句的正文。

    `plain` 是该 OCR 行去掉公式后的规整文字：本地那截里若冒出**这行正文里没有的
    单词**，说明锚点对错了地方(要删的是正文而不是公式) → 判为不匹配。

    ⚠️ 判"有没有"时必须**连 LaTeX 一起看**：公式字形本身也是字母拼出来的，
    `close_{t,s}` 在本地被拼成 `closet,s`，`closet` 这个"词"当然不在正文里 ——
    只看正文就会把**正确**的定位判成错的，整条公式放弃、碎片留在页面上
    (实测页 2 的 `close_{t,s}` 就是这么丢的)。
    """
    g = (gap or "").strip()
    if len(g) > 160:
        return False
    words = re.findall(r"[A-Za-z]{4,}", g)
    if len(words) > 1:
        return False
    if words and words[0].lower() not in (plain or "") + _norm_keep(latex):
        return False
    return bool(g) or bool((latex or "").strip())


def _page_index(local_texts: list[dict]) -> tuple[list[dict], str]:
    """把本地各行拼成**整页**规整文本，并记录每行的规整区间与 raw 下标映射。

    OCR 按句子给文本、PDF 按行排版，公式前后的锚点文字常常跨行，所以匹配在整页
    规整文本上做；再用 `_meta_at()` 把匹配位置映射回具体的行与字符。
    """
    metas: list[dict] = []
    parts: list[str] = []
    pos = 0
    for bi, blk in enumerate(local_texts):
        for li, ln in enumerate(blk.get("lines") or []):
            raw = "".join(r.get("t") or "" for r in ln.get("runs") or [])
            norm, ridx = _norm_keep_idx(raw)
            metas.append({"bi": bi, "li": li, "idx": len(metas), "raw": raw,
                          "norm": norm, "ridx": ridx, "base": pos})
            parts.append(norm)
            pos += len(norm)
    return metas, "".join(parts)


def _meta_at(metas: list[dict], pos: int) -> tuple[Optional[dict], int]:
    """规整下标 pos 落在哪一行；返回 (行, 行内规整下标)；越界返回 (None, -1)。"""
    lo, hi = 0, len(metas) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        m = metas[mid]
        if pos < m["base"]:
            hi = mid - 1
        elif pos >= m["base"] + len(m["norm"]):
            lo = mid + 1
        else:
            return m, pos - m["base"]
    return None, -1


def _apply_replacements(runs: list[dict], raw: str, repls: list,
                        body: dict) -> Optional[list[dict]]:
    """按 [(a, b, latex), …] 重建 runs。

    区间 [a, b) 换成 `\\(latex\\)`(用正文字体)，区间**外**的原文与样式(粗体/斜体/
    字号/上标)原样保留 —— 所以前后空白、标点都不会丢。区间重叠则返回 None(整行不动)。

    ⚠️ 定界符用 `\\(…\\)` 而**不是** `$…$`：前端对单个 `$` 有「看起来像数学」的额外
    判定(见 `_splitInlineMath`)，像 `$h = 1$` 这种不含 \\ _ ^ { } 的会被当普通文字，
    页面上就显示成带美元符号的原文了；`\\(…\\)` 没有这层判定，一律渲染。
    """
    styles: list[dict] = []
    for r in runs:
        styles.extend([r] * len(r.get("t") or ""))
    out: list[dict] = []

    def _push(txt: str, style: dict) -> None:
        if not txt:
            return
        if out and _same_style(out[-1], style):
            out[-1]["t"] += txt
        else:
            out.append({"t": txt, "s": style.get("s", 10.0),
                        "fam": style.get("fam", "serif"),
                        "b": bool(style.get("b")), "i": bool(style.get("i")),
                        "up": bool(style.get("up")),
                        "c": style.get("c", "#000000")})

    cur = 0
    for a, b, latex in sorted(repls):
        if a < cur or b > len(raw) or a > b:
            return None
        for i in range(cur, a):
            _push(raw[i], styles[i])
        _push(f"\\({latex}\\)", body)
        cur = b
    for i in range(cur, len(raw)):
        _push(raw[i], styles[i])
    return out


def _all_pos(hay: str, needle: str) -> list[int]:
    """`needle` 在 `hay` 里的所有出现下标。"""
    out: list[int] = []
    st = 0
    while needle:
        j = hay.find(needle, st)
        if j < 0:
            break
        out.append(j)
        st = j + 1
    return out


def _plain_math(latex: str) -> str:
    """LaTeX 的「纯文本形态」：只剩字母数字，用来和切出来的原字形比对。

    前端 `r2-math.js::_plainMath` 有同一套规则(那边的用途是 Surya 的块内对齐)；
    这里服务于「锚点歧义消解」，两边口径一致才不会各判各的。
    """
    s = re.sub(r"\\[A-Za-z]+", " ", latex or "")
    s = re.sub(r"\\[ ,;:!]", " ", s)
    s = re.sub(r"[{}_^]", "", s)
    return re.sub(r"[^0-9A-Za-z]", "", s).lower()


def _bag_ratio(latex: str, glyphs: str) -> float:
    """乱序字符多重集相似度(0~1)：这条 LaTeX 与这堆原字形"是不是一路货"。

    用**多重集**而不是顺序比对：LaTeX 里上下标的先后与 PDF 字形顺序天然不一致。
    用途见 `_find_anchor_pair(rank=…)`。
    """
    a = _plain_math(latex)
    b = re.sub(r"[^0-9A-Za-z]", "", glyphs or "").lower()
    if not a or not b:
        return 0.0
    ca, cb = _count_chars(a), _count_chars(b)
    inter = sum(min(v, cb.get(k, 0)) for k, v in ca.items())
    return 2.0 * inter / (len(a) + len(b))


def _count_chars(s: str) -> dict:
    out: dict[str, int] = {}
    for ch in s:
        out[ch] = out.get(ch, 0) + 1
    return out


def _lines_text(lines: list[dict]) -> str:
    """几行原字形拼成纯文本(只用于相似度比对)。"""
    return " ".join("".join(r.get("t") or "" for r in (ln.get("runs") or []))
                    for ln in lines or [])

def _has_glyphs(text: str) -> bool:
    """这段残留里有没有真的「字形」(字母/数字/汉字/希腊字母/数学符号)。

    ⚠️ **不能只认 ASCII**：公式字形里有大量非 ASCII 字形(σ α β × − ∑ ′ ⟨ ⟩)，
    把它们当成"标点"会让 `_tighten` 从中间切、`_has_glyphs` 判成"没有内容" ——
    实测 `and \\( \\sigma_p \\) is the …` 因此从 `p` 起切，σ 留在正文里，
    页面上出现 `σ` + `σp` 两个 sigma 叠在一起。真正的标点(,.;:()[]"'/%)
    都还是 ASCII，不在这一类里，照旧往前让给正文。
    """
    return bool(_GLYPH_RE.search(text or ""))


def _math_glyphs(text: str, latex: str, plain: str) -> bool:
    """这段本地字形就是「这条公式的字形」，而不是正文/别的句子(空段直接算过)。"""
    t = (text or "").strip()
    return not t or _gap_ok(t, latex, plain)


def _tighten(raw: str, a: int, b: int) -> tuple[int, int]:
    """把 `[a, b)` 收成**真正的公式字形区间**。

    * 开头的空白与标点往后让 —— 它们是正文的标点，不是公式字形。典型例子：
      `(zz800)6.` 里 "6" 才是公式字形，前面的 ")" 是括号的收尾；不往后让就会把
      ")" 一起删掉，变成 "(zz800"。
    * 结尾的空白也留给渲染层(KaTeX 自己带基线)，不然会把后面的空格吞掉。
    全部都是空白/标点时返回空区间，调用方按"没找到"处理。
    """
    m = _NON_GLYPH_PREFIX_RE.match(raw[a:b])
    if m:
        a += len(m.group(0))
    seg = raw[a:b]
    core = seg.strip()
    return a, a + len(core)

def _math_ish(text: str) -> bool:
    """像公式而不是像一句正文：长单词(≥4 字母)不超过 1 个。"""
    t = (text or "").strip()
    if not t:
        return False
    return len(re.findall(r"[A-Za-z]{4,}", t)) <= 1

def _anchor_of(ocr_lines: list[dict], idx: int, step: int) -> str:
    """取第 idx 行前后最近的一段**连续普通文字**当锚点(取尾部/头部 16 个规整字符)。

    ⚠️ 必须按「文字片段」取，不能把整行去掉公式后拼起来 —— 片段之间夹着行内公式，
    拼起来会跨过公式，在本地文本里根本找不到（本地那里是公式字形，不是空的）。
    """
    for i in range(idx + step, -1 if step < 0 else len(ocr_lines), step):
        plains = ocr_lines[i].get("plains") or []
        for p in (reversed(plains) if step < 0 else plains):
            if len(p) >= 4:
                return p[-16:] if step < 0 else p[:16]
    return ""
