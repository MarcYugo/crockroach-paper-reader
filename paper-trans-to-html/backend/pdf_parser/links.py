"""PDF 链接注释采集（内部跳转 → 引用；外部 URI → 参考文献的外链）。

三个解析后端（`pymupdf` / `surya` / `unlimited_ocr`）都在 `entry.parse_pdf` 里
统一调用本模块 —— 链接注释来自 **PDF 本身**（LaTeX/hyperref 排的论文里，正文的
`[12]` 就是一个内部跳转链接、目标指向参考文献列表里那一条），与“文字版式来自
哪个后端”无关，所以采集一份、三个后端共用，写进：

    pages[].links = [
      {"kind": "goto", "rect": [x0, y0, x1, y1],        # 源侧矩形（PDF 坐标，pt）
       "page": 目标页(0-based), "to": [x, y]},           # 目标位置（可能没有）
      {"kind": "uri",  "rect": [x0, y0, x1, y1], "uri": "https://…"},
    ]

坐标口径与块 bbox 完全一致（PyMuPDF 的左上原点、单位 pt）：已实测确认
（PyMuPDF 1.28.2，`data/tmp/probe_links.py`）——

  * `page.get_links()` 的 `"page"` 是 **0-based**（用注解原始字典里的页对象号
    与各页 xref 对比验证，不依赖任何“插入时页码是几基”的假设）；
  * `"rect"` / `"to"` 均为左上原点坐标；`"to" == (0,0)` 表示该目标没有有效坐标
    （如 `/Fit` 式目标）—— 这种记作“没有 `to`”。

所以 `backend/pdf_refs/refs.py` 可以直接拿它们做“矩形 ∩ 块 / 目标点 → 条目”的几何匹配。

采集失败（打不开 PDF / 注释损坏）**绝不影响解析主流程**：返回空表即可。
"""
from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger("uvicorn.error")

__all__ = ["collect_links"]


def _rect4(rect) -> list | None:
    """fitz.Rect → `[x0, y0, x1, y1]`（pt，两位小数）；空矩形返回 None。"""
    try:
        x0, y0, x1, y1 = float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)
    except Exception:
        return None
    if not (x1 > x0 and y1 > y0):
        return None
    return [round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)]


def collect_links(pdf_path: str | Path) -> dict:
    """逐页采集链接注释 → `{页下标: [link, …]}`（没有链接的页不出现）。

    只收 `LINK_GOTO`（页内跳转，引用标记就是它）与 `LINK_URI`（参考文献条目上
    的 DOI/arXiv 外链）；`LINK_NAMED` 若被解析成 goto 也算 goto，未解析的跳过。
    """
    try:
        import fitz
    except ImportError:              # 解析器本身就依赖 PyMuPDF，理论上到不了这里
        return {}
    out: dict = {}
    try:
        doc = fitz.open(str(pdf_path))
    except Exception as exc:
        log.warning("链接注释采集：打不开 %s（%s）", pdf_path, exc)
        return {}
    try:
        n = int(doc.page_count)
        for pi in range(n):
            try:
                raw = doc[pi].get_links() or []
            except Exception:
                continue
            got = []
            for lk in raw:
                kind = lk.get("kind")
                rect = _rect4(lk.get("from")) if lk.get("from") is not None else None
                if rect is None:
                    continue
                if kind == fitz.LINK_GOTO:
                    tp = lk.get("page")
                    if not isinstance(tp, int) or not (0 <= tp < n):
                        continue
                    item = {"kind": "goto", "rect": rect, "page": tp}
                    pt = lk.get("to")
                    if pt is not None:
                        try:
                            x, y = float(pt.x), float(pt.y)
                            if x > 0.5 or y > 0.5:     # (0,0) = /Fit 式目标：没有有效坐标
                                item["to"] = [round(x, 2), round(y, 2)]
                        except Exception:
                            pass
                    got.append(item)
                elif kind == fitz.LINK_URI:
                    uri = str(lk.get("uri") or "").strip()
                    if uri:
                        got.append({"kind": "uri", "rect": rect, "uri": uri[:500]})
            if got:
                out[pi] = got
    finally:
        try:
            doc.close()
        except Exception:
            pass
    return out
