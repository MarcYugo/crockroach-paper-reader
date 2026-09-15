"""后端一 · PyMuPDF 本地抽取(兜底)。

见包文档 `backend/pdf_parser/__init__.py`。
"""

from __future__ import annotations

import fitz  # PyMuPDF
from pathlib import Path
from typing import Optional

from .images import (_extract_raster, _extract_vector_figures,
                     _invisible_text_rects, _suppress_ghost_text)
from .options import (DEFAULT_OPTIONS)
from .text_layer import (_parse_text_blocks)


# =====================================================================
#  后端一 · PyMuPDF 本地抽取(兜底)
# =====================================================================
def _strip_font_keys(texts: list[dict]) -> None:
    """出页面前剥掉 runs 里的原始字体名(`f`)。

    它只服务于「公式候选区」判定(见 `_font_math_index`)，前端不认识、doc.json 也
    没必要为它变大 —— 于是解析完就丢掉，对外输出与改动前逐字一致。
    """
    for blk in texts or []:
        for ln in blk.get("lines") or []:
            for r in ln.get("runs") or []:
                r.pop("f", None)


def _local_page(page, doc, images_dir, pno: int, idx0: int = 0,
                opts: Optional[dict] = None) -> tuple[dict, int]:
    """纯 PyMuPDF 抽一页：文字块 + 内嵌点阵图 + 矢量图表。返回 (页数据, 下一图片序号)"""
    opts = opts or DEFAULT_OPTIONS
    texts = _parse_text_blocks(page)
    images = _extract_raster(page, doc, images_dir, pno, idx0=idx0)
    images += _extract_vector_figures(page, images_dir, pno, texts, images, idx0=idx0)
    # 图里已经有的文字不再叠一层渲染，否则会与图内字形错位成重影
    texts, ghost = _suppress_ghost_text(texts, images, _invisible_text_rects(page), opts)
    _strip_font_keys(texts)
    page_out = {
        "w": page.rect.width,
        "h": page.rect.height,
        "texts": texts,
        "images": images,
        "text": "\n\n".join(t["text"] for t in texts if t.get("text")),
    }
    if ghost:
        page_out["ghost_text_removed"] = ghost
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
    doc = fitz.open(str(pdf_path))
    try:
        if images_dir is not None:
            images_dir.mkdir(parents=True, exist_ok=True)

        pages_out = []
        ghost_total = 0
        for pno in range(doc.page_count):
            page = doc.load_page(pno)
            # 归一化旋转，统一坐标系(文字/点阵图/矢量区域一致)
            if page.rotation % 360 != 0:
                page.set_rotation(0)
            page_out, _idx = _local_page(page, doc, images_dir, pno, opts=opts)
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
            "parser_info": {
                "ghost_text_removed": ghost_total,   # 为去重影剔除的“图内文字”行数
            },
        }
    finally:
        doc.close()
