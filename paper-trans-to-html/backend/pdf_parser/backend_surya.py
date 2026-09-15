"""后端二 · Surya 2 推理服务(版面 + OCR)，行框由 PyMuPDF 补齐。

见包文档 `backend/pdf_parser/__init__.py`。
"""

from __future__ import annotations

import re
import fitz  # PyMuPDF
from pathlib import Path
from typing import Optional
from .. import surya_parser

from .backend_pymupdf import (_local_page, _pdf_outline, _strip_font_keys)
from .images import (_extract_raster, _extract_vector_figures,
                     _invisible_text_rects, _suppress_ghost_text)
from .text_layer import (_join_lines, _parse_text_blocks, _union_line_rect)


# =====================================================================
#  后端二 · Surya 2 推理服务(版面 + OCR)，行框由 PyMuPDF 补齐
# =====================================================================
def _page_lines(text_blocks: list[dict]) -> tuple[list[dict], list[int]]:
    """摊平 PyMuPDF 文字块 → (行列表, 每行所属原块索引)；行内保留 runs(字体样式)"""
    lines: list[dict] = []
    owner: list[int] = []
    for bi, blk in enumerate(text_blocks):
        for ln in blk.get("lines", []):
            lines.append(ln)
            owner.append(bi)
    return lines, owner

def _synth_lines(block: dict, rect: fitz.Rect) -> list[dict]:
    """无文本层(扫描件)时，用 OCR 文本按块高均分近似还原行框。"""
    parts = [p.strip() for p in re.split(r"\n+", block.get("text") or "") if p.strip()]
    if not parts:
        return []
    lh = max(1.0, rect.height / len(parts))
    size = max(6.0, min(48.0, round(lh * 0.78, 1)))
    return [{
        "x": rect.x0, "y": rect.y0 + i * lh, "w": rect.width, "h": lh,
        "runs": [{"t": t, "s": size, "fam": "serif", "b": False, "i": False,
                  "up": False, "c": "#000000"}],
    } for i, t in enumerate(parts)]


def _merge_surya_page(page, doc, img, blocks: list[dict], *, pno: int,
                      images_dir: Optional[Path], img_idx: int, opts: dict
                      ) -> tuple[dict, dict, int]:
    """把 Surya 的版面 block(渲染像素) + PyMuPDF 的精确行框(pt) 合并成一页数据。

    规则：
      * 图片类 block → 裁剪落盘为 images 条目(带 label/order 便于追溯)；
      * 其余 block → 取「中心落在该 block 内」的 PyMuPDF 行组成文字块(保留字体
        样式与精确位置)；没有文本层时用 OCR 文本合成行框；
      * Surya 没覆盖到的 PyMuPDF 行 → 单独成块追加，保证不漏文字；
      * 整页没有识别到插图 → 交由 PyMuPDF 的图片管线兜底。
    """
    sx = page.rect.width / float(img.width or 1)
    sy = page.rect.height / float(img.height or 1)
    lines, owner = _page_lines(_parse_text_blocks(page))
    used = [False] * len(lines)

    image_labels: set[str] = opts["image_labels"]
    pad = opts["image_pad"]
    texts: list[dict] = []
    images: list[dict] = []
    counter: dict[str, int] = {}

    for blk in blocks:
        label = blk.get("label") or "Text"
        counter[label] = counter.get(label, 0) + 1
        if blk.get("error"):
            continue
        box_px = blk.get("bbox")
        if not box_px:
            continue
        rect = fitz.Rect(box_px[0] * sx, box_px[1] * sy, box_px[2] * sx, box_px[3] * sy)

        # ---------- 图片类 block：只能按 label 判断(skipped/html 一定是空) ----------
        if label in image_labels:
            _bbox_px, crop = surya_parser.crop_block(img, bbox=box_px, pad=pad)
            if crop is None or min(crop.size) < opts["min_image_size"]:
                continue
            fname = surya_parser.crop_filename(pno, img_idx)
            if images_dir is not None:
                surya_parser.save_crop(crop, images_dir, fname)
            x0 = max(0.0, rect.x0 - pad * sx)
            y0 = max(0.0, rect.y0 - pad * sy)
            images.append({
                "x": x0, "y": y0,
                "w": min(page.rect.width - x0, rect.width + 2 * pad * sx),
                "h": min(page.rect.height - y0, rect.height + 2 * pad * sy),
                "file": fname,
                "label": label,
                "order": blk.get("reading_order"),
                "baked": True,      # 页面渲染裁剪，框内文字已烘焙进像素(去重影依据)
            })
            img_idx += 1
            continue

        # ---------- 文字 / 表格 / 公式类 block ----------
        picked = [i for i, ln in enumerate(lines)
                  if not used[i] and rect.contains(fitz.Point(
                      ln["x"] + ln["w"] / 2.0, ln["y"] + ln["h"] / 2.0))]
        entry: dict = {
            "label": label,
            "kind": blk.get("kind") or surya_parser.label_kind(label),
            "order": blk.get("reading_order"),
        }
        if picked:
            for i in picked:
                used[i] = True
            picked.sort(key=lambda i: (lines[i]["y"], lines[i]["x"]))
            sel = [lines[i] for i in picked]
            x0, y0, x1, y1 = _union_line_rect(sel)
            entry.update({"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0,
                          "text": _join_lines(sel), "lines": sel})
        else:
            body = (blk.get("text") or "").strip()
            if not body:
                continue      # 既不是图、又没有文字 → 丢弃(空白/装饰区域)
            entry.update({"x": rect.x0, "y": rect.y0, "w": rect.width,
                          "h": rect.height, "text": body,
                          "lines": _synth_lines(blk, rect)})
        # 公式块：Surya 的 <math> 里是 LaTeX 源码，抽出来交前端 MathJax 排版
        # (text 仍保留给翻译/兑底，两条路互不影响)
        if entry["kind"] == "formula":
            latex = surya_parser.latex_from_html(blk.get("html"))
            entry["latex"] = latex or (blk.get("text") or "").strip()
        if opts["keep_html"] and blk.get("html"):
            entry["html"] = blk["html"]
        texts.append(entry)

    # ---------- 兜底 1：Surya 漏掉的文字行 ----------
    leftovers: dict[int, list[dict]] = {}
    for i, ln in enumerate(lines):
        if not used[i]:
            leftovers.setdefault(owner[i], []).append(ln)
    for bi in sorted(leftovers):
        sel = sorted(leftovers[bi], key=lambda l: (l["y"], l["x"]))
        body = _join_lines(sel)
        if not body:
            continue
        x0, y0, x1, y1 = _union_line_rect(sel)
        texts.append({"label": "Text", "kind": "text",
                      "x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0,
                      "text": body, "lines": sel})

    # ---------- 兜底 2：整页没识别到插图 → 用 PyMuPDF 的图片管线找图 ----------
    if not images and opts["local_image_fallback"]:
        found = _extract_raster(page, doc, images_dir, pno, idx0=img_idx)
        found += _extract_vector_figures(page, images_dir, pno, texts, found, idx0=img_idx)
        for im in found:
            im["source"] = "pymupdf"
        images.extend(found)
        img_idx += len(found)

    # 兜底 3：剔除“已经烘焙进插图里”的文字行/块，避免与图内文字错位重影
    texts, ghost = _suppress_ghost_text(texts, images, _invisible_text_rects(page), opts)

    # 统一编号：id 供前端标注/笔记引用，须在页内稳定且唯一
    final_texts: list[dict] = []
    for i, t in enumerate(texts):
        if t.get("order") is None:
            t.pop("order", None)
        item = {"id": f"p{pno}b{i}"}
        item.update(t)
        final_texts.append(item)
    _strip_font_keys(final_texts)

    page_out = {
        "w": page.rect.width,
        "h": page.rect.height,
        "texts": final_texts,
        "images": images,
        "text": "\n\n".join(t["text"] for t in final_texts if t.get("text")),
        "image_size": [img.width, img.height],   # 渲染像素尺寸(便于核对坐标换算)
        "block_labels": counter,                 # 本页版面标签统计(排障用)
    }
    if ghost:
        page_out["ghost_text_removed"] = ghost   # 为去重影剔除的“图内文字”行数
    return page_out, counter, img_idx


def _parse_with_surya(pdf_path: Path, images_dir: Optional[Path], opts: dict) -> dict:
    """走 Surya 2 推理服务解析整份 PDF(逐页 OCR，内存占用可控)。"""
    client = surya_parser.SuryaClient(url=opts["surya_url"], backend=opts["surya_backend"])
    ok, msg = client.check()
    if not ok:
        raise surya_parser.SuryaUnavailable(msg)

    doc = fitz.open(str(pdf_path))
    try:
        if images_dir is not None:
            images_dir.mkdir(parents=True, exist_ok=True)
        pages_out: list[dict] = []
        labels_total: dict[str, int] = {}
        ghost_total = 0
        img_idx = 0
        for pno in range(doc.page_count):
            page = doc.load_page(pno)
            if page.rotation % 360 != 0:
                page.set_rotation(0)
            image = surya_parser.render_page_image(page, opts["dpi"])
            blocks = client.ocr_page(image)
            if blocks:
                page_out, counter, img_idx = _merge_surya_page(
                    page, doc, image, blocks, pno=pno, images_dir=images_dir,
                    img_idx=img_idx, opts=opts)
                for label, cnt in counter.items():
                    labels_total[label] = labels_total.get(label, 0) + cnt
            else:
                # 服务没返回任何 block(异常页)：整页退回本地抽取，保证不丢内容
                page_out, img_idx = _local_page(page, doc, images_dir, pno,
                                                idx0=img_idx, opts=opts)
                page_out["fallback"] = "pymupdf"
            ghost_total += page_out.get("ghost_text_removed", 0)
            pages_out.append(page_out)

        return {
            "num_pages": len(pages_out),
            "page_w": pages_out[0]["w"] if pages_out else 0,
            "page_h": pages_out[0]["h"] if pages_out else 0,
            "pdf_title": (doc.metadata or {}).get("title") or "",
            "pages": pages_out,
            "toc": _pdf_outline(doc),          # PDF 自带书签(没有就是空列表)
            "parser": "surya",
            "parser_info": {
                "url": opts["surya_url"],
                "backend": opts["surya_backend"],
                "dpi": opts["dpi"],
                "block_labels": labels_total,
                "ghost_text_removed": ghost_total,   # 为去重影剔除的行数
            },
        }
    finally:
        client.close()
        doc.close()
