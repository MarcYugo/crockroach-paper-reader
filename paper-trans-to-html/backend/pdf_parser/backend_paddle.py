"""后端三 · PaddleOCR-VL：整页合并 + 后端入口。

见包文档 `backend/pdf_parser/__init__.py`。
"""

from __future__ import annotations

import fitz  # PyMuPDF
from pathlib import Path
from typing import Optional
from .. import paddle_parser

from .backend_pymupdf import (_local_page, _pdf_outline, _strip_font_keys)
from .images import (_extract_raster, _extract_vector_figures,
                     _invisible_text_rects, _render_region_png,
                     _suppress_ghost_text)
from .math_block import (_accumulate_stats, _inject_ocr_math)
from .ocr_blocks import (_OCR_IMAGE_LABELS, _OCR_KIND, _OCR_MATH_LABELS,
                         _drop_lines_in_rects, _ocr_block_lines,
                         _synth_ocr_blocks, _text_layer_invisible,
                         extract_latex, looks_like_latex, parse_ocr_blocks)
from .text_layer import (_join_lines, _parse_text_blocks)


def _merge_paddle_page(page, doc, ocr_text: str, *, pno: int,
                       images_dir: Optional[Path], img_idx: int,
                       opts: dict, math_stats: Optional[dict] = None) -> tuple[dict, int]:
    """把一页的 PaddleOCR-VL 结果 + PyMuPDF 的版式/插图合并成一页数据。

    规则：
      * 有**可见**文字层的电子版 PDF：文字与版式一律以 PyMuPDF 的精确行框/字体样式
        为准，OCR **只补公式**(行内就地替换成 `$latex$`、独立公式合成 formula 块，
        见 `_inject_ocr_math`)与插图；
      * 纯扫描件(无可见文字层)：用 OCR 文本按页边距合成行框(近似版式)；
      * 若 OCR 输出带「标签 + 归一化坐标」(见 `parse_ocr_blocks`)：文字块按坐标还原
        行框(真实版式)，image 类块按坐标从页面栅格化成插图；
      * 整页没识别到插图时用 PyMuPDF 的图片管线兜底；
      * 最后统一去重影，剔除“已经烘焙进插图”的文字行。

    `math_stats` 非空时把本页注入的公式条数累加进去(供 parser_info 汇报)。
    """
    local_texts = _parse_text_blocks(page)
    visible_layer = bool(local_texts) and not _text_layer_invisible(page, local_texts)
    blocks = parse_ocr_blocks(ocr_text)

    texts: list[dict] = []
    images: list[dict] = []

    if blocks:
        # 模型坐标为 0~1000 的归一化值(每轴独立，与渲染 DPI 无关)：
        # pt = 归一化值 / 1000 × 页面尺寸
        sx = page.rect.width / 1000.0
        sy = page.rect.height / 1000.0
        formula_rects: list[fitz.Rect] = []
        for bi, blk in enumerate(blocks):
            label = blk["label"]
            bx0, by0, bx1, by1 = blk["bbox"]
            rect = fitz.Rect(bx0 * sx, by0 * sy, bx1 * sx, by1 * sy)

            # ---------- 图片类块：按坐标裁成 PNG(页面像素快照) ----------
            if label in _OCR_IMAGE_LABELS:
                if (rect.width < opts["min_image_size"]
                        or rect.height < opts["min_image_size"]):
                    continue
                fname = _render_region_png(page, rect, images_dir, pno, img_idx)
                if not fname:
                    continue
                images.append({"x": rect.x0, "y": rect.y0, "w": rect.width,
                               "h": rect.height, "file": fname, "label": label,
                               "baked": True})
                img_idx += 1
                continue

            # ---------- 文字类块 ----------
            body = (blk["text"] or "").strip()
            if not body and label not in _OCR_MATH_LABELS:
                continue
            # 公式：标签就是公式，或整块基本被数学定界符包住 → 抽 LaTeX 给前端
            # (MathJax)；正文里的行内提及不算整块公式，避免误伤正常段落。
            kind = _OCR_KIND.get(label, "text")
            latex = ""
            if label in _OCR_MATH_LABELS:
                latex = extract_latex(body)[0] or body
                kind = "formula"
            else:
                cand, ratio = extract_latex(body)
                if cand and ratio >= 0.6:
                    latex, kind = cand, "formula"
                elif looks_like_latex(body):        # 无定界符的裸 LaTeX
                    latex, kind = body, "formula"

            if kind == "formula" and latex:
                # 公式块：不管页面有没有可信文字层，都用 OCR 的 LaTeX 渲染
                lines = _ocr_block_lines(body, rect) or [{
                    "x": rect.x0, "y": rect.y0, "w": rect.width, "h": rect.height,
                    "runs": [{"t": body or latex, "s": 10, "fam": "serif",
                              "b": False, "i": False, "up": False, "c": "#000000"}],
                }]
                texts.append({
                    "id": f"p{pno}b{bi}", "label": label, "kind": "formula",
                    "x": rect.x0, "y": rect.y0, "w": rect.width, "h": rect.height,
                    "text": _join_lines(lines), "lines": lines, "latex": latex,
                })
                formula_rects.append(rect)
                continue

            if visible_layer:
                continue            # 电子版的非公式文字：交给 PyMuPDF，避免 OCR 噪声
            lines = _ocr_block_lines(body, rect)
            if not lines:
                continue
            texts.append({
                "id": f"p{pno}b{bi}", "label": label,
                "kind": _OCR_KIND.get(label, "text"),
                "x": rect.x0, "y": rect.y0, "w": rect.width, "h": rect.height,
                "text": _join_lines(lines), "lines": lines,
            })
        if visible_layer:
            # 本地文字层可信 → 用它；但公式区域改由 OCR 的 LaTeX 渲染，把那些区域里的
            # 本地文字行删掉(免得乱码字形与渲染后的公式叠在一起)。
            texts = _drop_lines_in_rects(local_texts, formula_rects) + texts

        # 模型没识别到插图 → 用 PyMuPDF 的图片管线兜底(内嵌位图 + 矢量图)
        if not images and opts["local_image_fallback"]:
            found = _extract_raster(page, doc, images_dir, pno, idx0=img_idx)
            found += _extract_vector_figures(page, images_dir, pno, texts,
                                             found, idx0=img_idx)
            for im in found:
                im["source"] = "pymupdf"
            images.extend(found)
            img_idx += len(found)
    else:
        # 输出不含坐标格式：整页纯文本 + PyMuPDF 图片管线
        images = _extract_raster(page, doc, images_dir, pno, idx0=img_idx)
        images += _extract_vector_figures(page, images_dir, pno, local_texts,
                                          images, idx0=img_idx)
        img_idx += len(images)
        if visible_layer:
            # 本地文字层可信 → 正文/版式一律用它；但**公式**用 OCR 的 LaTeX 补
            # (行内就地替换、跨块/独立公式合成 formula 块)，本地行框一个都不动。
            if opts["paddle_ocr_math"]:
                mstats, img_idx = _inject_ocr_math(
                    local_texts, ocr_text, pno=pno, page=page, images=images,
                    images_dir=images_dir, img_idx=img_idx)
                _accumulate_stats(math_stats, mstats)
            texts = local_texts
        else:
            synth = _synth_ocr_blocks(ocr_text, page, pno)
            texts = synth or local_texts

    texts, ghost = _suppress_ghost_text(texts, images, _invisible_text_rects(page), opts)

    final_texts: list[dict] = []
    for i, t in enumerate(texts):
        if not t.get("id"):
            t = {"id": f"p{pno}b{i}", **t}
        final_texts.append(t)
    _strip_font_keys(final_texts)

    page_out = {
        "w": page.rect.width,
        "h": page.rect.height,
        "texts": final_texts,
        "images": images,
        "text": "\n\n".join(t["text"] for t in final_texts if t.get("text")),
        "ocr_chars": len((ocr_text or "").strip()),
    }
    if ghost:
        page_out["ghost_text_removed"] = ghost
    if visible_layer:
        page_out["layout"] = "pymupdf"          # 文字层可信，OCR 只补了插图
    elif blocks:
        page_out["layout"] = "ocr-blocks"       # 按模型给的坐标还原版式
    else:
        page_out["layout"] = "ocr-synth"        # 无坐标格式，按文本合成版式
    return page_out, img_idx


def _parse_with_paddle(pdf_path: Path, images_dir: Optional[Path],
                       opts: dict) -> dict:
    """走 PaddleOCR-VL 推理服务解析整份 PDF(逐页 OCR，图片由 PyMuPDF 补齐)。

    服务端只吃图片，所以这里用同一个 PyMuPDF 文档**逐页渲染 → 送模型 → 合并版式**。
    单页请求失败只让该页退回本地抽取(计入 parser_info.ocr_failed_pages)；整份一页
    都没成功才抛错，让上层按回退链换后端。
    """
    client = paddle_parser.PaddleClient(
        url=opts["paddle_url"], model=opts["paddle_model"],
        api_key=opts["paddle_api_key"], timeout=opts["paddle_timeout"])
    ok, msg = client.check()
    if not ok:
        raise paddle_parser.PaddleUnavailable(msg)

    doc = fitz.open(str(pdf_path))
    try:
        if images_dir is not None:
            images_dir.mkdir(parents=True, exist_ok=True)
        pages_out: list[dict] = []
        img_idx = 0
        ghost_total = 0
        ocr_pages = 0
        ocr_chars = 0
        failed = 0
        math_total: dict = {"inline": 0, "display": 0}
        for pno in range(doc.page_count):
            page = doc.load_page(pno)
            if page.rotation % 360 != 0:
                page.set_rotation(0)
            ocr_text = ""
            try:
                png = paddle_parser.render_page_png(page, opts["dpi"])
                ocr_text = (client.ocr_image(
                    png, prompt=opts["paddle_prompt"],
                    max_tokens=opts["paddle_max_tokens"],
                    timeout=opts["paddle_timeout"]) or "").strip()
            except paddle_parser.PaddleUnavailable:
                failed += 1
            if ocr_text:
                page_out, img_idx = _merge_paddle_page(
                    page, doc, ocr_text, pno=pno, images_dir=images_dir,
                    img_idx=img_idx, opts=opts, math_stats=math_total)
                ocr_pages += 1
                ocr_chars += len(ocr_text)
            else:
                # 该页没拿到 OCR 文本(异常页)：整页退回本地抽取，保证不丢内容
                page_out, img_idx = _local_page(page, doc, images_dir, pno,
                                                idx0=img_idx, opts=opts)
                page_out["fallback"] = "pymupdf"
            ghost_total += page_out.get("ghost_text_removed", 0)
            pages_out.append(page_out)

        if doc.page_count and not ocr_pages:
            # 一页都没成功 → 服务/模型有问题：让上层按回退链换后端，
            # 别把纯 PyMuPDF 的结果标成 paddle 后端。
            raise paddle_parser.PaddleUnavailable(
                f"PaddleOCR-VL 未返回任何可用文本({failed}/{doc.page_count} 页失败)")

        return {
            "num_pages": len(pages_out),
            "page_w": pages_out[0]["w"] if pages_out else 0,
            "page_h": pages_out[0]["h"] if pages_out else 0,
            "pdf_title": (doc.metadata or {}).get("title") or "",
            "pages": pages_out,
            "toc": _pdf_outline(doc),          # PDF 自带书签(没有就是空列表)
            "parser": "paddle",
            "parser_info": {
                "url": opts["paddle_url"],
                "model": opts["paddle_model"],
                "prompt": opts["paddle_prompt"],
                "max_tokens": opts["paddle_max_tokens"],
                "dpi": opts["dpi"],
                "ocr_pages": ocr_pages,        # 走 OCR 的页数(其余页为本地兜底)
                "ocr_chars": ocr_chars,        # OCR 文本总字符数
                "ocr_failed_pages": failed,    # 请求失败、退回本地的页数
                "math_inline": math_total["inline"],   # 注入的行内公式条数
                "math_inline_split": math_total.get("inline_split", 0),
                                                       # 其中跨块(切行+合成覆盖块)的条数
                "math_display": math_total["display"],  # 注入的独立公式条数
                # OCR 给出了 LaTeX、但**没能在本地文本里定位到**的公式条数：它们的
                # 字形会原样留在正文里(覆盖不到)。是「公式没盖住原文」的直接指标。
                "math_anchor_missed": math_total.get("anchor_missed", 0),
                # 定位质量(字体候选区，独立于 OCR)：标出 math_font_lines 行像公式的
                # 字形，其中 math_font_used 行被 OCR 的公式用上；剩下的 math_font_missed
                # 既是潜在漏检(OCR 没给 LaTeX)，也可能是字体误检。
                "math_font_lines": math_total.get("font_lines", 0),
                "math_font_used": math_total.get("font_used", 0),
                "math_font_missed": math_total.get("font_missed", 0),
                "math_font_weak_pages": math_total.get("font_weak_pages", 0),
                # 公式落地之后仍在正文里的数学字形行数/字符数 —— 「公式没盖住原文」
                # 的直接指标，正常应远小于 math_font_lines。
                "math_leftover_lines": math_total.get("leftover_lines", 0),
                "math_leftover_chars": math_total.get("leftover_chars", 0),
                "ghost_text_removed": ghost_total,
            },
        }
    finally:
        client.close()
        doc.close()
