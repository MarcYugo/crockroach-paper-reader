"""公式框与 Figure 图片增强 · detection-service-group（`formula_table_service_group` 服务组）。

新流程（PyMuPDF 主干 + 服务组补公式）：

  1. 本地侧把公式字形擦成**等长空格**（`backend_pymupdf._blank_math_and_prune`），
     `pages[]._formula_slots` 只用于决定哪些页可以落公式覆盖框；
  2. 全部页面渲染 PNG（`detection_service_group.collect_images`）→
     `POST /v1/router/predict`：yolov13 检测公式框 → pp-formulanet-plus-l 识别 LaTeX；
  3. 服务组 Figure 裁图与 PDF 坐标写入 `pages[].images`，供阅读器使用；
  4. 公式框写进 `pages[].formula_boxes`：

         {"bbox_norm": [x中心, y中心, 宽, 高],   # 归一化(YOLO 口径)
          "latex": "\\frac{…}",
          "score": 0.93, "class_name": "DisplayedFormulaLine"}

     前端（`frontend/js/r2-math.js`）按 `bbox_norm × 当前页面显示宽高` 在页面上覆盖
     KaTeX —— 公式内容由**服务组**给出，位置由**公式框**给出，两者都不依赖本地空格。

⚠️ 与旧「OCR 配对」流程的区别：**不再配对、不再内联文本、不改版式** ——
空格位保持原样（承载排版宽度），公式以覆盖层形式画在它上面。

⚠️ 失败一律**不抛异常**：服务组不可用/超时/报错都只是加一条 warning；文字版式仍使用
PyMuPDF 结果，图片不回退到 PyMuPDF 提取（见 `entry.parse_pdf` 的 `_formula_postprocess`）。

统计键（`parser_info`）：`formula_slots`（空位处数，只作展示/排障）、`formula_boxes`
（落地的公式框条数）、`formula_box_pages`（有公式框的页数）、`formula_box_failed`
（带识别错误的条数）、`formula_box_ms`（本次增强总耗时 ms），以及 `figure_images` /
`figure_image_pages` / `figure_image_failed`。
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import math
import time
from pathlib import Path

from .. import detection_service_group
from .images import _suppress_ghost_text


def _figure_image(detection: dict, page: dict, page_index: int,
                  image_index: int, images_dir: Path | None) -> dict | None:
    crop = detection.get("crop")
    if not isinstance(crop, dict):
        return None
    encoded = str(crop.get("data") or "").strip()
    if encoded.startswith("data:"):
        encoded = encoded.partition(",")[2]
    if not encoded or images_dir is None:
        return None

    image_format = str(crop.get("format") or "png").strip().lower()
    if image_format in ("jpg", "jpeg"):
        image_format = "jpg"
    elif image_format not in ("png", "webp"):
        return None
    try:
        data = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        return None
    if not data:
        return None

    bounds = detection.get("bbox_pdf")
    if isinstance(bounds, (list, tuple)) and len(bounds) == 4:
        try:
            x0, y0, x1, y1 = (float(v) for v in bounds)
        except (TypeError, ValueError):
            return None
    else:
        norm = detection.get("bbox_norm")
        if not isinstance(norm, (list, tuple)) or len(norm) != 4:
            return None
        try:
            cx, cy, width, height = (float(v) for v in norm)
        except (TypeError, ValueError):
            return None
        x0 = (cx - width / 2) * float(page["w"])
        y0 = (cy - height / 2) * float(page["h"])
        x1 = (cx + width / 2) * float(page["w"])
        y1 = (cy + height / 2) * float(page["h"])
    if not all(math.isfinite(v) for v in (x0, y0, x1, y1)) or x1 <= x0 or y1 <= y0:
        return None

    digest = hashlib.sha256(data).hexdigest()[:12]
    fname = f"p{page_index + 1:03d}_i{image_index}.{image_format}"
    images_dir.mkdir(parents=True, exist_ok=True)
    (images_dir / fname).write_bytes(data)
    return {
        "x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0,
        "file": fname, "source": "detection_service_group",
        "class_name": str(detection.get("class_name") or ""),
        "cache_key": digest,
    }


def apply_formula_boxes(result: dict, pdf_path: Path, opts: dict,
                        warnings: list[str], *, images_dir: Path | None = None,
                        predict=None) -> dict:
    """对 PyMuPDF 解析结果补充服务组公式框和 Figure 裁图。

    * 只处理**有公式空位**的页（`_formula_slots`）—— 没有擦除过公式的页即使检测到
      公式也没有对照物，画上覆盖层只会压住正文原字形；
    * 预测覆盖全部页面，以便检测公式以外的 Figure；公式框仍只写入有公式空位的页面；
    * `predict` 可注入（测试用桩函数），默认走 `detection_service_group.predict`；
    * 返回同一个 `result`（就地写 `pages[].images`、`pages[].formula_boxes` 和统计）。
    """
    stats = result.setdefault("parser_info", {})
    pages = result.get("pages") or []
    # 空位处数：只作展示/排障（旧流程也把它写在这里，工具与重解析摘要都在读）
    stats["formula_slots"] = sum(len(pg.get("_formula_slots") or []) for pg in pages)
    formula_pages = {i for i, pg in enumerate(pages) if pg.get("_formula_slots")}
    if not pages:
        return result

    stat = {"formula_boxes": 0, "formula_box_pages": 0,
            "formula_box_failed": 0, "formula_box_ms": 0,
            "figure_images": 0, "figure_image_pages": 0,
            "figure_image_failed": 0}
    t0 = time.time()
    try:
        page_numbers = range(1, len(pages) + 1)
        rendered = detection_service_group.collect_images(pdf_path, page_numbers, opts)
        # Figure 裁剪图是实际落盘的插图，不是可关闭的公式渲染兜底。
        if predict is None:
            resp = detection_service_group.predict(rendered, opts, keep_crops=True)
        else:
            resp = predict(rendered, opts)
    except Exception as exc:
        # 服务不可用/超时（`DetectionGroupUnavailable`）与渲染/回包结构等意外错误：
        # 一律只降级成一条 warning —— 绝不因为增强坏了让整篇解析失败。
        stat["formula_box_ms"] = int((time.time() - t0) * 1000)
        stats.update(stat)
        warnings.append(f"detection-service-group 公式框与图片增强已跳过：{exc}")
        return result

    found = detection_service_group.extract_formula_boxes(resp or {})
    for page_index, boxes in sorted(found.items()):
        if (page_index not in formula_pages or not (0 <= page_index < len(pages))
                or not boxes):
            continue
        pages[page_index]["formula_boxes"] = boxes
        stat["formula_boxes"] += len(boxes)
        stat["formula_box_pages"] += 1
        stat["formula_box_failed"] += sum(1 for b in boxes if b.get("error"))

    figures = detection_service_group.extract_figure_detections(
        resp or {}, opts.get("image_labels"))
    page_image_counts: dict[int, int] = {}
    image_masks: dict[int, list[dict]] = {}
    for detection in figures:
        try:
            page_index = int(detection.get("page_index"))
        except (TypeError, ValueError):
            stat["figure_image_failed"] += 1
            continue
        if not (0 <= page_index < len(pages)):
            stat["figure_image_failed"] += 1
            continue
        image_index = page_image_counts.get(page_index, 0)
        image = _figure_image(detection, pages[page_index], page_index,
                              image_index, images_dir)
        if image is None:
            stat["figure_image_failed"] += 1
            continue
        page_image_counts[page_index] = image_index + 1
        pages[page_index].setdefault("images", []).append(image)
        image_masks.setdefault(page_index, []).append({**image, "baked": True})
        stat["figure_images"] += 1

    stat["figure_image_pages"] = len(page_image_counts)
    if opts.get("drop_text_in_images"):
        for page_index, masks in image_masks.items():
            page = pages[page_index]
            page["texts"], removed = _suppress_ghost_text(
                page.get("texts") or [], masks, [], opts)
            page["text"] = "\n\n".join(
                text.get("text") or "" for text in page["texts"]
                if text.get("text"))
            if removed:
                page["ghost_text_removed"] = (
                    page.get("ghost_text_removed", 0) + removed)

    stat["formula_box_ms"] = int((time.time() - t0) * 1000)
    stats.update(stat)

    if not stat["formula_boxes"] and formula_pages:
        warnings.append("detection-service-group 未返回可用的公式框/LaTeX"
                        "（页面里的公式保持为空白）")
    elif stat["formula_box_failed"]:
        warnings.append(f"detection-service-group 有 {stat['formula_box_failed']} 条公式"
                        "识别失败（对应位置保持为空白）")
    if stat["figure_image_failed"]:
        warnings.append(f"detection-service-group 有 {stat['figure_image_failed']} 张图片"
                        "缺少有效裁剪图或页面坐标，未能显示")
    return result
