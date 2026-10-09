"""插图：去重影、内嵌点阵图、矢量图形聚类/栅格化、给公式「挖洞」。

见包文档 `backend/pdf_parser/__init__.py`。
"""

from __future__ import annotations

import fitz  # PyMuPDF
from pathlib import Path
from typing import Optional

from .text_layer import (_join_lines, _text_line_rects, _union_line_rect)


_IMG_OK = {"png", "jpeg", "jpg", "webp", "gif"}

# =====================================================================
#  去重影 · 剔除“已经画在插图里”的文字行
# =====================================================================
#  图里本来就有的文字，如果再按文字层叠一遍，就会和插图里的字形错开(z 轴上下
#  两层)，看起来就是重影/双写。命中下面任一条的行会被删掉：
#    1) 整行落在**页面渲染类插图**里 —— Surya 裁剪图、矢量图栅格化图都是这一块
#       页面的像素快照，文字已经烘焙进图里了(插图带 baked=True)；
#    2) 整行落在**内嵌位图**里，且该行是**不可见文字**(render mode 3 / alpha=0)
#       —— 这是扫描件/图片的 OCR 文字层，只用于检索复制，看得见的笔画在底图里。
#  为避免误删贴边的正文，还要求“行框被插图覆盖 ≥ 阈值”且“行中心点在插图内”。
def _image_rect(im: dict) -> fitz.Rect:
    return fitz.Rect(im["x"], im["y"], im["x"] + im["w"], im["y"] + im["h"])


def _merge_rects(rects, gap: float = 1.0) -> list[fitz.Rect]:
    """合并相交(或相距小于 gap)的矩形，避免同一块面积被重复计算。"""
    out: list[fitz.Rect] = []
    pad = gap / 2.0
    for r in rects:
        cur = fitz.Rect(r)
        i = 0
        while i < len(out):
            if (cur + (-pad, -pad, pad, pad)).intersects(out[i]):
                cur = cur | out[i]
                out.pop(i)
                i = 0
                continue
            i += 1
        out.append(cur)
    return out


def _covered_ratio(rect: fitz.Rect, covers: list[fitz.Rect]) -> float:
    """rect 被 covers 覆盖的面积比例(0~1)；covers 需已合并，避免重复计数。"""
    area = abs(rect.get_area())
    if area <= 0:
        return 0.0
    return min(1.0, sum(_overlap_area(rect, c) for c in covers) / area)


def _invisible_text_rects(page) -> list[fitz.Rect]:
    """收集「不可见文字」的 span 外框(OCR 文字层)。

    PyMuPDF 的 get_text() 不看渲染模式，会把扫描件的隐形文字层一并抽出来；
    逐个 span 用 get_texttrace() 拿到渲染模式：3(既不填充也不描边)/7(仅裁剪)
    或不透明度为 0 的都算不可见。老版本 PyMuPDF 没有该接口时返回空列表，
    退化为“只按页面渲染类插图判断”。
    """
    trace = getattr(page, "get_texttrace", None)
    if trace is None:
        return []
    try:
        spans = trace()
    except Exception:  # pragma: no cover - 个别损坏页面
        return []

    rects: list[fitz.Rect] = []
    for sp in spans or []:
        try:
            mode = int(sp.get("type", 0))
        except (TypeError, ValueError):
            mode = 0
        invisible = mode in (3, 7)      # 3=invisible, 7=clip(不上色)
        if not invisible:
            alpha = sp.get("opacity")
            if isinstance(alpha, (int, float)) and float(alpha) <= 0.01:
                invisible = True
        if not invisible:
            continue
        bbox = sp.get("bbox")
        if not bbox:
            continue
        try:
            r = fitz.Rect(bbox)
        except Exception:
            continue
        if not r.is_empty:
            rects.append(r)
    return rects


def _suppress_ghost_text(texts: list[dict], images: list[dict],
                         invisible: list[fitz.Rect], opts: dict
                         ) -> tuple[list[dict], int]:
    """删掉“已经烘焙进插图”的文字行，返回 (新文字块, 删除的行数)。

    只动行，不动插图；某个文字块被删空就整块丢弃。文字块 bbox/纯文本会按
    剩余的行重算，保证前端定位与翻译内容都不受影响。
    """
    if not opts.get("drop_text_in_images") or not images or not texts:
        return texts, 0

    threshold = float(opts.get("text_in_image_overlap", 0.6))
    covers = _merge_rects([_image_rect(im) for im in images])
    baked = _merge_rects([_image_rect(im) for im in images if im.get("baked")])
    hidden = _merge_rects(invisible) if invisible else []

    def is_ghost(rect: fitz.Rect) -> bool:
        if rect.is_empty or _covered_ratio(rect, covers) < threshold:
            return False
        center = fitz.Point((rect.x0 + rect.x1) / 2.0, (rect.y0 + rect.y1) / 2.0)
        if not any(c.contains(center) for c in covers):
            return False                       # 行中心在插图外 → 只是贴边
        if baked and _covered_ratio(rect, baked) >= threshold:
            return True                        # 插图像素里已经有这行字
        return bool(hidden) and _covered_ratio(rect, hidden) >= threshold

    out: list[dict] = []
    removed = 0
    for blk in texts:
        old = blk.get("lines") or []
        if blk.get("kind") == "formula" and blk.get("latex"):
            # OCR 补上的公式块：版式就是原公式所在的位置，字形由 KaTeX 覆盖渲染，
            # 不做去重影(否则会被当成「图里的字」整块删掉，白补了)。
            out.append(blk)
            continue
        kept = [ln for ln in old
                if not is_ghost(fitz.Rect(ln["x"], ln["y"],
                                          ln["x"] + ln["w"], ln["y"] + ln["h"]))]
        if not kept:
            removed += len(old)
            continue                            # 整块都在图里 → 整块丢弃
        if len(kept) == len(old):
            out.append(blk)
            continue
        removed += len(old) - len(kept)
        blk = dict(blk)
        blk["lines"] = kept
        blk["text"] = _join_lines(kept)
        x0, y0, x1, y1 = _union_line_rect(kept)
        blk.update({"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0})
        out.append(blk)
    return out, removed


# =====================================================================
#  图片 · 内嵌点阵图
# =====================================================================
def _extract_raster(page, doc, images_dir, pno, idx0: int = 0) -> list[dict]:
    """抽取内嵌点阵图；文件名从 idx0 开始编号(便于与 Surya 裁剪图共用一个序号)。"""
    placed: list[dict] = []
    try:
        infos = page.get_image_info(xrefs=True)
    except Exception:
        return placed

    pw, ph = page.rect.width, page.rect.height
    page_area = pw * ph
    # 同一 bbox 存在多张(如 RGB + 软遮罩)时，保留“分辨率更高、非灰度”的一张
    best: dict[tuple, dict] = {}
    for info in infos:
        xref = info.get("xref") or -1
        bbox = info.get("bbox")
        if xref <= 0 or not bbox:
            continue
        x0, y0, x1, y1 = bbox
        w, h = x1 - x0, y1 - y0
        if w < 3 or h < 3:
            continue
        key = (round(x0, 1), round(y0, 1), round(w, 1), round(h, 1))
        iw, ih = info.get("width") or 0, info.get("height") or 0
        # colorspace 在 PyMuPDF 1.24+ 是分量数(int，1=灰度)，旧版是字符串；
        # cs-name 则一直是字符串，两者都兜住，避免 .lower() 直接报错。
        cs = info.get("colorspace")
        cs_name = str(info.get("cs-name") or "").strip().lower()
        gray = cs == 1 if isinstance(cs, int) else False
        if not gray:
            gray = "gray" in cs_name or "gray" in str(cs).lower()
        pref = 0 if gray else 1
        score = (iw * ih, pref)
        cur = best.get(key)
        if cur is None or score > cur[0]:
            best[key] = (score, info)

    used_files: dict[int, str] = {}
    draw_rects: Optional[list[fitz.Rect]] = None   # 懒加载：这一页的矢量图形
    for (_score, info) in best.values():
        xref = info["xref"]
        x0, y0, x1, y1 = info["bbox"]
        w, h = x1 - x0, y1 - y0
        if w * h > 0.97 * page_area:  # 整页背景
            continue
        # 有些图是「点阵底图 + 矢量线条叠在上面」画出来的：坐标轴、折线、标注常常是**矢量**。
        # 只抽底图会得到一张空背景 —— 实测一张红折线图的底图是一块灰格纹，抽出来就只剩灰格纹，
        # 坐标轴和红线全丢。这类图改成按页面区域**栅格化**，把叠在上面的矢量一起拍进去。
        box = fitz.Rect(x0, y0, x1, y1)
        if draw_rects is None:
            try:
                draw_rects = [fitz.Rect(d["rect"]) for d in page.get_drawings()
                              if d.get("rect") and not fitz.Rect(d["rect"]).is_empty
                              and (d.get("fill") is not None
                                   or d.get("color") is not None)]
            except Exception:
                draw_rects = []
        # 只对「不大」的图走合成渲染：整页扫描件走这条路会按 zoom 重渲一整页，太贵
        # （扫描件一般也没有矢量叠加，本来就会落到下面的抽原图分支）。
        if (w * h <= 0.25 * page_area
                and any(_overlap_area(box, dr) > 0.25 * (abs(dr.get_area()) or 1.0)
                        for dr in draw_rects)):
            # 按底图的原生像素数定缩放，尽量不掉清晰度
            zoom = min(6.0, max(2.0, (info.get("width") or 0) / max(w, 1.0)))
            fname = f"p{pno:03d}_i{idx0 + len(placed)}.png"
            if images_dir is not None:
                page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=box).save(
                    str(images_dir / fname))
            placed.append({"x": x0, "y": y0, "w": w, "h": h, "file": fname})
            continue
        try:
            im = doc.extract_image(xref)
        except Exception:
            continue
        if not im.get("image"):
            continue
        ext = (im.get("ext") or "png").lower()
        if ext not in _IMG_OK:
            ext = "png"
        if xref not in used_files:
            fname = f"p{pno:03d}_i{idx0 + len(placed)}.{ext}"
            if images_dir is not None:
                (images_dir / fname).write_bytes(im["image"])
            used_files[xref] = fname
        placed.append({"x": x0, "y": y0, "w": w, "h": h, "file": used_files[xref]})
    return placed


# =====================================================================
#  图片 · 矢量图形聚类 / 栅格化
# =====================================================================
#  判据对齐 `pymupdf_extract.py` 的 `page_figures`（同名样本、同一批图必须得到同一结果）。
#  关键差别：**候选不设逐元素门槛**，是不是图形由**合并后的区域**决定 —— 真图的线条
#  可以极细、节点可以很小，逐条卡线宽/面积会把整张图剔光。
_FIG_MIN_PARTS = 3        # 成图至少要有几个绘图元素（孤立细线只有 1~2 个）
_FIG_MIN_W = 30.0         # 图形区域最小宽（磅）
_FIG_MIN_H = 20.0         # 图形区域最小高（磅）
_FIG_GAP = 5.0            # 聚类间隙（磅）：两元素相距超过它就算两张图
_FIG_PAD = 2.0            # 栅格化外扩（磅）：描边压在区域边界上，不外扩会被裁掉半条


def _cluster_rects(rects, gap: float = 6.0) -> list[fitz.Rect]:
    """把彼此靠近的图形外框合并成若干“图形区域”。"""
    items = list(rects)
    items.sort(key=lambda r: r.width * r.height, reverse=True)
    regions: list[fitz.Rect] = []
    pad = gap / 2.0
    for r in items:
        rp = r + (-pad, -pad, pad, pad)
        merged = False
        for i in range(len(regions)):
            if rp.intersects(regions[i]):
                regions[i] = regions[i] | r  # 并集
                merged = True
                break
        if not merged:
            regions.append(fitz.Rect(r))
    return regions


def _overlap_area(a: fitz.Rect, b: fitz.Rect) -> float:
    r = a & b
    if r.is_empty:
        return 0.0
    return abs(r.get_area())


def _render_region_png(page, rect: fitz.Rect, images_dir, pno, idx) -> Optional[str]:
    try:
        z = min(3.0, max(1.0, 1500.0 / max(rect.width, rect.height, 1)))
        pix = page.get_pixmap(matrix=fitz.Matrix(z, z), clip=rect)
        fname = f"p{pno:03d}_i{idx}.png"
        if images_dir is not None:
            pix.save(str(images_dir / fname))
        return fname
    except Exception:
        return None


def _extract_vector_figures(page, images_dir, pno, texts, placed_raster,
                            idx0: int = 0) -> list[dict]:
    """识别矢量线条/色块绘制的图表，栅格化后作为图片块输出。"""
    out: list[dict] = []
    if not hasattr(page, "get_drawings"):
        return out
    try:
        drawings = list(page.get_drawings())
    except Exception:
        return out

    pw, ph = page.rect.width, page.rect.height
    # 候选形状**不设逐元素门槛**（照 `pymupdf_extract.page_figures` 的做法）。
    # 旧规则要求「填充色，或描边线宽 ≥0.15」「单元素 ≥12x12 且面积 ≥1600」，实测
    # samples/latex_sample.pdf 的流程图**一条都过不了**：图外框线宽 0.058~0.121pt，
    # 节点圆只有 700~1400pt² —— 于是 shapes 为空、p3/p4 的矢量图从来没被抽出来。
    # 「是不是图形」交给合并后的区域判（见下面的元素数 / 最小区域 / 二维尺寸三闸）。
    shapes: list[fitz.Rect] = []
    for d in drawings:
        r = d.get("rect")
        if not r or r.is_empty:
            continue
        r = fitz.Rect(r)
        if r.get_area() > 0.96 * pw * ph:  # 整页色块/背景
            continue
        shapes.append(r)

    if len(shapes) < _FIG_MIN_PARTS:
        return out
    # 先剔掉「背景型」大色块：一整块占了大半页宽的填充矩形是页面底色/装饰框，
    # 不是图表。不剔掉的话它会**把周围零散的小图形(真图)串成一大片"伪图"**，
    # 接着去重影就按「插图像素里已经有这行字」把这片区域里的正文整段删掉
    # —— 实测一个 536x310 的浅色填充矩形让 141 行正文凭空消失。
    shapes = [r for r in shapes
              if not (r.width > 0.5 * pw and r.get_area() > 0.25 * pw * ph)]
    if len(shapes) < _FIG_MIN_PARTS:
        return out
    # 聚类用定点式的 `_merge_rects` 而不是单趟的 `_cluster_rects`，间隙取 5.0 —— 与脚本
    # `page_figures(gap=5.0)` 同一套。旧的 gap=8 会把两个子图并成一张（样本 p4 的两个
    # 子图相距 6.68pt，恰好卡在 5 和 8 之间，脚本输出 2 张图）。
    regions = _merge_rects(shapes, gap=_FIG_GAP)

    text_rects = _text_line_rects(texts)
    used = [fitz.Rect(it["x"], it["y"], it["x"] + it["w"], it["y"] + it["h"]) for it in placed_raster]

    idx = idx0 + len(placed_raster)
    for reg in regions:
        if reg.width < _FIG_MIN_W or reg.height < _FIG_MIN_H:
            continue
        # 区域里必须有**成组的二维**元素：分式线/根号横线/表格横线都是又长又扁的单条线，
        # 「元素数 ≥3」挡掉孤立细线，「至少一个二维元素」挡掉“长横线 + 端点圆点”这种
        # （实测表格横线：区域 154x0，先被最小高度挡掉）。
        inside = [r for r in shapes if r in reg]      # Rect in Rect 等价于被包含
        if len(inside) < _FIG_MIN_PARTS:
            continue
        if not any(r.width > 1 and r.height > 1 for r in inside):
            continue
        # 往外扩一点再栅格化：细线框的描边压在区域边界上，不外扩会被裁掉半条。
        # 扩出来的矩形同时就是输出的 x/y/w/h —— 前端把图铺在这个框上，两者必须一致。
        reg = fitz.Rect(reg.x0 - _FIG_PAD, reg.y0 - _FIG_PAD,
                        reg.x1 + _FIG_PAD, reg.y1 + _FIG_PAD)
        reg.intersect(page.rect)
        w, h = reg.width, reg.height
        # 与文字高度重叠 → 多为表格/公式/装饰，不作为独立图。
        # 但阈值要**分大小看**：小区域(≤3% 页面积)常常是「图里带框的文字/标签」，
        # 文字占比本来就高(实测 Figure 1 的框 39%~43%)；只有大区域才可能是
        # 「把正文圈进来」的伪图(实测 31%~35%)。所以小区域放宽到 70%，大区域仍 25%。
        area = abs(reg.get_area()) or 1.0
        txt_overlap = sum(_overlap_area(reg, tr) for tr in text_rects)
        limit = 0.70 if area <= 0.03 * pw * ph else 0.25
        if txt_overlap > limit * area:
            continue
        # 与已有点阵图几乎重合 → 不必重复
        img_cover = sum(_overlap_area(reg, ir) for ir in used)
        if img_cover > 0.6 * reg.get_area():
            continue
        fname = _render_region_png(page, reg, images_dir, pno, idx)
        if not fname:
            continue
        idx += 1
        # baked：这是页面的像素快照，区域内的文字已经烘焙进 PNG(去重影时以此为准)
        out.append({"x": reg.x0, "y": reg.y0, "w": w, "h": h,
                    "file": fname, "baked": True})
    return out

def _carve_images(images: list[dict], page, images_dir: Optional[Path],
                  pno: int, hole: fitz.Rect, idx: int) -> int:
    """把盖住 `hole` 的「烘焙图」挖个洞，返回新的图片序号。

    公式要交给 KaTeX 渲染，但它原来常常是画在矢量图表里的（页面上那块像素快照
    PNG 里就含着这条公式）—— 不挖掉就会和图里的公式**重影**。做法是把那张图按
    洞拆成上/下/左/右四块，各自重新栅格化（不需要动像素，也不会引入新依赖）。
    挖不动（没盖到 / 整张都被盖）时按结果自然处理。`images` 就地替换。
    """
    if not images or images_dir is None:
        return idx
    out: list[dict] = []
    for im in images:
        rect = _image_rect(im)
        if not im.get("baked") or not rect.intersects(hole):
            out.append(im)
            continue
        inter = rect & hole
        if inter.is_empty or inter.get_area() <= 0:
            out.append(im)
            continue
        pieces = [
            fitz.Rect(rect.x0, rect.y0, rect.x1, inter.y0),      # 上
            fitz.Rect(rect.x0, inter.y1, rect.x1, rect.y1),      # 下
            fitz.Rect(rect.x0, inter.y0, inter.x0, inter.y1),    # 左
            fitz.Rect(inter.x1, inter.y0, rect.x1, inter.y1),    # 右
        ]
        for piece in pieces:
            if piece.width < 2 or piece.height < 2:
                continue
            fname = _render_region_png(page, piece, images_dir, pno, idx)
            if not fname:
                continue
            idx += 1
            out.append({**im, "x": piece.x0, "y": piece.y0,
                        "w": piece.width, "h": piece.height, "file": fname})
    images[:] = out
    return idx
