# -*- coding: utf-8 -*-
"""
YOLOv13 公式检测推理脚本：输入 PDF，输出 JSON + 可视化结果
============================================================

用法（在项目根目录执行）:
    python inference.py --pdf paper.pdf                       # 单份 PDF
    python inference.py --pdf paper.pdf --out outputs/paper    # 指定输出目录
    python inference.py --pdf pdfs/                            # 目录内所有 PDF
    python inference.py --pdf a.pdf b.pdf --weights runs/train/exp1/weights/best.pt
    python inference.py --pdf paper.pdf --dedup-thr 0.6        # 调整同类框去重阈值（默认 0.8）
    python inference.py --pdf paper.pdf --no-dedup             # 关闭同类框去重

默认权重:
    runs/train/exp1/weights/best.pt（若存在），否则自动在 runs/train/ 下查找
    第一份 weights/best.pt（例如 runs/train/arxivformula_yolov13l/weights/best.pt）。

输出（默认写在 outputs/<pdf 文件名>/ 下）:
    <stem>.json                 检测结果（每个检测项含坐标框、类别、置信度）
    vis_pages/page_0001.png ... 每页可视化图片（画框 + 类别 + 置信度）
    <stem>_vis.pdf              可视化 PDF（标注框和类别，页数/页面尺寸与原 PDF 一致）

JSON 结构:
    {
      "pdf":     {文件信息},
      "model":   {权重/推理参数},
      "classes": [{id, category_id, name}, ...],
      "pages":   [{page_index, page_number, width, height, pdf_width, pdf_height,
                   scale_x, scale_y, image, num_detections, num_duplicates_removed}, ...],
      "detections": [
        {
          "page_index": 0,            # 0 基页码
          "page_number": 1,           # 1 基页码
          "class_id": 0,              # 训练类别 index（0~5）
          "category_id": 1,           # info.json 原始 id = class_id + 1
          "class_name": "InlineFormula",
          "confidence": 0.93,
          "bbox": [x1, y1, x2, y2],   # 渲染图像素坐标（左上原点）
          "bbox_pdf": [x1, y1, x2, y2],  # PDF 点坐标（左上原点，供 PDF 阅读器直接使用）
          "bbox_norm": [x_c, y_c, w, h]  # 归一化 xywh（与 YOLO 标签一致）
        }, ...
      ],
      "dedup":   {enabled, overlap_threshold, removed_count,
                  removed: [检测项 + {overlap_ratio, iou, kept_bbox, kept_confidence}, ...]},
      "summary": {total_detections, duplicates_removed, per_class, ...}
    }
    detections 按阅读顺序（页 -> 上 -> 左）排序；注意 class_id + 1 才是 info.json 中的原始 id
    （与 train.py 中的说明一致）。

公式框检查（同类去重）:
    同一页 + 同一类别内，若两个框的重叠比 >= --dedup-thr（默认 0.8），只保留面积较大的那个，
    面积较小的框被判定为重复并移除；较大框本身不做合并/修改（面积相同时保留置信度更高的）。
    重叠比 = max(IoU, 交集 / 较小框面积)，因此"小框被大框包住"（同一行内公式被重复检出、
    公式编号/子式被大框套嵌等）也会被判为重复。被移除的框连同上交它的那个大框一并记入
    dedup.removed 便于核查，可用 --no-dedup 关闭，或 --dedup-thr 1.0 只去除完全被包含/重合的框。
    注意：保留哪个框只看面积（符合"较小的框去重"），因此大框的置信度可能低于被移除的小框。

依赖安装（如尚未安装）:
    pip install -r yolov13/requirements.txt
    pip install pymupdf pillow
    # 仅推理时的最小依赖（本仓库已装则可忽略）:
    pip install torch torchvision numpy opencv-python pyyaml psutil py-cpuinfo thop

调参提示:
    --dpi 默认 150（Letter/A4 约 1240~1275 px 宽，接近数据集页面渲染尺度）；调大更清晰但更慢；
    --imgsz 默认 640（与训练一致）；小字号行内公式漏检时可试 --imgsz 960/1280 并适当调低 --conf；
    --dedup-thr 默认 0.8，调大（如 0.95）只去除几乎完全重合的框，调小（如 0.6）判重更激进。
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
YOLOV13_ROOT = PROJECT_ROOT / "yolov13"  # 内置的 YOLOv13 源码目录

# 让脚本优先使用本仓库内置的 YOLOv13（而不是环境中可能已安装的其它 ultralytics）
if YOLOV13_ROOT.is_dir() and str(YOLOV13_ROOT) not in sys.path:
    sys.path.insert(0, str(YOLOV13_ROOT))

try:
    from ultralytics import YOLO
    from ultralytics.utils.plotting import Annotator, colors
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "导入 ultralytics 失败，请先安装依赖:\n"
        f"    pip install -r {YOLOV13_ROOT / 'requirements.txt'}\n"
        f"    pip install -e {YOLOV13_ROOT}\n"
        f"原始错误: {exc}"
    )

try:
    import pymupdf as fitz  # PyMuPDF >= 1.24 的推荐导入名
except ImportError:  # pragma: no cover
    try:
        import fitz  # 旧版本 PyMuPDF
    except ImportError as exc:
        raise SystemExit(f"渲染 PDF 需要 PyMuPDF，请先安装:  pip install pymupdf\n原始错误: {exc}")

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"保存可视化图片需要 Pillow，请先安装:  pip install pillow\n原始错误: {exc}")

# info.json 原始 id <-> 训练类别 index 的映射（当权重里没带 names 时兜底）
CLASS_MAP = {
    1: "InlineFormula",
    2: "DisplayedFormulaLine",
    3: "FormulaNumber",
    4: "DisplayedFormulaBlock",
    5: "Table",
    6: "Figure",
}
DEFAULT_WEIGHTS = PROJECT_ROOT / "runs" / "train" / "exp1" / "weights" / "best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv13 公式检测：PDF -> JSON + 可视化")
    parser.add_argument("--pdf", nargs="+", required=True, help="输入 PDF 文件或包含 PDF 的目录（可多个）")
    parser.add_argument(
        "--weights",
        default=None,
        help=f"模型权重（默认: {DEFAULT_WEIGHTS}；不存在时自动在 runs/train 下查找 best.pt）",
    )
    parser.add_argument("--out", default=None, help="输出目录（默认 outputs/<pdf 文件名>；多个输入时作为父目录）")
    parser.add_argument("--json", default=None, help="JSON 输出路径（仅单个输入时可用；默认 <out>/<stem>.json）")
    parser.add_argument("--dpi", type=int, default=150, help="PDF 渲染 DPI（默认 150；调大更清晰但更慢）")
    parser.add_argument("--imgsz", type=int, default=640, help="推理输入尺寸（默认 640，与训练一致）")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值（默认 0.25）")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU 阈值（默认 0.7）")
    parser.add_argument("--max-det", type=int, default=300, help="每页最多保留的检测框数（默认 300）")
    parser.add_argument("--batch", type=int, default=8, help="每批送入模型的页数（默认 8）")
    parser.add_argument("--device", default="", help='计算设备: 留空自动选择；"0" 第一块 GPU；"cpu"')
    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help="只保留指定类别，可用类别名或 id/category_id，如 --classes InlineFormula Table",
    )
    parser.add_argument("--line-width", type=int, default=2, help="可视化线宽（像素，默认 2）")
    parser.add_argument("--font-size", type=int, default=None, help="可视化字体大小（默认按图像尺寸自适应）")
    parser.add_argument("--no-vis", action="store_true", help="不输出可视化图片与可视化 PDF")
    parser.add_argument("--max-pages", type=int, default=0, help="最多处理前 N 页（0 表示全部）")
    parser.add_argument("--no-dedup", action="store_true", help="关闭同类框去重（默认开启）")
    parser.add_argument(
        "--dedup-thr",
        type=float,
        default=0.8,
        help="同类框去重阈值（0~1，默认 0.8）；重叠比 = max(IoU, 交集/较小框面积)，达到阈值即去掉较小的框",
    )
    return parser.parse_args()


def resolve_weights(weights: str | None) -> Path:
    """确定权重路径：命令行 > 默认路径(runs/train/exp1) > runs/train 下任意 best.pt。"""
    if weights:
        path = Path(weights)
        if not path.is_file():
            raise SystemExit(f"权重文件不存在: {path}")
        return path
    if DEFAULT_WEIGHTS.is_file():
        return DEFAULT_WEIGHTS
    candidates = sorted((PROJECT_ROOT / "runs" / "train").glob("**/weights/best.pt"))
    if not candidates:
        raise SystemExit(
            "未找到权重文件，请用 --weights 指定，例如:\n"
            "    python inference.py --pdf paper.pdf --weights runs/train/exp1/weights/best.pt"
        )
    return candidates[0]


def collect_pdfs(inputs: list[str]) -> list[Path]:
    """把文件/目录入参展开成 PDF 文件列表（目录不递归）。"""
    pdfs: list[Path] = []
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            pdfs.extend(sorted(p for p in path.iterdir() if p.suffix.lower() == ".pdf"))
        elif path.is_file():
            pdfs.append(path)
        else:
            raise SystemExit(f"输入不存在: {path}")
    if not pdfs:
        raise SystemExit("未找到任何 PDF 文件。")
    return pdfs


def resolve_class_filter(classes: list[str] | None, names: dict[int, str]) -> set[int] | None:
    """把 --classes 的类别名/id 转成模型类别 index 集合。"""
    if not classes:
        return None
    keep: set[int] = set()
    name_to_idx = {v: k for k, v in names.items()}
    for token in classes:
        if token in name_to_idx:
            keep.add(name_to_idx[token])
        elif token.isdigit():
            # 允许直接写模型 index，或写 info.json 的 category_id（index + 1）
            idx = int(token)
            keep.add(idx if idx in names else idx - 1)
        else:
            raise SystemExit(f"未知类别: {token}（可选: {', '.join(names.values())}）")
    invalid = keep - set(names)
    if invalid:
        raise SystemExit(f"类别 id 超出范围: {sorted(invalid)}（合法范围 0~{max(names)} 或 1~{max(names) + 1}）")
    return keep


def render_page(page: "fitz.Page", dpi: int) -> np.ndarray:
    """把 PDF 页渲染成 RGB 图像（np.ndarray, HWC）。"""
    pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csRGB, alpha=False)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3]


def build_detections(res, page_index: int, scale_x: float, scale_y: float, keep: set[int] | None) -> list[dict]:
    """把一页的 Results 转成检测项列表（坐标框 + 类别 + 置信度）。"""
    items: list[dict] = []
    boxes = res.boxes
    if boxes is None or len(boxes) == 0:
        return items

    names = res.names
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    clses = boxes.cls.cpu().numpy().astype(int)
    img_h, img_w = res.orig_shape[:2]

    for (x1, y1, x2, y2), conf, cls in zip(xyxy, confs, clses):
        if keep is not None and cls not in keep:
            continue
        x1, y1 = float(np.clip(x1, 0, img_w)), float(np.clip(y1, 0, img_h))
        x2, y2 = float(np.clip(x2, 0, img_w)), float(np.clip(y2, 0, img_h))
        w, h = x2 - x1, y2 - y1
        items.append(
            {
                "page_index": page_index,
                "page_number": page_index + 1,
                "class_id": int(cls),
                "category_id": int(cls) + 1,  # info.json 原始 id
                "class_name": names.get(int(cls), f"class_{int(cls)}"),
                "confidence": round(float(conf), 4),
                "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                "bbox_pdf": [
                    round(x1 / scale_x, 2),
                    round(y1 / scale_y, 2),
                    round(x2 / scale_x, 2),
                    round(y2 / scale_y, 2),
                ],
                "bbox_norm": [
                    round((x1 + w / 2) / img_w, 6),
                    round((y1 + h / 2) / img_h, 6),
                    round(w / img_w, 6),
                    round(h / img_h, 6),
                ],
            }
        )
    return items


def box_area(box: list[float]) -> float:
    """xyxy 框的面积（像素平方）。"""
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def overlap_ratio(box_a: list[float], box_b: list[float]) -> tuple[float, float]:
    """返回 (重叠比, IoU)。重叠比 = max(IoU, 交集 / 较小框面积)，小框被大框包住时为 1.0。"""
    inter_w = max(0.0, min(box_a[2], box_b[2]) - max(box_a[0], box_b[0]))
    inter_h = max(0.0, min(box_a[3], box_b[3]) - max(box_a[1], box_b[1]))
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0, 0.0
    area_a, area_b = box_area(box_a), box_area(box_b)
    union = area_a + area_b - inter
    iou = inter / union if union > 0 else 0.0
    smaller = min(area_a, area_b)
    return (max(inter / smaller, iou) if smaller > 0 else 0.0), iou


def dedup_same_class(items: list[dict], thr: float) -> tuple[list[dict], list[dict]]:
    """公式框检查：同页同类别内重叠比 >= thr 时，只保留面积较大的框（去掉较小的框）。

    返回 (保留的检测项, 被去掉的检测项)。被去掉的项会额外带上 overlap_ratio / iou / kept_bbox，
    便于核查是哪个更大的同类框把它去掉的；面积相同则保留置信度更高的那个。较大框不做合并。
    """
    groups: dict[tuple[int, int], list[dict]] = {}
    for item in items:
        groups.setdefault((item["page_index"], item["class_id"]), []).append(item)

    kept: list[dict] = []
    removed: list[dict] = []
    for key in sorted(groups):
        keepers: list[dict] = []
        # 面积从大到小处理，保证先留下较大的框；面积相同时先留置信度高的
        for cand in sorted(groups[key], key=lambda d: (-box_area(d["bbox"]), -d["confidence"])):
            duplicate = None
            for keeper in keepers:
                ratio, iou = overlap_ratio(cand["bbox"], keeper["bbox"])
                if ratio >= thr:
                    duplicate = (ratio, iou, keeper)
                    break
            if duplicate is None:
                keepers.append(cand)
            else:
                ratio, iou, keeper = duplicate
                removed.append(
                    {
                        **cand,
                        "overlap_ratio": round(ratio, 4),
                        "iou": round(iou, 4),
                        "kept_bbox": keeper["bbox"],
                        "kept_confidence": keeper["confidence"],
                    }
                )
        kept.extend(keepers)
    return kept, removed


def plot_detections(res, items: list[dict], line_width: int, font_size: int | None) -> np.ndarray:
    """只绘制通过公式框检查（同类去重后）的框，保证可视化图片/PDF 与 JSON 一致；返回 BGR 图像。

    未用 Results.plot() 是因为它会画出所有原始框（含已去重的重复框）。
    """
    annotator = Annotator(res.orig_img.copy(), line_width, font_size, example=res.names)
    for d in items:
        annotator.box_label(d["bbox"], f'{d["class_name"]} {d["confidence"]:.2f}', color=colors(d["class_id"], True))
    return annotator.result()


def resolve_ckpt_path(model: "YOLO", weights: Path) -> str:
    """取模型实际加载的权重路径（拿不到时回退到命令行/默认解析出的路径）。"""
    ckpt = getattr(model, "ckpt_path", None)
    return str(Path(ckpt)) if ckpt else str(weights)


def main() -> None:
    args = parse_args()

    weights = resolve_weights(args.weights)
    pdfs = collect_pdfs(args.pdf)
    if args.json and len(pdfs) > 1:
        raise SystemExit("--json 仅在单个输入 PDF 时可用；多个输入时请用 --out 指定输出目录。")
    if not 0.0 < args.dedup_thr <= 1.0:
        raise SystemExit(f"--dedup-thr 需在 (0, 1] 之间，当前为 {args.dedup_thr}")

    print("=" * 72)
    print(f"权重: {weights}")
    print(f"PDF : {len(pdfs)} 个 -> {', '.join(p.name for p in pdfs)}")
    print(f"参数: imgsz={args.imgsz}, conf={args.conf}, iou={args.iou}, dpi={args.dpi}, device={args.device or 'auto'}")
    print(f"检查: 同类框去重 {'关闭' if args.no_dedup else f'开启（阈值 {args.dedup_thr}）'}")
    print("=" * 72)

    model = YOLO(str(weights))
    names = {int(k): v for k, v in (model.names or {}).items()}
    if not names:  # 权重里没有 names 时用 info.json 的映射兜底
        names = {idx - 1: name for idx, name in CLASS_MAP.items()}
    keep = resolve_class_filter(args.classes, names)

    for pdf_path in pdfs:
        run_one_pdf(pdf_path, model, weights, names, keep, args, batch_mode=len(pdfs) > 1)


def run_one_pdf(
    pdf_path: Path,
    model: "YOLO",
    weights: Path,
    names: dict[int, str],
    keep: set[int] | None,
    args: argparse.Namespace,
    batch_mode: bool,
) -> None:
    out_dir = Path(args.out) if args.out else PROJECT_ROOT / "outputs" / pdf_path.stem
    if batch_mode and args.out:
        out_dir = out_dir / pdf_path.stem  # 批量时每个 PDF 一个子目录
    out_dir.mkdir(parents=True, exist_ok=True)

    vis_dir = out_dir / "vis_pages"
    json_path = Path(args.json) if args.json else out_dir / f"{pdf_path.stem}.json"
    vis_pdf_path = out_dir / f"{pdf_path.stem}_vis.pdf"
    if not args.no_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    num_pages = doc.page_count if args.max_pages <= 0 else min(doc.page_count, args.max_pages)
    print(f"\n[{pdf_path.name}] 共 {doc.page_count} 页，处理 {num_pages} 页")

    pages_meta: list[dict] = []
    detections: list[dict] = []
    dedup_removed: list[dict] = []
    vis_pages: list[tuple[int, Path]] = []

    for start in range(0, num_pages, max(1, args.batch)):
        end = min(start + max(1, args.batch), num_pages)
        images, metas = [], []
        for idx in range(start, end):
            page = doc[idx]
            img = render_page(page, args.dpi)
            images.append(Image.fromarray(img))  # 传 PIL(RGB) 给模型，ultralytics 会转成 BGR
            metas.append((idx, page, img.shape[1] / page.rect.width, img.shape[0] / page.rect.height))

        predict_kwargs = dict(imgsz=args.imgsz, conf=args.conf, iou=args.iou, max_det=args.max_det, verbose=False)
        if args.device:
            predict_kwargs["device"] = args.device
        results = model.predict(images, **predict_kwargs)

        for res, (idx, page, scale_x, scale_y) in zip(results, metas):
            items = build_detections(res, idx, scale_x, scale_y, keep)
            removed: list[dict] = []
            if not args.no_dedup:  # 公式框检查：同页同类别去重，只留较大的框
                items, removed = dedup_same_class(items, args.dedup_thr)
                dedup_removed.extend(removed)
            items.sort(key=lambda d: (d["bbox"][1], d["bbox"][0]))  # 阅读顺序：先上后左
            detections.extend(items)

            image_rel = ""
            if not args.no_vis:
                annotated = plot_detections(res, items, args.line_width, args.font_size)  # 仅画保留的框（BGR）
                page_png = vis_dir / f"page_{idx + 1:04d}.png"
                Image.fromarray(annotated[:, :, ::-1]).save(page_png)  # BGR -> RGB
                vis_pages.append((idx, page_png))
                image_rel = page_png.relative_to(out_dir).as_posix()

            pages_meta.append(
                {
                    "page_index": idx,
                    "page_number": idx + 1,
                    "width": int(res.orig_shape[1]),
                    "height": int(res.orig_shape[0]),
                    "pdf_width": round(page.rect.width, 2),
                    "pdf_height": round(page.rect.height, 2),
                    "scale_x": round(scale_x, 6),
                    "scale_y": round(scale_y, 6),
                    "image": image_rel,
                    "num_detections": len(items),
                    "num_duplicates_removed": len(removed),
                }
            )
            note = f"（同类去重移除 {len(removed)} 个较小框）" if removed else ""
            print(f"  第 {idx + 1} 页: {len(items)} 个检测框{note}")

    # ---- 输出可视化 PDF（页数/页面尺寸与原 PDF 一致，内容为标注后的页面图像）----
    if vis_pages:
        vis_doc = fitz.open()
        for idx, png in vis_pages:
            rect = doc[idx].rect
            new_page = vis_doc.new_page(width=rect.width, height=rect.height)
            new_page.insert_image(new_page.rect, filename=str(png))
        vis_doc.save(vis_pdf_path, deflate=True)
        vis_doc.close()
    doc.close()

    per_class = Counter(d["class_name"] for d in detections)
    payload = {
        "pdf": {"path": str(pdf_path.resolve()), "file_name": pdf_path.name, "num_pages": num_pages},
        "model": {
            "weights": resolve_ckpt_path(model, weights),
            "imgsz": args.imgsz,
            "conf": args.conf,
            "iou": args.iou,
            "max_det": args.max_det,
            "dpi": args.dpi,
            "device": args.device or "auto",
        },
        "classes": [
            {"id": idx, "category_id": idx + 1, "name": name}
            for idx, name in sorted(names.items())
            if keep is None or idx in keep
        ],
        "pages": pages_meta,
        "detections": detections,
        "dedup": {
            "enabled": not args.no_dedup,
            "overlap_threshold": args.dedup_thr,
            "removed_count": len(dedup_removed),
            "removed": dedup_removed,
        },
        "summary": {
            "total_detections": len(detections),
            "duplicates_removed": len(dedup_removed),
            "per_class": {name: int(per_class[name]) for name in sorted(per_class)},
        },
    }

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"  JSON      : {json_path}")
    if not args.no_vis:
        print(f"  可视化图片: {vis_dir}（{len(vis_pages)} 张）")
        if vis_pages:
            print(f"  可视化 PDF: {vis_pdf_path}")
    print(f"  检测总数  : {len(detections)} {payload['summary']['per_class']}")
    if not args.no_dedup:
        print(f"  同类去重  : 移除 {len(dedup_removed)} 个较小框（阈值 {args.dedup_thr}）")


if __name__ == "__main__":
    main()
