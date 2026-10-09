# -*- coding: utf-8 -*-
"""YOLOv13 公式检测服务测试脚本：输入 PDF，调用 /predict，校验返回的 JSON。

前置：服务已启动（见 README.md），例如
    cd yolov13_formula_detection_service && docker compose up -d

用法:
    python test_detection_service.py                          # 用仓库自带的 latex_sample.pdf
    python test_detection_service.py --pdf paper.pdf --url http://127.0.0.1:9000
    python test_detection_service.py --pdf paper.pdf --out result.json --max-pages 2
"""

import argparse
import base64
import io
import json
import sys
from pathlib import Path

import pymupdf as fitz  # PyMuPDF
import requests
from PIL import Image

_SERVICE_DIR = Path(__file__).resolve().parent
# 默认输入：优先服务目录下的 latex_sample.pdf，其次仓库根目录
DEFAULT_PDF = next(
    (p for p in (_SERVICE_DIR / "latex_sample.pdf", _SERVICE_DIR.parent / "latex_sample.pdf") if p.is_file()),
    _SERVICE_DIR / "latex_sample.pdf",
)


def render_pages(pdf_path: Path, dpi: int, max_pages: int) -> list[dict]:
    """把 PDF 每页渲染成 PNG（base64）并带上页面尺寸（pt）。"""
    images = []
    with fitz.open(pdf_path) as doc:
        num_pages = doc.page_count if max_pages <= 0 else min(doc.page_count, max_pages)
        for index in range(num_pages):
            page = doc[index]
            pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csRGB, alpha=False)
            images.append(
                {
                    "name": f"page_{index + 1:04d}.png",
                    "page_index": index,
                    "pdf_width": page.rect.width,
                    "pdf_height": page.rect.height,
                    "data": base64.b64encode(pix.tobytes("png")).decode("ascii"),
                }
            )
    return images


def summarize(body: dict) -> None:
    print(f"  检测总数: {body['summary']['total_detections']}  去重移除: {body['summary']['duplicates_removed']}")
    print(f"  分类统计: {json.dumps(body['summary']['per_class'], ensure_ascii=False)}")
    print(f"  耗时    : {body['timing']}")
    for image in body["images"]:
        print(f"  第 {image['page_number']} 页: {image['num_detections']} 个框（{image['width']}x{image['height']}）")


def check(body: dict, num_pages: int) -> None:
    """最小校验：结构完整、坐标合理、框内区域图像可解码。"""
    assert body["status"] == "ok", f"status={body['status']}"
    assert len(body["images"]) == num_pages, f"返回 {len(body['images'])} 页，期望 {num_pages} 页"
    assert body["detections"], "没有检测到任何目标"

    pages = {image["page_index"]: image for image in body["images"]}
    for det in body["detections"]:
        page = pages[det["page_index"]]
        x1, y1, x2, y2 = det["bbox"]
        assert 0 <= x1 < x2 <= page["width"], f"bbox 越界: {det['bbox']} / {page['width']}x{page['height']}"
        assert 0 <= y1 < y2 <= page["height"], f"bbox 越界: {det['bbox']} / {page['width']}x{page['height']}"
        assert det["category_id"] == det["class_id"] + 1, det
        assert 0.0 < det["confidence"] <= 1.0, det

        crop = det.get("crop")
        assert crop and crop["encoding"] == "base64", "缺少框内区域图像"
        image = Image.open(io.BytesIO(base64.b64decode(crop["data"])))
        rect = crop["rect"]
        assert image.size == (rect[2] - rect[0], rect[3] - rect[1]), f"裁剪尺寸不符: {image.size} / {rect}"
    print(f"  校验通过: {len(body['detections'])} 个检测框（含框内区域图像）")


def main() -> int:
    parser = argparse.ArgumentParser(description="测试公式检测服务：输入 PDF，输出/校验 JSON")
    parser.add_argument("--pdf", default=str(DEFAULT_PDF), help=f"输入 PDF（默认 {DEFAULT_PDF}）")
    parser.add_argument("--url", default="http://127.0.0.1:9000", help="服务地址（默认 http://127.0.0.1:8000）")
    parser.add_argument("--out", default=None, help="把返回的 JSON 保存到该文件")
    parser.add_argument("--dpi", type=int, default=300, help="PDF 渲染 DPI（默认 150，与 inference.py 一致）")
    parser.add_argument("--max-pages", type=int, default=1, help="最多测试前 N 页（默认 1，0 表示全部）")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.is_file():
        print(f"找不到 PDF: {pdf_path}")
        return 1

    print(f"服务地址: {args.url}")
    try:
        health = requests.get(f"{args.url.rstrip('/')}/health", timeout=10)
        print(f"健康检查: {health.status_code} {health.text[:200]}")
    except requests.RequestException as exc:
        print(f"无法访问服务: {exc}")
        return 1

    images = render_pages(pdf_path, args.dpi, args.max_pages)
    print(f"输入 PDF: {pdf_path}（渲染 {len(images)} 页，DPI {args.dpi}）")

    response = requests.post(f"{args.url.rstrip('/')}/predict", json={"images": images}, timeout=600)
    print(f"POST /predict -> {response.status_code}")
    if response.status_code != 200:
        print(response.text[:2000])
        return 1

    body = response.json()
    summarize(body)
    if args.out:
        Path(args.out).write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  JSON 已保存: {args.out}")

    try:
        check(body, len(images))
    except AssertionError as exc:
        print(f"  校验失败: {exc}")
        return 1

    print("TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
