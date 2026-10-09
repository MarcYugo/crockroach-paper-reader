# -*- coding: utf-8 -*-
"""YOLOv13 公式检测推理封装。

后处理（坐标换算、类别过滤、同类框去重）直接复用仓库根目录 inference.py 的实现，
因此服务输出与 `python inference.py` 的 JSON 语义完全一致；服务侧额外提供框内区域图像的裁剪与编码。
"""

from __future__ import annotations

import base64
import io
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image

from config import PROJECT_ROOT, ServiceConfig

# 复用 inference.py 的推理后处理逻辑（该模块会把仓库内置的 yolov13/ 加入 sys.path）
for _path in (PROJECT_ROOT, PROJECT_ROOT / "yolov13"):
    _entry = str(_path)
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from inference import (  # noqa: E402
    CLASS_MAP,
    build_detections,
    dedup_same_class,
    resolve_class_filter,
    resolve_ckpt_path,
)
from ultralytics import YOLO  # noqa: E402

from schemas import EffectiveParams  # noqa: E402


@dataclass
class DecodedImage:
    """一张待检测的图像（PDF 渲染页）。"""

    image: Image.Image  # RGB，交给模型
    array: np.ndarray  # HWC RGB，用于裁剪框内区域
    name: str
    page_index: int
    pdf_width: float | None = None
    pdf_height: float | None = None


def decode_base64_image(data: str) -> bytes:
    """解码 base64 图片内容（兼容 data:image/png;base64,... 前缀与换行）。"""
    text = data.strip()
    if text.startswith("data:"):
        separator = text.find(",")
        if separator == -1:
            raise ValueError("data URL 格式不正确（缺少逗号）")
        text = text[separator + 1 :]
    text = "".join(text.split())
    if not text:
        raise ValueError("图片 base64 内容为空")
    try:
        return base64.b64decode(text, validate=True)
    except Exception as exc:
        raise ValueError(f"base64 解码失败: {exc}") from None


def open_image(raw: bytes) -> tuple[Image.Image, np.ndarray]:
    """把图片字节流解码成 RGB 的 PIL 图像与 numpy 数组。"""
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except Exception as exc:
        raise ValueError(f"无法解析图像: {exc}") from None
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image, np.array(image, dtype=np.uint8)


def crop_and_encode(array: np.ndarray, bbox: Sequence[float], params: EffectiveParams) -> dict[str, Any] | None:
    """裁出框内区域（可选外扩）并编码成 base64。"""
    height, width = array.shape[:2]
    pad = params.crop_padding
    x1 = max(0, min(int(np.floor(bbox[0])) - pad, width - 1))
    y1 = max(0, min(int(np.floor(bbox[1])) - pad, height - 1))
    x2 = max(1, min(int(np.ceil(bbox[2])) + pad, width))
    y2 = max(1, min(int(np.ceil(bbox[3])) + pad, height))
    if x2 <= x1 or y2 <= y1:
        return None

    patch = array[y1:y2, x1:x2]
    buffer = io.BytesIO()
    if params.crop_format in ("jpg", "jpeg"):
        # subsampling=0 保留公式细节，避免色度抽样糊掉细笔画
        Image.fromarray(patch).save(buffer, format="JPEG", quality=params.crop_quality, subsampling=0)
        media_type = "jpeg"
    else:
        Image.fromarray(patch).save(buffer, format="PNG", compress_level=6)
        media_type = "png"

    payload = buffer.getvalue()
    return {
        "format": media_type,
        "encoding": "base64",
        "width": int(patch.shape[1]),
        "height": int(patch.shape[0]),
        "bytes": len(payload),
        "rect": [x1, y1, x2, y2],
        "data": base64.b64encode(payload).decode("ascii"),
    }


class FormulaDetector:
    """加载 YOLOv13 权重并执行公式检测（推理串行化，供多线程 Web 服务安全调用）。"""

    def __init__(self, config: ServiceConfig) -> None:
        self.config = config
        if not config.weights.is_file():
            raise FileNotFoundError(f"未找到模型权重: {config.weights}（可用环境变量 MODEL_PATH 指定）")

        self._lock = threading.Lock()
        self.model = YOLO(str(config.weights))
        names = {int(key): str(value) for key, value in (self.model.names or {}).items()}
        self.names: dict[int, str] = names or {idx - 1: name for idx, name in CLASS_MAP.items()}  # 权重无 names 时兜底
        self.device_arg = self._device_arg()
        self.device_name = self._resolve_device_name()
        self.warmup_ms: float | None = None
        if config.warmup:
            self.warmup()

    # ------------------------------------------------------------------ #
    # 设备 / 预热
    # ------------------------------------------------------------------ #
    def _device_arg(self) -> str | None:
        device = (self.config.device or "").strip()
        return None if device == "" or device.lower() == "auto" else device

    def _resolve_device_name(self) -> str:
        if self.device_arg:
            return self.device_arg
        return "cuda:0" if torch.cuda.is_available() else "cpu"

    def warmup(self) -> None:
        """用一张空白图跑一次推理，触发 CUDA 初始化，避免首个请求超时。"""
        dummy = np.full((self.config.imgsz, self.config.imgsz, 3), 255, dtype=np.uint8)
        start = time.perf_counter()
        with self._lock:
            self.model.predict(dummy, imgsz=self.config.imgsz, conf=self.config.conf, iou=self.config.iou,
                               max_det=self.config.max_det, verbose=False, **self._device_kwargs())
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.warmup_ms = round((time.perf_counter() - start) * 1000.0, 1)

    def _device_kwargs(self) -> dict[str, Any]:
        return {"device": self.device_arg} if self.device_arg else {}

    # ------------------------------------------------------------------ #
    # 元信息
    # ------------------------------------------------------------------ #
    def model_info(self) -> dict[str, Any]:
        cuda_available = torch.cuda.is_available()
        return {
            "weights": resolve_ckpt_path(self.model, self.config.weights),
            "classes": self._class_list(None),
            "device": self.device_name,
            "cuda_available": cuda_available,
            "gpu": torch.cuda.get_device_name(0) if cuda_available else None,
            "torch": torch.__version__,
            "warmup_ms": self.warmup_ms,
        }

    def _class_list(self, keep: set[int] | None) -> list[dict[str, Any]]:
        return [
            {"id": idx, "category_id": idx + 1, "name": name}
            for idx, name in sorted(self.names.items())
            if keep is None or idx in keep
        ]

    def _class_filter(self, classes: list[str] | None) -> set[int] | None:
        if not classes:
            return None
        try:
            return resolve_class_filter(list(classes), self.names)
        except SystemExit as exc:  # inference.py 的参数解析以 SystemExit 报错
            raise ValueError(str(exc)) from None

    # ------------------------------------------------------------------ #
    # 推理
    # ------------------------------------------------------------------ #
    def predict(self, images: Sequence[DecodedImage], params: EffectiveParams) -> dict[str, Any]:
        if not images:
            raise ValueError("images 不能为空")

        keep = self._class_filter(params.classes)
        started = time.perf_counter()
        infer_ms = 0.0
        image_metas: list[dict[str, Any]] = []
        detections: list[dict[str, Any]] = []
        removed_all: list[dict[str, Any]] = []

        batch_size = max(1, params.batch)
        with self._lock:
            for start in range(0, len(images), batch_size):
                chunk = list(images[start : start + batch_size])
                begin = time.perf_counter()
                results = self.model.predict(
                    [item.image for item in chunk],
                    imgsz=params.imgsz,
                    conf=params.conf,
                    iou=params.iou,
                    max_det=params.max_det,
                    verbose=False,
                    **self._device_kwargs(),
                )
                infer_ms += (time.perf_counter() - begin) * 1000.0

                for result, item in zip(results, chunk):
                    items, removed = self._postprocess(result, item, keep, params)
                    detections.extend(items)
                    removed_all.extend(removed)
                    image_metas.append(
                        {
                            "image_index": len(image_metas),
                            "page_index": item.page_index,
                            "page_number": item.page_index + 1,
                            "file_name": item.name,
                            "width": int(result.orig_shape[1]),
                            "height": int(result.orig_shape[0]),
                            "num_detections": len(items),
                            "num_duplicates_removed": len(removed),
                        }
                    )

        per_class = Counter(item["class_name"] for item in detections)
        return {
            "status": "ok",
            "model": {
                "weights": resolve_ckpt_path(self.model, self.config.weights),
                "imgsz": params.imgsz,
                "conf": params.conf,
                "iou": params.iou,
                "max_det": params.max_det,
                "batch": params.batch,
                "device": self.device_name,
            },
            "classes": self._class_list(keep),
            "images": image_metas,
            "detections": detections,
            "dedup": {
                "enabled": params.dedup,
                "overlap_threshold": params.dedup_thr,
                "removed_count": len(removed_all),
                "removed": removed_all,
            },
            "summary": {
                "num_images": len(images),
                "total_detections": len(detections),
                "duplicates_removed": len(removed_all),
                "per_class": {name: int(per_class[name]) for name in sorted(per_class)},
            },
            "timing": {
                "inference_ms": round(infer_ms, 1),
                "total_ms": round((time.perf_counter() - started) * 1000.0, 1),
            },
        }

    def _postprocess(
        self,
        result: Any,
        item: DecodedImage,
        keep: set[int] | None,
        params: EffectiveParams,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """单页后处理：坐标换算 -> 类别过滤 -> 同类框去重 -> 阅读顺序排序 -> 裁剪框内区域。"""
        image_height, image_width = result.orig_shape[:2]
        # 传入 PDF 页面尺寸时把像素坐标换算成 PDF 点坐标（与 inference.py 的 scale_x/scale_y 一致）
        scale_x = image_width / item.pdf_width if item.pdf_width else 1.0
        scale_y = image_height / item.pdf_height if item.pdf_height else 1.0

        items = build_detections(result, item.page_index, scale_x, scale_y, keep)
        removed: list[dict[str, Any]] = []
        if params.dedup:
            items, removed = dedup_same_class(items, params.dedup_thr)
        items.sort(key=lambda det: (det["bbox"][1], det["bbox"][0]))

        if params.return_crops:
            for detection in items:
                detection["crop"] = crop_and_encode(item.array, detection["bbox"], params)
        return items, removed


__all__ = [
    "DecodedImage",
    "FormulaDetector",
    "crop_and_encode",
    "decode_base64_image",
    "open_image",
]
