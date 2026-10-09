# -*- coding: utf-8 -*-
"""HTTP 请求/响应模型，以及请求参数与服务默认值的合并与校验。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

from config import ServiceConfig

CROP_FORMATS = ("png", "jpg", "jpeg")


# --------------------------------------------------------------------------- #
# 请求
# --------------------------------------------------------------------------- #
class ImageInput(BaseModel):
    """一张 PDF 渲染页面图片。"""

    data: str = Field(..., description="图片内容的 base64（可带 data:image/png;base64, 前缀）")
    name: str | None = Field(None, description="文件名/页名，仅用于回显")
    page_index: int | None = Field(None, ge=0, description="0 基页码；缺省时按请求中的顺序")
    pdf_width: float | None = Field(None, gt=0, description="原始 PDF 页面宽度（pt），用于换算 bbox_pdf")
    pdf_height: float | None = Field(None, gt=0, description="原始 PDF 页面高度（pt），用于换算 bbox_pdf")


class PredictParams(BaseModel):
    """单次请求的推理参数；留空则使用服务默认值（环境变量）。"""

    conf: float | None = Field(None, description="置信度阈值")
    iou: float | None = Field(None, description="NMS IoU 阈值")
    imgsz: int | None = Field(None, description="推理输入尺寸")
    max_det: int | None = Field(None, description="每张图最多保留的检测框数")
    batch: int | None = Field(None, description="每批送入模型的图片数")
    dedup: bool | None = Field(None, description="是否开启同类框去重（公式框检查）")
    dedup_thr: float | None = Field(None, description="同类框去重阈值")
    classes: list[str] | None = Field(None, description="只保留指定类别，可用类别名或 id/category_id")
    return_crops: bool | None = Field(None, description="是否返回框内区域图像（base64）")
    crop_format: Literal["png", "jpg", "jpeg"] | None = Field(None, description="框内区域图像格式")
    crop_quality: int | None = Field(None, description="jpg 质量（1~100）")
    crop_padding: int | None = Field(None, description="框外扩像素数")


class PredictRequest(BaseModel):
    """JSON 形式的检测请求。"""

    images: list[ImageInput] = Field(..., min_length=1, description="按页顺序排列的 PDF 渲染图")
    params: PredictParams | None = None


# --------------------------------------------------------------------------- #
# 响应（仅用于 OpenAPI 文档展示，实际响应为等价的 dict）
# --------------------------------------------------------------------------- #
class CropResult(BaseModel):
    format: str = Field(..., description="图像格式：png / jpeg")
    encoding: str = "base64"
    width: int
    height: int
    bytes: int = Field(..., description="编码前的字节数")
    rect: list[int] = Field(..., description="实际裁剪区域 [x1, y1, x2, y2]（图像像素坐标）")
    data: str = Field(..., description="base64 编码的框内区域图像")


class Detection(BaseModel):
    page_index: int = Field(..., description="0 基页码")
    page_number: int = Field(..., description="1 基页码")
    class_id: int = Field(..., description="训练类别 index（0~5）")
    category_id: int = Field(..., description="info.json 原始 id = class_id + 1")
    class_name: str
    confidence: float
    bbox: list[float] = Field(..., description="渲染图像素坐标 [x1, y1, x2, y2]")
    bbox_pdf: list[float] = Field(..., description="PDF 点坐标 [x1, y1, x2, y2]（未传 pdf_width/height 时与 bbox 相同）")
    bbox_norm: list[float] = Field(..., description="归一化 xywh [x_c, y_c, w, h]")
    crop: CropResult | None = None


class ImageMeta(BaseModel):
    image_index: int
    page_index: int
    page_number: int
    file_name: str
    width: int
    height: int
    num_detections: int
    num_duplicates_removed: int


class RemovedDetection(Detection):
    overlap_ratio: float
    iou: float
    kept_bbox: list[float]
    kept_confidence: float


class DedupInfo(BaseModel):
    enabled: bool
    overlap_threshold: float
    removed_count: int
    removed: list[RemovedDetection]


class Summary(BaseModel):
    num_images: int
    total_detections: int
    duplicates_removed: int
    per_class: dict[str, int]


class Timing(BaseModel):
    inference_ms: float
    total_ms: float


class PredictResponse(BaseModel):
    """检测结果：检测框 + 类别 + 框内区域图像（base64），全部包含在 JSON 中。"""

    status: str
    model: dict
    classes: list[dict]
    images: list[ImageMeta]
    detections: list[Detection]
    dedup: DedupInfo
    summary: Summary
    timing: Timing


# --------------------------------------------------------------------------- #
# 参数归一化
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EffectiveParams:
    """请求参数与服务默认值合并后的最终推理参数。"""

    conf: float
    iou: float
    imgsz: int
    max_det: int
    batch: int
    dedup: bool
    dedup_thr: float
    classes: list[str] | None
    return_crops: bool
    crop_format: str
    crop_quality: int
    crop_padding: int


def parse_class_list(raw: str | None) -> list[str] | None:
    """把 "InlineFormula,Table" / "InlineFormula Table" 解析成类别列表。"""
    if not raw or not raw.strip():
        return None
    tokens = [token for token in re.split(r"[,;\s]+", raw.strip()) if token]
    return tokens or None


def parse_int_list(raw: str | None) -> list[int | None]:
    """解析 "0,1,,3" 形式的整数列表（空位表示 None）。"""
    if not raw or not raw.strip():
        return []
    values: list[int | None] = []
    for token in re.split(r"[,;]+", raw.strip()):
        token = token.strip()
        if token == "":
            values.append(None)
            continue
        try:
            values.append(int(token))
        except ValueError:
            raise ValueError(f"无法解析整数 {token!r}") from None
    return values


def parse_size_list(raw: str | None) -> list[tuple[float, float] | None]:
    """解析 "612x792,595x842" 形式的 PDF 页面尺寸（pt）。"""
    if not raw or not raw.strip():
        return []
    sizes: list[tuple[float, float] | None] = []
    for token in re.split(r"[,;]+", raw.strip()):
        token = token.strip().lower()
        if token == "":
            sizes.append(None)
            continue
        match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*[x*]\s*(\d+(?:\.\d+)?)", token)
        if match is None:
            raise ValueError(f"无法解析页面尺寸 {token!r}，应为 WxH（如 612x792）")
        sizes.append((float(match.group(1)), float(match.group(2))))
    return sizes


def resolve_params(params: PredictParams | None, config: ServiceConfig) -> EffectiveParams:
    """合并请求参数与服务默认值并做范围校验；非法时抛 ValueError（由接口转成 HTTP 400）。"""
    params = params or PredictParams()

    def pick(value, default):
        return default if value is None else value

    effective = EffectiveParams(
        conf=float(pick(params.conf, config.conf)),
        iou=float(pick(params.iou, config.iou)),
        imgsz=int(pick(params.imgsz, config.imgsz)),
        max_det=int(pick(params.max_det, config.max_det)),
        batch=int(pick(params.batch, config.batch)),
        dedup=bool(pick(params.dedup, config.dedup)),
        dedup_thr=float(pick(params.dedup_thr, config.dedup_thr)),
        classes=params.classes or None,
        return_crops=bool(pick(params.return_crops, config.return_crops)),
        crop_format=str(pick(params.crop_format, config.crop_format)).lower(),
        crop_quality=int(pick(params.crop_quality, config.crop_quality)),
        crop_padding=int(pick(params.crop_padding, config.crop_padding)),
    )

    if not 0.0 < effective.conf <= 1.0:
        raise ValueError(f"conf 需在 (0, 1] 之间，当前为 {effective.conf}")
    if not 0.0 < effective.iou <= 1.0:
        raise ValueError(f"iou 需在 (0, 1] 之间，当前为 {effective.iou}")
    if effective.imgsz < 32:
        raise ValueError(f"imgsz 至少为 32，当前为 {effective.imgsz}")
    if not 1 <= effective.max_det <= 3000:
        raise ValueError(f"max_det 需在 [1, 3000] 之间，当前为 {effective.max_det}")
    if not 1 <= effective.batch <= 64:
        raise ValueError(f"batch 需在 [1, 64] 之间，当前为 {effective.batch}")
    if not 0.0 < effective.dedup_thr <= 1.0:
        raise ValueError(f"dedup_thr 需在 (0, 1] 之间，当前为 {effective.dedup_thr}")
    if effective.crop_format not in CROP_FORMATS:
        raise ValueError(f"crop_format 只支持 {', '.join(CROP_FORMATS)}，当前为 {effective.crop_format}")
    if not 1 <= effective.crop_quality <= 100:
        raise ValueError(f"crop_quality 需在 [1, 100] 之间，当前为 {effective.crop_quality}")
    if not 0 <= effective.crop_padding <= 512:
        raise ValueError(f"crop_padding 需在 [0, 512] 之间，当前为 {effective.crop_padding}")
    return effective
