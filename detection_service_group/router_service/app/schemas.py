"""HTTP 请求/响应模型。

请求模型与下游 yolov13 检测服务的字段保持一致，额外增加 ``session_id``，
用于在同一个 router service 内区分不同客户端（不同 PDF）的缓存池数据。
响应模型仅用于 OpenAPI 文档展示，实际返回的是等价的 dict。
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field


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
    """转发给检测服务的推理参数；留空的字段使用 router 的服务默认值。"""

    conf: float | None = Field(None, description="置信度阈值")
    iou: float | None = Field(None, description="NMS IoU 阈值")
    imgsz: int | None = Field(None, description="推理输入尺寸")
    max_det: int | None = Field(None, description="每张图最多保留的检测框数")
    batch: int | None = Field(None, description="每批送入模型的图片数")
    dedup: bool | None = Field(None, description="是否开启同类框去重")
    dedup_thr: float | None = Field(None, description="同类框去重阈值")
    classes: list[str] | None = Field(None, description="只保留指定类别，可用类别名或 id/category_id")
    crop_format: Literal["png", "jpg", "jpeg"] | None = Field(None, description="框内区域图像格式")
    crop_quality: int | None = Field(None, ge=1, le=100, description="jpg 质量（1~100）")
    crop_padding: int | None = Field(None, ge=0, description="框外扩像素数")


class RouterPredictRequest(BaseModel):
    """一次完整的「检测 → 分流 → 识别 → 合并」请求。"""

    images: list[ImageInput] = Field(..., min_length=1, description="按页顺序排列的 PDF 渲染图")
    params: PredictParams | None = None
    session_id: str | None = Field(
        None,
        max_length=128,
        description="客户端会话 id；缺省时由服务生成。同一 pdf 的多页请用同一 session_id",
    )
    keep_crops: bool = Field(True, description="是否在响应中保留框内区域图像（base64）")


# --------------------------------------------------------------------------- #
# 响应（仅用于 OpenAPI 文档展示）
# --------------------------------------------------------------------------- #
class FormulaResult(BaseModel):
    latex: str | None = Field(None, description="pp-formulanet-plus-l 识别出的 LaTeX")
    error: str | None = None
    elapsed_ms: float | None = None


class TableResult(BaseModel):
    html: str | None = Field(None, description="slanet_plus 识别出的表格结构 HTML")
    structure_score: float | None = None
    num_cells: int | None = None
    error: str | None = None
    elapsed_ms: float | None = None


class RouterTiming(BaseModel):
    detection_ms: float
    recognition_ms: float
    total_ms: float
    worker_wait_ms: float


class RouterSummary(BaseModel):
    session_id: str
    formula_count: int
    table_count: int
    figure_count: int
    other_count: int
    formula_ok: int
    table_ok: int
    failed: int


class RouterPredictResponse(BaseModel):
    """在 yolov13 原始响应上追加分组后的检测框与 router 汇总信息。"""

    status: str
    session_id: str
    images: list[dict[str, Any]]
    recognized_detections: list[dict[str, Any]]
    unrecognized_detections: list[dict[str, Any]]
    summary: dict[str, Any]
    router: dict[str, Any]


class SessionState(BaseModel):
    session_id: str
    created_at: float
    updated_at: float
    requests: int
    succeeded: int
    failed: int
    formula_submitted: int
    formula_completed: int
    table_submitted: int
    table_completed: int
    in_flight: int
    last_error: str | None = None
    result_count: int = 0


# --------------------------------------------------------------------------- #
# multipart 表单里的列表字段解析（与 yolov13 检测服务的写法保持一致）
# --------------------------------------------------------------------------- #
def parse_class_list(raw: str | None) -> list[str] | None:
    """"InlineFormula,Table" / "InlineFormula Table" -> 类别列表。"""
    if not raw or not raw.strip():
        return None
    tokens = [token for token in re.split(r"[,;\s]+", raw.strip()) if token]
    return tokens or None


def parse_int_list(raw: str | None) -> list[int | None]:
    """解析 "0,1,,3"（空位表示 None）。"""
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
