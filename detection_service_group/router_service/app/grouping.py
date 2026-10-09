"""检测结果分流：把 yolov13 的检测框划分成公式 / 表格 / 图片三类。

分类依据是 ``class_name``（也兼容用 ``category_id`` / ``class_id`` 数字匹配），
类别列表可通过环境变量覆盖，方便换成其它训练集。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

FORMULA = "formula"
TABLE = "table"
FIGURE = "figure"
OTHER = "other"

RECOGNIZED_KINDS = (FORMULA, TABLE)


@dataclass
class ClassifiedDetection:
    """一条检测框 + 它的归属类别 + 待识别图像。"""

    index: int
    detection: dict[str, Any]
    kind: str
    image: str | None
    page_index: int

    @property
    def needs_recognition(self) -> bool:
        return self.kind in RECOGNIZED_KINDS and bool(self.image)


class Taxonomy:
    """类别名 → 分组 的映射。"""

    def __init__(
        self,
        formula_classes: Iterable[str],
        table_classes: Iterable[str],
        figure_classes: Iterable[str],
    ) -> None:
        self.formula = {name.lower() for name in formula_classes}
        self.table = {name.lower() for name in table_classes}
        self.figure = {name.lower() for name in figure_classes}

    def classify(self, detection: dict[str, Any]) -> str:
        name = str(detection.get("class_name") or "").strip().lower()
        if name in self.formula:
            return FORMULA
        if name in self.table:
            return TABLE
        if name in self.figure:
            return FIGURE
        # 类别名对不上时，允许直接按 category_id / class_id 配置（如 "1,2,3"）
        for key in ("category_id", "class_id"):
            value = detection.get(key)
            if value is None:
                continue
            token = str(value)
            if token in self.formula:
                return FORMULA
            if token in self.table:
                return TABLE
            if token in self.figure:
                return FIGURE
        return OTHER


def crop_data(detection: dict[str, Any]) -> str | None:
    """取出检测框内的 base64 图像；下游识别服务需要它作为输入。"""
    crop = detection.get("crop")
    if not isinstance(crop, dict):
        return None
    data = crop.get("data")
    if isinstance(data, str) and data.strip():
        return data
    return None


def classify_detections(detections: Iterable[dict[str, Any]], taxonomy: Taxonomy) -> list[ClassifiedDetection]:
    classified: list[ClassifiedDetection] = []
    for index, detection in enumerate(detections):
        if not isinstance(detection, dict):
            continue
        page_index = detection.get("page_index")
        classified.append(
            ClassifiedDetection(
                index=index,
                detection=detection,
                kind=taxonomy.classify(detection),
                image=crop_data(detection),
                page_index=page_index if isinstance(page_index, int) else 0,
            )
        )
    return classified
