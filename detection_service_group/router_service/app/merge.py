"""结果合并。

在 yolov13 检测服务的原始响应上保留元数据，并将检测框分组：

* 命中配置的公式类增加 ``formula``：pp-formulanet-plus-l 返回的 LaTeX；
* 命中配置的表格类增加 ``table``：slanet_plus 返回的结构 HTML 等；
* 被路由到识别服务的检测框放入 ``recognized_detections``，其它框放入
  ``unrecognized_detections`` 并保持原样。

另外追加顶层 ``session_id`` 与 ``router`` 汇总块（分类计数、耗时、错误）。
"""

from __future__ import annotations

import copy
from typing import Any

from .grouping import FIGURE, FORMULA, OTHER, TABLE
from .pipeline import PendingDetection


def build_response(
    detect_result: dict[str, Any],
    pendings: list[PendingDetection],
    outcomes: list[dict[str, Any]],
    *,
    session_id: str,
    request_id: str,
    detection_ms: float,
    recognition_ms: float,
    total_ms: float,
    keep_crops: bool,
) -> dict[str, Any]:
    response = copy.deepcopy(detect_result)
    detections = response.get("detections")
    if not isinstance(detections, list):
        detections = []

    counts = {FORMULA: 0, TABLE: 0, FIGURE: 0, OTHER: 0}
    ok_counts = {FORMULA: 0, TABLE: 0}
    failed = 0
    recognized_indexes = {
        pending.classified.index
        for pending in pendings
        if pending.classified.kind in (FORMULA, TABLE)
    }

    for pending, outcome in zip(pendings, outcomes, strict=True):
        classified = pending.classified
        counts[classified.kind] = counts.get(classified.kind, 0) + 1
        if not (0 <= classified.index < len(detections)):
            continue
        target = detections[classified.index]
        if not isinstance(target, dict):
            continue

        if classified.kind == FORMULA:
            target["formula"] = _formula_block(outcome)
            ok_counts[FORMULA] += 1 if outcome.get("ok") else 0
        elif classified.kind == TABLE:
            target["table"] = _table_block(outcome)
            ok_counts[TABLE] += 1 if outcome.get("ok") else 0
        else:
            continue
        if not outcome.get("ok"):
            failed += 1

    if not keep_crops:
        for detection in detections:
            if isinstance(detection, dict):
                detection.pop("crop", None)

    response.pop("detections", None)
    response["recognized_detections"] = [
        detection for index, detection in enumerate(detections) if index in recognized_indexes
    ]
    response["unrecognized_detections"] = [
        detection for index, detection in enumerate(detections) if index not in recognized_indexes
    ]
    response["session_id"] = session_id
    response["router"] = {
        "request_id": request_id,
        "session_id": session_id,
        "counts": {
            "formula": counts.get(FORMULA, 0),
            "table": counts.get(TABLE, 0),
            "figure": counts.get(FIGURE, 0),
            "other": counts.get(OTHER, 0),
            "total_detections": len(detections),
        },
        "recognition": {
            "formula_ok": ok_counts.get(FORMULA, 0),
            "table_ok": ok_counts.get(TABLE, 0),
            "failed": failed,
        },
        "timing": {
            "detection_ms": round(detection_ms, 2),
            "recognition_ms": round(recognition_ms, 2),
            "total_ms": round(total_ms, 2),
        },
        "pipeline": "detection producer -> formula/table buffer pools -> recognition consumers",
    }
    return response


def _formula_block(outcome: dict[str, Any]) -> dict[str, Any]:
    data = outcome.get("data") or {}
    return {
        "latex": data.get("latex"),
        "error": outcome.get("error"),
        "elapsed_ms": outcome.get("elapsed_ms"),
        "wait_ms": outcome.get("wait_ms"),
    }


def _table_block(outcome: dict[str, Any]) -> dict[str, Any]:
    data = outcome.get("data") or {}
    return {
        "html": data.get("html"),
        "structure_score": data.get("structure_score"),
        "num_cells": data.get("num_cells"),
        "error": outcome.get("error"),
        "elapsed_ms": outcome.get("elapsed_ms"),
        "wait_ms": outcome.get("wait_ms"),
    }
