"""服务配置。

所有配置均通过环境变量注入，方便在 docker-compose / K8s 中按机器规格调整，
而无需重新构建镜像。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any


def _str(name: str, default: str) -> str:
    value = os.getenv(name)
    return default if value is None or value.strip() == "" else value.strip()


def _int(name: str, default: int) -> int:
    try:
        return int(_str(name, str(default)))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(_str(name, str(default)))
    except ValueError:
        return default


def _bool(name: str, default: bool) -> bool:
    value = _str(name, "true" if default else "false").lower()
    return value in {"1", "true", "yes", "y", "on"}


def _list(name: str, default: str) -> tuple[str, ...]:
    raw = _str(name, default)
    tokens = [token.strip() for token in raw.replace(";", ",").split(",")]
    return tuple(token for token in tokens if token)


# --------------------------------------------------------------------------- #
# 默认类别分组（与 yolov13_formula_detection_service 的类别表一致）
# --------------------------------------------------------------------------- #
DEFAULT_FORMULA_CLASSES = (
    "InlineFormula",
    "DisplayedFormulaLine",
    "FormulaNumber",
    "DisplayedFormulaBlock",
)
DEFAULT_TABLE_CLASSES = ("Table",)
DEFAULT_FIGURE_CLASSES = ("Figure",)


@dataclass(frozen=True)
class DetectionOptions:
    """转发给 yolov13 检测服务的默认推理参数（请求可用 params 覆盖）。"""

    conf: float
    iou: float
    imgsz: int
    max_det: int
    batch: int
    dedup: bool
    dedup_thr: float
    crop_format: str
    crop_quality: int
    crop_padding: int


@dataclass(frozen=True)
class Settings:
    # ---- 下游服务地址 ---- #
    detection_url: str
    formula_url: str
    table_url: str

    # ---- 本服务 ---- #
    host: str
    port: int
    log_level: str

    # ---- 请求准入（多客户端并发） ---- #
    max_concurrent_requests: int
    max_queue: int
    queue_timeout_s: float
    request_timeout_s: float
    session_ttl_s: float

    # ---- 缓冲池 / 消费者 ---- #
    formula_workers: int
    table_workers: int
    formula_batch_size: int
    table_batch_size: int
    batch_window_ms: int
    pool_max_size: int
    pool_overflow: str  # "block" | "drop"

    # ---- HTTP 客户端 ---- #
    connect_timeout_s: float
    read_timeout_s: float
    max_connections: int

    # ---- 检测与分流 ---- #
    force_return_crops: bool
    formula_classes: tuple[str, ...]
    table_classes: tuple[str, ...]
    figure_classes: tuple[str, ...]
    detection: DetectionOptions = field(default_factory=lambda: DetectionOptions(
        conf=0.25,
        iou=0.7,
        imgsz=640,
        max_det=300,
        batch=8,
        dedup=True,
        dedup_thr=0.8,
        crop_format="png",
        crop_quality=90,
        crop_padding=0,
    ))

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            detection_url=_str("DETECTION_URL", "http://127.0.0.1:9000").rstrip("/"),
            formula_url=_str("FORMULA_URL", "http://127.0.0.1:9001").rstrip("/"),
            table_url=_str("TABLE_URL", "http://127.0.0.1:9002").rstrip("/"),
            host=_str("SERVICE_HOST", "0.0.0.0"),
            port=_int("SERVICE_PORT", _int("PORT", 9003)),
            log_level=_str("LOG_LEVEL", "INFO"),
            max_concurrent_requests=max(1, _int("MAX_CONCURRENT_REQUESTS", 2)),
            max_queue=max(1, _int("MAX_QUEUE", 64)),
            queue_timeout_s=max(0.1, _float("QUEUE_TIMEOUT", 30.0)),
            request_timeout_s=max(1.0, _float("REQUEST_TIMEOUT", 300.0)),
            session_ttl_s=max(0.0, _float("SESSION_TTL", 1800.0)),
            formula_workers=max(1, _int("FORMULA_WORKERS", 2)),
            table_workers=max(1, _int("TABLE_WORKERS", 1)),
            formula_batch_size=max(1, _int("FORMULA_BATCH_SIZE", 4)),
            table_batch_size=max(1, _int("TABLE_BATCH_SIZE", 1)),
            batch_window_ms=max(0, _int("BATCH_WINDOW_MS", 20)),
            pool_max_size=max(1, _int("POOL_MAX_SIZE", 512)),
            pool_overflow=_str("POOL_OVERFLOW", "block").lower(),
            connect_timeout_s=max(0.5, _float("HTTP_CONNECT_TIMEOUT", 10.0)),
            read_timeout_s=max(1.0, _float("HTTP_READ_TIMEOUT", 300.0)),
            max_connections=max(1, _int("HTTP_MAX_CONNECTIONS", 32)),
            force_return_crops=_bool("FORCE_RETURN_CROPS", True),
            formula_classes=_list("FORMULA_CLASSES", ",".join(DEFAULT_FORMULA_CLASSES)),
            table_classes=_list("TABLE_CLASSES", ",".join(DEFAULT_TABLE_CLASSES)),
            figure_classes=_list("FIGURE_CLASSES", ",".join(DEFAULT_FIGURE_CLASSES)),
            detection=DetectionOptions(
                conf=_float("DETECT_CONF", 0.25),
                iou=_float("DETECT_IOU", 0.7),
                imgsz=_int("DETECT_IMGSZ", 640),
                max_det=_int("DETECT_MAX_DET", 300),
                batch=_int("DETECT_BATCH", 8),
                dedup=_bool("DETECT_DEDUP", True),
                dedup_thr=_float("DETECT_DEDUP_THR", 0.8),
                crop_format=_str("DETECT_CROP_FORMAT", "png").lower(),
                crop_quality=_int("DETECT_CROP_QUALITY", 90),
                crop_padding=_int("DETECT_CROP_PADDING", 0),
            ),
        )

    def detection_default_params(self) -> dict[str, Any]:
        """转发给检测服务的默认参数；crop 必须返回，否则无法继续识别。"""
        detection = self.detection
        return {
            "conf": detection.conf,
            "iou": detection.iou,
            "imgsz": detection.imgsz,
            "max_det": detection.max_det,
            "batch": detection.batch,
            "dedup": detection.dedup,
            "dedup_thr": detection.dedup_thr,
            "return_crops": self.force_return_crops,
            "crop_format": detection.crop_format,
            "crop_quality": detection.crop_quality,
            "crop_padding": detection.crop_padding,
        }

    def public_info(self) -> dict[str, Any]:
        return {
            "detection_url": self.detection_url,
            "formula_url": self.formula_url,
            "table_url": self.table_url,
            "max_concurrent_requests": self.max_concurrent_requests,
            "max_queue": self.max_queue,
            "queue_timeout_s": self.queue_timeout_s,
            "request_timeout_s": self.request_timeout_s,
            "session_ttl_s": self.session_ttl_s,
            "formula_workers": self.formula_workers,
            "table_workers": self.table_workers,
            "formula_batch_size": self.formula_batch_size,
            "table_batch_size": self.table_batch_size,
            "batch_window_ms": self.batch_window_ms,
            "pool_max_size": self.pool_max_size,
            "pool_overflow": self.pool_overflow,
            "force_return_crops": self.force_return_crops,
            "formula_classes": list(self.formula_classes),
            "table_classes": list(self.table_classes),
            "figure_classes": list(self.figure_classes),
            "detection_defaults": self.detection_default_params(),
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """进程内单例；测试可用 get_settings.cache_clear() 重置。"""
    return Settings.from_env()
