# -*- coding: utf-8 -*-
"""服务配置：所有可调参数均可通过环境变量覆盖（见 README.md）。"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SERVICE_ROOT.parent  # 仓库根目录（含 inference.py 与 yolov13/）
DEFAULT_WEIGHTS = SERVICE_ROOT / "models" / "yolov13_arxivformula_sft" / "yolov13s_ep5_bs32_lr_0.01.pt"


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or value.strip() == "" else value.strip()


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw.strip())
    except ValueError:
        raise ValueError(f"环境变量 {name} 需要是整数，当前为 {raw!r}") from None


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw.strip())
    except ValueError:
        raise ValueError(f"环境变量 {name} 需要是数字，当前为 {raw!r}") from None


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    value = raw.strip().lower()
    if value in ("1", "true", "yes", "y", "on"):
        return True
    if value in ("0", "false", "no", "n", "off"):
        return False
    raise ValueError(f"环境变量 {name} 需要是布尔值（true/false），当前为 {raw!r}")


@dataclass(frozen=True)
class ServiceConfig:
    """推理默认值 + 资源限制。"""

    weights: Path = DEFAULT_WEIGHTS
    device: str = ""  # ""=自动；"0"=第一块 GPU；"cpu"=仅 CPU
    imgsz: int = 640
    conf: float = 0.25
    iou: float = 0.7
    max_det: int = 300
    batch: int = 8
    dedup: bool = True
    dedup_thr: float = 0.8
    return_crops: bool = True
    crop_format: str = "png"
    crop_quality: int = 90
    crop_padding: int = 0
    warmup: bool = True
    max_images: int = 50
    max_pixels: int = 40_000_000
    max_upload_mb: int = 30
    log_level: str = "info"

    @classmethod
    def from_env(cls) -> "ServiceConfig":
        return cls(
            weights=Path(_env_str("MODEL_PATH", str(DEFAULT_WEIGHTS))),
            device=_env_str("DEVICE", ""),
            imgsz=_env_int("IMGSZ", 640),
            conf=_env_float("CONF", 0.25),
            iou=_env_float("IOU", 0.7),
            max_det=_env_int("MAX_DET", 300),
            batch=_env_int("BATCH", 8),
            dedup=_env_bool("DEDUP", True),
            dedup_thr=_env_float("DEDUP_THR", 0.8),
            return_crops=_env_bool("RETURN_CROPS", True),
            crop_format=_env_str("CROP_FORMAT", "png").lower(),
            crop_quality=_env_int("CROP_QUALITY", 90),
            crop_padding=_env_int("CROP_PADDING", 0),
            warmup=_env_bool("WARMUP", True),
            max_images=_env_int("MAX_IMAGES", 50),
            max_pixels=_env_int("MAX_PIXELS", 40_000_000),
            max_upload_mb=_env_int("MAX_UPLOAD_MB", 30),
            log_level=_env_str("LOG_LEVEL", "info").lower(),
        )

    def as_dict(self) -> dict:
        data = asdict(self)
        data["weights"] = str(self.weights)
        return data
