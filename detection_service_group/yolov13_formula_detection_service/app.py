# -*- coding: utf-8 -*-
"""YOLOv13 公式检测 HTTP 服务（FastAPI）。

输入 PDF 渲染后的页面图片，输出检测框、类别、置信度以及框内区域图像（base64），全部以 JSON 返回。
接口说明与示例见同目录 README.md。
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from config import ServiceConfig
from detector import DecodedImage, FormulaDetector, decode_base64_image, open_image
from schemas import (
    PredictParams,
    PredictRequest,
    PredictResponse,
    parse_class_list,
    parse_int_list,
    parse_size_list,
    resolve_params,
)

logger = logging.getLogger("formula_detection_service")


class _State:
    """进程内共享状态：配置、模型、加载错误。"""

    def __init__(self) -> None:
        self.config: ServiceConfig | None = None
        self.detector: FormulaDetector | None = None
        self.error: str | None = None


state = _State()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = ServiceConfig.from_env()
    state.config = config
    logger.setLevel(config.log_level.upper())

    # 默认参数在这里校验一次，便于配置错误在启动阶段就暴露
    try:
        resolve_params(None, config)
    except ValueError as exc:
        state.error = f"环境变量配置非法: {exc}"
        logger.error(state.error)

    if state.error is None:
        logger.info("加载模型 %s（device=%s）", config.weights, config.device or "auto")
        started = time.perf_counter()
        try:
            state.detector = await asyncio.get_running_loop().run_in_executor(None, FormulaDetector, config)
            logger.info(
                "模型就绪：device=%s，耗时 %.1fs%s",
                state.detector.device_name,
                time.perf_counter() - started,
                f"，预热 {state.detector.warmup_ms}ms" if state.detector.warmup_ms is not None else "",
            )
        except Exception as exc:  # 模型加载失败时仍启动服务，/predict 返回 503，便于排查
            state.error = f"{type(exc).__name__}: {exc}"
            logger.exception("模型加载失败，/predict 将返回 503")

    yield
    state.detector = None


app = FastAPI(
    title="YOLOv13 公式检测服务",
    version="1.0.0",
    description="输入 PDF 渲染图，输出检测框 + 类别 + 框内区域图像（JSON）。",
    lifespan=lifespan,
)


@app.exception_handler(ValueError)
async def value_error_handler(request, exc: ValueError) -> JSONResponse:
    """参数/图片非法统一返回 400，而不是 500。"""
    return JSONResponse(status_code=400, content={"detail": str(exc)})


# --------------------------------------------------------------------------- #
# 辅助函数
# --------------------------------------------------------------------------- #
def require_detector() -> FormulaDetector:
    if state.detector is None:
        raise HTTPException(status_code=503, detail=f"模型尚未就绪: {state.error or '正在加载'}")
    return state.detector


def _config() -> ServiceConfig:
    if state.config is None:
        return ServiceConfig.from_env()
    return state.config


def _check_pixel_budget(array, config: ServiceConfig, name: str) -> None:
    height, width = array.shape[:2]
    if height * width > config.max_pixels:
        raise HTTPException(
            status_code=413,
            detail=f"图片 {name} 分辨率 {width}x{height} 超过上限 {config.max_pixels} 像素（可用 MAX_PIXELS 调整）",
        )


def _check_file_size(size: int, config: ServiceConfig, name: str) -> None:
    limit = config.max_upload_mb * 1024 * 1024
    if size > limit:
        raise HTTPException(
            status_code=413,
            detail=f"图片 {name} 大小 {size / 1048576:.1f}MB 超过上限 {config.max_upload_mb}MB（可用 MAX_UPLOAD_MB 调整）",
        )


def _check_image_count(count: int, config: ServiceConfig) -> None:
    if count > config.max_images:
        raise HTTPException(
            status_code=413,
            detail=f"单次请求最多 {config.max_images} 张图片，当前 {count} 张（可用 MAX_IMAGES 调整）",
        )


def _build_image(raw: bytes, name: str, page_index: int, pdf_size, config: ServiceConfig) -> DecodedImage:
    _check_file_size(len(raw), config, name)
    image, array = open_image(raw)
    _check_pixel_budget(array, config, name)
    pdf_width, pdf_height = pdf_size if pdf_size else (None, None)
    return DecodedImage(
        image=image,
        array=array,
        name=name,
        page_index=page_index,
        pdf_width=pdf_width,
        pdf_height=pdf_height,
    )


def _run(detector: FormulaDetector, images: list[DecodedImage], params) -> dict[str, Any]:
    _check_image_count(len(images), detector.config)
    try:
        return detector.predict(images, params)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    except Exception as exc:  # noqa: BLE001
        logger.exception("推理失败")
        raise HTTPException(status_code=500, detail=f"推理失败: {exc}") from None


# --------------------------------------------------------------------------- #
# 接口
# --------------------------------------------------------------------------- #
@app.get("/", include_in_schema=False)
def root() -> dict[str, Any]:
    return {
        "service": "yolov13-formula-detection",
        "docs": "/docs",
        "health": "/health",
        "info": "/info",
        "predict": ["POST /predict（JSON + base64）", "POST /predict/upload（multipart 文件上传）"],
    }


@app.get("/health", summary="健康检查（模型未就绪时返回 503）")
def health():
    if state.detector is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "error" if state.error else "loading",
                "model_loaded": False,
                "error": state.error,
            },
        )
    return {
        "status": "ok",
        "model_loaded": True,
        "weights": str(state.detector.config.weights),
        "device": state.detector.device_name,
    }


@app.get("/info", summary="服务与模型信息（类别表、默认参数、限制）")
def info() -> dict[str, Any]:
    config = _config()
    return {
        "service": {"name": "yolov13-formula-detection", "version": app.version},
        "model": state.detector.model_info() if state.detector is not None else None,
        "model_loaded": state.detector is not None,
        "error": state.error,
        "defaults": {
            key: value
            for key, value in config.as_dict().items()
            if key not in ("weights", "warmup", "log_level", "max_images", "max_pixels", "max_upload_mb")
        },
        "limits": {
            "max_images": config.max_images,
            "max_pixels": config.max_pixels,
            "max_upload_mb": config.max_upload_mb,
        },
    }


@app.post(
    "/predict",
    summary="检测 PDF 渲染图（JSON 请求，base64 图片）",
    responses={200: {"model": PredictResponse, "description": "检测结果：框 + 类别 + 框内区域图像"}},
)
def predict(payload: PredictRequest) -> dict[str, Any]:
    detector = require_detector()
    params = resolve_params(payload.params, detector.config)

    _check_image_count(len(payload.images), detector.config)
    images: list[DecodedImage] = []
    for index, item in enumerate(payload.images):
        name = item.name or f"image_{index:04d}"
        try:
            raw = decode_base64_image(item.data)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"第 {index} 张图片（{name}）{exc}") from None
        try:
            images.append(
                _build_image(
                    raw,
                    name,
                    item.page_index if item.page_index is not None else index,
                    (item.pdf_width, item.pdf_height) if item.pdf_width and item.pdf_height else None,
                    detector.config,
                )
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"第 {index} 张图片（{name}）{exc}") from None

    return _run(detector, images, params)


@app.post(
    "/predict/upload",
    summary="检测 PDF 渲染图（multipart 文件上传）",
    responses={200: {"model": PredictResponse, "description": "检测结果：框 + 类别 + 框内区域图像"}},
)
def predict_upload(
    files: list[UploadFile] | None = File(None, description="PDF 渲染页图片，按页顺序上传（字段名 files，可重复）"),
    file: UploadFile | None = File(None, description="单张图片的简写字段，等价于 files"),
    conf: float | None = Form(None),
    iou: float | None = Form(None),
    imgsz: int | None = Form(None),
    max_det: int | None = Form(None),
    batch: int | None = Form(None),
    dedup: bool | None = Form(None),
    dedup_thr: float | None = Form(None),
    classes: str | None = Form(None, description="逗号/空格分隔的类别名或 id，如 InlineFormula,Table"),
    return_crops: bool | None = Form(None),
    crop_format: str | None = Form(None),
    crop_quality: int | None = Form(None),
    crop_padding: int | None = Form(None),
    page_indices: str | None = Form(None, description="逗号分隔的 0 基页码，缺省按上传顺序"),
    pdf_sizes: str | None = Form(None, description="逗号分隔的 PDF 页面尺寸 WxH（pt），用于换算 bbox_pdf"),
) -> dict[str, Any]:
    detector = require_detector()

    uploads = list(files or [])
    if file is not None:
        uploads.append(file)
    if not uploads:
        raise HTTPException(status_code=400, detail="请至少上传一张图片（字段名 files 或 file）")
    _check_image_count(len(uploads), detector.config)

    try:
        class_list = parse_class_list(classes)
        page_index_list = parse_int_list(page_indices)
        pdf_size_list = parse_size_list(pdf_sizes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None

    params = resolve_params(
        PredictParams(
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            max_det=max_det,
            batch=batch,
            dedup=dedup,
            dedup_thr=dedup_thr,
            classes=class_list,
            return_crops=return_crops,
            crop_format=crop_format,
            crop_quality=crop_quality,
            crop_padding=crop_padding,
        ),
        detector.config,
    )

    images: list[DecodedImage] = []
    for index, upload in enumerate(uploads):
        name = upload.filename or f"image_{index:04d}"
        page_index = page_index_list[index] if index < len(page_index_list) and page_index_list[index] is not None else index
        pdf_size = pdf_size_list[index] if index < len(pdf_size_list) else None
        try:
            raw = upload.file.read()
        finally:
            upload.file.close()
        try:
            images.append(_build_image(raw, name, page_index, pdf_size, detector.config))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"图片 {name} {exc}") from None

    return _run(detector, images, params)


if __name__ == "__main__":  # 本地调试：python app.py
    import uvicorn

    # 端口优先取 SERVICE_PORT，其次 PORT（与容器内 CMD 的取值顺序一致）
    port = os.environ.get("SERVICE_PORT") or os.environ.get("PORT") or "8000"
    uvicorn.run(
        app,
        host=os.environ.get("SERVICE_HOST", "0.0.0.0"),
        port=int(port),
        log_level=os.environ.get("LOG_LEVEL", "info"),
    )
