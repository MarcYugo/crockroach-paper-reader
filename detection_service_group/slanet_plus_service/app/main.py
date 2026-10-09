"""SLANet_plus 表格结构识别 HTTP 服务。

接口一览
--------
GET  /health                               存活探针（不依赖模型）
GET  /ready                                就绪探针（模型加载完成才返回 200）
GET  /v1/info                              服务与模型配置信息
GET  /v1/stats                             并发/吞吐统计
GET  /metrics                              Prometheus 文本格式指标
POST /v1/table/structure/recognition          上传图片文件识别（支持多张）
POST /v1/table/structure/recognition/base64   base64 图片识别（支持多张）
POST /v1/table/structure/recognition/url      URL 图片识别（支持多张）

说明
----
SLANet_plus 只做「表格结构识别」：输入是**裁剪好的表格区域图片**，
输出是表格的 HTML 骨架（`<table><tr><td colspan=..></td>...`），
cell 内的文字为空。若需要 cell 文字，请叠加文本检测/识别模型，
或由上游服务（router_service）把识别到的文本回填进 HTML 骨架。
"""

from __future__ import annotations

import base64
import binascii
import logging
import os
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app import __version__
from app.config import settings
from app.engine import (
    PoolBusyError,
    PoolNotReadyError,
    PredictorPool,
    extract_cells,
    extract_score,
    extract_structure,
)

# --------------------------------------------------------------------------- #
# 日志
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=getattr(logging, settings.log_level, logging.INFO),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("slanet-plus")

# 临时文件目录（图片字节先落盘再交给 PaddleOCR，兼容性最好）
TEMP_DIR: str = settings.temp_dir
_pool: Optional[PredictorPool] = None
_started_at = time.time()

# 只接收图片：PDF 必须由客户端渲染/裁剪成图片后再上传（见 test_slanet_service.py）。
# 表格结构识别要的是「表格区域」，整页 PDF 直接喂进去识别效果无意义，所以显式拒绝。
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
_PDF_SUFFIXES = {".pdf"}


def _prepare_temp_dir(preferred: str) -> str:
    """优先使用内存盘目录，不可用时回退到系统临时目录。"""
    for candidate in (preferred, tempfile.gettempdir()):
        try:
            os.makedirs(candidate, exist_ok=True)
            probe = os.path.join(candidate, ".probe")
            with open(probe, "w", encoding="utf-8") as fh:
                fh.write("ok")
            os.remove(probe)
            return candidate
        except OSError:
            logger.warning("临时目录 %s 不可写，尝试下一个", candidate)
    raise RuntimeError("找不到可写的临时目录")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global TEMP_DIR, _pool
    TEMP_DIR = _prepare_temp_dir(settings.temp_dir)
    logger.info("临时目录：%s", TEMP_DIR)
    # 设备（GPU/CPU）在配置层已自动探测，这里显式打印决策原因便于排查
    logger.info("设备选择：%s", settings.device_note)
    _pool = PredictorPool(settings)
    logger.info("开始加载 SLANet_plus，pool_size=%d ...", settings.pool_size)
    await run_in_threadpool(_pool.start)   # 同步阻塞加载放到线程里，避免卡住事件循环
    logger.info("服务就绪：%s", settings.public_info())
    try:
        yield
    finally:
        if _pool is not None:
            _pool.stop()


app = FastAPI(
    title="SLANet_plus Table Structure Recognition Service",
    description="基于 PaddleOCR 的表格结构识别推理服务（支持并发/批量）",
    version=__version__,
    lifespan=lifespan,
)


def get_pool() -> PredictorPool:
    if _pool is None or not _pool.ready:
        raise HTTPException(status_code=503, detail="模型尚未加载完成")
    return _pool


# --------------------------------------------------------------------------- #
# 请求/响应模型
# --------------------------------------------------------------------------- #
class Base64Request(BaseModel):
    images: List[str] = Field(..., description="base64 字符串列表，可带 data:image/png;base64, 前缀")
    batch_size: Optional[int] = Field(None, description="批大小，默认取服务配置")
    include_raw: bool = Field(False, description="是否返回原始结构 token / cell 坐标 / PaddleX 原始结果")


class UrlRequest(BaseModel):
    urls: List[str] = Field(..., description="图片 URL 列表")
    batch_size: Optional[int] = Field(None)
    include_raw: bool = Field(False)


# --------------------------------------------------------------------------- #
# 工具函数
# --------------------------------------------------------------------------- #
def _check_batch(n: int, limit: int) -> None:
    if n == 0:
        raise HTTPException(status_code=400, detail="未提供任何图片")
    if n > limit:
        raise HTTPException(status_code=400, detail=f"单次最多 {limit} 张图片，当前 {n} 张")


def _write_temp(data: bytes, suffix: str) -> str:
    if suffix not in _IMAGE_SUFFIXES:
        suffix = ".png"
    handle, path = tempfile.mkstemp(dir=TEMP_DIR, suffix=suffix)
    with os.fdopen(handle, "wb") as fh:
        fh.write(data)
    return path


def _safe_suffix(filename: Optional[str], content_type: Optional[str]) -> str:
    if content_type == "application/pdf" or (
        filename and os.path.splitext(filename)[1].lower() in _PDF_SUFFIXES
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "服务只接收图片，不支持直接上传 PDF。SLANet_plus 需要的是「裁剪好的表格区域」，"
                "请在客户端用 PyMuPDF 把表格区域渲染成 PNG 再上传，参考 test_slanet_service.py。"
            ),
        )
    if filename:
        suffix = os.path.splitext(filename)[1].lower()
        if suffix in _IMAGE_SUFFIXES:
            return suffix
    if content_type:
        mapping = {
            "image/png": ".png",
            "image/jpeg": ".jpg",
            "image/jpg": ".jpg",
            "image/bmp": ".bmp",
            "image/webp": ".webp",
            "image/tiff": ".tiff",
        }
        if content_type in mapping:
            return mapping[content_type]
    return ".png"


def _validate_size(data: bytes) -> None:
    if len(data) > settings.max_image_mb * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"图片超过 {settings.max_image_mb}MB 限制",
        )


async def _infer(
    paths: List[str],
    batch_size: Optional[int],
    metas: List[Dict[str, Any]],
    include_raw: bool,
) -> Dict[str, Any]:
    """调用 predictor 池并组装统一响应。"""
    pool = get_pool()
    effective_batch = max(1, min(batch_size or settings.default_batch_size, settings.max_batch_size))
    # 批大小不能超过本次图片数，否则 Paddle 会补零浪费算力
    effective_batch = min(effective_batch, len(paths))

    begin = time.perf_counter()
    raw_results = await run_in_threadpool(pool.predict, paths, effective_batch)
    elapsed_ms = (time.perf_counter() - begin) * 1000

    results: List[Dict[str, Any]] = []
    for index, meta in enumerate(metas):
        data = raw_results[index] if index < len(raw_results) else {}
        html, tokens = extract_structure(data)
        cells = extract_cells(data)
        item: Dict[str, Any] = {
            "index": index,
            "filename": meta.get("filename") or os.path.basename(paths[index]),
            # 表格 HTML 骨架（cell 文字为空，需由上游回填）
            "html": html,
            "structure_score": extract_score(data),
            "num_cells": len(cells),
            "error": None,
        }
        if include_raw:
            item["structure"] = tokens
            item["cells"] = cells
            item["raw"] = data
        results.append(item)

    return {
        "count": len(results),
        "batch_size": effective_batch,
        "elapsed_ms": round(elapsed_ms, 2),
        "per_image_ms": round(elapsed_ms / max(1, len(results)), 2),
        "results": results,
    }


def _cleanup(paths: List[str]) -> None:
    for path in paths:
        try:
            os.remove(path)
        except OSError:
            pass


# --------------------------------------------------------------------------- #
# 全局异常处理
# --------------------------------------------------------------------------- #
@app.exception_handler(PoolBusyError)
async def _busy_handler(request: Request, exc: PoolBusyError) -> JSONResponse:
    logger.warning("拒绝请求（过载）：%s", exc)
    return JSONResponse(
        status_code=503,
        content={"detail": str(exc)},
        headers={"Retry-After": "1"},  # 提示客户端重试
    )


@app.exception_handler(PoolNotReadyError)
async def _not_ready_handler(request: Request, exc: PoolNotReadyError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(exc)}, headers={"Retry-After": "5"})


@app.middleware("http")
async def _access_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
    begin = time.perf_counter()
    response = await call_next(request)
    cost_ms = (time.perf_counter() - begin) * 1000
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-Ms"] = f"{cost_ms:.2f}"
    if request.url.path not in {"/health", "/metrics"}:
        logger.info(
            "%s %s -> %s (%.1fms) rid=%s",
            request.method,
            request.url.path,
            response.status_code,
            cost_ms,
            request_id,
        )
    return response


# --------------------------------------------------------------------------- #
# 探针 / 观测
# --------------------------------------------------------------------------- #
@app.get("/health", summary="存活探针", tags=["ops"])
async def health() -> Dict[str, Any]:
    pool_ready = bool(_pool and _pool.ready)
    return {
        "status": "ok" if pool_ready else "loading",
        "model_ready": pool_ready,
        "uptime_s": round(time.time() - _started_at, 1),
    }


@app.get("/ready", summary="就绪探针", tags=["ops"])
async def ready() -> JSONResponse:
    pool_ready = bool(_pool and _pool.ready)
    return JSONResponse(status_code=200 if pool_ready else 503, content={"ready": pool_ready})


@app.get("/v1/info", summary="服务信息", tags=["ops"])
async def info() -> Dict[str, Any]:
    return {
        "service": "slanet-plus",
        "version": __version__,
        "model": "SLANet_plus",
        "model_ready": bool(_pool and _pool.ready),
        "config": settings.public_info(),
    }


@app.get("/v1/stats", summary="并发统计", tags=["ops"])
async def stats() -> Dict[str, Any]:
    if _pool is None:
        raise HTTPException(status_code=503, detail="模型尚未加载完成")
    return _pool.stats()


@app.get("/metrics", summary="Prometheus 指标", tags=["ops"], response_class=PlainTextResponse)
async def metrics() -> PlainTextResponse:
    data = _pool.stats() if _pool else {
        "ready": False, "device": settings.device, "inflight": 0, "waiting": 0, "idle": 0,
        "requests_total": 0, "images_total": 0, "errors_total": 0,
        "rejected_total": 0, "uptime_s": 0, "pool_size": settings.pool_size,
    }
    lines = [
        "# HELP slanet_up 模型是否就绪 (1/0)",
        "# TYPE slanet_up gauge",
        f"slanet_up {1 if data['ready'] else 0}",
        "# HELP slanet_inflight 正在处理的请求数",
        "# TYPE slanet_inflight gauge",
        f"slanet_inflight {data['inflight']}",
        "# HELP slanet_waiting 排队等待的请求数",
        "# TYPE slanet_waiting gauge",
        f"slanet_waiting {data['waiting']}",
        "# HELP slanet_idle 空闲 predictor 数",
        "# TYPE slanet_idle gauge",
        f"slanet_idle {data['idle']}",
        "# HELP slanet_pool_size predictor 总数",
        "# TYPE slanet_pool_size gauge",
        f"slanet_pool_size {data['pool_size']}",
        "# HELP slanet_requests_total 累计推理请求数",
        "# TYPE slanet_requests_total counter",
        f"slanet_requests_total {data['requests_total']}",
        "# HELP slanet_images_total 累计识别图片数",
        "# TYPE slanet_images_total counter",
        f"slanet_images_total {data['images_total']}",
        "# HELP slanet_errors_total 累计推理失败数",
        "# TYPE slanet_errors_total counter",
        f"slanet_errors_total {data['errors_total']}",
        "# HELP slanet_rejected_total 因过载被拒绝的请求数",
        "# TYPE slanet_rejected_total counter",
        f"slanet_rejected_total {data['rejected_total']}",
        "# HELP slanet_uptime_seconds 服务运行时长",
        "# TYPE slanet_uptime_seconds gauge",
        f"slanet_uptime_seconds {data['uptime_s']}",
    ]
    return PlainTextResponse("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
# 推理接口
# --------------------------------------------------------------------------- #
@app.post("/v1/table/structure/recognition", summary="表格结构识别（多文件上传）", tags=["inference"])
async def recognition_files(
    files: Optional[List[UploadFile]] = File(default=None, description="表格区域图片，可多个"),
    file: Optional[UploadFile] = File(default=None, description="单文件写法（与 files 等价）"),
    batch_size: Optional[int] = Form(default=None, description="批大小"),
    include_raw: bool = Form(default=False, description="是否返回结构 token / cell 坐标 / 原始结果"),
) -> Dict[str, Any]:
    uploads: List[UploadFile] = []
    if files:
        uploads.extend(files)
    if file is not None:
        uploads.append(file)
    _check_batch(len(uploads), settings.max_batch_size)

    paths: List[str] = []
    metas: List[Dict[str, Any]] = []
    try:
        for upload in uploads:
            data = await upload.read()
            _validate_size(data)
            suffix = _safe_suffix(upload.filename, upload.content_type)
            paths.append(await run_in_threadpool(_write_temp, data, suffix))
            metas.append({"filename": upload.filename})
        return await _infer(paths, batch_size, metas, include_raw)
    finally:
        _cleanup(paths)


@app.post("/v1/table/structure/recognition/base64", summary="表格结构识别（base64）", tags=["inference"])
async def recognition_base64(payload: Base64Request) -> Dict[str, Any]:
    _check_batch(len(payload.images), settings.max_batch_size)

    paths: List[str] = []
    metas: List[Dict[str, Any]] = []
    try:
        for index, raw in enumerate(payload.images):
            content = raw.split(",", 1)[1] if raw.startswith("data:") and "," in raw else raw
            try:
                data = base64.b64decode(content, validate=False)
            except (binascii.Error, ValueError) as exc:
                raise HTTPException(status_code=400, detail=f"第 {index} 张图片 base64 解码失败：{exc}")
            _validate_size(data)
            paths.append(await run_in_threadpool(_write_temp, data, ".png"))
            metas.append({"filename": f"image_{index}.png"})
        return await _infer(paths, payload.batch_size, metas, payload.include_raw)
    finally:
        _cleanup(paths)


@app.post("/v1/table/structure/recognition/url", summary="表格结构识别（URL）", tags=["inference"])
async def recognition_url(payload: UrlRequest) -> Dict[str, Any]:
    _check_batch(len(payload.urls), settings.max_url_images)

    paths: List[str] = []
    metas: List[Dict[str, Any]] = []
    try:
        async with httpx.AsyncClient(timeout=settings.url_timeout, follow_redirects=True) as client:
            for index, url in enumerate(payload.urls):
                try:
                    response = await client.get(url)
                    response.raise_for_status()
                except httpx.HTTPError as exc:
                    raise HTTPException(status_code=400, detail=f"下载失败 {url}: {exc}")
                data = response.content
                _validate_size(data)
                suffix = _safe_suffix(
                    os.path.basename(url.split("?")[0]), response.headers.get("content-type")
                )
                paths.append(await run_in_threadpool(_write_temp, data, suffix))
                metas.append({"filename": os.path.basename(url.split("?")[0]) or f"image_{index}"})
        return await _infer(paths, payload.batch_size, metas, payload.include_raw)
    finally:
        _cleanup(paths)


@app.get("/", include_in_schema=False)
async def root() -> Dict[str, str]:
    return {"service": "slanet-plus", "docs": "/docs", "health": "/health"}
