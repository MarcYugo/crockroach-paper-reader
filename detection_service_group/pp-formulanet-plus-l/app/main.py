"""PP-FormulaNet_plus-L 公式识别 HTTP 服务。

接口一览
--------
GET  /health                          存活探针（不依赖模型）
GET  /ready                           就绪探针（模型加载完成才返回 200）
GET  /v1/info                         服务与模型配置信息
GET  /v1/stats                        并发/吞吐统计
GET  /metrics                         Prometheus 文本格式指标
POST /v1/formula/recognition          上传图片文件识别（支持多张）
POST /v1/formula/recognition/base64   base64 图片识别（支持多张）
POST /v1/formula/recognition/url      URL 图片识别（支持多张）
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
from app.engine import PoolBusyError, PoolNotReadyError, PredictorPool, extract_formula

# --------------------------------------------------------------------------- #
# 日志
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=getattr(logging, settings.log_level, logging.INFO),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("pp-formulanet")

# 临时文件目录（图片字节先落盘再交给 PaddleOCR，兼容性最好）
TEMP_DIR: str = settings.temp_dir
_pool: Optional[PredictorPool] = None
_started_at = time.time()

# 只接收图片：PDF 必须由客户端渲染/裁剪成图片后再上传（见 test_pp_formulanet_service.py）。
# PaddleOCR 虽然也吃 PDF，但那是“一页当一个公式”，对论文页面会给出无意义结果，所以显式拒绝。
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
    logger.info("开始加载 PP-FormulaNet_plus-L，pool_size=%d ...", settings.pool_size)
    await run_in_threadpool(_pool.start)   # 同步阻塞加载放到线程里，避免卡住事件循环
    logger.info("服务就绪：%s", settings.public_info())
    try:
        yield
    finally:
        if _pool is not None:
            _pool.stop()


app = FastAPI(
    title="PP-FormulaNet_plus-L Formula Recognition Service",
    description="基于 PaddleOCR 的公式识别推理服务（支持并发/批量）",
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
    include_raw: bool = Field(False, description="是否返回 PaddleX 原始结果")


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
                "服务只接收图片，不支持直接上传 PDF（PDF 会被当成“一页一个公式”）。"
                "请在客户端用 PyMuPDF 把页面或公式区域渲染成 PNG 再上传，"
                "参考 test_pp_formulanet_service.py。"
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
        item: Dict[str, Any] = {
            "index": index,
            "filename": meta.get("filename") or os.path.basename(paths[index]),
            "rec_formula": extract_formula(data),
            "error": None,
        }
        if include_raw:
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
        "service": "pp-formulanet-plus-l",
        "version": __version__,
        "model": "PP-FormulaNet_plus-L",
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
        "# HELP ppformulanet_up 模型是否就绪 (1/0)",
        "# TYPE ppformulanet_up gauge",
        f"ppformulanet_up {1 if data['ready'] else 0}",
        "# HELP ppformulanet_inflight 正在处理的请求数",
        "# TYPE ppformulanet_inflight gauge",
        f"ppformulanet_inflight {data['inflight']}",
        "# HELP ppformulanet_waiting 排队等待的请求数",
        "# TYPE ppformulanet_waiting gauge",
        f"ppformulanet_waiting {data['waiting']}",
        "# HELP ppformulanet_idle 空闲 predictor 数",
        "# TYPE ppformulanet_idle gauge",
        f"ppformulanet_idle {data['idle']}",
        "# HELP ppformulanet_pool_size predictor 总数",
        "# TYPE ppformulanet_pool_size gauge",
        f"ppformulanet_pool_size {data['pool_size']}",
        "# HELP ppformulanet_requests_total 累计推理请求数",
        "# TYPE ppformulanet_requests_total counter",
        f"ppformulanet_requests_total {data['requests_total']}",
        "# HELP ppformulanet_images_total 累计识别图片数",
        "# TYPE ppformulanet_images_total counter",
        f"ppformulanet_images_total {data['images_total']}",
        "# HELP ppformulanet_errors_total 累计推理失败数",
        "# TYPE ppformulanet_errors_total counter",
        f"ppformulanet_errors_total {data['errors_total']}",
        "# HELP ppformulanet_rejected_total 因过载被拒绝的请求数",
        "# TYPE ppformulanet_rejected_total counter",
        f"ppformulanet_rejected_total {data['rejected_total']}",
        "# HELP ppformulanet_uptime_seconds 服务运行时长",
        "# TYPE ppformulanet_uptime_seconds gauge",
        f"ppformulanet_uptime_seconds {data['uptime_s']}",
    ]
    return PlainTextResponse("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
# 推理接口
# --------------------------------------------------------------------------- #
@app.post("/v1/formula/recognition", summary="公式识别（多文件上传）", tags=["inference"])
async def recognition_files(
    files: Optional[List[UploadFile]] = File(default=None, description="图片文件，可多个"),
    file: Optional[UploadFile] = File(default=None, description="单文件写法（与 files 等价）"),
    batch_size: Optional[int] = Form(default=None, description="批大小"),
    include_raw: bool = Form(default=False, description="是否返回原始结果"),
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


@app.post("/v1/formula/recognition/base64", summary="公式识别（base64）", tags=["inference"])
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


@app.post("/v1/formula/recognition/url", summary="公式识别（URL）", tags=["inference"])
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
    return {"service": "pp-formulanet-plus-l", "docs": "/docs", "health": "/health"}
