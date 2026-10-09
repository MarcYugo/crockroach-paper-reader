"""router_service：公式/表格识别流水线路由服务（FastAPI）。

请求链路：

    client ──▶ router /v1/router/predict
                 │  ① 转发 PDF 渲染图给 yolov13 检测服务（生产者）
                 │  ② 按 class_name 把检测框分成公式 / 表格 / 图片
                 │  ③ 公式框、表格框分别投入公式缓存池、表格缓存池（缓冲池）
                 │  ④ 两条消费者流水线并行调用 pp-formulanet-plus-l / slanet_plus
                 ▼  ⑤ 结果回填 + 合并，返回「检测结果 + 识别内容」的总数据

接口说明与示例见同目录 README.md。
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from base64 import b64encode
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

from . import __version__
from .clients import DownstreamClients, DownstreamError
from .config import Settings, get_settings
from .grouping import Taxonomy, classify_detections
from .merge import build_response
from .pipeline import Pipeline
from .schemas import (
    PredictParams,
    RouterPredictRequest,
    RouterPredictResponse,
    parse_class_list,
    parse_int_list,
    parse_size_list,
)
from .sessions import SessionStore, new_request_id

logger = logging.getLogger("router_service")


class _State:
    """进程内共享状态。"""

    def __init__(self) -> None:
        self.settings: Settings | None = None
        self.clients: DownstreamClients | None = None
        self.sessions: SessionStore | None = None
        self.taxonomy: Taxonomy | None = None
        self.pipeline: Pipeline | None = None
        self.gate: RequestGate | None = None
        self.started_at: float = time.time()


state = _State()


class RequestGate:
    """请求准入控制：同时最多 ``MAX_CONCURRENT_REQUESTS`` 个请求在处理。

    多出来的请求在这里排队（最多等 ``QUEUE_TIMEOUT`` 秒），避免把下游三个
    服务压垮；超时返回 503。
    """

    def __init__(self, limit: int, timeout_s: float, max_queue: int) -> None:
        self.limit = limit
        self.timeout_s = timeout_s
        self.max_queue = max_queue
        self._semaphore = asyncio.Semaphore(limit)
        self.active = 0
        self.waiting = 0
        self.rejected = 0

    @asynccontextmanager
    async def slot(self):
        if self.waiting >= self.max_queue:
            self.rejected += 1
            raise HTTPException(status_code=503, detail=f"服务繁忙：排队已满（{self.max_queue}）")
        self.waiting += 1
        try:
            await asyncio.wait_for(self._semaphore.acquire(), timeout=self.timeout_s)
        except asyncio.TimeoutError:
            self.rejected += 1
            raise HTTPException(
                status_code=503,
                detail=f"服务繁忙：并发上限 {self.limit}，排队超过 {self.timeout_s:.0f}s",
            ) from None
        finally:
            self.waiting -= 1
        self.active += 1
        try:
            yield
        finally:
            self.active -= 1
            self._semaphore.release()

    def stats(self) -> dict[str, Any]:
        return {
            "limit": self.limit,
            "max_queue": self.max_queue,
            "active": self.active,
            "waiting": self.waiting,
            "rejected": self.rejected,
            "queue_timeout_s": self.timeout_s,
        }


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    settings = get_settings()
    state.settings = settings
    state.started_at = time.time()
    logger.setLevel(settings.log_level.upper())

    state.taxonomy = Taxonomy(settings.formula_classes, settings.table_classes, settings.figure_classes)
    state.sessions = SessionStore(ttl_s=settings.session_ttl_s)
    state.gate = RequestGate(settings.max_concurrent_requests, settings.queue_timeout_s, settings.max_queue)

    clients = DownstreamClients(settings)
    await clients.start()
    state.clients = clients

    pipeline = Pipeline(settings, clients, state.sessions)
    await pipeline.start()
    state.pipeline = pipeline

    logger.info(
        "router_service 就绪：检测=%s，公式=%s，表格=%s",
        settings.detection_url,
        settings.formula_url,
        settings.table_url,
    )
    try:
        yield
    finally:
        if state.pipeline is not None:
            await state.pipeline.stop()
        if state.clients is not None:
            await state.clients.aclose()


app = FastAPI(
    title="公式/表格识别路由服务",
    version=__version__,
    description=(
        "把 PDF 渲染图转发给 YOLOv13 检测服务，按类别分流到公式缓存池与表格缓存池，"
        "再由 pp-formulanet-plus-l / slanet_plus 并行识别，最后合并返回。"
    ),
    lifespan=lifespan,
)


# --------------------------------------------------------------------------- #
# 依赖
# --------------------------------------------------------------------------- #
def _settings() -> Settings:
    if state.settings is None:
        return get_settings()
    return state.settings


def require_pipeline() -> Pipeline:
    if state.pipeline is None:
        raise HTTPException(status_code=503, detail="流水线尚未就绪")
    return state.pipeline


def require_clients() -> DownstreamClients:
    if state.clients is None:
        raise HTTPException(status_code=503, detail="HTTP 客户端尚未就绪")
    return state.clients


def require_sessions() -> SessionStore:
    if state.sessions is None:
        raise HTTPException(status_code=503, detail="会话存储尚未就绪")
    return state.sessions


def require_gate() -> RequestGate:
    if state.gate is None:
        raise HTTPException(status_code=503, detail="请求网关尚未就绪")
    return state.gate


def require_taxonomy() -> Taxonomy:
    if state.taxonomy is None:
        return Taxonomy(_settings().formula_classes, _settings().table_classes, _settings().figure_classes)
    return state.taxonomy


# --------------------------------------------------------------------------- #
# 核心流程
# --------------------------------------------------------------------------- #
def build_detection_params(settings: Settings, params: PredictParams | None) -> dict[str, Any]:
    """请求参数覆盖服务默认值；``return_crops`` 由服务强制（识别需要框内图像）。"""
    merged = settings.detection_default_params()
    if params is not None:
        for key, value in params.model_dump(exclude_none=True).items():
            if key == "return_crops":
                continue
            merged[key] = value
    merged["return_crops"] = settings.force_return_crops
    return merged


async def run_pipeline(
    *,
    images: list[dict[str, Any]],
    params: PredictParams | None,
    session_id: str | None,
    keep_crops: bool,
) -> dict[str, Any]:
    settings = _settings()
    clients = require_clients()
    pipeline = require_pipeline()
    sessions = require_sessions()
    gate = require_gate()
    taxonomy = require_taxonomy()

    request_id = new_request_id()
    record = sessions.ensure(session_id)
    sid = record.session_id

    total_start = time.perf_counter()
    async with gate.slot():
        sessions.enter_request(sid)
        try:
            detect_params = build_detection_params(settings, params)

            # ① 生产者：检测
            detection_start = time.perf_counter()
            try:
                detect_result = await clients.detect(images, detect_params)
            except DownstreamError as exc:
                raise HTTPException(status_code=502, detail=f"检测服务调用失败：{exc}") from None
            detection_ms = (time.perf_counter() - detection_start) * 1000

            detections = detect_result.get("detections")
            if not isinstance(detections, list):
                raise HTTPException(status_code=502, detail="检测服务响应缺少 detections 字段")

            # ② 分流：公式 / 表格 / 图片
            classified = classify_detections(detections, taxonomy)

            # ③ 投递到缓存池，④ 等待消费者回填结果
            recognition_start = time.perf_counter()
            pendings = await pipeline.submit(sid, classified)
            outcomes = await pipeline.gather(pendings, settings.request_timeout_s)
            recognition_ms = (time.perf_counter() - recognition_start) * 1000

            # ⑤ 合并
            total_ms = (time.perf_counter() - total_start) * 1000
            result = build_response(
                detect_result,
                pendings,
                outcomes,
                session_id=sid,
                request_id=request_id,
                detection_ms=detection_ms,
                recognition_ms=recognition_ms,
                total_ms=total_ms,
                keep_crops=keep_crops,
            )
        except HTTPException as exc:
            sessions.record_request(sid, False, str(exc.detail))
            raise
        except asyncio.CancelledError:
            sessions.record_request(sid, False, "请求被取消")
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("请求 %s 处理失败", request_id)
            sessions.record_request(sid, False, f"{type(exc).__name__}: {exc}")
            raise HTTPException(status_code=500, detail=f"路由处理失败: {exc}") from None

    failed = result["router"]["recognition"]["failed"]
    sessions.record_request(sid, True, None if not failed else f"{failed} 个检测框识别失败")
    sessions.store_result(sid, request_id, result)
    logger.info(
        "请求完成 %s：session=%s，检测 %d ms，识别 %d ms，总计 %d ms，公式 %d / 表格 %d / 失败 %d",
        request_id,
        sid,
        detection_ms,
        recognition_ms,
        total_ms,
        result["router"]["counts"]["formula"],
        result["router"]["counts"]["table"],
        failed,
    )
    return result


# --------------------------------------------------------------------------- #
# 探针 / 观测
# --------------------------------------------------------------------------- #
@app.get("/", include_in_schema=False)
def root() -> dict[str, Any]:
    return {
        "service": "router-service",
        "version": __version__,
        "docs": "/docs",
        "health": "/health",
        "info": "/v1/info",
        "predict": ["POST /v1/router/predict（JSON + base64）", "POST /v1/router/predict/upload（multipart）"],
    }


@app.get("/health", summary="健康检查（自身存活；下游状态见 downstream 字段）")
async def health() -> dict[str, Any]:
    clients = state.clients
    downstream: dict[str, Any] = {}
    if clients is not None:
        checks = await asyncio.gather(
            clients.detect_health(),
            clients.formula_health(),
            clients.table_health(),
            return_exceptions=True,
        )
        for name, outcome in zip(("detection", "formula", "table"), checks, strict=True):
            if isinstance(outcome, BaseException):
                downstream[name] = {"status": "error", "detail": str(outcome)}
            else:
                downstream[name] = {"status": str(outcome.get("status", "ok"))}
    ready = state.pipeline is not None and state.clients is not None
    return {
        "status": "ok" if ready else "loading",
        "service": "router-service",
        "version": __version__,
        "uptime_s": round(time.time() - state.started_at, 1),
        "downstream": downstream,
    }


@app.get("/ready", summary="就绪探针（下游任一不可用则返回 503）")
async def ready() -> JSONResponse:
    clients = state.clients
    if state.pipeline is None or clients is None:
        return JSONResponse(status_code=503, content={"ready": False, "reason": "流水线尚未就绪"})
    checks = await asyncio.gather(
        clients.detect_health(),
        clients.formula_health(),
        clients.table_health(),
        return_exceptions=True,
    )
    statuses = ["detection", "formula", "table"]
    details = {
        name: ("ok" if not isinstance(outcome, BaseException) else str(outcome))
        for name, outcome in zip(statuses, checks, strict=True)
    }
    all_ok = all(value == "ok" for value in details.values())
    return JSONResponse(status_code=200 if all_ok else 503, content={"ready": all_ok, "downstream": details})


@app.get("/v1/info", summary="服务信息与配置")
def info() -> dict[str, Any]:
    settings = _settings()
    return {
        "service": "router-service",
        "version": __version__,
        "config": settings.public_info(),
        "downstream": {
            "detection": settings.detection_url,
            "formula": settings.formula_url,
            "table": settings.table_url,
        },
    }


@app.get("/v1/stats", summary="流水线与请求统计")
def stats() -> dict[str, Any]:
    pipeline = state.pipeline
    sessions = state.sessions
    return {
        "uptime_s": round(time.time() - state.started_at, 1),
        "gate": state.gate.stats() if state.gate is not None else None,
        "pipeline": pipeline.stats() if pipeline is not None else None,
        "sessions": sessions.snapshot() if sessions is not None else [],
    }


@app.get("/v1/pools", summary="公式 / 表格缓存池状态")
def pools() -> dict[str, Any]:
    pipeline = state.pipeline
    if pipeline is None:
        raise HTTPException(status_code=503, detail="流水线尚未就绪")
    return pipeline.stats()


@app.get("/metrics", summary="Prometheus 指标", response_class=PlainTextResponse)
def metrics() -> PlainTextResponse:
    pipeline = state.pipeline
    gate = state.gate
    lines = [
        "# HELP router_requests_active 正在处理的请求数",
        "# TYPE router_requests_active gauge",
        f"router_requests_active {gate.active if gate else 0}",
        "# HELP router_requests_waiting 排队等待准入的请求数",
        "# TYPE router_requests_waiting gauge",
        f"router_requests_waiting {gate.waiting if gate else 0}",
        "# HELP router_requests_rejected_total 因排队超时被拒绝的请求数",
        "# TYPE router_requests_rejected_total counter",
        f"router_requests_rejected_total {gate.rejected if gate else 0}",
    ]
    if pipeline is not None:
        for name, pool in (("formula", pipeline.formula_pool), ("table", pipeline.table_pool)):
            stats = pool.stats
            lines.extend(
                [
                    f"# HELP router_{name}_pool_size 缓存池中待处理任务数",
                    f"# TYPE router_{name}_pool_size gauge",
                    f"router_{name}_pool_size {pool.size}",
                    f"# HELP router_{name}_pool_in_flight 正在识别中的任务数",
                    f"# TYPE router_{name}_pool_in_flight gauge",
                    f"router_{name}_pool_in_flight {pool.in_flight}",
                    f"# HELP router_{name}_submitted_total 累计投递任务数",
                    f"# TYPE router_{name}_submitted_total counter",
                    f"router_{name}_submitted_total {stats.submitted}",
                    f"# HELP router_{name}_failed_total 累计识别失败数",
                    f"# TYPE router_{name}_failed_total counter",
                    f"router_{name}_failed_total {stats.failed}",
                ]
            )
    return PlainTextResponse("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
# 会话
# --------------------------------------------------------------------------- #
@app.get("/v1/router/sessions", summary="列出所有会话")
def list_sessions() -> dict[str, Any]:
    sessions = require_sessions()
    return {"count": len(sessions.snapshot()), "sessions": sessions.snapshot()}


@app.get("/v1/router/session/{session_id}", summary="查看单个会话状态")
def get_session(session_id: str) -> dict[str, Any]:
    record = require_sessions().get(session_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"会话不存在：{session_id}")
    return record.as_dict()


@app.get("/v1/router/session/{session_id}/results", summary="查看会话最近若干次请求的合并结果")
def get_session_results(session_id: str) -> dict[str, Any]:
    sessions = require_sessions()
    if sessions.get(session_id) is None:
        raise HTTPException(status_code=404, detail=f"会话不存在：{session_id}")
    results = sessions.list_results(session_id)
    return {"session_id": session_id, "count": len(results), "results": results}


@app.delete("/v1/router/session/{session_id}", summary="清除会话（释放该客户端的缓存结果）")
def delete_session(session_id: str) -> dict[str, Any]:
    removed = require_sessions().drop(session_id)
    if not removed:
        raise HTTPException(status_code=404, detail=f"会话不存在：{session_id}")
    return {"session_id": session_id, "removed": True}


# --------------------------------------------------------------------------- #
# 推理
# --------------------------------------------------------------------------- #
@app.post(
    "/v1/router/predict",
    summary="检测 + 公式/表格识别（JSON 请求，base64 图片）",
    responses={200: {"model": RouterPredictResponse, "description": "检测结果 + 公式 LaTeX + 表格 HTML"}},
)
async def predict(payload: RouterPredictRequest) -> dict[str, Any]:
    images = [image.model_dump(exclude_none=True) for image in payload.images]
    return await run_pipeline(
        images=images,
        params=payload.params,
        session_id=payload.session_id,
        keep_crops=payload.keep_crops,
    )


@app.post(
    "/v1/router/predict/upload",
    summary="检测 + 公式/表格识别（multipart 文件上传）",
    responses={200: {"model": RouterPredictResponse, "description": "检测结果 + 公式 LaTeX + 表格 HTML"}},
)
async def predict_upload(
    files: list[UploadFile] | None = File(None, description="PDF 渲染页图片，按页顺序上传（字段名 files，可重复）"),
    file: UploadFile | None = File(None, description="单张图片的简写字段，等价于 files"),
    session_id: str | None = Form(None, description="客户端会话 id，同一 PDF 复用同一个值"),
    keep_crops: bool = Form(True, description="是否在响应中保留框内区域图像"),
    conf: float | None = Form(None),
    iou: float | None = Form(None),
    imgsz: int | None = Form(None),
    max_det: int | None = Form(None),
    batch: int | None = Form(None),
    dedup: bool | None = Form(None),
    dedup_thr: float | None = Form(None),
    classes: str | None = Form(None, description="逗号/空格分隔的类别名或 id，如 InlineFormula,Table"),
    crop_format: str | None = Form(None),
    crop_quality: int | None = Form(None),
    crop_padding: int | None = Form(None),
    page_indices: str | None = Form(None, description="逗号分隔的 0 基页码，缺省按上传顺序"),
    pdf_sizes: str | None = Form(None, description="逗号分隔的 PDF 页面尺寸 WxH（pt），用于换算 bbox_pdf"),
) -> dict[str, Any]:
    uploads = list(files or [])
    if file is not None:
        uploads.append(file)
    if not uploads:
        raise HTTPException(status_code=400, detail="请至少上传一张图片（字段名 files 或 file）")

    try:
        class_list = parse_class_list(classes)
        page_index_list = parse_int_list(page_indices)
        pdf_size_list = parse_size_list(pdf_sizes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None

    params = PredictParams(
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        max_det=max_det,
        batch=batch,
        dedup=dedup,
        dedup_thr=dedup_thr,
        classes=class_list,
        crop_format=crop_format if crop_format in (None, "png", "jpg", "jpeg") else None,
        crop_quality=crop_quality,
        crop_padding=crop_padding,
    )

    images: list[dict[str, Any]] = []
    for index, upload in enumerate(uploads):
        try:
            raw = await upload.read()
        finally:
            await upload.close()
        if not raw:
            raise HTTPException(status_code=400, detail=f"第 {index} 张图片内容为空")
        page_index = (
            page_index_list[index]
            if index < len(page_index_list) and page_index_list[index] is not None
            else index
        )
        pdf_size = pdf_size_list[index] if index < len(pdf_size_list) else None
        item: dict[str, Any] = {
            "data": b64encode(raw).decode("ascii"),
            "name": upload.filename or f"image_{index:04d}",
            "page_index": page_index,
        }
        if pdf_size:
            item["pdf_width"], item["pdf_height"] = pdf_size
        images.append(item)

    return await run_pipeline(
        images=images,
        params=params,
        session_id=session_id,
        keep_crops=keep_crops,
    )


if __name__ == "__main__":  # 本地调试：python -m app.main
    import uvicorn

    _settings_value = get_settings()
    uvicorn.run(
        app,
        host=os.environ.get("SERVICE_HOST", _settings_value.host),
        port=int(os.environ.get("SERVICE_PORT") or os.environ.get("PORT") or _settings_value.port),
        log_level=os.environ.get("LOG_LEVEL", _settings_value.log_level).lower(),
    )
