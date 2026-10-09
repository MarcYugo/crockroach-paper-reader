"""生产者-消费者流水线。

* 生产者：``/v1/router/predict`` 拿到 yolov13 的检测结果后，把公式框、表格框分别
  投递到公式池 / 表格池（``submit``）。
* 缓冲池：``BufferPool`` 持有待识别任务，慢消费者不会丢数据（block 策略自带背压）。
* 消费者：每个池有若干 worker 协程，按批取走任务并调用 pp-formulanet-plus-l /
  slanet_plus，结果回填到 future（``gather`` 等待）。

因此「检测 → 公式识别」与「检测 → 表格识别」是两条并行的流水线，
多个客户端的请求也会共用同一组 worker，从而形成持续的流水（而非一请求一模型）。
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

from .clients import DownstreamClients, DownstreamError
from .config import Settings
from .grouping import FORMULA, TABLE, ClassifiedDetection
from .pools import BufferPool, PoolFull, PoolItem
from .sessions import SessionStore

logger = logging.getLogger(__name__)


@dataclass
class PendingDetection:
    """一条检测框对应的待回收结果。"""

    classified: ClassifiedDetection
    future: asyncio.Future | None = None
    immediate: dict[str, Any] | None = None


class Pipeline:
    def __init__(self, settings: Settings, clients: DownstreamClients, sessions: SessionStore) -> None:
        self.settings = settings
        self.clients = clients
        self.sessions = sessions
        self.formula_pool = BufferPool(
            "formula",
            max_size=settings.pool_max_size,
            batch_size=settings.formula_batch_size,
            batch_window_ms=settings.batch_window_ms,
            overflow=settings.pool_overflow,
        )
        self.table_pool = BufferPool(
            "table",
            max_size=settings.pool_max_size,
            batch_size=settings.table_batch_size,
            batch_window_ms=settings.batch_window_ms,
            overflow=settings.pool_overflow,
        )
        self._tasks: list[asyncio.Task[None]] = []

    # ----------------------------------------------------------------- #
    # 生命周期
    # ----------------------------------------------------------------- #
    async def start(self) -> None:
        for index in range(self.settings.formula_workers):
            self._tasks.append(asyncio.create_task(self._worker_loop(self.formula_pool, FORMULA), name=f"formula-{index}"))
        for index in range(self.settings.table_workers):
            self._tasks.append(asyncio.create_task(self._worker_loop(self.table_pool, TABLE), name=f"table-{index}"))
        logger.info(
            "流水线已启动：公式 worker=%d（批 %d）、表格 worker=%d（批 %d）、批量窗口 %dms",
            self.settings.formula_workers,
            self.settings.formula_batch_size,
            self.settings.table_workers,
            self.settings.table_batch_size,
            self.settings.batch_window_ms,
        )

    async def stop(self) -> None:
        for pool in (self.formula_pool, self.table_pool):
            pool.close()
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

    # ----------------------------------------------------------------- #
    # 生产者侧
    # ----------------------------------------------------------------- #
    async def submit(self, session_id: str, classified: list[ClassifiedDetection]) -> list[PendingDetection]:
        """把分类后的检测框投递到对应缓存池。

        ``POOL_OVERFLOW=block`` 时池满会等待消费者腾出空位（背压，不丢数据）；
        ``drop`` 时立即把该条任务标记为失败。
        """
        pendings: list[PendingDetection] = []
        loop = asyncio.get_running_loop()

        for item in classified:
            if item.kind not in (FORMULA, TABLE):
                pendings.append(PendingDetection(classified=item, immediate=None))
                continue
            if not item.image:
                pendings.append(
                    PendingDetection(
                        classified=item,
                        immediate={"ok": False, "data": {}, "error": "检测结果未包含框内图像（crop），无法识别", "wait_ms": 0.0},
                    )
                )
                continue

            future: asyncio.Future = loop.create_future()
            pool = self.formula_pool if item.kind == FORMULA else self.table_pool
            pool_item = PoolItem(
                item_id=f"{session_id}:{item.kind}:{item.index}",
                session_id=session_id,
                kind=item.kind,
                image=item.image,
                detection_index=item.index,
                page_index=item.page_index,
                class_name=str(item.detection.get("class_name") or ""),
                future=future,
            )
            try:
                await pool.submit(pool_item)
            except PoolFull as exc:
                future.set_result({"ok": False, "data": {}, "error": str(exc), "wait_ms": 0.0})
            pendings.append(PendingDetection(classified=item, future=future))

        self.sessions.record_submitted(
            session_id, FORMULA, sum(1 for p in pendings if p.future is not None and p.classified.kind == FORMULA)
        )
        self.sessions.record_submitted(
            session_id, TABLE, sum(1 for p in pendings if p.future is not None and p.classified.kind == TABLE)
        )
        return pendings

    # ----------------------------------------------------------------- #
    # 消费者侧
    # ----------------------------------------------------------------- #
    async def gather(self, pendings: list[PendingDetection], timeout_s: float) -> list[dict[str, Any]]:
        """等待所有识别结果；超时只告警，不把仍在处理的任务标记为失败。"""
        futures: list[asyncio.Future] = [p.future for p in pendings if p.future is not None]
        warning_interval = max(0.1, timeout_s)
        wait_started = time.perf_counter()
        while pending_futures := [future for future in futures if not future.done()]:
            await asyncio.wait(pending_futures, timeout=warning_interval)
            pending_count = sum(not future.done() for future in futures)
            if pending_count:
                elapsed = time.perf_counter() - wait_started
                logger.warning(
                    "有 %d 条识别任务已等待 %.1fs，仍未完成；继续等待并保留任务",
                    pending_count,
                    elapsed,
                )

        outcomes: list[dict[str, Any]] = []
        for pending_item in pendings:
            if pending_item.immediate is not None:
                outcomes.append(pending_item.immediate)
                continue
            future = pending_item.future
            if future is None:
                outcomes.append({"ok": True, "data": {}, "error": None, "wait_ms": 0.0})
            else:
                outcomes.append(_safe_outcome(future))
        return outcomes

    async def _worker_loop(self, pool: BufferPool, kind: str) -> None:
        while True:
            batch = await pool.get_batch()
            try:
                await self._run_batch(pool, kind, batch)
            except asyncio.CancelledError:
                message = f"{kind} 消费者已停止"
                pool.complete(batch, [_error_outcome(item, message) for item in batch], error="cancelled")
                raise
            except Exception as exc:  # noqa: BLE001 - worker 必须永不退出
                logger.exception("%s 消费者处理批次失败", kind)
                pool.complete(batch, [_error_outcome(item, f"{type(exc).__name__}: {exc}") for item in batch], error=str(exc))

    async def _run_batch(self, pool: BufferPool, kind: str, batch: list[PoolItem]) -> None:
        images = [item.image for item in batch]
        started = time.perf_counter()
        try:
            retry_count = 0
            while True:
                try:
                    if kind == FORMULA:
                        raw_results = await self.clients.recognize_formulas(
                            images, self.settings.formula_batch_size
                        )
                    else:
                        raw_results = await self.clients.recognize_tables(
                            images, self.settings.table_batch_size
                        )
                    break
                except DownstreamError as exc:
                    if not _is_formula_busy_error(kind, exc):
                        raise
                    delay_s = min(0.5 * (2 ** min(retry_count, 4)), 8.0)
                    retry_count += 1
                    logger.warning(
                        "公式服务繁忙；保留当前 %d 张任务，%.1fs 后进行第 %d 次重试：%s",
                        len(batch),
                        delay_s,
                        retry_count,
                        exc,
                    )
                    await asyncio.sleep(delay_s)
        except Exception as exc:  # noqa: BLE001 - 下游异常转成每条任务的错误结果
            message = str(exc)
            logger.warning("%s 批次（%d 张）识别失败：%s", kind, len(batch), message)
            pool.complete(batch, [_error_outcome(item, message) for item in batch], error=message)
            self._record_outcomes(batch, kind, ok=False, error=message)
            return

        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        per_image_ms = round(elapsed_ms / max(1, len(batch)), 2)
        outcomes: list[dict[str, Any]] = []
        ok_count = 0
        for item, raw in zip(batch, raw_results, strict=True):
            outcome = _format_outcome(item, kind, raw, per_image_ms)
            outcomes.append(outcome)
            ok_count += 1 if outcome["ok"] else 0
        pool.complete(batch, outcomes)
        self._record_outcomes(batch, kind, ok=None, error=None, outcomes=outcomes)
        logger.debug("%s 批次完成：%d 张，耗时 %.1fms，成功 %d", kind, len(batch), elapsed_ms, ok_count)

    def _record_outcomes(
        self,
        batch: list[PoolItem],
        kind: str,
        *,
        ok: bool | None,
        error: str | None,
        outcomes: list[dict[str, Any]] | None = None,
    ) -> None:
        for index, item in enumerate(batch):
            if outcomes is None:
                result_ok = bool(ok)
                result_error = error
            else:
                result_ok = bool(outcomes[index]["ok"])
                result_error = outcomes[index]["error"]
            self.sessions.record_outcome(item.session_id, kind, result_ok, result_error)

    # ----------------------------------------------------------------- #
    # 观测
    # ----------------------------------------------------------------- #
    def stats(self) -> dict[str, Any]:
        return {
            "formula_pool": self.formula_pool.stats.as_dict(self.formula_pool),
            "table_pool": self.table_pool.stats.as_dict(self.table_pool),
            "workers": {
                "formula": self.settings.formula_workers,
                "table": self.settings.table_workers,
            },
        }


def _safe_outcome(future: asyncio.Future) -> dict[str, Any]:
    try:
        return dict(future.result())
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "data": {}, "error": f"{type(exc).__name__}: {exc}", "wait_ms": 0.0}


def _error_outcome(item: PoolItem, error: str) -> dict[str, Any]:
    return item.result(ok=False, error=error)


def _is_formula_busy_error(kind: str, exc: DownstreamError) -> bool:
    """只重试公式 predictor 池满导致的 503，其他下游错误仍显式失败。"""
    return (
        kind == FORMULA
        and exc.service == "formula"
        and exc.status_code == 503
        and ("所有 predictor 均被占用" in str(exc) or "排队等待超过" in str(exc))
    )


def _format_outcome(item: PoolItem, kind: str, raw: Any, elapsed_ms: float) -> dict[str, Any]:
    """把下游单条结果整理成统一结构，并带上 ``elapsed_ms``。"""
    result = item.result(ok=False, data={}, error=None)
    result["elapsed_ms"] = elapsed_ms
    if not isinstance(raw, dict):
        result["error"] = f"下游返回结构异常: {type(raw).__name__}"
        return result

    downstream_error = raw.get("error")
    if kind == FORMULA:
        latex = raw.get("rec_formula")
        data = {"latex": latex if isinstance(latex, str) and latex.strip() else None}
    else:
        html = raw.get("html")
        data = {
            "html": html if isinstance(html, str) and html.strip() else None,
            "structure_score": raw.get("structure_score"),
            "num_cells": raw.get("num_cells"),
        }

    main_value = data["latex"] if kind == FORMULA else data["html"]
    if downstream_error:
        result["error"] = str(downstream_error)
    elif main_value is None:
        result["error"] = "下游未返回识别内容"
    else:
        result["ok"] = True
    result["data"] = data
    return result
