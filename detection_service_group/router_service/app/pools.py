"""缓冲池。

生产者（检测服务返回的检测框）把待识别任务放进池子，消费者（worker 协程）
按批取走并调用下游识别服务，结果回填到 ``PoolItem.future``。

池子有最大容量：``POOL_OVERFLOW=block`` 时提交会等待空位（自动背压，保证不丢数据），
``POOL_OVERFLOW=drop`` 时直接丢弃并立即返回错误，避免慢消费者拖垮整个服务。
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PoolItem:
    """缓存池中的一条待识别任务。"""

    item_id: str
    session_id: str
    kind: str  # "formula" | "table"
    image: str  # 框内区域图像的 base64
    detection_index: int
    page_index: int
    class_name: str
    enqueued_at: float = field(default_factory=time.perf_counter)
    future: asyncio.Future | None = None

    def result(self, *, ok: bool, data: dict[str, Any] | None = None, error: str | None = None) -> dict[str, Any]:
        return {
            "ok": ok,
            "data": data or {},
            "error": error,
            "wait_ms": round((time.perf_counter() - self.enqueued_at) * 1000, 2),
        }


@dataclass
class PoolStats:
    submitted: int = 0
    consumed: int = 0
    rejected: int = 0
    failed: int = 0
    batches: int = 0
    wait_ms_total: float = 0.0
    last_batch_size: int = 0

    def as_dict(self, pool: "BufferPool") -> dict[str, Any]:
        avg_wait = self.wait_ms_total / self.consumed if self.consumed else 0.0
        return {
            "name": pool.name,
            "size": pool.size,
            "in_flight": pool.in_flight,
            "max_size": pool.max_size,
            "submitted": self.submitted,
            "consumed": self.consumed,
            "rejected": self.rejected,
            "failed": self.failed,
            "batches": self.batches,
            "last_batch_size": self.last_batch_size,
            "avg_wait_ms": round(avg_wait, 2),
        }


class PoolFull(RuntimeError):
    """池子已满且策略为 drop。"""


class BufferPool:
    """带批量取件与统计的异步缓冲池。"""

    def __init__(
        self,
        name: str,
        *,
        max_size: int,
        batch_size: int,
        batch_window_ms: int,
        overflow: str = "block",
    ) -> None:
        self.name = name
        self.max_size = max_size
        self.batch_size = max(1, batch_size)
        self.batch_window_s = max(0, batch_window_ms) / 1000.0
        self.overflow = overflow
        self.stats = PoolStats()
        self._queue: asyncio.Queue[PoolItem] = asyncio.Queue(maxsize=max_size)
        self._in_flight = 0
        self._closed = False

    # ----------------------------------------------------------------- #
    # 生产者侧
    # ----------------------------------------------------------------- #
    @property
    def size(self) -> int:
        return self._queue.qsize()

    @property
    def in_flight(self) -> int:
        return self._in_flight

    async def submit(self, item: PoolItem) -> None:
        """放入一条任务；满时按 ``overflow`` 策略阻塞等待或抛 ``PoolFull``。"""
        if self._closed:
            raise PoolFull(f"{self.name} 池已关闭")
        if self.overflow == "drop":
            try:
                self._queue.put_nowait(item)
            except asyncio.QueueFull:
                self.stats.rejected += 1
                raise PoolFull(f"{self.name} 池已满（{self.max_size}），已按 POOL_OVERFLOW=drop 丢弃") from None
        else:
            # block：队列满时 put 会等消费者腾出空位，形成背压
            await self._queue.put(item)
        self.stats.submitted += 1

    # ----------------------------------------------------------------- #
    # 消费者侧
    # ----------------------------------------------------------------- #
    async def get_batch(self) -> list[PoolItem]:
        """取一批任务：先阻塞等第一条，再在 ``batch_window`` 内尽量凑满。"""
        first = await self._queue.get()
        batch = [first]
        if self.batch_size > 1 and self.batch_window_s > 0:
            loop = asyncio.get_running_loop()
            deadline = loop.time() + self.batch_window_s
            while len(batch) < self.batch_size:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                try:
                    batch.append(await asyncio.wait_for(self._queue.get(), remaining))
                except asyncio.TimeoutError:
                    break
        elif self.batch_size > 1:
            while len(batch) < self.batch_size:
                try:
                    batch.append(self._queue.get_nowait())
                except asyncio.QueueEmpty:
                    break

        self._in_flight += len(batch)
        self.stats.batches += 1
        self.stats.last_batch_size = len(batch)
        for item in batch:
            self.stats.wait_ms_total += (time.perf_counter() - item.enqueued_at) * 1000
        return batch

    def complete(self, batch: list[PoolItem], results: list[dict[str, Any]], error: str | None = None) -> None:
        """把一批任务的结果回填到各自的 future。"""
        self._in_flight = max(0, self._in_flight - len(batch))
        for item, outcome in zip(batch, results, strict=True):
            if item.future is not None and not item.future.done():
                item.future.set_result(outcome)
        self.stats.consumed += len(batch)
        if error is not None:
            self.stats.failed += len(batch)

    def close(self) -> None:
        """标记为关闭：拒绝新任务，并把池中未处理的任务以错误结果回填，避免调用方永久等待。

        消费者协程由 Pipeline 在停机时取消。
        """
        self._closed = True
        while True:
            try:
                item = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if item.future is not None and not item.future.done():
                item.future.set_result(item.result(ok=False, error="服务正在停止，任务未执行"))

    @property
    def closed(self) -> bool:
        return self._closed
