"""会话（session）存储。

多客户端并发时，缓存池里会同时存在来自不同 PDF 的任务；每条任务都带
``session_id``，结果按 session 归档，互不干扰。会话记录同时用于限流统计
与结果回查（``GET /v1/router/session/{session_id}/results``）。
"""

from __future__ import annotations

import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

DEFAULT_KEEP_RESULTS = 8


def new_session_id() -> str:
    return f"sess_{uuid.uuid4().hex[:16]}"


def new_request_id() -> str:
    return f"req_{uuid.uuid4().hex[:12]}"


@dataclass
class SessionRecord:
    session_id: str
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    requests: int = 0
    succeeded: int = 0
    failed: int = 0
    formula_submitted: int = 0
    formula_completed: int = 0
    table_submitted: int = 0
    table_completed: int = 0
    in_flight: int = 0
    last_error: str | None = None
    # 最近若干次请求的合并结果（OrderedDict 便于按插入顺序淘汰）
    results: OrderedDict[str, dict[str, Any]] = field(default_factory=OrderedDict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "requests": self.requests,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "formula_submitted": self.formula_submitted,
            "formula_completed": self.formula_completed,
            "table_submitted": self.table_submitted,
            "table_completed": self.table_completed,
            "in_flight": self.in_flight,
            "last_error": self.last_error,
            "result_count": len(self.results),
        }


class SessionStore:
    """进程内会话表；带 TTL 清理，避免长跑服务内存无界增长。"""

    def __init__(self, ttl_s: float = 1800.0, keep_results: int = DEFAULT_KEEP_RESULTS) -> None:
        self.ttl_s = ttl_s
        self.keep_results = max(1, keep_results)
        self._sessions: dict[str, SessionRecord] = {}

    # ----------------------------------------------------------------- #
    # 生命周期
    # ----------------------------------------------------------------- #
    def ensure(self, session_id: str | None) -> SessionRecord:
        """取（或创建）会话；每次调用记一次请求。"""
        self.purge_expired()
        sid = session_id or new_session_id()
        record = self._sessions.get(sid)
        if record is None:
            record = SessionRecord(session_id=sid)
            self._sessions[sid] = record
        record.updated_at = time.time()
        record.requests += 1
        return record

    def get(self, session_id: str) -> SessionRecord | None:
        return self._sessions.get(session_id)

    def drop(self, session_id: str) -> bool:
        return self._sessions.pop(session_id, None) is not None

    def purge_expired(self) -> int:
        if self.ttl_s <= 0:
            return 0
        deadline = time.time() - self.ttl_s
        stale = [sid for sid, record in self._sessions.items() if record.updated_at < deadline and record.in_flight == 0]
        for sid in stale:
            self._sessions.pop(sid, None)
        return len(stale)

    def snapshot(self) -> list[dict[str, Any]]:
        self.purge_expired()
        return [record.as_dict() for record in self._sessions.values()]

    # ----------------------------------------------------------------- #
    # 计数
    # ----------------------------------------------------------------- #
    def record_submitted(self, session_id: str, kind: str, count: int) -> None:
        record = self._sessions.get(session_id)
        if record is None:
            return
        if kind == "formula":
            record.formula_submitted += count
        else:
            record.table_submitted += count

    def record_outcome(self, session_id: str, kind: str, ok: bool, error: str | None = None) -> None:
        record = self._sessions.get(session_id)
        if record is None:
            return
        if kind == "formula":
            record.formula_completed += 1
        else:
            record.table_completed += 1
        if not ok:
            record.failed += 1
            if error:
                record.last_error = error
        record.updated_at = time.time()

    def record_request(self, session_id: str, ok: bool, error: str | None = None) -> None:
        record = self._sessions.get(session_id)
        if record is None:
            return
        if ok:
            record.succeeded += 1
        else:
            record.failed += 1
        if error:
            # 请求整体成功但个别检测框识别失败时，也会把原因记下来便于排查
            record.last_error = error
        record.in_flight = max(0, record.in_flight - 1)
        record.updated_at = time.time()

    def enter_request(self, session_id: str) -> None:
        record = self._sessions.get(session_id)
        if record is not None:
            record.in_flight += 1

    # ----------------------------------------------------------------- #
    # 结果归档
    # ----------------------------------------------------------------- #
    def store_result(self, session_id: str, request_id: str, payload: dict[str, Any]) -> None:
        record = self._sessions.get(session_id)
        if record is None:
            return
        record.results[request_id] = payload
        while len(record.results) > self.keep_results:
            record.results.popitem(last=False)
        record.updated_at = time.time()

    def list_results(self, session_id: str) -> list[dict[str, Any]]:
        record = self._sessions.get(session_id)
        if record is None:
            return []
        return [
            {"request_id": request_id, "result": payload}
            for request_id, payload in record.results.items()
        ]
