"""下游服务的异步 HTTP 客户端。

三个下游服务（yolov13 检测 / pp-formulanet-plus-l 公式识别 / slanet_plus 表格识别）
共用同一个 ``httpx.AsyncClient``，连接池复用，避免每次请求重新建连。
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Mapping

import httpx

from .config import Settings

logger = logging.getLogger(__name__)

# 下游服务单次请求可接受的图片数上限（与两个识别服务默认的 MAX_BATCH_SIZE 一致）
_MAX_BATCH = 8


class DownstreamError(RuntimeError):
    """下游服务返回非 2xx，或响应结构不符合预期。"""

    def __init__(self, service: str, message: str, status_code: int | None = None) -> None:
        super().__init__(f"[{service}] {message}")
        self.service = service
        self.status_code = status_code


def _chunks(items: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _detail(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text[:300]
    if isinstance(body, Mapping):
        return str(body.get("detail") or body.get("message") or body)[:300]
    return str(body)[:300]


class DownstreamClients:
    """封装对三个下游服务的调用。"""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: httpx.AsyncClient | None = None

    async def start(self) -> None:
        if self._client is not None:
            return
        timeout = httpx.Timeout(
            connect=self._settings.connect_timeout_s,
            read=self._settings.read_timeout_s,
            write=self._settings.read_timeout_s,
            pool=self._settings.connect_timeout_s,
        )
        limits = httpx.Limits(
            max_connections=self._settings.max_connections,
            max_keepalive_connections=self._settings.max_connections,
        )
        self._client = httpx.AsyncClient(timeout=timeout, limits=limits)

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("DownstreamClients 尚未启动")
        return self._client

    async def _post(self, service: str, url: str, json_body: Mapping[str, Any]) -> dict[str, Any]:
        try:
            response = await self.client.post(url, json=dict(json_body))
        except httpx.TimeoutException as exc:
            raise DownstreamError(service, f"请求超时: {exc}") from None
        except httpx.HTTPError as exc:
            raise DownstreamError(service, f"网络错误: {exc}") from None
        if response.status_code >= 400:
            raise DownstreamError(service, _detail(response), response.status_code)
        try:
            payload = response.json()
        except ValueError:
            raise DownstreamError(service, "响应不是合法 JSON") from None
        if not isinstance(payload, dict):
            raise DownstreamError(service, f"响应结构异常: {type(payload).__name__}")
        return payload

    # ----------------------------------------------------------------- #
    # 生产者：yolov13 公式检测
    # ----------------------------------------------------------------- #
    async def detect(self, images: list[dict[str, Any]], params: Mapping[str, Any]) -> dict[str, Any]:
        """调用 yolov13 ``/predict``，返回原始检测 JSON。"""
        return await self._post(
            "detection",
            f"{self._settings.detection_url}/predict",
            {"images": images, "params": dict(params)},
        )

    async def detect_health(self) -> dict[str, Any]:
        return await self._get("detection", f"{self._settings.detection_url}/health")

    # ----------------------------------------------------------------- #
    # 消费者：公式 / 表格识别
    # ----------------------------------------------------------------- #
    async def recognize_formulas(self, images: list[str], batch_size: int) -> list[dict[str, Any]]:
        """调用 pp-formulanet-plus-l，返回按输入顺序排列的 results 列表。"""
        outputs: list[dict[str, Any]] = []
        size = max(1, min(batch_size, _MAX_BATCH))
        for group in _chunks(images, size):
            payload = await self._post(
                "formula",
                f"{self._settings.formula_url}/v1/formula/recognition/base64",
                {"images": group, "batch_size": len(group)},
            )
            outputs.extend(_collect(payload, len(group), "formula"))
        return outputs

    async def recognize_tables(self, images: list[str], batch_size: int) -> list[dict[str, Any]]:
        """调用 slanet_plus，返回按输入顺序排列的 results 列表。"""
        outputs: list[dict[str, Any]] = []
        size = max(1, min(batch_size, _MAX_BATCH))
        for group in _chunks(images, size):
            payload = await self._post(
                "table",
                f"{self._settings.table_url}/v1/table/structure/recognition/base64",
                {"images": group, "batch_size": len(group)},
            )
            outputs.extend(_collect(payload, len(group), "table"))
        return outputs

    async def formula_health(self) -> dict[str, Any]:
        return await self._get("formula", f"{self._settings.formula_url}/health")

    async def table_health(self) -> dict[str, Any]:
        return await self._get("table", f"{self._settings.table_url}/health")

    async def _get(self, service: str, url: str) -> dict[str, Any]:
        try:
            response = await self.client.get(url)
        except httpx.HTTPError as exc:
            raise DownstreamError(service, f"网络错误: {exc}") from None
        if response.status_code >= 400:
            raise DownstreamError(service, _detail(response), response.status_code)
        try:
            payload = response.json()
        except ValueError:
            raise DownstreamError(service, "响应不是合法 JSON") from None
        return payload if isinstance(payload, dict) else {"value": payload}


def _collect(payload: Mapping[str, Any], expected: int, service: str) -> list[dict[str, Any]]:
    """把下游 ``results`` 补齐/截断到与输入等长，保证顺序可以一一对应。"""
    results = payload.get("results")
    if not isinstance(results, list):
        raise DownstreamError(service, "响应缺少 results 字段")
    if len(results) < expected:
        logger.warning("%s 返回 %d 条结果，少于请求的 %d 张图片", service, len(results), expected)
        results = list(results) + [{} for _ in range(expected - len(results))]
    return [item if isinstance(item, dict) else {"error": str(item)} for item in results[:expected]]
