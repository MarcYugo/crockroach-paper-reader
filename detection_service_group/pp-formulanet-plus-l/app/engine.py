"""predictor 池：解决 Paddle 推理的并发问题。

设计要点
--------
1. Paddle Inference 的 predictor **不是线程安全**的，因此这里维护 N 个
   `FormulaRecognition` 实例（N = PREDICTOR_POOL_SIZE），每次请求独占一个，
   用完归还，从而实现 N 路并行推理。
2. 池外再套一层「准入信号量」（pool_size + max_queue），排队超时直接抛
   `PoolBusyError`，由 API 层转成 HTTP 503，避免请求无限堆积把服务拖死。
3. 批量请求（多张图）复用同一个 predictor 的 `batch_size` 能力，吞吐更高。
"""

from __future__ import annotations

import inspect
import logging
import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from app.config import Settings

logger = logging.getLogger(__name__)


class PoolBusyError(RuntimeError):
    """排队超时（服务过载）→ HTTP 503。"""


class PoolNotReadyError(RuntimeError):
    """模型尚未加载完成 → HTTP 503。"""


@dataclass
class _Worker:
    """一个 predictor 实例 + 统计信息。"""

    index: int
    model: Any
    requests: int = 0
    images: int = 0
    busy_seconds: float = 0.0
    busy: bool = False


# --------------------------------------------------------------------------- #
# 结果解析
# --------------------------------------------------------------------------- #
def result_to_dict(res: Any) -> Dict[str, Any]:
    """把 PaddleX/PaddleOCR 的 Result 对象转成 dict（不同版本形态略有差异）。"""
    data = getattr(res, "json", None)
    if data is None:
        data = getattr(res, "res", None)
    if data is None:
        data = res
    if callable(data):  # 某些版本 json 是方法
        try:
            data = data()
        except TypeError:
            pass
    if isinstance(data, dict) and len(data) == 1 and "res" in data:
        data = data["res"]
    if not isinstance(data, dict):
        return {"raw": str(res)}
    return data


def extract_formula(data: Dict[str, Any]) -> Optional[str]:
    """从结果里取出 LaTeX 源码。"""
    for key in ("rec_formula", "rec_text", "text", "formula"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


# --------------------------------------------------------------------------- #
# 池实现
# --------------------------------------------------------------------------- #
class PredictorPool:
    """`FormulaRecognition` 实例池。"""

    def __init__(self, cfg: Settings) -> None:
        self.cfg = cfg
        self._queue: "queue.Queue[_Worker]" = queue.Queue()
        self._workers: List[_Worker] = []
        self._admission = threading.BoundedSemaphore(cfg.pool_size + cfg.max_queue)
        self._lock = threading.Lock()
        self._started = False
        self._closed = False
        self._inflight = 0
        self._waiting = 0
        self._inflight_lock = threading.Lock()
        self._rejected = 0
        self._requests_total = 0
        self._images_total = 0
        self._errors_total = 0
        self._started_at = 0.0

    # ---------------- 生命周期 ---------------- #
    @property
    def ready(self) -> bool:
        return self._started and not self._closed

    def start(self) -> None:
        """加载模型。会阻塞数十秒，建议在 lifespan 里丢到线程池执行。"""
        with self._lock:
            if self._started:
                return
            from paddleocr import FormulaRecognition  # 延迟导入，加快模块加载

            self._check_model_dir()
            kwargs = self._build_kwargs(FormulaRecognition)
            logger.info("开始加载模型：%s（device=%s，%s）", kwargs, self.cfg.device, self.cfg.device_note)

            for index in range(self.cfg.pool_size):
                begin = time.perf_counter()
                model = FormulaRecognition(**kwargs)
                worker = _Worker(index=index, model=model)
                self._workers.append(worker)
                self._queue.put(worker)
                logger.info(
                    "predictor #%d 加载完成，耗时 %.2fs", index, time.perf_counter() - begin
                )

            self._started = True
            self._started_at = time.time()

        if self.cfg.warmup:
            self._warmup()

    def stop(self) -> None:
        self._closed = True
        self._started = False
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._workers.clear()
        logger.info("predictor 池已关闭")

    def _check_model_dir(self) -> None:
        """提前检查模型目录，挂载没配好时给出明确提示（而不是等 Paddle 抛异常）。"""
        model_dir = self.cfg.model_dir
        if not os.path.isdir(model_dir):
            logger.error("模型目录不存在：%s，请检查 volume 挂载", model_dir)
            return
        missing = [
            name
            for name in ("inference.json", "inference.pdiparams", "inference.yml")
            if not os.path.isfile(os.path.join(model_dir, name))
        ]
        if missing:
            logger.error(
                "模型目录 %s 缺少必需文件 %s，请确认模型下载完整（可用 scripts/check_model.py 校验）",
                model_dir,
                missing,
            )

    def _build_kwargs(self, predictor_cls: Any) -> Dict[str, Any]:
        """根据当前 PaddleOCR 版本的签名过滤参数，避免版本差异导致启动失败。"""
        kwargs: Dict[str, Any] = {
            # PaddleOCR 3.0.3 的 FormulaRecognition 默认 model_name 是
            # PP-FormulaNet_plus-M，而本地 L 模型的 config.json 里是
            # PP-FormulaNet_plus-L。这里必须显式传入，否则会触发
            # "Model name mismatch，please input the correct model dir."
            "model_name": self.cfg.model_name,
            "model_dir": self.cfg.model_dir,
            "device": self.cfg.device,
            "cpu_threads": self.cfg.cpu_threads,
            "enable_mkldnn": self.cfg.enable_mkldnn,
            "use_tensorrt": self.cfg.use_tensorrt,
            "precision": self.cfg.precision,
        }
        if not self.cfg.use_tensorrt:
            kwargs.pop("precision")

        try:
            params = inspect.signature(predictor_cls.__init__).parameters
        except (TypeError, ValueError):
            return kwargs

        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
            return kwargs

        supported = {k: v for k, v in kwargs.items() if k in params}
        dropped = sorted(set(kwargs) - set(supported))
        if dropped:
            logger.warning("当前 PaddleOCR 版本不支持参数 %s，已忽略", dropped)
        return supported

    def _warmup(self) -> None:
        """用一张合成小图预热，避免首个真实请求触发算子编译/显存分配。"""
        path = os.path.join(self.cfg.temp_dir, "_warmup.png")
        try:
            os.makedirs(self.cfg.temp_dir, exist_ok=True)
            if not _make_warmup_image(path):
                return
            begin = time.perf_counter()
            self.predict([path], batch_size=1)
            logger.info("预热完成，耗时 %.2fs", time.perf_counter() - begin)
        except Exception:  # 预热失败不应阻止服务启动
            logger.warning("预热失败（不影响服务启动）", exc_info=True)
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    # ---------------- 推理 ---------------- #
    def predict(
        self, paths: Sequence[str], batch_size: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """对一组图片路径做推理，返回与输入等长的结果列表。"""
        if not self.ready:
            raise PoolNotReadyError("模型尚未就绪")

        paths = list(paths)
        if not paths:
            return []

        batch = max(1, min(batch_size or self.cfg.default_batch_size, self.cfg.max_batch_size))

        if not self._admission.acquire(timeout=self.cfg.queue_timeout):
            with self._inflight_lock:
                self._rejected += 1
            raise PoolBusyError(
                f"服务繁忙：排队等待超过 {self.cfg.queue_timeout:g}s，请稍后重试"
            )

        try:
            with self._inflight_lock:
                self._waiting += 1
            try:
                worker = self._queue.get(timeout=self.cfg.queue_timeout)
            except queue.Empty:
                with self._inflight_lock:
                    self._rejected += 1
                raise PoolBusyError(
                    f"服务繁忙：所有 predictor 均被占用超过 {self.cfg.queue_timeout:g}s"
                )
            finally:
                with self._inflight_lock:
                    self._waiting -= 1

            try:
                with self._inflight_lock:
                    self._inflight += 1
                begin = time.perf_counter()
                raw_results = worker.model.predict(input=paths, batch_size=batch)
                elapsed = time.perf_counter() - begin
                results = [result_to_dict(item) for item in raw_results]
            except Exception:
                with self._inflight_lock:
                    self._errors_total += 1
                raise
            finally:
                with self._inflight_lock:
                    self._inflight -= 1
                worker.requests += 1
                worker.images += len(paths)
                worker.busy_seconds += time.perf_counter() - begin
                self._queue.put(worker)

            with self._inflight_lock:
                self._requests_total += 1
                self._images_total += len(paths)
            return results
        finally:
            self._admission.release()

    # ---------------- 观测 ---------------- #
    def stats(self) -> Dict[str, Any]:
        with self._inflight_lock:
            inflight = self._inflight
            waiting = self._waiting
            rejected = self._rejected
            requests_total = self._requests_total
            images_total = self._images_total
            errors_total = self._errors_total
        idle = self._queue.qsize()
        busy = self.cfg.pool_size - idle
        return {
            "ready": self.ready,
            "device": self.cfg.device,
            "uptime_s": round(time.time() - self._started_at, 1) if self._started_at else 0.0,
            "pool_size": self.cfg.pool_size,
            "idle": idle,
            "busy": busy,
            "inflight": inflight,
            "waiting": waiting,
            "capacity": self.cfg.pool_size + self.cfg.max_queue,
            "requests_total": requests_total,
            "images_total": images_total,
            "errors_total": errors_total,
            "rejected_total": rejected,
            "workers": [
                {
                    "index": w.index,
                    "requests": w.requests,
                    "images": w.images,
                    "busy_seconds": round(w.busy_seconds, 2),
                }
                for w in self._workers
            ],
        }


def _make_warmup_image(path: str) -> bool:
    """生成一张含公式文本的白色图片用于预热。"""
    try:
        import cv2
        import numpy as np
    except Exception:
        return False
    try:
        image = np.full((192, 768, 3), 255, dtype=np.uint8)
        cv2.putText(
            image,
            "E = mc^2 + \\int_0^1 x dx",
            (24, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.4,
            (0, 0, 0),
            3,
        )
        return bool(cv2.imwrite(path, image))
    except Exception:
        return False
