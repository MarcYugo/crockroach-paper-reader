"""服务配置。

所有配置均通过环境变量注入，方便在 docker-compose / K8s 中按机器规格调整，
而无需重新构建镜像。
"""

from __future__ import annotations

import importlib
import logging
import os
from dataclasses import dataclass
from functools import lru_cache

logger = logging.getLogger(__name__)


def _str(name: str, default: str) -> str:
    value = os.getenv(name)
    return default if value is None or value == "" else value


def _int(name: str, default: int) -> int:
    try:
        return int(_str(name, str(default)))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(_str(name, str(default)))
    except ValueError:
        return default


def _bool(name: str, default: bool) -> bool:
    value = _str(name, "true" if default else "false").strip().lower()
    return value in {"1", "true", "yes", "y", "on"}


# --------------------------------------------------------------------------- #
# 设备自动检测：没有 GPU 就用 CPU
# --------------------------------------------------------------------------- #
_DEVICE_AUTO_TOKENS = {"auto", "default", "detect"}
_GPU_DISABLED_TOKENS = {"", "-1", "none", "void"}


@lru_cache(maxsize=1)
def cuda_device_count() -> int:
    """探测 Paddle 可用的 CUDA 设备数量，0 表示只能用 CPU。

    覆盖常见「无 GPU」情形：装了 CPU 版 paddle、宿主机没有 NVIDIA 驱动、
    容器启动时没有挂载 GPU（缺 --gpus / nvidia-container-toolkit）。
    """
    try:
        paddle = importlib.import_module("paddle")  # 动态导入：没装 Paddle 的环境也能读配置
    except Exception as exc:  # pragma: no cover - 取决于运行环境
        logger.info("无法导入 Paddle（%s），按 CPU 处理", exc)
        return 0

    try:
        if not paddle.device.is_compiled_with_cuda():
            logger.info("当前 Paddle 为 CPU 版本（is_compiled_with_cuda=False）")
            return 0
    except Exception as exc:  # pragma: no cover - 老版本 API 差异
        logger.warning("检测 Paddle 是否支持 CUDA 失败：%s", exc)
        return 0

    try:
        count = int(paddle.device.cuda.device_count())
    except Exception as exc:  # 缺少驱动 / 未挂载 GPU 时会走到这里
        logger.warning("查询 CUDA 设备数量失败（宿主机缺少 NVIDIA 驱动或容器未挂载 GPU）：%s", exc)
        return 0

    if count <= 0:
        logger.info("Paddle 未发现可用 CUDA 设备")
    return max(0, count)


def resolve_device(requested: str) -> tuple[str, bool, str]:
    """把 DEVICE 解析为最终设备，返回 ``(device, auto, note)``。

    * ``auto=True`` 表示该值由自动检测或自动回退得到；
    * ``note`` 是人类可读的决策原因，会写进日志与 /v1/info，便于排查。

    规则：
      1. DEVICE 为空 / auto / default → 按检测结果选 gpu:0 或 cpu；
      2. DEVICE=gpu:xx 但检测不到 GPU → 自动回退 cpu（除非 DEVICE_STRICT=true）；
      3. CUDA_VISIBLE_DEVICES 被显式置空或 -1 → 直接按 cpu 处理。
    """
    raw = (requested or "").strip()
    lower = raw.lower()

    if lower in _DEVICE_AUTO_TOKENS:
        count = cuda_device_count()
        if count > 0:
            return "gpu:0", True, f"DEVICE=auto：检测到 {count} 张可用 CUDA 设备，使用 gpu:0"
        return "cpu", True, "DEVICE=auto：未检测到可用 GPU，使用 cpu"

    if lower.startswith("gpu"):
        visible = os.getenv("CUDA_VISIBLE_DEVICES")
        if visible is not None and visible.strip().lower() in _GPU_DISABLED_TOKENS:
            return "cpu", True, f"DEVICE={raw} 但 CUDA_VISIBLE_DEVICES={visible!r} 禁用了 GPU，改用 cpu"
        if cuda_device_count() <= 0:
            if _bool("DEVICE_STRICT", False):
                return raw, False, f"DEVICE={raw} 未检测到 GPU，但 DEVICE_STRICT=true，保留该值"
            return "cpu", True, f"DEVICE={raw} 但未检测到可用 GPU，自动回退 cpu"

    return raw, False, f"DEVICE={raw}（显式指定）"


def default_pool_size(is_cpu: bool) -> int:
    """CPU 上并行实例过多只会互相抢核，默认给保守值。"""
    if not is_cpu:
        return 2
    cpu_total = os.cpu_count() or 1
    return 1 if cpu_total <= 4 else 2


@dataclass(frozen=True)
class Settings:
    """运行时配置。"""

    # ---- 模型 ----
    model_name: str         # 公式识别模型名，必须与配置文件中的 Global.model_name 一致
    model_dir: str          # 本地模型目录（容器内路径，由 volume 挂载）
    device: str             # auto / cpu / gpu:0 / gpu:0,1（auto=自动探测，无 GPU 自动用 CPU）
    device_auto: bool       # device 是否由自动检测/自动回退得到
    device_note: str        # 设备决策原因（写日志 + /v1/info，便于排查）
    cpu_threads: int        # 单个 predictor 的 CPU 线程数
    enable_mkldnn: bool     # CPU 推理时启用 MKL-DNN
    use_tensorrt: bool      # 启用 Paddle Inference 的 TensorRT 子图加速（仅 GPU）
    precision: str          # TensorRT 精度：fp32 / fp16
    warmup: bool            # 启动时用合成图跑一次，避免首请求抖动

    # ---- 并发 ----
    pool_size: int          # predictor 实例个数（真正的并行推理数）
    max_queue: int          # 允许排队的请求数，超出直接 503
    queue_timeout: float    # 排队等待超时（秒），超时返回 503
    default_batch_size: int # 默认批大小
    max_batch_size: int     # 单请求允许的最大图片数（= 批大小上限）

    # ---- 输入限制 ----
    max_image_mb: float     # 单张图片体积上限（MB）
    max_url_images: int     # URL 方式单次最大图片数
    url_timeout: float      # 下载 URL 图片的超时（秒）

    # ---- 其它 ----
    temp_dir: str           # 临时文件目录（建议放 tmpfs，如 /dev/shm）
    log_level: str

    @classmethod
    def from_env(cls) -> "Settings":
        device, device_auto, device_note = resolve_device(_str("DEVICE", "auto"))
        is_cpu = device.strip().lower().startswith("cpu")
        cpu_total = os.cpu_count() or 1
        pool_size = max(1, _int("PREDICTOR_POOL_SIZE", default_pool_size(is_cpu)))

        use_tensorrt = _bool("USE_TENSORRT", False)
        if use_tensorrt and is_cpu:
            logger.warning("USE_TENSORRT=true 在 CPU 上不可用，已自动关闭")
            use_tensorrt = False

        # CPU 推理靠单实例内部多线程，实例数 × 线程数 ≈ 物理核数比较合适；
        # GPU 场景 CPU 线程只负责预处理，保持较小的默认值。
        default_threads = max(1, cpu_total // pool_size) if is_cpu else 4

        return cls(
            model_name=_str("MODEL_NAME", "PP-FormulaNet_plus-L"),
            model_dir=_str("MODEL_DIR", "/models/PP-FormulaNet_plus-L"),
            device=device,
            device_auto=device_auto,
            device_note=device_note,
            cpu_threads=max(1, _int("CPU_THREADS", default_threads)),
            enable_mkldnn=_bool("ENABLE_MKLDNN", True),
            use_tensorrt=use_tensorrt,
            precision=_str("PRECISION", "fp32"),
            warmup=_bool("WARMUP", True),
            pool_size=pool_size,
            max_queue=max(0, _int("MAX_QUEUE", 64)),
            queue_timeout=max(0.1, _float("QUEUE_TIMEOUT", 30)),
            default_batch_size=max(1, _int("DEFAULT_BATCH_SIZE", 1)),
            max_batch_size=max(1, _int("MAX_BATCH_SIZE", 8)),
            max_image_mb=max(0.1, _float("MAX_IMAGE_MB", 20)),
            max_url_images=max(1, _int("MAX_URL_IMAGES", 8)),
            url_timeout=max(1.0, _float("URL_TIMEOUT", 15)),
            temp_dir=_str("TEMP_DIR", "/dev/shm/pp-formulanet"),
            log_level=_str("LOG_LEVEL", "INFO").upper(),
        )

    def public_info(self) -> dict:
        """对外暴露的、不含敏感信息的配置快照。"""
        return {
            "model_dir": self.model_dir,
            "device": self.device,
            "device_auto": self.device_auto,
            "device_note": self.device_note,
            "pool_size": self.pool_size,
            "max_queue": self.max_queue,
            "queue_timeout_s": self.queue_timeout,
            "default_batch_size": self.default_batch_size,
            "max_batch_size": self.max_batch_size,
            "cpu_threads": self.cpu_threads,
            "use_tensorrt": self.use_tensorrt,
            "precision": self.precision,
            "warmup": self.warmup,
        }


settings = Settings.from_env()
