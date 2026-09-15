"""解析配置：后端选择/回退参数、环境变量覆盖、后端可用性探测、`status()`。

见包文档 `backend/pdf_parser/__init__.py`。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional
from .. import paddle_parser
from .. import surya_parser


# =====================================================================
#  解析配置(后端选择 / Surya 服务参数)
# =====================================================================
CONFIG_FILE = Path(__file__).resolve().parent.parent / "config.json"

DEFAULT_OPTIONS: dict = {
    "backend": "auto",                 # auto | paddle | surya | pymupdf
    "surya_url": surya_parser.DEFAULT_URL,
    "surya_backend": surya_parser.DEFAULT_BACKEND,   # llamacpp | vllm
    "paddle_url": paddle_parser.DEFAULT_URL,
    "paddle_model": paddle_parser.DEFAULT_MODEL,     # 服务端 --served-model-name
    "paddle_prompt": paddle_parser.DEFAULT_PROMPT,   # OCR: / Table Recognition: ...
    "paddle_max_tokens": paddle_parser.DEFAULT_MAX_TOKENS,  # 单页生成上限
    "paddle_timeout": 300,             # 单页 OCR 请求超时(秒)
    "paddle_api_key": "",              # 服务端设了 --api-key 时填
    "paddle_ocr_math": True,           # 保留本地版式，用 OCR 的 LaTeX 补公式
    "dpi": 192,                        # 页面渲染 DPI(paddle / surya 都用)
    "image_labels": list(surya_parser.DEFAULT_IMAGE_LABELS),
    "image_pad": 4,                    # 插图裁剪四周外扩像素
    "min_image_size": 32,              # 宽或高小于该值的插图丢弃
    "keep_html": True,                 # 把 block 的 html(表格/公式)写进 doc.json
    "local_image_fallback": True,      # 某页没识别到图时，用 PyMuPDF 兜底找图
    "fallback": True,                  # Surya 失败时回退 PyMuPDF
    # 去重影：插图里已经“烘焙”了文字时，不再把同一批文字叠在图上渲染
    "drop_text_in_images": True,       # 关掉则保留全部文字行(可能叠字)
    "text_in_image_overlap": 0.6,      # 行框被插图覆盖超过该比例 → 判为图内文字
}

_ENV_KEYS = {
    "backend": "PDF_PARSER_BACKEND",
    "surya_url": "SURYA_INFERENCE_URL",
    "surya_backend": "SURYA_INFERENCE_BACKEND",
    "paddle_url": "PADDLE_OCR_URL",
    "paddle_model": "PADDLE_OCR_MODEL",
    "paddle_prompt": "PADDLE_OCR_PROMPT",
    "paddle_max_tokens": "PADDLE_OCR_MAX_TOKENS",
    "paddle_timeout": "PADDLE_OCR_TIMEOUT",
    "paddle_api_key": "PADDLE_OCR_API_KEY",
    "paddle_ocr_math": "PADDLE_OCR_MATH",
    "dpi": "PDF_PARSER_DPI",
    "image_labels": "PDF_PARSER_IMAGE_LABELS",
    "drop_text_in_images": "PDF_PARSER_DROP_TEXT_IN_IMAGES",
}


def _to_bool(val, default: bool = False) -> bool:
    """config.json / 环境变量里的 true/false/1/0/yes/no 统一成 bool。"""
    if isinstance(val, bool):
        return val
    if val is None:
        return default
    if isinstance(val, (int, float)):
        return bool(val)
    s = str(val).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off", ""):
        return False
    return default


def load_options(config_file: Optional[Path] = None,
                 overrides: Optional[dict] = None,
                 backend: Optional[str] = None) -> tuple[dict, list[str]]:
    """合并解析配置：默认值 < config.json[parser] < 环境变量 < 显式参数。"""
    opts = dict(DEFAULT_OPTIONS)
    warnings: list[str] = []

    cfg_file = Path(config_file) if config_file else CONFIG_FILE
    if cfg_file.exists():
        try:
            raw = json.loads(cfg_file.read_text(encoding="utf-8"))
            section = raw.get("parser") or {}
            if not isinstance(section, dict):
                raise ValueError("parser 段应为对象")
            for key, val in section.items():
                if key in opts and val is not None:
                    opts[key] = val
        except Exception as exc:
            warnings.append(f"读取解析配置 {cfg_file} 失败，使用默认值：{exc}")

    for key, env in _ENV_KEYS.items():
        val = os.environ.get(env)
        if val:
            opts[key] = val
    if overrides:
        opts.update({k: v for k, v in overrides.items() if v is not None and k in opts})
    if backend:
        opts["backend"] = backend

    mode = str(opts["backend"] or "").strip().lower()
    if mode not in ("auto", "paddle", "surya", "pymupdf"):
        warnings.append(f"未知的解析后端 {opts['backend']!r}，已按 auto 处理")
        mode = "auto"
    opts["backend"] = mode
    try:
        opts["dpi"] = max(48, min(400, int(opts["dpi"])))
    except (TypeError, ValueError):
        opts["dpi"] = DEFAULT_OPTIONS["dpi"]
    for key in ("image_pad", "min_image_size"):
        try:
            opts[key] = int(opts[key])
        except (TypeError, ValueError):
            opts[key] = DEFAULT_OPTIONS[key]
    for key in ("keep_html", "local_image_fallback", "fallback",
                "drop_text_in_images", "paddle_ocr_math"):
        opts[key] = _to_bool(opts[key], DEFAULT_OPTIONS[key])
    try:
        opts["text_in_image_overlap"] = min(
            1.0, max(0.0, float(opts["text_in_image_overlap"])))
    except (TypeError, ValueError):
        opts["text_in_image_overlap"] = DEFAULT_OPTIONS["text_in_image_overlap"]
    opts["image_labels"] = surya_parser.resolve_image_labels(opts["image_labels"])
    opts["surya_url"] = surya_parser.normalize_url(opts["surya_url"])
    opts["surya_backend"] = (str(opts["surya_backend"] or "").strip().lower()
                             or surya_parser.DEFAULT_BACKEND)
    opts["paddle_url"] = paddle_parser.normalize_url(opts["paddle_url"])
    opts["paddle_model"] = (str(opts["paddle_model"] or "").strip()
                            or paddle_parser.DEFAULT_MODEL)
    opts["paddle_prompt"] = (str(opts["paddle_prompt"] or "").strip()
                             or paddle_parser.DEFAULT_PROMPT)
    opts["paddle_api_key"] = str(opts["paddle_api_key"] or "").strip()
    for key, lo in (("paddle_max_tokens", 64), ("paddle_timeout", 30)):
        try:
            opts[key] = max(lo, int(opts[key]))
        except (TypeError, ValueError):
            opts[key] = DEFAULT_OPTIONS[key]
    return opts, warnings


def client_ready() -> tuple[bool, str]:
    """本地 Surya **客户端**依赖是否齐全。返回 (是否齐全, 缺失说明)。

    推理跑在外部服务上，但本进程要自己渲染页面(Pillow) 并构造请求/解析 block
    (surya-ocr)，缺任一的都会在解析中途抛 SuryaUnavailable 再回退 PyMuPDF。
    这里只查模块是否存在(不真正 import)，避免拖慢启动或产生副作用。
    """
    import importlib.util
    missing = []
    if importlib.util.find_spec("PIL") is None:
        missing.append("Pillow(pip install Pillow)")
    if importlib.util.find_spec("surya") is None:
        missing.append("surya-ocr(pip install -U surya-ocr)")
    if missing:
        return False, "本机缺少 Surya 客户端依赖：" + "、".join(missing)
    return True, ""


def status(config_file: Optional[Path] = None) -> dict:
    """解析后端状态(供 /api/config 展示与排障)。

    effective 必须同时考虑「服务连得上」和「本机客户端依赖齐全」——
    只看服务会误报 surya 可用，实际解析仍会回退。auto 按
    PaddleOCR-VL > Surya > PyMuPDF 的优先级给出实际生效的那个。
    """
    try:
        opts, warnings = load_options(config_file)
    except Exception as exc:  # pragma: no cover - 配置损坏时不应影响服务
        return {"backend": "pymupdf", "effective": "pymupdf", "message": str(exc)}

    p_ok, p_models, p_msg = paddle_parser.check_server(
        opts["paddle_url"], timeout=paddle_parser.PROBE_TIMEOUT,
        model=opts["paddle_model"])
    s_ok, _models, s_msg = surya_parser.check_server(
        opts["surya_url"], timeout=surya_parser.PROBE_TIMEOUT)
    ready, why = client_ready()          # Surya 客户端依赖(surya-ocr + Pillow)
    surya_usable = s_ok and ready

    mode = opts["backend"]
    if mode == "pymupdf":
        effective = "pymupdf"
    elif mode == "paddle":
        effective = "paddle" if p_ok else "pymupdf"
    elif mode == "surya":
        effective = "surya" if surya_usable else "pymupdf"
    else:                                 # auto：优先级 paddle > surya > pymupdf
        effective = "paddle" if p_ok else ("surya" if surya_usable else "pymupdf")

    # 给前端一段“为什么是它”的说明
    messages: list[str] = []
    if effective == "paddle":
        messages.append(p_msg)
    elif effective == "surya":
        messages.append(s_msg)
    else:
        if not p_ok:
            messages.append(p_msg)
        if not surya_usable:
            messages.append(f"服务就绪，但{why}" if s_ok and why else s_msg)

    return {
        "backend": mode,
        "effective": effective,
        "env_override": bool(os.environ.get("PDF_PARSER_BACKEND")),
        "paddle_url": opts["paddle_url"],
        "paddle_ready": p_ok,
        "paddle_model": opts["paddle_model"],
        "paddle_models": p_models,
        "paddle_message": p_msg,
        "surya_url": opts["surya_url"],
        "surya_backend": opts["surya_backend"],
        "surya_ready": s_ok,
        "surya_client_ready": ready,
        "surya_message": s_msg,
        "dpi": opts["dpi"],
        "message": "；".join(m for m in messages if m),
        "warnings": warnings,
    }
