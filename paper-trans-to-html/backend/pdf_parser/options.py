"""解析配置：后端选择、环境变量覆盖、`status()`。

见包文档 `backend/pdf_parser/__init__.py`。

「文字版式后端」有两个实现：

  * **PyMuPDF**（`backend_pymupdf.py`）：本地抽取文字层；Figure 图片和公式框由
    **detection-service-group**（`formula_table_service_group` 服务组）提供
    （见 `backend_detection_service_group.py`）；
  * **Surya 2**（`../surya_parser.py` + `backend_surya.py`）：整页 OCR，文字 /
    公式 / 插图全部来自 Surya，前端按 `parser === "surya"` 独立渲染。

`status()` 的返回字段是对外契约（`/api/settings/parser` 与网页
「⚙ 设置 → 存储 / OCR」面板都在读）：`detection_service_group_*` 与 `surya_*`
是两套协议；`surya_ready` 是**真探测**（`GET /models`，超时 `PROBE_TIMEOUT`），
`surya_client_ready` 是本机 surya-ocr 客户端依赖是否装好；两者都就绪才能用
`surya` 后端解析。旧的 `paddle_*` 配置项**不再被读取**（键保留在 config.json 里，
仅作记录/回退参考）。

默认常量的家在各客户端模块：detection-service-group 在 `../detection_service_group.py`、
Surya 在 `../surya_parser.py` —— config.json 里 `parser` 段的键与这里一一对应，
改动作时两边一起对齐。

⚠️ 命名沿革：这套配置以前叫 `ft_group_*` / `FT_GROUP_*`（后端取值 `ftgroup`）。
改名后**旧键不再读取**，只在检测到旧键/旧环境变量时各加一条 warning（不影响运行）；
后端取值 `ftgroup` 会被就地映射成 `detection_service_group`。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

from ..detection_service_group import (
    DEFAULT_TIMEOUT as DEFAULT_DETECTION_SERVICE_GROUP_TIMEOUT,
    DEFAULT_URL as DEFAULT_DETECTION_SERVICE_GROUP_URL,
    PROBE_TIMEOUT as DETECTION_SERVICE_GROUP_PROBE_TIMEOUT,
    check_server as detection_service_group_check,
    configured_dpi)
from ..surya_parser import (
    DEFAULT_BACKEND as DEFAULT_SURYA_BACKEND,
    DEFAULT_IMAGE_LABELS,
    DEFAULT_URL as DEFAULT_SURYA_URL,
    PROBE_TIMEOUT as SURYA_PROBE_TIMEOUT,
    check_server as surya_check,
    client_ready as surya_client_ready)


# =====================================================================
#  解析配置(后端选择 / 服务参数)
# =====================================================================
# 仓库根的 `config.json`（与 `app.py` 的 `CONFIG_FILE` 是**同一个文件**）。
# ⚠️ 拆包时在这里踩过坑：原来单文件 `backend/pdf_parser.py` 上溯 2 层正好是仓库根，
#    拆成 `backend/pdf_parser/options.py` 后少了一层，指向了 `backend/config.json`
#    —— 那个文件根本不存在。后果：网页面板“切换解析后端”写的是根 `config.json`，
#    而解析时 `parse_pdf()` 读的是这个空路径（文件不存在就静默用默认值）→ 永远走
#    `auto`。而 `app.py` 的 `status(CONFIG_FILE)` 传的是根路径，所以面板里显示的
#    是“已保存的值”，实际解析却是另一回事 —— 表现就是“手动选后端不生效”。
#    改这里时顺手对一下 `app.py::CONFIG_FILE`（两者必须指向同一份）。
CONFIG_FILE = Path(__file__).resolve().parents[2] / "config.json"

# 合法的后端取值。旧配置里的 `paddle` 会在 `load_options` 里映射成 `auto`，
# `ftgroup` 会映射成 `detection_service_group`（都是历史上的取值，保留识别只为把
# 老配置安顿好）。`surya` 是独立解析链（见 `backend_surya.py` / `../surya_parser.py`）。
PARSER_MODES = ("auto", "detection_service_group", "surya", "pymupdf")

# ---- 图片/公式检测增强(detection-service-group)与 Surya 的默认参数 ----
# 默认常量住在各自的客户端模块（导入见文件头）：
#   detection-service-group -> ../detection_service_group.py
#   surya                   -> ../surya_parser.py（默认地址与 surya_doc_parse_service 端口一致）

DEFAULT_OPTIONS: dict = {
    "backend": "auto",                # auto | detection_service_group | surya | pymupdf
    "surya_url": DEFAULT_SURYA_URL,
    "surya_backend": DEFAULT_SURYA_BACKEND,
    "detection_service_group_url": DEFAULT_DETECTION_SERVICE_GROUP_URL,       # 服务组地址
    "detection_service_group_timeout": DEFAULT_DETECTION_SERVICE_GROUP_TIMEOUT,  # 预测超时(秒)
    "detection_service_group_conf": 0.25,        # 检测置信度阈值(传给服务组)
    "detection_service_group_enable": True,      # 用服务组补充公式框与 Figure 图片
    "detection_service_group_keep_crops": True,  # 公式裁图兜底；Figure 显示所需裁图始终开启
    "dpi": configured_dpi(),           # 与服务组直接渲染时读取同一份 config.json
    "image_dpi": 0,                    # 插图裁剪的独立渲染 dpi（0=与 dpi 相同；只影响裁图清晰度）
    "image_labels": list(DEFAULT_IMAGE_LABELS),
    "image_pad": 4,                    # 插图裁剪四周外扩像素
    "min_image_size": 32,              # 宽或高小于该值的插图丢弃
    "keep_html": True,                 # 把 block 的 html(表格/公式)写进 doc.json
    "fallback": True,                  # [已失效] 旧「OCR 失败回退 PyMuPDF」开关
    # 去重影：插图里已经“烘焙”了文字时，不再把同一批文字叠在图上渲染
    "drop_text_in_images": True,       # 关掉则保留全部文字行(可能叠字)
    "text_in_image_overlap": 0.6,      # 行框被插图覆盖超过该比例 → 判为图内文字
}

_ENV_KEYS = {
    "backend": "PDF_PARSER_BACKEND",
    "surya_url": "SURYA_INFERENCE_URL",
    "surya_backend": "SURYA_INFERENCE_BACKEND",
    "detection_service_group_url": "DETECTION_SERVICE_GROUP_URL",
    "detection_service_group_timeout": "DETECTION_SERVICE_GROUP_TIMEOUT",
    "detection_service_group_conf": "DETECTION_SERVICE_GROUP_CONF",
    "detection_service_group_enable": "DETECTION_SERVICE_GROUP_ENABLE",
    "detection_service_group_keep_crops": "DETECTION_SERVICE_GROUP_KEEP_CROPS",
    "dpi": "PDF_PARSER_DPI",
    "image_dpi": "PDF_PARSER_IMAGE_DPI",
    "image_labels": "PDF_PARSER_IMAGE_LABELS",
    "drop_text_in_images": "PDF_PARSER_DROP_TEXT_IN_IMAGES",
}

# 旧名（`ft_group_*` 时代）—— 改名后不再读取，检测到就提醒一次，避免「配了不生效」。
_LEGACY_KEYS = {
    "ft_group_url": "detection_service_group_url",
    "ft_group_timeout": "detection_service_group_timeout",
    "ft_group_conf": "detection_service_group_conf",
    "ft_group_enable": "detection_service_group_enable",
    "ft_group_keep_crops": "detection_service_group_keep_crops",
}
_LEGACY_ENV_KEYS = {
    "FT_GROUP_URL": "DETECTION_SERVICE_GROUP_URL",
    "FT_GROUP_TIMEOUT": "DETECTION_SERVICE_GROUP_TIMEOUT",
    "FT_GROUP_CONF": "DETECTION_SERVICE_GROUP_CONF",
    "FT_GROUP_ENABLE": "DETECTION_SERVICE_GROUP_ENABLE",
    "FT_GROUP_KEEP_CROPS": "DETECTION_SERVICE_GROUP_KEEP_CROPS",
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


def _normalize_url(url: Any, default: str) -> str:
    """补全默认地址，并去掉结尾多余的 '/'。"""
    return (str(url or "").strip() or default).rstrip("/")


def _resolve_image_labels(spec: Any = None) -> set[str]:
    """None -> 默认图片类标签；'Picture,Figure' / ['Picture'] -> 集合"""
    if not spec:
        return set(DEFAULT_IMAGE_LABELS)
    if isinstance(spec, str):
        return {s.strip() for s in spec.split(",") if s.strip()}
    return {str(s).strip() for s in spec if str(s).strip()}


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
            stale = [k for k in section if k in _LEGACY_KEYS]
            if stale:
                warnings.append(
                    "解析配置里的 " + "、".join(sorted(stale)) + " 已改名为"
                    " detection_service_group_*（本次未读取，请更新 config.json）")
            if "local_image_fallback" in section:
                warnings.append("解析配置项 local_image_fallback 已停用；"
                                "Figure 图片只使用检测端（detection-service-group / "
                                "Surya）返回的裁图")
        except Exception as exc:
            warnings.append(f"读取解析配置 {cfg_file} 失败，使用默认值：{exc}")
    elif not os.environ.get(_ENV_KEYS["backend"]):
        # 没配置文件、又没环境变量覆盖 → 这次解析用的是内置默认值。以前这里完全静默，
        # 于是“手动选后端不生效”无从下手；现在跟着解析结果一起报出来(落在 doc.json
        # 的 warnings 里)。
        warnings.append(f"未找到解析配置 {cfg_file}，本次按默认配置解析"
                        "(网页「设置 → 存储 / OCR」保存的值就写在这个文件里)")

    for key, env in _ENV_KEYS.items():
        val = os.environ.get(env)
        if val:
            opts[key] = val
    stale_env = [k for k in _LEGACY_ENV_KEYS if os.environ.get(k)]
    if stale_env:
        warnings.append("环境变量 " + "、".join(sorted(stale_env))
                        + " 已改名为 DETECTION_SERVICE_GROUP_*（本次未读取）")
    if overrides:
        opts.update({k: v for k, v in overrides.items() if v is not None and k in opts})
    if backend:
        opts["backend"] = backend

    mode = str(opts["backend"] or "").strip().lower()
    if mode == "paddle":
        # 旧配置兼容：PaddleOCR-VL 的「配对 + 内联」链路已清空，
        # 公式增强改走 detection-service-group（auto 时同样会启用）。
        warnings.append("解析后端 'paddle' 已停用（配对链路已清空），本次按 auto 处理"
                        "（公式增强走 detection-service-group）")
        mode = "auto"
    if mode == "ftgroup":
        # 旧名兼容：`ftgroup` 已改名为 `detection_service_group`。
        warnings.append("解析后端 'ftgroup' 已改名为 'detection_service_group'，"
                        "本次直接按新名处理")
        mode = "detection_service_group"
    if mode not in PARSER_MODES:
        warnings.append(f"未知的解析后端 {opts['backend']!r}，已按 auto 处理")
        mode = "auto"
    opts["backend"] = mode
    try:
        opts["dpi"] = max(48, min(400, int(opts["dpi"])))
    except (TypeError, ValueError):
        opts["dpi"] = DEFAULT_OPTIONS["dpi"]
    try:
        # 插图裁剪分辨率：0 = 跟随 dpi；上限 600（render_pages 的渲染上限）
        opts["image_dpi"] = max(0, min(600, int(opts["image_dpi"])))
    except (TypeError, ValueError):
        opts["image_dpi"] = DEFAULT_OPTIONS["image_dpi"]
    for key in ("image_pad", "min_image_size"):
        try:
            opts[key] = int(opts[key])
        except (TypeError, ValueError):
            opts[key] = DEFAULT_OPTIONS[key]
    for key in ("keep_html", "fallback",
                "drop_text_in_images", "detection_service_group_enable",
                "detection_service_group_keep_crops"):
        opts[key] = _to_bool(opts[key], DEFAULT_OPTIONS[key])
    try:
        opts["text_in_image_overlap"] = min(
            1.0, max(0.0, float(opts["text_in_image_overlap"])))
    except (TypeError, ValueError):
        opts["text_in_image_overlap"] = DEFAULT_OPTIONS["text_in_image_overlap"]
    opts["image_labels"] = _resolve_image_labels(opts["image_labels"])
    opts["surya_url"] = _normalize_url(opts["surya_url"], DEFAULT_SURYA_URL)
    opts["surya_backend"] = (str(opts["surya_backend"] or "").strip().lower()
                             or DEFAULT_SURYA_BACKEND)
    opts["detection_service_group_url"] = _normalize_url(
        opts["detection_service_group_url"], DEFAULT_DETECTION_SERVICE_GROUP_URL)
    key = "detection_service_group_timeout"
    try:
        opts[key] = max(30, int(opts[key]))
    except (TypeError, ValueError):
        opts[key] = DEFAULT_OPTIONS[key]
    key = "detection_service_group_conf"
    try:
        opts[key] = min(1.0, max(0.0, float(opts[key])))
    except (TypeError, ValueError):
        opts[key] = DEFAULT_OPTIONS[key]
    return opts, warnings


def status(config_file: Optional[Path] = None) -> dict:
    """解析后端状态(供 /api/config 与 /api/settings/parser 展示)。

    返回字段是对外契约（前端 `common.js::renderParser` 在读）：
      * `effective` —— 实际生效的版式后端：`surya`（配置为 surya 时）或 `pymupdf`；
      * `detection_service_group_ready` 是**真探测**（`GET /health`，超时
        `PROBE_TIMEOUT`=2s，免得面板拉状态时卡住）；它用于检测 Figure 并识别公式；
      * `detection_service_group_enabled` 说明当前配置下**本次解析会不会**跑检测增强
        （surya 后端不跑）；
      * `surya_ready` 是**真探测**（`GET /models`）**且**本机 surya-ocr 客户端依赖
        就绪（`surya_client_ready`）—— 两者都满足才能用 surya 后端解析；
        `surya_service_ready` 单独给出服务端探测结果，`surya_message` 是明细。
    """
    try:
        opts, warnings = load_options(config_file)
    except Exception as exc:  # pragma: no cover - 配置损坏时不应影响服务
        return {"backend": "pymupdf", "effective": "pymupdf", "message": str(exc)}

    try:
        f_ok, f_msg = detection_service_group_check(
            opts["detection_service_group_url"],
            timeout=DETECTION_SERVICE_GROUP_PROBE_TIMEOUT)
    except Exception as exc:   # 探测本身出错不该影响状态接口
        f_ok, f_msg = False, f"探测失败：{exc}"
    try:
        s_ok, s_msg = surya_check(opts["surya_url"], timeout=SURYA_PROBE_TIMEOUT)
    except Exception as exc:   # 同上：探测异常只是"不可用"
        s_ok, s_msg = False, f"探测失败：{exc}"
    try:
        c_ok, c_msg = surya_client_ready()
    except Exception as exc:
        c_ok, c_msg = False, f"客户端依赖探测失败：{exc}"

    backend = opts["backend"]
    enhance_on = bool(opts.get("detection_service_group_enable")) and backend in (
        "auto", "detection_service_group")
    if backend == "surya":
        if s_ok and c_ok:
            message = ("文字版式、公式（<math>→LaTeX）与插图由 Surya 2 整页 OCR 提供，"
                       "前端按 Surya 数据独立渲染；detection-service-group 不参与")
        else:
            why = "推理服务不可用" if not s_ok else "本机缺少 surya-ocr 客户端依赖"
            message = f"解析后端现为 surya，但{why} —— 上传 / 重解析会失败（见下方明细）"
    else:
        message = ("文字版式走 PyMuPDF 本地解析；detection-service-group "
                   "用于提供 Figure 图片并识别公式框与 LaTeX"
                   + ("（服务已就绪）" if f_ok else "（服务不可用，本次跳过检测增强）"))
    return {
        "backend": backend,              # 用户保存的值(auto/detection_service_group/…)
        "effective": "surya" if backend == "surya" else "pymupdf",   # 实际生效的版式后端
        "env_override": bool(os.environ.get(_ENV_KEYS["backend"])),
        "detection_service_group_url": opts["detection_service_group_url"],
        "detection_service_group_ready": f_ok,
        "detection_service_group_enabled": enhance_on,
        "detection_service_group_message": f_msg,
        "surya_url": opts["surya_url"],
        "surya_backend": opts["surya_backend"],
        # 服务端 + 客户端都就绪才算"能用"；细分给 surya_service_ready / client_ready
        "surya_ready": bool(s_ok and c_ok),
        "surya_service_ready": bool(s_ok),
        "surya_client_ready": bool(c_ok),
        "surya_message": f"{s_msg}；{c_msg}",
        "dpi": opts["dpi"],
        "image_dpi": opts["image_dpi"],
        "message": message,
        "warnings": warnings,
    }
