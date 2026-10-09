# -*- coding: utf-8 -*-
"""公式 / 表格识别流水线（router_service + 三个模型服务）部署后的端到端测试。

被测服务（默认端口，见根目录 .env / docker-compose.yml）：

    :9000  yolov13_formula_detection_service     检测（生产者）
    :9001  pp-formulanet-plus-l                  公式识别（消费者）
    :9002  slanet_plus_service                   表格识别（消费者）
    :9003  router_service                        对外主入口（检测 + 分流 + 合并）

测试内容：

    ① 四个服务的探针（/health、/ready）与 router 的观测接口
    ② 主接口 POST /v1/router/predict：真实 PDF 渲染图走完整流水线，
       校验「检测结果 + 公式 LaTeX + 表格 HTML」的合并结构与计数一致性
    ③ keep_crops=false 时裁剪图被移除
    ④ 参数透传、空图片的入参校验
    ⑤ multipart 上传通道与 JSON 通道结果一致
    ⑥ 会话（session）：状态、结果回查、列表、删除
    ⑦ 双客户端并发：不同 session_id 同时请求，结果不串扰

用法（服务已 docker compose up -d 之后）：

    python test_group_service.py                                  # 默认地址 + 仓库自带 latex_sample.pdf
    python test_group_service.py --pdf latex_sample.pdf --max-pages 2 --json report.json --out result.json
    python test_group_service.py --url http://192.168.1.50:9003    # 远程部署
    python test_group_service.py --wait 300                        # 首启等模型加载完再测
    python test_group_service.py --json report.json --out result.json
    python test_group_service.py --skip-direct --skip-concurrency  # 只测 router 主流程

依赖：pip install requests pymupdf（可选 pillow，用于校验裁剪图能否解码）
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import io
import json
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

import pymupdf
import requests

try:  # 可选依赖：只用于校验裁剪图尺寸
    from PIL import Image
except ImportError:  # pragma: no cover - 环境缺 pillow 时跳过尺寸校验
    Image = None

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_PDF = REPO_ROOT / "latex_sample.pdf"

# 与 router_service/app/config.py 的默认类别保持一致（/v1/info 拿不到时兜底）
DEFAULT_FORMULA_CLASSES = ("InlineFormula", "DisplayedFormulaLine", "FormulaNumber", "DisplayedFormulaBlock")
DEFAULT_TABLE_CLASSES = ("Table",)
DEFAULT_FIGURE_CLASSES = ("Figure",)

PASS, FAIL, WARN, SKIP = "PASS", "FAIL", "WARN", "SKIP"


# --------------------------------------------------------------------------- #
# 结果记录
# --------------------------------------------------------------------------- #
@dataclass
class Report:
    """收集断言结果；只有 FAIL 会让进程以非 0 退出。"""

    records: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    passed: int = 0
    failed: int = 0
    warned: int = 0
    skipped: int = 0
    _section: str = ""

    def section(self, title: str) -> None:
        self._section = title
        print(f"\n=== {title} ===")

    def _record(self, status: str, name: str, detail: str = "") -> None:
        self.records.append({"section": self._section, "status": status, "name": name, "detail": detail})
        suffix = f"  ({detail})" if detail else ""
        print(f"  [{status}] {name}{suffix}")

    def check(self, name: str, condition: bool, detail: str = "") -> bool:
        if condition:
            self.passed += 1
            self._record(PASS, name)
        else:
            self.failed += 1
            self._record(FAIL, name, detail)
        return bool(condition)

    def warn_if(self, name: str, condition: bool, detail: str = "") -> None:
        if condition:
            self.passed += 1
            self._record(PASS, name)
        else:
            self.warn(name, detail)

    def warn(self, name: str, detail: str = "") -> None:
        self.warned += 1
        self._record(WARN, name, detail)

    def skip(self, name: str, detail: str = "") -> None:
        self.skipped += 1
        self._record(SKIP, name, detail)

    def note(self, text: str) -> None:
        self.notes.append(text)
        print(f"  · {text}")

    def summary(self) -> int:
        total = self.passed + self.failed + self.warned + self.skipped
        print("\n" + "=" * 72)
        print(f"总计 {total} 项：通过 {self.passed} / 失败 {self.failed} / 告警 {self.warned} / 跳过 {self.skipped}")
        if self.failed:
            print("\n失败项：")
            for record in self.records:
                if record["status"] == FAIL:
                    print(f"  - [{record['section']}] {record['name']}  {record['detail']}")
        print("=" * 72)
        return 0 if self.failed == 0 else 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "totals": {
                "passed": self.passed,
                "failed": self.failed,
                "warned": self.warned,
                "skipped": self.skipped,
            },
            "notes": self.notes,
            "records": self.records,
        }


# --------------------------------------------------------------------------- #
# HTTP 封装
# --------------------------------------------------------------------------- #
@dataclass
class HttpResult:
    status: int
    body: Any
    elapsed_ms: float
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and 200 <= self.status < 300

    @property
    def json(self) -> dict[str, Any]:
        return self.body if isinstance(self.body, dict) else {}

    def brief(self) -> str:
        if self.error:
            return self.error
        text = self.body if isinstance(self.body, str) else json.dumps(self.body, ensure_ascii=False)
        return f"{self.status} {str(text)[:200]}"


class GroupClient:
    """对四个服务的 HTTP 调用（连接复用）。"""

    def __init__(self, router_url: str, detection_url: str, formula_url: str, table_url: str, timeout: float) -> None:
        self.router = router_url.rstrip("/")
        self.detection = detection_url.rstrip("/")
        self.formula = formula_url.rstrip("/")
        self.table = table_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "formula-table-group-test/1.0"

    def request(self, method: str, url: str, *, timeout: float | None = None, **kwargs: Any) -> HttpResult:
        start = time.perf_counter()
        try:
            response = self.session.request(method, url, timeout=timeout or self.timeout, **kwargs)
        except requests.RequestException as exc:  # 网络层失败也变成一条可读结果
            return HttpResult(0, None, (time.perf_counter() - start) * 1000, f"{type(exc).__name__}: {exc}")
        elapsed = (time.perf_counter() - start) * 1000
        try:
            body: Any = response.json()
        except ValueError:
            body = response.text
        return HttpResult(response.status_code, body, elapsed)

    def get(self, url: str, **kwargs: Any) -> HttpResult:
        return self.request("GET", url, **kwargs)

    def post_json(self, url: str, payload: dict[str, Any], **kwargs: Any) -> HttpResult:
        return self.request("POST", url, json=payload, **kwargs)

    def close(self) -> None:
        self.session.close()


# --------------------------------------------------------------------------- #
# PDF 渲染
# --------------------------------------------------------------------------- #
def render_pages(pdf_path: Path, dpi: int, max_pages: int) -> tuple[list[dict[str, Any]], list[bytes]]:
    """把 PDF 每页渲染成 PNG：返回（接口用的 base64 payload, multipart 用的原始字节）。"""
    payloads: list[dict[str, Any]] = []
    raws: list[bytes] = []
    with pymupdf.open(pdf_path) as doc:
        count = doc.page_count if max_pages <= 0 else min(doc.page_count, max_pages)
        for index in range(count):
            page = doc[index]
            pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB, alpha=False)
            data = pix.tobytes("png")
            raws.append(data)
            payloads.append(
                {
                    "name": f"page_{index + 1:04d}.png",
                    "page_index": index,
                    "pdf_width": round(page.rect.width, 2),
                    "pdf_height": round(page.rect.height, 2),
                    "data": base64.b64encode(data).decode("ascii"),
                }
            )
    return payloads, raws


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #
def derive_url(base: str, port: int) -> str:
    """把 base（如 http://127.0.0.1:9003）换成同主机的另一个端口。"""
    parsed = urlparse(base)
    host = parsed.hostname or "127.0.0.1"
    return urlunparse((parsed.scheme or "http", f"{host}:{port}", "", "", "", ""))


def class_list(value: Any, default: tuple[str, ...]) -> set[str]:
    if isinstance(value, (list, tuple)) and value:
        return {str(item).strip().lower() for item in value if str(item).strip()}
    return {item.lower() for item in default}


def classify(detection: dict[str, Any], taxonomy: dict[str, set[str]]) -> str:
    name = str(detection.get("class_name") or "").strip().lower()
    if name in taxonomy["formula"]:
        return "formula"
    if name in taxonomy["table"]:
        return "table"
    if name in taxonomy["figure"]:
        return "figure"
    for key in ("category_id", "class_id"):
        token = str(detection.get(key))
        for kind in ("formula", "table", "figure"):
            if token in taxonomy[kind]:
                return kind
    return "other"


def decode_crop_size(data: str) -> tuple[int, int] | None:
    if Image is None:
        return None
    try:
        image = Image.open(io.BytesIO(base64.b64decode(data)))
        return image.size
    except Exception:  # noqa: BLE001 - 数据坏了按 None 处理，由调用方断言
        return None


def wait_until_ready(client: GroupClient, seconds: float, interval: float = 5.0) -> bool:
    """轮询 router 的 /ready，等模型加载完；seconds<=0 直接返回当前状态。"""
    deadline = time.time() + seconds
    while True:
        result = client.get(f"{client.router}/ready", timeout=15)
        if result.status == 200 and result.json.get("ready") is True:
            return True
        if seconds <= 0 or time.time() >= deadline:
            return False
        downstream = result.json.get("downstream", result.brief())
        print(f"  ... 服务未就绪（{downstream}），{interval:.0f}s 后重试")
        time.sleep(interval)


# --------------------------------------------------------------------------- #
# 测试步骤
# --------------------------------------------------------------------------- #
def test_direct_probes(client: GroupClient, report: Report) -> None:
    report.section("① 下游服务直接探针")
    for label, base, ready_key, has_ready_endpoint in (
        ("检测服务 yolov13", client.detection, "model_loaded", False),
        ("公式识别 pp-formulanet-plus-l", client.formula, "model_ready", True),
        ("表格识别 slanet_plus", client.table, "model_ready", True),
    ):
        health = client.get(f"{base}/health", timeout=15)
        report.check(f"{label} /health 可达", health.ok, health.brief())
        if health.ok:
            report.check(f"{label} 模型已加载", health.json.get(ready_key) is True, health.brief())
        if has_ready_endpoint:
            ready = client.get(f"{base}/ready", timeout=15)
            report.check(f"{label} /ready 返回 200", ready.status == 200, ready.brief())
        else:
            report.skip(f"{label} /ready", "检测服务以 /health.model_loaded 报告就绪状态")


def taxonomy_from_config(config: dict[str, Any]) -> dict[str, set[str]]:
    """按部署实际配置的类别分组判定检测框归属（拿不到就用默认值）。"""
    return {
        "formula": class_list(config.get("formula_classes"), DEFAULT_FORMULA_CLASSES),
        "table": class_list(config.get("table_classes"), DEFAULT_TABLE_CLASSES),
        "figure": class_list(config.get("figure_classes"), DEFAULT_FIGURE_CLASSES),
    }


def test_router_probes(client: GroupClient, report: Report) -> dict[str, Any]:
    report.section("② router_service 探针与观测接口")
    health = client.get(f"{client.router}/health", timeout=30)
    report.check("router /health 返回 200", health.ok, health.brief())
    downstream = health.json.get("downstream", {})
    for name in ("detection", "formula", "table"):
        report.check(f"router /health 认为 {name} 正常", downstream.get(name, {}).get("status") == "ok", str(downstream.get(name)))

    ready = client.get(f"{client.router}/ready", timeout=30)
    report.check("router /ready 返回 200", ready.status == 200, ready.brief())

    info = client.get(f"{client.router}/v1/info", timeout=15)
    report.check("/v1/info 返回 200", info.ok, info.brief())
    config = info.json.get("config", {}) if info.ok else {}
    for key in ("detection_url", "formula_url", "table_url"):
        report.check(f"/v1/info 配置含 {key}", bool(config.get(key)), str(config.get(key)))
    report.note(
        "下游地址: "
        f"detection={config.get('detection_url')} formula={config.get('formula_url')} table={config.get('table_url')}"
    )
    report.check(
        "/v1/info 类别分组非空",
        bool(config.get("formula_classes")) and bool(config.get("table_classes")),
        str(config.get("formula_classes")),
    )

    stats = client.get(f"{client.router}/v1/stats", timeout=15)
    report.check("/v1/stats 返回 200", stats.ok, stats.brief())
    report.check(
        "/v1/stats 含 gate / pipeline / sessions",
        all(key in stats.json for key in ("gate", "pipeline", "sessions")),
        str(list(stats.json)),
    )

    pools = client.get(f"{client.router}/v1/pools", timeout=15)
    report.check(
        "/v1/pools 含 formula_pool / table_pool / workers",
        pools.ok and all(key in pools.json for key in ("formula_pool", "table_pool", "workers")),
        pools.brief(),
    )
    report.warn_if(
        "缓存池当前无积压",
        pools.ok and pools.json.get("formula_pool", {}).get("size") == 0 and pools.json.get("table_pool", {}).get("size") == 0,
        pools.brief(),
    )

    metrics = client.get(f"{client.router}/metrics", timeout=15)
    report.check("/metrics 返回 Prometheus 文本", metrics.ok and "router_requests_active" in str(metrics.body), metrics.brief())

    bad = client.post_json(f"{client.router}/v1/router/predict", {"images": []}, timeout=30)
    report.check("空 images 请求被拒绝（4xx）", 400 <= bad.status < 500, bad.brief())

    return config


def verify_merged_response(body: dict[str, Any], pages: int, taxonomy: dict[str, set[str]], report: Report, tag: str) -> dict[str, int]:
    """校验主接口响应的结构与计数；返回分类计数。"""
    recognized = body.get("recognized_detections")
    unrecognized = body.get("unrecognized_detections")
    if not isinstance(recognized, list):
        recognized = []
    if not isinstance(unrecognized, list):
        unrecognized = []
    detections = recognized + unrecognized
    counts = {"formula": 0, "table": 0, "figure": 0, "other": 0}
    ok = {"formula": 0, "table": 0}
    failed = 0
    crop_checked = 0

    problems: dict[str, list[str]] = {
        "formula_block": [],
        "formula_content": [],
        "table_block": [],
        "table_content": [],
        "misplaced": [],
        "crop": [],
    }

    for position, detection in enumerate(detections):
        if not isinstance(detection, dict):
            continue
        kind = classify(detection, taxonomy)
        counts[kind] += 1
        label = f"#{position} {detection.get('class_name')}"
        in_recognized = position < len(recognized)
        if in_recognized != (kind in {"formula", "table"}):
            problems["misplaced"].append(f"{label} 位于错误的检测框分组")
        if kind == "formula":
            block = detection.get("formula")
            if not isinstance(block, dict):
                problems["formula_block"].append(label)
            else:
                if not (block.get("latex") or block.get("error")):
                    problems["formula_content"].append(f"{label} -> {block}")
                if "table" in detection:
                    problems["misplaced"].append(f"{label} 同时带 table 块")
                if block.get("error") is None:
                    ok["formula"] += 1
                else:
                    failed += 1
        elif kind == "table":
            block = detection.get("table")
            if not isinstance(block, dict):
                problems["table_block"].append(label)
            else:
                if not (block.get("html") or block.get("error")):
                    problems["table_content"].append(f"{label} -> {block}")
                if "formula" in detection:
                    problems["misplaced"].append(f"{label} 同时带 formula 块")
                if block.get("error") is None:
                    ok["table"] += 1
                else:
                    failed += 1
        elif "formula" in detection or "table" in detection:
            problems["misplaced"].append(f"{label} 不应带识别块")

        crop = detection.get("crop")
        if isinstance(crop, dict) and isinstance(crop.get("data"), str):
            crop_checked += 1
            if Image is not None and decode_crop_size(crop["data"]) is None:
                problems["crop"].append(label)

    report.check(f"{tag} 公式框都带 formula 块", not problems["formula_block"], ", ".join(problems["formula_block"][:3]))
    report.check(f"{tag} 公式框都有 latex（或可见 error）", not problems["formula_content"], ", ".join(problems["formula_content"][:3]))
    report.check(f"{tag} 表格框都带 table 块", not problems["table_block"], ", ".join(problems["table_block"][:3]))
    report.check(f"{tag} 表格框都有 html（或可见 error）", not problems["table_content"], ", ".join(problems["table_content"][:3]))
    report.check(f"{tag} 识别块没有错配到其它类别", not problems["misplaced"], ", ".join(problems["misplaced"][:3]))
    report.check(f"{tag} 裁剪图都能解码", not problems["crop"], ", ".join(problems["crop"][:3]))

    router = body.get("router", {})
    reported = router.get("counts", {}) if isinstance(router, dict) else {}
    reported_rec = router.get("recognition", {}) if isinstance(router, dict) else {}
    report.check(
        f"{tag} 检测框分组数量等于原检测总数",
        len(detections) == reported.get("total_detections"),
        f"grouped={len(detections)} total={reported.get('total_detections')}",
    )
    report.check(
        f"{tag} router.counts 与实际分类一致",
        all(reported.get(k) == counts[k] for k in ("formula", "table", "figure", "other"))
        and reported.get("total_detections") == len(detections),
        f"reported={reported} actual={counts} total={len(detections)}",
    )
    report.check(
        f"{tag} recognition 计数一致",
        reported_rec.get("formula_ok") == ok["formula"]
        and reported_rec.get("table_ok") == ok["table"]
        and reported_rec.get("failed") == failed,
        f"reported={reported_rec} actual={{'formula_ok': {ok['formula']}, 'table_ok': {ok['table']}, 'failed': {failed}}}",
    )

    images = body.get("images")
    report.check(f"{tag} images 与输入页数一致", isinstance(images, list) and len(images) == pages, f"{len(images) if isinstance(images, list) else None} vs {pages}")
    report.check(f"{tag} detections 非空", bool(detections), f"{len(detections)} 个框")
    report.check(f"{tag} 未识别类别与识别类别分开放", not problems["misplaced"], ", ".join(problems["misplaced"][:3]))

    timing = router.get("timing", {}) if isinstance(router, dict) else {}
    detection_ms = timing.get("detection_ms", -1)
    total_ms = timing.get("total_ms", -1)
    report.check(f"{tag} timing 数值合理", detection_ms >= 0 and total_ms >= detection_ms, str(timing))
    report.check(f"{tag} 顶层保留检测服务原字段", "status" in body and "model" in body, str(list(body)))

    report.warn_if(
        f"{tag} 至少识别出 1 个公式/表格框",
        counts["formula"] + counts["table"] > 0,
        f"formula={counts['formula']} table={counts['table']}（检查 PDF/检测阈值，或换一页再测）",
    )
    if failed:
        report.warn(f"{tag} 有识别失败的框", f"{failed} 个框的识别结果带 error，详见各框的 formula/table.error")
    report.note(
        f"{tag} 分类计数: formula={counts['formula']} table={counts['table']} figure={counts['figure']} other={counts['other']}；"
        f"裁剪图 {crop_checked} 张；耗时 detection={detection_ms}ms total={total_ms}ms"
    )
    return counts


def test_predict_json(
    client: GroupClient,
    report: Report,
    pages: list[dict[str, Any]],
    session_id: str,
    taxonomy: dict[str, set[str]],
    timeout: float,
) -> dict[str, Any] | None:
    report.section("③ 主接口 JSON + base64（完整流水线）")
    payload = {"session_id": session_id, "keep_crops": True, "images": pages, "params": {"conf": 0.25, "imgsz": 640}}
    result = client.post_json(f"{client.router}/v1/router/predict", payload, timeout=timeout)
    report.check("POST /v1/router/predict 返回 200", result.ok, result.brief())
    if not result.ok:
        return None
    body = result.json
    report.check("响应 status=ok", body.get("status") == "ok", str(body.get("status")))
    report.check("响应回显 session_id", body.get("session_id") == session_id, str(body.get("session_id")))
    router = body.get("router", {})
    report.check("响应带 router 汇总块", isinstance(router, dict) and "request_id" in router, str(list(router) if isinstance(router, dict) else router))
    report.check("响应带 pipeline 说明", "pipeline" in router if isinstance(router, dict) else False)
    verify_merged_response(body, len(pages), taxonomy, report, "JSON")
    return body


def test_keep_crops(client: GroupClient, report: Report, pages: list[dict[str, Any]], taxonomy: dict[str, set[str]], timeout: float) -> None:
    report.section("④ keep_crops=false（只要文字结果）")
    session_id = f"group-test-nocrop-{uuid.uuid4().hex[:8]}"
    payload = {"session_id": session_id, "keep_crops": False, "images": pages}
    result = client.post_json(f"{client.router}/v1/router/predict", payload, timeout=timeout)
    report.check("keep_crops=false 请求成功", result.ok, result.brief())
    if result.ok:
        detections = result.json.get("recognized_detections", []) + result.json.get("unrecognized_detections", [])
        report.check("所有框的 crop 字段已移除", all("crop" not in det for det in detections if isinstance(det, dict)), f"{len(detections)} 个框")
        verify_merged_response(result.json, len(pages), taxonomy, report, "no-crop")
    client.request("DELETE", f"{client.router}/v1/router/session/{session_id}", timeout=15)


def test_upload_channel(
    client: GroupClient,
    report: Report,
    raws: list[bytes],
    baseline: dict[str, Any] | None,
    taxonomy: dict[str, set[str]],
    timeout: float,
) -> None:
    report.section("⑤ multipart 上传通道")
    session_id = f"group-test-upload-{uuid.uuid4().hex[:8]}"
    files = [("files", (f"page_{index + 1:04d}.png", data, "image/png")) for index, data in enumerate(raws)]
    result = client.request(
        "POST",
        f"{client.router}/v1/router/predict/upload",
        files=files,
        data={"session_id": session_id},
        timeout=timeout,
    )
    report.check("POST /v1/router/predict/upload 返回 200", result.ok, result.brief())
    if result.ok:
        body = result.json
        report.check(
            "上传通道返回分组后的检测结构",
            body.get("session_id") == session_id
            and isinstance(body.get("recognized_detections"), list)
            and isinstance(body.get("unrecognized_detections"), list),
            str(list(body))[:200],
        )
        verify_merged_response(body, len(raws), taxonomy, report, "upload")
        if baseline is not None:
            base_counts = baseline.get("router", {}).get("counts", {})
            upload_counts = body.get("router", {}).get("counts", {})
            report.check(
                "上传通道与 JSON 通道的分类计数一致",
                base_counts == upload_counts,
                f"json={base_counts} upload={upload_counts}",
            )
    report.check("缺少图片的上传请求被拒绝", 400 <= client.request("POST", f"{client.router}/v1/router/predict/upload", data={}, timeout=30).status < 500)
    client.request("DELETE", f"{client.router}/v1/router/session/{session_id}", timeout=15)


def test_sessions(client: GroupClient, report: Report, session_id: str) -> None:
    report.section("⑥ 会话管理")
    state = client.get(f"{client.router}/v1/router/session/{session_id}", timeout=15)
    report.check("GET /v1/router/session/{id} 返回 200", state.ok, state.brief())
    if state.ok:
        report.check("会话记录到请求数", state.json.get("requests", 0) >= 1, str(state.json.get("requests")))
        report.check("会话记录到成功数", state.json.get("succeeded", 0) >= 1, str(state.json.get("succeeded")))
        report.check("会话无滞留请求", state.json.get("in_flight") == 0, str(state.json))
        report.warn_if(
            "会话未记录识别失败",
            state.json.get("failed") == 0,
            f"failed={state.json.get('failed')}（识别任务失败数会计入 failed；详见各检测框 error）",
        )
        report.check(
            "会话已回填识别统计",
            all(key in state.json for key in ("formula_submitted", "formula_completed", "table_submitted", "table_completed")),
            str(state.json),
        )

    results = client.get(f"{client.router}/v1/router/session/{session_id}/results", timeout=15)
    report.check("GET .../results 返回 200", results.ok, results.brief())
    items = results.json.get("results", []) if results.ok else []
    report.check("结果回查非空", bool(items), f"{len(items)} 条")
    if items:
        stored = items[-1].get("result", {})
        report.check("归档结果属于该会话", stored.get("session_id") == session_id, str(stored.get("session_id")))

    listing = client.get(f"{client.router}/v1/router/sessions", timeout=15)
    report.check(
        "会话列表包含本次会话",
        listing.ok and any(item.get("session_id") == session_id for item in listing.json.get("sessions", [])),
        listing.brief(),
    )

    removed = client.request("DELETE", f"{client.router}/v1/router/session/{session_id}", timeout=15)
    report.check("DELETE 会话返回 200", removed.ok, removed.brief())
    report.check("删除后查询返回 404", client.get(f"{client.router}/v1/router/session/{session_id}", timeout=15).status == 404)


def test_concurrency(client: GroupClient, report: Report, pages: list[dict[str, Any]], timeout: float) -> None:
    report.section("⑦ 双客户端并发（session 隔离）")
    session_a = f"group-test-A-{uuid.uuid4().hex[:8]}"
    session_b = f"group-test-B-{uuid.uuid4().hex[:8]}"

    def call(session_id: str) -> HttpResult:
        return client.post_json(
            f"{client.router}/v1/router/predict",
            {"session_id": session_id, "keep_crops": False, "images": pages},
            timeout=timeout,
        )

    wall_start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        future_a = pool.submit(call, session_a)
        future_b = pool.submit(call, session_b)
        result_a, result_b = future_a.result(), future_b.result()
    wall_ms = (time.perf_counter() - wall_start) * 1000

    report.check("并发请求 A 返回 200", result_a.ok, result_a.brief())
    report.check("并发请求 B 返回 200", result_b.ok, result_b.brief())
    if result_a.ok and result_b.ok:
        report.check("A 的结果归属 A", result_a.json.get("session_id") == session_a, str(result_a.json.get("session_id")))
        report.check("B 的结果归属 B", result_b.json.get("session_id") == session_b, str(result_b.json.get("session_id")))
        report.check(
            "两个会话结果不串扰",
            result_a.json.get("router", {}).get("session_id") == session_a
            and result_b.json.get("router", {}).get("session_id") == session_b
            and result_a.json.get("router", {}).get("request_id") != result_b.json.get("router", {}).get("request_id"),
        )

        server_ms = sum(result.json.get("router", {}).get("timing", {}).get("total_ms", 0.0) for result in (result_a, result_b))
        report.note(f"并发墙钟 {wall_ms:.0f}ms，两个请求服务端合计 {server_ms:.0f}ms（重叠越明显流水线并行越充分）")
        report.warn_if(
            "并发请求有明显重叠（墙钟 < 合计耗时）",
            wall_ms < server_ms * 0.98,
            f"wall={wall_ms:.0f}ms server_sum={server_ms:.0f}ms（单请求本就很快或下游串行时属正常）",
        )

    for session_id in (session_a, session_b):
        state = client.get(f"{client.router}/v1/router/session/{session_id}", timeout=15)
        report.check(f"会话 {session_id[-6:]} 已归档", state.ok and state.json.get("succeeded", 0) >= 1, state.brief())
        client.request("DELETE", f"{client.router}/v1/router/session/{session_id}", timeout=15)


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(
        description="公式/表格识别流水线部署后的端到端测试",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--url", default="http://127.0.0.1:9003", help="router_service 地址")
    parser.add_argument("--detection-url", default=None, help="检测服务地址（默认取 router 主机的 9000 端口）")
    parser.add_argument("--formula-url", default=None, help="公式识别地址（默认取 router 主机的 9001 端口）")
    parser.add_argument("--table-url", default=None, help="表格识别地址（默认取 router 主机的 9002 端口）")
    parser.add_argument("--pdf", default=str(DEFAULT_PDF), help="测试用 PDF")
    parser.add_argument("--dpi", type=int, default=200, help="PDF 渲染精度")
    parser.add_argument("--max-pages", type=int, default=1, help="最多处理前 N 页（0 = 全部）")
    parser.add_argument("--timeout", type=float, default=600.0, help="单次预测请求超时（秒）")
    parser.add_argument("--wait", type=float, default=0.0, help="开始测试前等 /ready 的秒数（首启模型加载用）")
    parser.add_argument("--json", default=None, help="把测试报告写入该 JSON 文件")
    parser.add_argument("--out", default=None, help="把第一次预测的完整响应写入该 JSON 文件")
    parser.add_argument("--skip-direct", action="store_true", help="跳过下游服务直接探针（下游不对外暴露时用）")
    parser.add_argument("--skip-concurrency", action="store_true", help="跳过双客户端并发测试")
    args = parser.parse_args()

    pdf_path = Path(args.pdf).expanduser()
    if not pdf_path.is_file():
        print(f"找不到 PDF: {pdf_path}")
        return 2

    detection_url = args.detection_url or derive_url(args.url, 9000)
    formula_url = args.formula_url or derive_url(args.url, 9001)
    table_url = args.table_url or derive_url(args.url, 9002)

    report = Report()
    print("=" * 72)
    print("公式 / 表格识别流水线 —— 部署后测试")
    print(f"  router    : {args.url.rstrip('/')}")
    print(f"  detection : {detection_url}")
    print(f"  formula   : {formula_url}")
    print(f"  table     : {table_url}")
    print(f"  PDF       : {pdf_path}（DPI {args.dpi}，前 {args.max_pages or '全部'} 页）")
    print("=" * 72)

    pages, raws = render_pages(pdf_path, args.dpi, args.max_pages)
    if not pages:
        print("PDF 没有可渲染的页面")
        return 2

    client = GroupClient(args.url, detection_url, formula_url, table_url, args.timeout)
    baseline: dict[str, Any] | None = None
    start = time.time()
    try:
        if not wait_until_ready(client, args.wait):
            report.section("⓪ 就绪检查")
            report.check("router /ready 就绪", False, "服务未就绪；首启可用 --wait 300 等待模型加载")

        if args.skip_direct:
            report.section("① 下游服务直接探针")
            report.skip("下游直接探针", "--skip-direct")
        else:
            test_direct_probes(client, report)

        config = test_router_probes(client, report)
        taxonomy = taxonomy_from_config(config)

        session_id = f"group-test-main-{uuid.uuid4().hex[:8]}"
        baseline = test_predict_json(client, report, pages, session_id, taxonomy, args.timeout)
        if baseline is not None:
            report.section("③b 参数透传")
            passthrough = client.post_json(
                f"{client.router}/v1/router/predict",
                {"session_id": session_id, "keep_crops": False, "images": pages, "params": {"conf": 0.9, "imgsz": 960}},
                timeout=args.timeout,
            )
            report.check("带 params（conf/imgsz）的请求成功", passthrough.ok, passthrough.brief())
            if passthrough.ok:
                base_total = baseline.get("router", {}).get("counts", {}).get("total_detections", 0)
                new_total = passthrough.json.get("router", {}).get("counts", {}).get("total_detections", 0)
                report.check("高 conf 阈值不会多检出框", new_total <= base_total, f"conf=0.9 -> {new_total}，conf=0.25 -> {base_total}")
        else:
            report.section("③b 参数透传")
            report.skip("参数透传", "主接口不可用")

        test_keep_crops(client, report, pages, taxonomy, args.timeout)
        test_upload_channel(client, report, raws, baseline, taxonomy, args.timeout)
        if baseline is not None:
            test_sessions(client, report, session_id)
        else:
            report.section("⑥ 会话管理")
            report.skip("会话管理", "主接口不可用")
        if args.skip_concurrency:
            report.section("⑦ 双客户端并发（session 隔离）")
            report.skip("双客户端并发", "--skip-concurrency")
        else:
            test_concurrency(client, report, pages, args.timeout)
    finally:
        client.close()

    elapsed = time.time() - start
    print(f"\n总耗时 {elapsed:.1f}s")
    exit_code = report.summary()

    if args.json:
        payload = report.as_dict()
        payload["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        payload["elapsed_s"] = round(elapsed, 2)
        payload["targets"] = {
            "router": args.url,
            "detection": detection_url,
            "formula": formula_url,
            "table": table_url,
            "pdf": str(pdf_path),
            "dpi": args.dpi,
            "pages": len(pages),
        }
        Path(args.json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"测试报告已写入: {args.json}")

    if args.out and baseline is not None:
        Path(args.out).write_text(json.dumps(baseline, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"主接口响应已写入: {args.out}")

    print("TEST PASSED" if exit_code == 0 else "TEST FAILED")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
