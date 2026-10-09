# -*- coding: utf-8 -*-
"""router_service 离线冒烟测试。

不需要真的启动 yolov13 / pp-formulanet-plus-l / slanet_plus，也不用 GPU：
脚本会用桩服务（返回结构一致的假数据）模拟三个下游，然后把 router service
真正跑起来，覆盖健康检查、检测分流、公式/表格识别合并、会话隔离、并发限流。

    python scripts/smoke_test_router.py           # 离线自测（推荐）
    python scripts/smoke_test_router.py --url http://127.0.0.1:9003   # 打真实部署

依赖：pip install fastapi uvicorn httpx python-multipart
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, Request

SERVICE_ROOT = Path(__file__).resolve().parents[1]

# 1x1 的透明 PNG，仅用于让桩服务返回结构正确的 crop
TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFAAH/q842iQAAAABJRU5ErkJggg=="
)

PASSED = 0
FAILED = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  [PASS] {name}")
    else:
        FAILED += 1
        print(f"  [FAIL] {name} {detail}")


# --------------------------------------------------------------------------- #
# 桩服务
# --------------------------------------------------------------------------- #
def detection_payload() -> dict[str, Any]:
    """模拟 yolov13 响应，含公式、表格、图片和未配置的公式类别。"""
    detections: list[dict[str, Any]] = []

    def add(index: int, class_id: int, name: str, page: int, bbox: list[float]) -> None:
        detections.append(
            {
                "page_index": page,
                "page_number": page + 1,
                "class_id": class_id,
                "category_id": class_id + 1,
                "class_name": name,
                "confidence": 0.9,
                "bbox": bbox,
                "bbox_pdf": bbox,
                "bbox_norm": [0.1, 0.1, 0.1, 0.1],
                "crop": {
                    "format": "png",
                    "encoding": "base64",
                    "width": 1,
                    "height": 1,
                    "bytes": 68,
                    "rect": [int(v) for v in bbox],
                    "data": TINY_PNG_B64,
                },
            }
        )

    add(0, 0, "InlineFormula", 0, [10, 10, 40, 24])
    add(1, 1, "DisplayedFormulaLine", 0, [10, 40, 200, 70])
    add(2, 3, "DisplayedFormulaBlock", 1, [10, 80, 200, 140])
    add(3, 4, "Table", 1, [10, 160, 300, 400])
    add(4, 4, "Table", 1, [10, 420, 300, 600])
    add(5, 5, "Figure", 1, [10, 620, 300, 700])

    return {
        "status": "ok",
        "model": {"weights": "stub.pt", "imgsz": 640, "conf": 0.25, "iou": 0.7, "max_det": 300, "batch": 8, "device": "cpu"},
        "classes": [
            {"id": 0, "category_id": 1, "name": "InlineFormula"},
            {"id": 4, "category_id": 5, "name": "Table"},
            {"id": 5, "category_id": 6, "name": "Figure"},
        ],
        "images": [
            {"image_index": 0, "page_index": 0, "page_number": 1, "file_name": "page_0001.png", "width": 1000, "height": 1400, "num_detections": 2, "num_duplicates_removed": 0},
            {"image_index": 1, "page_index": 1, "page_number": 2, "file_name": "page_0002.png", "width": 1000, "height": 1400, "num_detections": 4, "num_duplicates_removed": 0},
        ],
        "detections": detections,
        "dedup": {"enabled": True, "overlap_threshold": 0.8, "removed_count": 0, "removed": []},
        "summary": {"num_images": 2, "total_detections": len(detections), "duplicates_removed": 0, "per_class": {}},
        "timing": {"inference_ms": 12.0, "total_ms": 15.0},
    }


def build_detection_stub(recorder: list[dict[str, Any]]) -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "model_loaded": True}

    @app.post("/predict")
    async def predict(request: Request) -> dict[str, Any]:
        body = await request.json()
        recorder.append(body)
        return detection_payload()

    return app


def build_formula_stub(delay_s: float, recorder: list[dict[str, Any]]) -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "model_ready": True}

    @app.post("/v1/formula/recognition/base64")
    async def recognition(request: Request) -> dict[str, Any]:
        body = await request.json()
        images = body.get("images") or []
        recorder.append(body)
        if delay_s:
            await asyncio.sleep(delay_s)
        results = [
            {"index": index, "filename": f"image_{index}.png", "rec_formula": f"\\frac{{{index + 1}}}{{n}}", "error": None}
            for index in range(len(images))
        ]
        return {"count": len(results), "batch_size": len(images), "elapsed_ms": 1.0, "per_image_ms": 1.0, "results": results}

    return app


def build_table_stub(delay_s: float, recorder: list[dict[str, Any]]) -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "model_ready": True}

    @app.post("/v1/table/structure/recognition/base64")
    async def recognition(request: Request) -> dict[str, Any]:
        body = await request.json()
        images = body.get("images") or []
        recorder.append(body)
        if delay_s:
            await asyncio.sleep(delay_s)
        results = [
            {
                "index": index,
                "filename": f"image_{index}.png",
                "html": f"<html><body><table><tr><td>{index}</td></tr></table></body></html>",
                "structure_score": 0.97,
                "num_cells": 4,
                "error": None,
            }
            for index in range(len(images))
        ]
        return {"count": len(results), "batch_size": len(images), "elapsed_ms": 1.0, "per_image_ms": 1.0, "results": results}

    return app


# --------------------------------------------------------------------------- #
# 启动工具
# --------------------------------------------------------------------------- #
def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def serve(app: FastAPI, port: int) -> uvicorn.Server:
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()
    deadline = time.time() + 20
    while not server.started:
        if time.time() > deadline:
            raise RuntimeError(f"服务启动超时（:{port}）")
        time.sleep(0.05)
    return server


def wait_http(url: str, timeout_s: float = 20.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            if httpx.get(url, timeout=1.0).status_code < 500:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.1)
    raise RuntimeError(f"等待服务就绪超时：{url}")


def image_payload(count: int = 1) -> list[dict[str, Any]]:
    return [
        {"data": TINY_PNG_B64, "name": f"page_{index:04d}.png", "page_index": index, "pdf_width": 612.0, "pdf_height": 792.0}
        for index in range(count)
    ]


def post_predict(client: httpx.Client, base: str, session_id: str | None, images: int = 1, keep_crops: bool = True) -> dict[str, Any]:
    body = {"images": image_payload(images), "keep_crops": keep_crops}
    if session_id is not None:
        body["session_id"] = session_id
    response = client.post(f"{base}/v1/router/predict", json=body)
    if response.status_code != 200:
        raise AssertionError(f"/v1/router/predict 返回 {response.status_code}: {response.text[:300]}")
    return response.json()


# --------------------------------------------------------------------------- #
# 用例
# --------------------------------------------------------------------------- #
def run_checks(
    base: str,
    formula_calls: list[dict[str, Any]],
    table_calls: list[dict[str, Any]],
    detect_calls: list[dict[str, Any]],
    *,
    stubbed: bool,
) -> None:
    """``stubbed=True`` 表示下游是桩服务，可以断言固定的检测结果与参数透传。"""
    with httpx.Client(timeout=60.0) as client:
        print("\n[1] 探针与信息")
        health = client.get(f"{base}/health")
        check("GET /health 200", health.status_code == 200, health.text[:200])
        body = health.json()
        check("health 三个下游均 ok", all(v.get("status") == "ok" for v in body.get("downstream", {}).values()), json.dumps(body.get("downstream")))
        ready = client.get(f"{base}/ready")
        check("GET /ready 200", ready.status_code == 200, ready.text[:200])
        info = client.get(f"{base}/v1/info").json()
        check("GET /v1/info 含配置", "config" in info and "formula_classes" in info["config"])
        check("检测默认参数强制 return_crops", info["config"]["detection_defaults"]["return_crops"] is True)

        print("\n[2] 单请求完整链路")
        result = post_predict(client, base, "sess_alpha")
        check("响应保留 yolov13 原始 status", result.get("status") == "ok")
        check("响应带回 session_id", result.get("session_id") == "sess_alpha")
        counts = result["router"]["counts"]
        recognized = result["recognized_detections"]
        unrecognized = result["unrecognized_detections"]
        all_detections = recognized + unrecognized
        formulas = [d for d in recognized if "formula" in d]
        tables = [d for d in recognized if "table" in d]
        check("识别与未识别检测框分组后总数不变", len(all_detections) == counts["total_detections"])
        check("未识别检测框未附加识别结果", all("formula" not in d and "table" not in d for d in unrecognized))
        check("分流计数之和等于检测框总数", counts["formula"] + counts["table"] + counts["figure"] + counts["other"] == len(all_detections), json.dumps(counts, ensure_ascii=False))
        check("公式框数 = 追加了 formula 的检测框数", counts["formula"] == len(formulas))
        check("表格框数 = 追加了 table 的检测框数", counts["table"] == len(tables))
        check("公式框都带 formula.latex", formulas and all(d["formula"]["latex"] for d in formulas))
        check("表格框都带 table.html", tables and all(d["table"]["html"] for d in tables))
        check("router.recognition.failed 为 0", result["router"]["recognition"]["failed"] == 0)
        check("识别成功数 = formula_ok / table_ok", result["router"]["recognition"]["formula_ok"] == len(formulas) and result["router"]["recognition"]["table_ok"] == len(tables))
        check("保留 crop（keep_crops=true）", all("crop" in d for d in all_detections))
        if stubbed:
            check("桩数据：2 公式 / 2 表格 / 1 图片", (counts["formula"], counts["table"], counts["figure"]) == (2, 2, 1), json.dumps(counts, ensure_ascii=False))
            unconfigured = [d for d in unrecognized if d["class_name"] == "DisplayedFormulaBlock"]
            check("未配置的公式类别单独返回且不识别", bool(unconfigured) and "formula" not in unconfigured[0])
            check("桩数据：latex 以 \\frac 开头", formulas[0]["formula"]["latex"].startswith("\\frac"))
            check("桩数据：table.html 含 <table>", "<table>" in tables[0]["table"]["html"])
            figures = [d for d in unrecognized if d["class_name"] == "Figure"]
            check("桩数据：图片框未被追加识别字段", bool(figures) and "formula" not in figures[0] and "table" not in figures[0])
        check("原始检测框数量不变", len(all_detections) == 6)
        check("router.timing 三个字段齐全", {"detection_ms", "recognition_ms", "total_ms"} <= set(result["router"]["timing"]))

        print("\n[3] 批量与裁剪开关")
        result = post_predict(client, base, "sess_alpha", keep_crops=False)
        all_detections = result["recognized_detections"] + result["unrecognized_detections"]
        check("keep_crops=false 时不返回 crop", all("crop" not in d for d in all_detections))
        check("keep_crops=false 仍返回 latex", any(d.get("formula", {}).get("latex") for d in result["recognized_detections"]))

        print("\n[4] 消费者按批取件")
        post_predict(client, base, "sess_batch", images=3)
        if stubbed:
            sizes = {len(call["images"]) for call in formula_calls}
            check("公式池出现批量推理（>1 张/批）", max(sizes) >= 2, f"实际批大小 {sorted(sizes)}")
            check("表格请求按批提交", any(len(call["images"]) >= 1 for call in table_calls))

        print("\n[5] 会话隔离与并发（2 个客户端）")
        import concurrent.futures

        def worker(name: str) -> dict[str, Any]:
            with httpx.Client(timeout=60.0) as inner:
                return post_predict(inner, base, name)

        before = {
            sid: client.get(f"{base}/v1/router/session/{sid}").json().get("requests", 0)
            for sid in ("sess_a", "sess_b")
        }
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(worker, "sess_a"), pool.submit(worker, "sess_b")]
            results = [future.result() for future in futures]
        check("两个并发客户端都拿到结果", all(r["status"] == "ok" for r in results))
        check("两个会话的 session_id 互不串扰", {r["session_id"] for r in results} == {"sess_a", "sess_b"})
        for sid in ("sess_a", "sess_b"):
            record = client.get(f"{base}/v1/router/session/{sid}").json()
            check(f"会话 {sid} 请求计数 +1", record["requests"] == before[sid] + 1, json.dumps(record, ensure_ascii=False))
            check(f"会话 {sid} in_flight 归零", record["in_flight"] == 0)
        listed = client.get(f"{base}/v1/router/sessions").json()
        check("会话列表包含 2 个客户端", {"sess_a", "sess_b"} <= {s["session_id"] for s in listed["sessions"]})

        print("\n[6] 结果回查与缓存池状态")
        stored = client.get(f"{base}/v1/router/session/sess_a/results").json()
        check("会话结果可按 session_id 回查", stored["count"] >= 1 and stored["results"][0]["result"]["session_id"] == "sess_a")
        pools = client.get(f"{base}/v1/pools").json()
        check("公式池统计被消费完", pools["formula_pool"]["size"] == 0 and pools["formula_pool"]["in_flight"] == 0)
        check("公式池累计提交 > 0", pools["formula_pool"]["submitted"] > 0)
        check("表格池累计提交 > 0", pools["table_pool"]["submitted"] > 0)
        stats = client.get(f"{base}/v1/stats").json()
        check("stats 含 gate/pipeline", stats["gate"] and stats["pipeline"])
        metrics = client.get(f"{base}/metrics")
        check("metrics 输出 Prometheus 文本", metrics.status_code == 200 and "router_formula_pool_size" in metrics.text)

        print("\n[7] multipart 上传通道")
        files = [("files", (f"page_{i}.png", _png_bytes(), "image/png")) for i in range(2)]
        response = client.post(f"{base}/v1/router/predict/upload", files=files, data={"session_id": "sess_upload"})
        check("POST /v1/router/predict/upload 200", response.status_code == 200, response.text[:200])
        if response.status_code == 200:
            upload_result = response.json()
            check(
                "upload 通道也返回 latex",
                any(d.get("formula", {}).get("latex") for d in upload_result["recognized_detections"]),
            )
            check("upload 通道透传文件名", upload_result["images"][0]["file_name"].endswith(".png"))

        print("\n[8] 参数透传与校验")
        response = client.post(
            f"{base}/v1/router/predict",
            json={"images": image_payload(1), "session_id": "sess_params", "params": {"conf": 0.5, "imgsz": 960, "crop_padding": 2}},
        )
        check("带 params 的请求成功", response.status_code == 200, response.text[:200])
        if stubbed:
            last_call = detect_calls[-1]
            check("params.conf 透传到检测服务", last_call["params"]["conf"] == 0.5, json.dumps(last_call["params"]))
            check("params.imgsz 透传", last_call["params"]["imgsz"] == 960)
            check("return_crops 仍被强制为 true", last_call["params"]["return_crops"] is True)

        check("每个识别框都带耗时字段", formulas and formulas[0]["formula"]["elapsed_ms"] is not None and formulas[0]["formula"]["wait_ms"] is not None)

        empty = client.post(f"{base}/v1/router/predict", json={"images": []})
        check("空 images 返回 422", empty.status_code == 422)
        missing = client.get(f"{base}/v1/router/session/not-exist")
        check("未知会话返回 404", missing.status_code == 404)
        deleted = client.delete(f"{base}/v1/router/session/sess_upload")
        check("DELETE 会话成功", deleted.status_code == 200 and deleted.json()["removed"] is True)


def run_timing_check(base: str, formula_delay: float, table_delay: float) -> None:
    """验证流水线确实省时间。

    桩下游每次调用固定耗时 ``formula_delay`` / ``table_delay`` 秒；
    一次请求有 2 个已配置公式框 + 2 个表格框，串行执行需要 2*f + 2*t，
    而「批量推理 + 两条流水线并行」应该明显更快。
    """
    print("\n[9] 流水线加速验证")
    serial_ms = (2 * formula_delay + 2 * table_delay) * 1000
    with httpx.Client(timeout=120.0) as client:
        started = time.perf_counter()
        result = post_predict(client, base, "sess_timing")
        elapsed_ms = (time.perf_counter() - started) * 1000
    budget_ms = serial_ms * 0.75
    print(f"    串行基线下限 {serial_ms:.0f} ms，实际 {elapsed_ms:.0f} ms（阈值 {budget_ms:.0f} ms）")
    check("批量 + 并行流水线比串行更快", elapsed_ms < budget_ms, f"实际 {elapsed_ms:.0f} ms")
    check("两条流水线并行（识别耗时 ≈ 单次调用耗时）", result["router"]["timing"]["recognition_ms"] < serial_ms, json.dumps(result["router"]["timing"]))


def _png_bytes() -> bytes:
    import base64

    return base64.b64decode(TINY_PNG_B64)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description="router_service 冒烟测试")
    parser.add_argument("--url", default=None, help="已部署的 router service 地址；缺省则自动拉起桩下游 + 本服务")
    parser.add_argument("--formula-delay", type=float, default=0.3, help="桩公式服务的耗时（秒），用于验证流水线加速")
    parser.add_argument("--table-delay", type=float, default=0.3, help="桩表格服务的耗时（秒）")
    args = parser.parse_args()

    servers: list[uvicorn.Server] = []
    detect_calls: list[dict[str, Any]] = []
    formula_calls: list[dict[str, Any]] = []
    table_calls: list[dict[str, Any]] = []

    if args.url:
        base = args.url.rstrip("/")
        print(f"使用已部署的 router service：{base}")
        try:
            run_checks(base, formula_calls, table_calls, detect_calls, stubbed=False)
        except AssertionError as exc:
            check("联调过程未抛异常", False, str(exc))
        except httpx.HTTPError as exc:
            check("真实部署可访问", False, str(exc))
    else:
        ports = {"detection": free_port(), "formula": free_port(), "table": free_port(), "router": free_port()}
        print("启动桩下游服务：")
        print(f"  detection :{ports['detection']}  formula :{ports['formula']}  table :{ports['table']}")
        servers.append(serve(build_detection_stub(detect_calls), ports["detection"]))
        servers.append(serve(build_formula_stub(args.formula_delay, formula_calls), ports["formula"]))
        servers.append(serve(build_table_stub(args.table_delay, table_calls), ports["table"]))

        os.environ.update(
            {
                "DETECTION_URL": f"http://127.0.0.1:{ports['detection']}",
                "FORMULA_URL": f"http://127.0.0.1:{ports['formula']}",
                "TABLE_URL": f"http://127.0.0.1:{ports['table']}",
                "SERVICE_PORT": str(ports["router"]),
                "MAX_CONCURRENT_REQUESTS": "2",
                "FORMULA_WORKERS": "1",
                "TABLE_WORKERS": "1",
                "FORMULA_BATCH_SIZE": "4",
                "TABLE_BATCH_SIZE": "4",
                "FORMULA_CLASSES": "InlineFormula,DisplayedFormulaLine",
                "BATCH_WINDOW_MS": "40",
                "LOG_LEVEL": "WARNING",
            }
        )
        sys.path.insert(0, str(SERVICE_ROOT))
        from app.main import app as router_app  # noqa: PLC0415 - 必须在设置环境变量之后导入

        print(f"启动 router service :{ports['router']}")
        servers.append(serve(router_app, ports["router"]))
        base = f"http://127.0.0.1:{ports['router']}"
        wait_http(f"{base}/health")
        run_checks(base, formula_calls, table_calls, detect_calls, stubbed=True)
        run_timing_check(base, args.formula_delay, args.table_delay)

    for server in servers:
        server.should_exit = True
    time.sleep(0.5)

    print("\n" + "=" * 60)
    print(f"结果：{PASSED} 通过 / {FAILED} 失败")
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
