#!/usr/bin/env python3
"""离线冒烟测试：不装 Paddle、不用 GPU，也能验证服务接线是否正确。

原理：把 `paddleocr` 换成一个返回假结果的桩模块，然后真正启动 uvicorn，
用 HTTP 请求跑一遍所有接口（含并发限流），确认 200/503 等行为符合预期。

用法：
    pip install fastapi uvicorn httpx python-multipart numpy opencv-python
    python scripts/smoke_test_api.py
"""

from __future__ import annotations

import concurrent.futures as futures
import json
import math
import os
import pathlib
import sys
import time
import types

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# --------------------------------------------------------------------------- #
# 1) 造一个 paddleocr 桩模块（必须在 import app 之前）
# --------------------------------------------------------------------------- #
FAKE_LATENCY = 0.2
ENDPOINT = "/v1/table/structure/recognition"


class _FakeResult:
    def __init__(self, index: int) -> None:
        self.json = {
            "res": {
                "input_path": None,
                "page_index": None,
                # 每个 cell 是一个 8 坐标多边形（xyxyxyxy），与 PaddleX 输出一致
                "bbox": [
                    [0, 0, 100, 0, 100, 50, 0, 50],
                    [100, 0, 200, 0, 200, 50, 100, 50],
                ],
                "structure": [
                    "<html>", "<body>", "<table>",
                    "<tr>", "<td>", f"cell{index}", "</td>", "</tr>",
                    "</table>", "</body>", "</html>",
                ],
                "structure_score": 0.98,
            }
        }


class _FakeTableStructureRecognition:
    instances = 0

    def __init__(self, **kwargs):
        type(self).instances += 1
        self.kwargs = kwargs

    def predict(self, input, batch_size: int = 1):  # noqa: A002 - 与 PaddleOCR 签名保持一致
        time.sleep(FAKE_LATENCY)
        return [_FakeResult(i) for i, _ in enumerate(list(input))]


fake_paddleocr = types.ModuleType("paddleocr")
fake_paddleocr.__version__ = "3.0.3-fake"
fake_paddleocr.TableStructureRecognition = _FakeTableStructureRecognition
sys.modules["paddleocr"] = fake_paddleocr

# --------------------------------------------------------------------------- #
# 2) 配置环境变量（必须在 import app.config 之前）
# --------------------------------------------------------------------------- #
TEMP_DIR = ROOT / ".smoke" / "tmp"
MODEL_DIR = ROOT / ".smoke" / "model"
TEMP_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
for _name in ("inference.json", "inference.pdiparams", "inference.yml"):
    (MODEL_DIR / _name).touch(exist_ok=True)   # 占位，避免启动时的模型目录告警

os.environ.update(
    MODEL_NAME="SLANet_plus",
    MODEL_DIR=str(MODEL_DIR),
    DEVICE="cpu",
    PREDICTOR_POOL_SIZE="2",   # 2 路并行
    MAX_QUEUE="2",             # 只能再排 2 个
    QUEUE_TIMEOUT="0.5",       # 排队 0.5s 超时 → 503
    MAX_BATCH_SIZE="4",
    TEMP_DIR=str(TEMP_DIR),
    WARMUP="true",
    LOG_LEVEL="WARNING",
)

import httpx  # noqa: E402
import uvicorn  # noqa: E402

from app.config import settings  # noqa: E402
from app.main import app  # noqa: E402

PORT = int(os.getenv("SMOKE_PORT", "18124"))
BASE = f"http://127.0.0.1:{PORT}"

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        PASSED.append(name)
        print(f"  [PASS] {name} {detail}")
    else:
        FAILED.append(name)
        print(f"  [FAIL] {name} {detail}")


def make_png() -> bytes:
    import cv2
    import numpy as np

    image = np.full((120, 480, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (10, 10), (470, 110), (0, 0, 0), 2)
    for x in (120, 240, 360):
        cv2.line(image, (x, 10), (x, 110), (0, 0, 0), 1)
    cv2.line(image, (10, 60), (470, 60), (0, 0, 0), 1)
    ok, buffer = cv2.imencode(".png", image)
    assert ok
    return buffer.tobytes()


def main() -> int:
    print(f"配置: pool_size={settings.pool_size} max_queue={settings.max_queue} "
          f"queue_timeout={settings.queue_timeout}")
    png = make_png()

    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=PORT, log_level="warning", access_log=False)
    )
    import threading

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(300):
        if server.started:
            break
        time.sleep(0.05)
    if not server.started:
        print("[FATAL] 服务启动失败")
        return 1

    try:
        print("\n--- 探针与信息 ---")
        response = httpx.get(f"{BASE}/health", timeout=30)
        check("GET /health 200", response.status_code == 200, str(response.json()))

        response = httpx.get(f"{BASE}/ready", timeout=30)
        check("GET /ready 200（模型已就绪）", response.status_code == 200)

        response = httpx.get(f"{BASE}/v1/info", timeout=30)
        check("GET /v1/info 200", response.status_code == 200)
        check("info.model_ready 为 true", response.json()["model_ready"] is True)
        check("info.model 为 SLANet_plus", response.json()["model"] == "SLANet_plus")

        check("predictor 实例数 == pool_size",
              _FakeTableStructureRecognition.instances == settings.pool_size,
              f"(实际 {_FakeTableStructureRecognition.instances})")

        print("\n--- 推理接口 ---")
        response = httpx.post(
            f"{BASE}{ENDPOINT}",
            files={"files": ("a.png", png, "image/png")},
            timeout=60,
        )
        body = response.json()
        check("上传单张 200", response.status_code == 200, json.dumps(body, ensure_ascii=False)[:160])
        check("返回 html", bool(body["results"][0]["html"]))
        check("html 拼接正确", body["results"][0]["html"].startswith("<html><body><table>"))
        check("num_cells 正确", body["results"][0]["num_cells"] == 2)

        response = httpx.post(
            f"{BASE}{ENDPOINT}",
            files=[
                ("files", ("a.png", png, "image/png")),
                ("files", ("b.png", png, "image/png")),
                ("files", ("c.png", png, "image/png")),
            ],
            data={"batch_size": "3"},
            timeout=60,
        )
        body = response.json()
        check("上传 3 张 200", response.status_code == 200)
        check("batch_size 生效", body["batch_size"] == 3, f"batch_size={body['batch_size']}")
        check("返回 3 条结果", body["count"] == 3)

        response = httpx.post(
            f"{BASE}{ENDPOINT}",
            files=[("files", (f"{i}.png", png, "image/png")) for i in range(5)],
            timeout=60,
        )
        check("超过 MAX_BATCH_SIZE 返回 400", response.status_code == 400, str(response.json()))

        response = httpx.post(
            f"{BASE}{ENDPOINT}",
            files={"files": ("a.pdf", png, "application/pdf")},
            timeout=60,
        )
        check("PDF 被拒绝 400", response.status_code == 400)

        import base64 as b64

        response = httpx.post(
            f"{BASE}{ENDPOINT}/base64",
            json={"images": [b64.b64encode(png).decode()], "batch_size": 1, "include_raw": True},
            timeout=60,
        )
        body = response.json()
        check("base64 接口 200", response.status_code == 200)
        check("include_raw 返回结构 token", isinstance(body["results"][0].get("structure"), list))
        check("include_raw 返回 cell 坐标", len(body["results"][0].get("cells", [])) == 2)

        response = httpx.post(
            f"{BASE}{ENDPOINT}/base64",
            json={"images": ["!!!not-base64!!!"]},
            timeout=60,
        )
        check("非法 base64 返回 400", response.status_code == 400)

        response = httpx.post(f"{BASE}{ENDPOINT}", timeout=60)
        check("未传文件返回 400", response.status_code == 400)

        print("\n--- 并发与限流 ---")
        # 用「共享连接池的同一个 client」发并发，避免每次新建连接带来的抖动，
        # 保证请求真正同时到达服务端。
        # 容量 = pool_size(2) + max_queue(2) = 4，单请求耗时 0.2s、排队超时 0.5s，
        # 因此 16 并发里必然有一部分在 0.5s 内拿不到资源 → 503。
        burst = 16
        shared = httpx.Client(
            timeout=60,
            limits=httpx.Limits(max_connections=burst, max_keepalive_connections=burst),
        )
        try:
            code = shared.post(
                f"{BASE}{ENDPOINT}",
                files={"files": ("a.png", png, "image/png")},
            ).status_code
            check("并发前热身请求 200", code == 200, f"code={code}")

            def one(_: int) -> int:
                return shared.post(
                    f"{BASE}{ENDPOINT}",
                    files={"files": ("a.png", png, "image/png")},
                ).status_code

            begin = time.perf_counter()
            with futures.ThreadPoolExecutor(max_workers=burst) as ex:
                codes = list(ex.map(one, range(burst)))
            cost = time.perf_counter() - begin
        finally:
            shared.close()

        ok = codes.count(200)
        busy = codes.count(503)
        check("并发下无 5xx 崩溃（仅 200/503）", set(codes) <= {200, 503}, f"codes={sorted(set(codes))}")
        check(
            "过载请求被 503 拒绝",
            busy > 0,
            f"200×{ok} / 503×{busy}，耗时 {cost:.2f}s",
        )
        # 理论上限：超时窗口内 2 路并行能翻台的次数 + 一次性可接纳的容量
        throughput_bound = settings.pool_size * math.ceil(settings.queue_timeout / FAKE_LATENCY)
        upper = throughput_bound + settings.pool_size + settings.max_queue
        check(
            "接纳数受并行度与超时窗口约束",
            settings.pool_size <= ok <= upper,
            f"200×{ok}（理论上限 {upper}）",
        )

        response = httpx.post(
            f"{BASE}{ENDPOINT}",
            files={"files": ("a.png", png, "image/png")},
            timeout=60,
        )
        check("过载后服务仍可用", response.status_code == 200)

        print("\n--- 观测 ---")
        stats = httpx.get(f"{BASE}/v1/stats", timeout=30).json()
        check("stats.inflight 归零", stats["inflight"] == 0, json.dumps(stats, ensure_ascii=False))
        check("stats.rejected_total > 0", stats["rejected_total"] > 0)
        check("stats.idle == pool_size", stats["idle"] == settings.pool_size)

        metrics = httpx.get(f"{BASE}/metrics", timeout=30).text
        check("/metrics 含关键指标", "slanet_requests_total" in metrics)

        print("\n--- 优雅关闭 ---")
        server.should_exit = True
        thread.join(timeout=15)
        check("uvicorn 正常退出", not thread.is_alive())

    finally:
        server.should_exit = True

    print(f"\n结果：{len(PASSED)} 通过 / {len(FAILED)} 失败")
    if FAILED:
        print("失败项：" + ", ".join(FAILED))
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
