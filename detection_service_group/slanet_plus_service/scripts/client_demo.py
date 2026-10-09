#!/usr/bin/env python3
"""调用示例：验证服务的文件上传 / base64 / URL 三种接口，并简单压一下并发。

用法：
    python3 scripts/client_demo.py --url http://127.0.0.1:9002 --image ./table.png
    python3 scripts/client_demo.py --url http://127.0.0.1:9002 --image ./table.png --concurrency 4
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures as futures
import json
import pathlib
import time

import httpx

ENDPOINT = "/v1/table/structure/recognition"


def check_health(base_url: str) -> None:
    response = httpx.get(f"{base_url}/health", timeout=10)
    response.raise_for_status()
    print("健康检查:", json.dumps(response.json(), ensure_ascii=False))


def infer_file(base_url: str, image: pathlib.Path, timeout: float = 120) -> dict:
    with image.open("rb") as fh:
        response = httpx.post(
            f"{base_url}{ENDPOINT}",
            files={"files": (image.name, fh, "image/png")},
            data={"batch_size": "1"},
            timeout=timeout,
        )
    response.raise_for_status()
    return response.json()


def infer_base64(base_url: str, image: pathlib.Path, timeout: float = 120) -> dict:
    payload = {"images": [base64.b64encode(image.read_bytes()).decode()], "batch_size": 1}
    response = httpx.post(f"{base_url}{ENDPOINT}/base64", json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:9002")
    parser.add_argument("--image", required=True, type=pathlib.Path)
    parser.add_argument("--concurrency", type=int, default=1, help="并发请求数（压测用）")
    parser.add_argument("--image-url", default=None, help="额外测试 URL 方式")
    args = parser.parse_args()

    base_url = args.url.rstrip("/")
    check_health(base_url)

    print("\n[1] 文件上传方式")
    result = infer_file(base_url, args.image)
    print(json.dumps(result, ensure_ascii=False, indent=2)[:1200])

    print("\n[2] base64 方式")
    result = infer_base64(base_url, args.image)
    print("html:", result["results"][0]["html"])
    print("structure_score:", result["results"][0]["structure_score"])

    if args.image_url:
        print("\n[3] URL 方式")
        response = httpx.post(
            f"{base_url}{ENDPOINT}/url",
            json={"urls": [args.image_url]},
            timeout=120,
        )
        response.raise_for_status()
        print("html:", response.json()["results"][0]["html"])

    if args.concurrency > 1:
        print(f"\n[4] 并发压测：{args.concurrency} 路并发 × 1 次")
        begin = time.perf_counter()
        with futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            outs = list(pool.map(lambda _: infer_file(base_url, args.image), range(args.concurrency)))
        cost = time.perf_counter() - begin
        ok = sum(1 for item in outs if item["results"][0]["html"])
        print(f"成功 {ok}/{args.concurrency}，总耗时 {cost:.2f}s，"
              f"平均 {cost / args.concurrency:.2f}s/张，QPS≈{args.concurrency / cost:.2f}")

    print("\n服务统计:")
    print(json.dumps(httpx.get(f"{base_url}/v1/stats", timeout=10).json(),
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
