#!/usr/bin/env python3
"""PaddleOCR-VL-1.6 (vLLM) 推理服务 —— 并发压力测试

只依赖标准库(urllib + threading), 不需要 requests / locust / asyncio。
图片素材复用同目录 test_paddleocr_vl_service.py 里的"现画数字图"(也可用 --image 传真实图)。

先跑功能自检, 再跑本脚本
-------------------------
    python test_paddleocr_vl_service.py       # 先确认能识别、流式正常
    python press_test.py                      # 再测并发上限

测量模型: 闭环(closed-loop)
---------------------------
起 N 个 worker 线程, 每个 worker「发一个请求 -> 等完整回复 -> 再发下一个」。
于是 --concurrency 就是"同时在飞的请求数", 和真实客户端行为一致; 吞吐由服务端算力
决定, 而不是客户端无脑灌请求。这比"一次性发 N 个"更能反映稳态承载能力。

对 vLLM 来说, 超出承载能力时**不会**报错, 而是表现为:
  * num_requests_waiting 上升(V1 的调度队列), 时延跟着涨;
  * KV cache 用满后开始抢占/重算(kv_cache_usage_perc 顶到 ~100%), 吞吐反而下降;
  * 客户端看到的成功率仍可能是 100%, 只是变慢。
所以判断"到顶了没有"主要看 waiting 峰值 + KV 峰值 + 时延拐点, 不能只看成功率。
(真正的 4xx/5xx 只会来自: 服务挂了 / OOM / 上下文超长 / 队列超时。)

每档(每个并发值)统计
--------------------
  * 请求数: 成功 / HTTP 错误(按状态码分类) / 连接失败 / 超时
  * 时延: 端到端 p50 / p90 / p99; 若开了 --stream 还有首字延迟 TTFT p50/p90
  * 吞吐: 请求/秒、输出 token/秒(用响应里的 usage)
  * 服务端: num_requests_running / num_requests_waiting / kv_cache_usage_perc 的峰值
            (后台按 --sample-interval 抓 /metrics)

用法示例
--------
    python press_test.py                                   # 默认阶梯 1,2,4,8, 每档 16 个请求
    python press_test.py --concurrency 1,2,4,8,16,32 --requests 32
    python press_test.py --duration 30                     # 每档压 30 秒(更适合长文档)
    python press_test.py --stream                          # 流式, 额外测 TTFT
    python press_test.py --pdf 论文.pdf --pages 1-5         # 多页轮流发(更接近真实负载)
    python press_test.py --pdf 扫描件.pdf --pages all --duration 60
    python press_test.py --image 某页.png --max-tokens 2048 --json-out press.json
    python press_test.py --url http://127.0.0.1:8080/v1

输入(--pdf / --image / 都不给)
--------------------------------
不指定时用脚本现画的数字图(载荷稳定、好复现)。想要真实负载就传 --pdf:
先用 PyMuPDF 把指定页渲染成 PNG(需 pip install PyMuPDF), 然后**每个请求轮流取一页**。
为什么要轮流: 固定同一张图会被服务端的多模态处理器缓存命中, 测出来的吞吐偏乐观。

想改服务端并发能力(改完要重建容器: docker compose up -d)
-------------------------------------------------------
    MAX_NUM_BATCHED_TOKENS   单批 token 上限(默认 16384): 调大能提高算力利用率, 吃显存
    --max-num-seqs           同时处理的序列数上限(默认 256): 显存小就调小
    GPU_MEMORY_UTILIZATION   KV cache 占显存比例(默认 0.90)
    MAX_MODEL_LEN            上下文上限(默认 32768): 调小可省显存、提高并发

退出码
------
0 = 跑完且至少一档满足阈值; 1 = 所有档都有失败/超阈值错误; 2 = 环境问题(未就绪/输入不存在)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass, field

OK, FAIL, ENV = 0, 1, 2

DEFAULT_URL = os.environ.get("PADDLEOCR_VL_URL", "http://127.0.0.1:8080/v1")
DEFAULT_MODEL = os.environ.get("PADDLEOCR_VL_MODEL", "PaddleOCR-VL-1.6-0.9B")
DEFAULT_TEXT = "8471 3625"
OCR_PROMPT = "OCR:"
MAX_LEVEL = 256          # 再往上压的就不是服务, 而是压测机自己了

try:                     # 复用功能自检脚本里的工具(同目录): 现画图 / PDF 渲染 / stdio 修正
    from test_paddleocr_vl_service import (DEFAULT_DPI, bytes_to_data_uri,
                                           png_dimensions, render_digits_png,
                                           render_pdf_pages, setup_stdio)
    setup_stdio()
    HAS_STDIO_FIX = True
except Exception:        # noqa: BLE001 - 独立使用时退化为必须 --image
    render_digits_png = None      # type: ignore[assignment]
    render_pdf_pages = None       # type: ignore[assignment]
    png_dimensions = None         # type: ignore[assignment]
    bytes_to_data_uri = None      # type: ignore[assignment]
    DEFAULT_DPI = 192
    HAS_STDIO_FIX = False


if not HAS_STDIO_FIX:
    # 拿不到上面那个模块时至少把编码问题自己解决掉(否则输出重定向到文件/管道会崩)
    for _s in (sys.stdout, sys.stderr):
        try:
            if not _s.isatty():
                _s.reconfigure(encoding="utf-8", errors="replace")
            else:
                _s.reconfigure(errors="replace")
        except (AttributeError, ValueError, OSError):
            pass


# ==========================================================================
# HTTP
# ==========================================================================
def http_text(url: str, timeout: float, headers: dict | None = None) -> tuple[int, str]:
    req = urllib.request.Request(url, method="GET")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")
    except (urllib.error.URLError, OSError, ValueError) as e:
        return 0, f"连接失败: {e}"


@dataclass
class Rec:
    """一次请求的观测结果。"""
    ok: bool = False
    status: int = 0
    latency: float = 0.0                 # 端到端秒
    ttft: float | None = None            # 首字延迟(仅 --stream)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    chars: int = 0
    error: str = ""


def post_chat(url: str, payload: dict, headers: dict, timeout: float,
              stream: bool) -> Rec:
    """发一次 /v1/chat/completions 并计时。失败不抛异常, 记在 Rec 里。"""
    rec = Rec()
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), method="POST")
    req.add_header("Content-Type", "application/json")
    for k, v in headers.items():
        req.add_header(k, v)

    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            rec.status = r.status
            if not stream:
                body = json.loads(r.read().decode("utf-8", "replace"))
                choices = body.get("choices") or []
                rec.chars = len((choices[0].get("message") or {}).get("content") or "") if choices else 0
                usage = body.get("usage") or {}
                rec.prompt_tokens = int(usage.get("prompt_tokens") or 0)
                rec.completion_tokens = int(usage.get("completion_tokens") or 0)
            else:
                buf: list[str] = []
                for raw in r:                       # SSE 按行读
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if obj.get("usage"):
                        usable = obj["usage"]
                        rec.prompt_tokens = int(usable.get("prompt_tokens") or 0)
                        rec.completion_tokens = int(usable.get("completion_tokens") or 0)
                    for choice in obj.get("choices") or []:
                        piece = (choice.get("delta") or {}).get("content")
                        if piece:
                            if rec.ttft is None:
                                rec.ttft = time.perf_counter() - t0
                            buf.append(piece)
                rec.chars = sum(len(x) for x in buf)
        rec.latency = time.perf_counter() - t0
        rec.ok = rec.status == 200
        if not rec.ok:
            rec.error = f"HTTP {rec.status}"
        return rec
    except urllib.error.HTTPError as e:
        rec.status = e.code
        rec.latency = time.perf_counter() - t0
        rec.error = f"HTTP {e.code} {e.read().decode('utf-8', 'replace')[:120]}"
        return rec
    except TimeoutError:
        rec.latency = time.perf_counter() - t0
        rec.error = f"超时(>{timeout:.0f}s)"
        return rec
    except (urllib.error.URLError, OSError, ValueError) as e:
        rec.latency = time.perf_counter() - t0
        rec.error = f"连接失败: {e}"
        return rec


# ==========================================================================
# 服务端指标采样(/metrics, Prometheus 文本)
# ==========================================================================
GAUGES = ("num_requests_running", "num_requests_waiting",
          "kv_cache_usage_perc", "gpu_cache_usage_perc")


class MetricsSampler(threading.Thread):
    """后台按固定间隔抓 vLLM 指标峰值 —— 压测端看不到排队情况, 这里补上。"""

    def __init__(self, root: str, interval: float, timeout: float) -> None:
        super().__init__(daemon=True, name="metrics-sampler")
        self.root = root
        self.interval = max(interval, 0.2)
        self.timeout = timeout
        self._stop = threading.Event()
        self.peaks: dict[str, float] = {}
        self.samples = 0
        self.available = True

    def run(self) -> None:
        first = True
        while True:
            # 先立刻采一次: 档位很短(几十毫秒)时, 等一个 interval 就永远采不到样本,
            # 于是排队/KV 峰值会全显示 0, 看起来像"服务很轻松", 属于误导。
            if not first and self._stop.wait(self.interval):
                return
            first = False
            status, body = http_text(f"{self.root}/metrics", self.timeout)
            if status != 200:
                self.available = False
                return
            self.samples += 1
            acc: dict[str, float] = {}
            for line in body.splitlines():
                if not line.startswith("vllm:"):
                    continue
                head, _, val = line.rpartition(" ")
                name = head.split("{")[0][5:]
                if name not in GAUGES:
                    continue
                try:
                    acc[name] = acc.get(name, 0.0) + float(val)
                except ValueError:
                    continue
            for k, v in acc.items():
                self.peaks[k] = max(self.peaks.get(k, 0.0), v)

    def stop(self) -> None:
        self._stop.set()

    def kv_peak(self) -> float:
        return max(self.peaks.get("kv_cache_usage_perc", 0.0),
                   self.peaks.get("gpu_cache_usage_perc", 0.0))


# ==========================================================================
# 单档压测
# ==========================================================================
@dataclass
class LevelResult:
    concurrency: int
    wall_s: float = 0.0
    claimed: int = 0
    records: list[Rec] = field(default_factory=list)
    peaks: dict[str, float] = field(default_factory=dict)
    metrics_ok: bool = True
    metric_samples: int = 0

    # ---- 派生统计 ----
    @property
    def ok(self) -> list[Rec]:
        return [r for r in self.records if r.ok]

    @property
    def rate(self) -> float:
        return len(self.ok) / len(self.records) if self.records else 0.0

    def latencies(self) -> list[float]:
        return [r.latency for r in self.records if r.ok]

    def ttfts(self) -> list[float]:
        return [r.ttft for r in self.records if r.ok and r.ttft is not None]  # type: ignore[misc]

    def to_dict(self) -> dict:
        errs = Counter(r.status for r in self.records if not r.ok)
        lat = self.latencies()
        ttft = self.ttfts()
        comp = sum(r.completion_tokens for r in self.ok)
        return {
            "concurrency": self.concurrency,
            "wall_s": round(self.wall_s, 3),
            "requests": len(self.records),
            "ok": len(self.ok),
            "success_rate": round(self.rate, 4),
            "errors_by_status": {str(k): v for k, v in errs.items()},
            "error_samples": sorted({r.error for r in self.records if not r.ok})[:5],
            "latency_ms": {"p50": round(pct(lat, 50) * 1000, 1),
                           "p90": round(pct(lat, 90) * 1000, 1),
                           "p99": round(pct(lat, 99) * 1000, 1)},
            "ttft_ms": ({"p50": round(pct(ttft, 50) * 1000, 1),
                         "p90": round(pct(ttft, 90) * 1000, 1)} if ttft else None),
            "throughput_req_s": round(len(self.ok) / self.wall_s, 2) if self.wall_s else 0.0,
            "throughput_out_tokens_s": round(comp / self.wall_s, 1)
            if self.wall_s and comp else None,
            "avg_chars": round(sum(r.chars for r in self.ok) / len(self.ok), 1) if self.ok else 0,
            "server_peaks": {k: round(v, 3) for k, v in (self.peaks or {}).items()},
            "metrics_available": self.metrics_ok,
            "metric_samples": self.metric_samples,
        }


def pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = min(len(s) - 1, max(0, int(round(p / 100.0 * (len(s) - 1)))))
    return s[k]


class LevelState:
    def __init__(self, total: int | None, deadline: float) -> None:
        self.lock = threading.Lock()
        self.records: list[Rec] = []
        self.claimed = 0
        self.total = total
        self.deadline = deadline

    def claim(self) -> bool:
        """领一个请求配额; 额度用完或到时间就返回 False(worker 退出)。"""
        with self.lock:
            if time.time() >= self.deadline:
                return False
            if self.total is not None and self.claimed >= self.total:
                return False
            self.claimed += 1
            return True

    def push(self, rec: Rec) -> None:
        with self.lock:
            self.records.append(rec)


def warmup(url: str, payload: dict, headers: dict, timeout: float,
           stream: bool, times: int) -> None:
    """先打几个请求热身: 首次请求会触发 CUDA 图捕获/编译, 计入统计会严重失真。"""
    for i in range(times):
        rec = post_chat(url, payload, headers, timeout, stream)
        mark = "✓" if rec.ok else "✗"
        print(f"    热身 {i + 1}/{times} {mark} {rec.latency * 1000:.0f} ms"
              + ("" if rec.ok else f"  {rec.error}"))
        if not rec.ok:
            break


def run_level(cfg: argparse.Namespace, concurrency: int, picker: "Picker", headers: dict,
              root: str) -> LevelResult:
    res = LevelResult(concurrency=concurrency)
    # 时长模式 = 每档压 --duration 秒; 否则每档固定 --requests 个请求
    total = None if cfg.duration else cfg.requests
    deadline = time.time() + cfg.duration if cfg.duration else float("inf")
    state = LevelState(total, deadline)

    sampler = MetricsSampler(root, cfg.sample_interval, min(cfg.timeout, 10.0))
    if cfg.sample_interval > 0:
        sampler.start()

    def worker() -> None:
        while state.claim():
            state.push(post_chat(f"{cfg.url}/chat/completions", picker.next(), headers,
                                 cfg.timeout, cfg.stream))

    threads = [threading.Thread(target=worker, daemon=True, name=f"w{i}")
               for i in range(concurrency)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    res.wall_s = time.perf_counter() - t0

    if cfg.sample_interval > 0:
        sampler.stop()
        sampler.join(timeout=2.0)
        res.peaks = sampler.peaks
        res.metrics_ok = sampler.available
        res.metric_samples = sampler.samples

    res.claimed = state.claimed
    res.records = state.records
    return res


# ==========================================================================
# 展示
# ==========================================================================
HEADER = (f"{'并发':>4} {'请求':>5} {'成功':>5} {'失败':>5} "
          f"{'p50(ms)':>8} {'p90(ms)':>8} {'p99(ms)':>8} "
          f"{'TTFT50':>7} {'req/s':>7} {'tok/s':>7} {'KV%':>6} {'等待峰值':>8}")


def fmt_level(r: LevelResult) -> str:
    d = r.to_dict()
    lat = d["latency_ms"]
    ttft = (d["ttft_ms"] or {}).get("p50")
    tok = d["throughput_out_tokens_s"]
    kv = r.peaks.get("kv_cache_usage_perc", r.peaks.get("gpu_cache_usage_perc", 0.0))
    wait = r.peaks.get("num_requests_waiting", 0.0)
    # 没采到样本时一律显示 '-', 不要用 0 冒充"很轻松"
    kv_s = f"{kv * 100:.1f}" if r.metric_samples else "-"
    wait_s = f"{wait:.0f}" if r.metric_samples else "-"
    return (
        f"{d['concurrency']:>4} {d['requests']:>5} {d['ok']:>5} {d['requests'] - d['ok']:>5} "
        f"{lat['p50']:>8.1f} {lat['p90']:>8.1f} {lat['p99']:>8.1f} "
        f"{f'{ttft:.0f}' if ttft else '-':>7} {d['throughput_req_s']:>7.2f} "
        f"{f'{tok:.0f}' if tok else '-':>7} {kv_s:>6} {wait_s:>8}"
    )


def explain(results: list[LevelResult], threshold: float, cfg: argparse.Namespace) -> None:
    """给出"稳定并发上限"和一点解读, 免得只丢一堆数字让人猜。"""
    good = [r for r in results if r.rate >= threshold and not any(
        s == 0 or s >= 500 for s in (rec.status for rec in r.records if not rec.ok))]
    print()
    print("=" * 110)
    if not good:
        print(f"结论: 没有任何档位达到成功率 >= {threshold:.0%} —— 服务可能没起来或已经过载。")
        print("      先跑 test_paddleocr_vl_service.py 确认功能正常, 再看容器日志。")
        return

    best = max(good, key=lambda r: r.concurrency)
    bd = best.to_dict()
    print(f"结论: 稳定并发上限 ≈ {bd['concurrency']}"
          f"(成功率 {bd['success_rate']:.0%}, p90 {bd['latency_ms']['p90']:.0f} ms, "
          f"{bd['throughput_req_s']:.2f} req/s"
          + (f", {bd['throughput_out_tokens_s']:.0f} tok/s" if bd["throughput_out_tokens_s"] else "")
          + ")")

    peak = max(results, key=lambda r: r.to_dict()["throughput_req_s"])
    if peak.concurrency != best.concurrency:
        pd = peak.to_dict()
        print(f"      峰值吞吐出现在并发 {pd['concurrency']}({pd['throughput_req_s']:.2f} req/s),"
              f" 但该档已开始排队/劣化 —— 吞吐最优 ≠ 延迟可接受。")

    # 排队与 KV 是 vLLM 的两个典型瓶颈信号
    waiting = max((r.peaks.get("num_requests_waiting", 0.0) for r in results), default=0.0)
    kv = max((r.to_dict()["server_peaks"].get("kv_cache_usage_perc",
                                               r.to_dict()["server_peaks"].get("gpu_cache_usage_perc", 0.0))
              for r in results), default=0.0)
    if all(r.metric_samples == 0 for r in results):
        print("      提示: 一次 /metrics 样本都没采到(档位太短, 或 --sample-interval 0), "
              "排队与 KV 峰值不可见 —— 加大 --requests / --duration 再看。")
    elif not results[0].metrics_ok:
        print("      提示: /metrics 不可用, 排队与 KV 峰值不可见。")
    else:
        if waiting >= 1:
            print(f"      服务端等待队列峰值 {waiting:.0f}: 并发已超过实际并行处理能力,"
                  f" 想再提高需加大 --max-num-seqs / 显存。")
        if kv >= 0.95:
            print(f"      KV cache 峰值 {kv:.0%} 已打满: 可调大 GPU_MEMORY_UTILIZATION,"
                  f" 或调小 MAX_MODEL_LEN / MAX_NUM_BATCHED_TOKENS。")
        if waiting < 1 and kv < 0.8:
            print("      没有观察到排队和 KV 打满: 瓶颈可能还在客户端侧"
                  "(单机线程发请求 / base64 编码图片), 可加大 --concurrency 再试。")

    if cfg.stream:
        print("      注: 本次为流式模式, TTFT 反映首字延迟, 端到端时延包含整个输出时长。")


# ==========================================================================
# main
# ==========================================================================
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PaddleOCR-VL-1.6 (vLLM) 并发压力测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例: python press_test.py --concurrency 1,2,4,8 --requests 16 --stream")
    p.add_argument("--url", default=DEFAULT_URL, help=f"服务地址(默认 {DEFAULT_URL})")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"模型名(默认 {DEFAULT_MODEL})")
    p.add_argument("--api-key", default=os.environ.get("PADDLEOCR_VL_API_KEY", "EMPTY"),
                   help="服务端设了 --api-key 时用它(默认 EMPTY)")
    p.add_argument("--concurrency", default="1,2,4,8",
                   help="并发阶梯, 逗号分隔(默认 1,2,4,8; 上限 %d)" % MAX_LEVEL)
    p.add_argument("--requests", type=int, default=16, help="每档请求数(默认 16)")
    p.add_argument("--duration", type=float, default=0.0,
                   help="每档压测时长秒数; 设了它就不看 --requests(默认 0=关)")
    p.add_argument("--warmup", type=int, default=2, help="正式开始前的热身请求数(默认 2)")
    p.add_argument("--stream", action="store_true",
                   help="用流式(SSE)请求, 额外统计首字延迟 TTFT")
    p.add_argument("--max-tokens", type=int, default=1024, help="生成长度上限(默认 1024)")
    p.add_argument("--timeout", type=float, default=600.0, help="单请求超时秒数(默认 600)")
    p.add_argument("--threshold", type=float, default=0.99, help="判定该档可用的成功率(默认 0.99)")
    p.add_argument("--sample-interval", type=float, default=1.0,
                   help="抓 /metrics 的间隔秒数(默认 1.0; 0 = 不抓)")
    src = p.add_argument_group("输入(不指定就用现画数字图)")
    src.add_argument("--pdf", default=None,
                     help="用 PDF 压测: 逐页渲染后, 每个请求轮流取一页")
    src.add_argument("--pages", default="1",
                     help="PDF 页码, 如 1 / 1-5 / 1,3,5-7 / all(默认 1)")
    src.add_argument("--dpi", type=int, default=DEFAULT_DPI,
                     help=f"PDF 渲染 DPI(默认 {DEFAULT_DPI}; 越大请求体越大)")
    src.add_argument("--image", default=None, help="用单张真实图片压测(与 --pdf 二选一)")
    src.add_argument("--text", default=DEFAULT_TEXT, help=f"现画图里的数字(默认 {DEFAULT_TEXT!r})")
    src.add_argument("--scale", type=int, default=26, help="现画图字号缩放(默认 26)")
    p.add_argument("--json-out", default=None, help="把结果写成 json 文件")
    p.add_argument("--wait-ready", type=float, default=0.0,
                   help="开始前等 /health 就绪的秒数(默认 0=不等)")
    return p.parse_args(argv)


class Picker:
    """多页时轮流发: 固定同一张图会被服务端多模态处理器缓存命中, 测出的吞吐偏乐观。"""

    def __init__(self, payloads: list[dict]) -> None:
        self.payloads = payloads
        self.i = 0
        self.lock = threading.Lock()

    def next(self) -> dict:
        with self.lock:
            payload = self.payloads[self.i % len(self.payloads)]
            self.i += 1
            return payload


def make_payload(cfg: argparse.Namespace, uri: str) -> dict:
    """把 data URI 包成一次 OCR 请求体(图片只编码一次并复用, 不把压测机 CPU 算进去)。"""
    return {
        "model": cfg.model,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": uri}},
            {"type": "text", "text": OCR_PROMPT},
        ]}],
        "temperature": 0.0,
        "max_tokens": cfg.max_tokens,
        **({"stream": True} if cfg.stream else {}),
    }


def build_payloads(cfg: argparse.Namespace) -> list[dict]:
    """按 --pdf / --image / 默认 构造请求体列表(每页一个; 只有一项时就等于单图压测)。"""
    if cfg.pdf:
        if render_pdf_pages is None:
            print("错误: 无法导入 test_paddleocr_vl_service.render_pdf_pages, "
                  "请把两个脚本放在同一目录, 或用 --image 指定图片。")
            sys.exit(ENV)
        if not os.path.isfile(cfg.pdf):
            print(f"错误: PDF 不存在: {cfg.pdf}")
            sys.exit(ENV)
        try:
            total, rendered = render_pdf_pages(cfg.pdf, cfg.pages, cfg.dpi)
        except SystemExit:
            raise
        except Exception as e:        # noqa: BLE001
            print(f"错误: 渲染 PDF 失败: {type(e).__name__}: {e}")
            sys.exit(ENV)
        sizes = ", ".join(
            f"第{no}页 {len(blob) / 1024:.0f}KB"
            + (f"({png_dimensions(blob)[0]}x{png_dimensions(blob)[1]})" if png_dimensions else "")
            for no, blob in rendered)
        print(f"  输入: {cfg.pdf}(共 {total} 页, 本次用 {len(rendered)} 页, dpi={cfg.dpi})")
        print(f"       {sizes}")
        print(f"       每个请求轮流取一页, 避免多模态处理器缓存命中导致结果偏乐观")
        return [make_payload(cfg, bytes_to_data_uri(blob)) for _, blob in rendered]

    if cfg.image:
        if not os.path.isfile(cfg.image):
            print(f"错误: 图片不存在: {cfg.image}")
            sys.exit(ENV)
        with open(cfg.image, "rb") as f:
            blob = f.read()
        import mimetypes
        mime = mimetypes.guess_type(cfg.image)[0] or "image/png"
        print(f"  输入: {cfg.image}({len(blob) / 1024:.1f} KB)")
        return [make_payload(cfg, bytes_to_data_uri(blob, mime))]

    if render_digits_png is None or bytes_to_data_uri is None:
        print("错误: 无法导入 test_paddleocr_vl_service(现画图/编码工具), "
              "请用 --image 指定图片, 或把两个脚本放在同一目录。")
        sys.exit(ENV)
    blob = render_digits_png(cfg.text, cfg.scale)
    print(f"  输入: 现画数字图 {cfg.text!r}({len(blob) / 1024:.1f} KB)")
    return [make_payload(cfg, bytes_to_data_uri(blob))]


def main(argv: list[str] | None = None) -> int:
    cfg = parse_args(argv)
    cfg.url = cfg.url.rstrip("/")
    root = cfg.url[:-3] if cfg.url.endswith("/v1") else cfg.url
    headers = {"Authorization": f"Bearer {cfg.api_key}"} if cfg.api_key else {}

    try:
        levels = sorted({int(x) for x in re.split(r"[,\s]+", cfg.concurrency) if x.strip()})
    except ValueError:
        print(f"错误: --concurrency 解析失败: {cfg.concurrency!r}(应为 1,2,4,8 这样的格式)")
        return ENV
    if not levels or levels[-1] > MAX_LEVEL:
        print(f"错误: 并发值需在 1..{MAX_LEVEL} 之间, 当前: {levels}")
        return ENV

    print("=" * 110)
    print("PaddleOCR-VL-1.6 (vLLM) 并发压力测试")
    print(f"  地址: {cfg.url}    模型: {cfg.model}")
    print(f"  模式: {'流式' if cfg.stream else '非流式'}"
          f"    每档: {f'{cfg.duration:g}s 时长' if cfg.duration else f'{cfg.requests} 个请求'}"
          f"    max_tokens: {cfg.max_tokens}")
    print("=" * 110)

    if cfg.wait_ready > 0:
        deadline = time.time() + cfg.wait_ready
        while True:
            status, _ = http_text(f"{root}/health", 10)
            if status == 200:
                print("  服务已就绪")
                break
            if time.time() >= deadline:
                print(f"  错误: 等待 {cfg.wait_ready:.0f}s 后 /health 仍不通。")
                print("        docker compose logs -f paddleocr-vl-vllm 看看卡在哪。")
                return ENV
            time.sleep(2.0)
    else:
        status, body = http_text(f"{root}/health", 10)
        if status != 200:
            print(f"  警告: /health 返回 HTTP {status} {body[:120]}")
            print("        服务可能还在加载模型 —— 建议加 --wait-ready 300 重试。")

    picker = Picker(build_payloads(cfg))
    if cfg.warmup > 0:
        print("\n-- 热身(首次请求会触发 CUDA 图捕获/编译, 必须排除在统计外) --")
        warmup(f"{cfg.url}/chat/completions", picker.payloads[0], headers, cfg.timeout,
               cfg.stream, cfg.warmup)

    print("\n-- 分档压测(闭环: 每档并发 = 同时在飞的请求数) --")
    print(HEADER)
    results: list[LevelResult] = []
    for n in levels:
        r = run_level(cfg, n, picker, headers, root)
        results.append(r)
        print(fmt_level(r), flush=True)
        d = r.to_dict()
        if d["error_samples"]:
            print(f"      错误样例: {d['error_samples']}")

    explain(results, cfg.threshold, cfg)

    if cfg.json_out:
        summary = {
            "url": cfg.url,
            "model": cfg.model,
            "stream": cfg.stream,
            "max_tokens": cfg.max_tokens,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "levels": [r.to_dict() for r in results],
        }
        with open(cfg.json_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\n结果已写入 {cfg.json_out}")

    any_good = any(r.rate >= cfg.threshold for r in results)
    return OK if any_good else FAIL


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n已中断")
        sys.exit(ENV)
