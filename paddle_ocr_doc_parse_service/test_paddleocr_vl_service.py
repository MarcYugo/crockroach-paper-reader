#!/usr/bin/env python3
"""PaddleOCR-VL-1.6 (vLLM) 推理服务 —— 部署后功能自检 / 端到端测试

HTTP 层只用标准库(urllib), 不需要 requests / openai。
输入可以是: 现画的数字图(默认, 零素材依赖) / 真实图片(--image) / **PDF(--pdf, 逐页渲染)**。
读 PDF 需要 PyMuPDF(主项目 paper-trans-to-html 用的就是它): pip install PyMuPDF
(没装 PyMuPDF 会自动退回 pypdfium2; 两个都没有才报错退出)

和本目录另外两个脚本的分工
--------------------------
  test_paddleocr_vl_service.py(本文件) 部署完先跑它: 确认服务真的能识别、能流式、参数都对
  press_test.py                        功能没问题后再压: 测并发上限与吞吐

输入模式
--------
  * 默认(什么都不给)  现画一张白底黑字数字图 —— 内容已知, 可**断言**识别结果对不对
  * --pdf 某.pdf       用 PyMuPDF 逐页渲染, **每页各发一次 OCR 请求**(最接近真实用法);
                       可 --pages 选页、--dpi 调清晰度、--expect 断言某段内容出现
  * --image 某.png     单张真实图片, 做结构性断言(有输出、长度合理)

检查项(逐条打印 ✓/✗)
--------------------
 * GET  /health                 服务是否就绪(vLLM 引擎起来后 /health 才 200)
 * GET  /v1/models              期望的模型别名是否在列表里(默认 PaddleOCR-VL-1.6-0.9B)
 * POST /v1/chat/completions    纯文本请求 —— HTTP 层与请求体格式是否正确
 * POST /v1/chat/completions    逐页 OCR(核心用例): 每页都跑一遍, 汇总页数/字数/耗时
 * POST /v1/chat/completions    数字断言(现画图)或 --expect 内容命中断言
 * POST /v1/chat/completions    Table / Formula / Chart 三种提示词是否都能正常返回
 * POST /v1/chat/completions    一次请求塞两张图(多图多模态是否可用)
 * POST /v1/chat/completions    stream=true 流式输出(含首字延迟 TTFT)
 * POST /v1/chat/completions    故意用不存在的模型名 —— 期望 4xx(说明 model 字段真的被校验)
 * GET  /metrics                顺手抓一次 vLLM 指标(抓不到只提示, 不算失败)

为什么要"现画"一张图
--------------------
OCR 服务的自检必须是**可断言**的: 现画一张白底黑字、内容已知的数字图, 再检查模型输出里
是否出现这些数字。这样不依赖任何外部素材, 也能发现"服务活着但识别结果串了/退化成乱码"
这类静默故障(历史上很常见)。有真实文档时用 --image 传进去, 此时改为做结构性断言
(有输出、不像复读、长度合理)。

用法示例
--------
    python test_paddleocr_vl_service.py                        # 默认地址, 用现画数字图自检
    python test_paddleocr_vl_service.py --wait-ready 300        # 刚 up -d 完, 等地起完再测
    python test_paddleocr_vl_service.py --pdf 论文.pdf --pages 1-3 --out-dir ./_out
    python test_paddleocr_vl_service.py --pdf 论文.pdf --pages all --dpi 150
    python test_paddleocr_vl_service.py --pdf 扫描件.pdf --expect "总 结"   # 断言某段内容出现
    python test_paddleocr_vl_service.py --image 某页.png        # 单张真实图片
    python test_paddleocr_vl_service.py --dump-dir ./_dbg --out-dir ./_dbg   # 存测试图+原始输出
    python test_paddleocr_vl_service.py --url http://other:8080/v1

退出码
------
0 = 全部通过
1 = 有关键项失败(识别不出数字 / 4xx 5xx / 流式不通)
2 = 环境问题(服务没起来、图片/PDF 不存在、没装 PDF 渲染库)

排查提示
--------
* /health 一直不通: docker compose logs -f paddleocr-vl-vllm 看是不是在加载权重/编译, 或者
  报 CUDA 相关错误(驱动太旧)、bfloat16 不支持(图灵卡要 --dtype float16)。
* /v1/models 里没有期望名字: 检查 compose 里的 --served-model-name, 或 --model 传实际名字。
* OCR 识别结果不对: 先 --dump-dir 把现画的图存下来看一眼(是否画歪/太小), 再用 --image 试真实图。
* PDF 整页识别质量一般: 模型侧 max_pixels≈100 万(约 1000x1000), 而 192dpi 的 A4 整页是
  1587x2245, 会被服务端缩放, 细节必然损失。PaddleOCR 官方做法是先版面检测再按区域切图识别;
  本脚本只做"整页直送"的连通性/质量粗测, 想更准请送裁剪后的区域图。
"""
from __future__ import annotations

import argparse
import base64
import difflib
import json
import mimetypes
import os
import re
import struct
import sys
import time
import urllib.error
import urllib.request
import zlib
from dataclasses import dataclass

OK, FAIL, ENV = 0, 1, 2

DEFAULT_URL = os.environ.get("PADDLEOCR_VL_URL", "http://127.0.0.1:8080/v1")
# 启动时 --served-model-name 挂了这两个别名, 任选其一都能用
DEFAULT_MODEL = os.environ.get("PADDLEOCR_VL_MODEL", "PaddleOCR-VL-1.6-0.9B")
DEFAULT_TEXT = "8471 3625"
# PDF 渲染 DPI: 与主项目 paper-trans-to-html 的默认值保持一致
DEFAULT_DPI = 192
# 模型 preprocessor_config.json 的 max_pixels: 超过它服务端会自己缩, 所以整页高 DPI 送进去
# 并不会"更清晰", 只是多花上传时间与显存 —— 用来给用户提示。
MODEL_MAX_PIXELS = 1003520
# PaddleOCR-VL 的任务提示词(见 vLLM 官方 recipe 与模型卡)
PROMPTS = {
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
    "chart": "Chart Recognition:",
}


# ==========================================================================
# 输出
# ==========================================================================
def setup_stdio() -> None:
    """让输出在 Windows 上不再是雷。

    Python 在真控制台下走 Windows 控制台 API, 中文/✓ 都没问题; 但一旦输出被重定向到文件
    或管道(CI、`| tee`、子进程捕获), 就退回 locale 编码(简中 Windows 是 GBK), 打印 ✓ 会直接
    抛 UnicodeEncodeError 把脚本弄挂。这里强制 UTF-8 + errors=replace, 保证不崩。
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            if not stream.isatty():
                stream.reconfigure(encoding="utf-8", errors="replace")
            else:
                stream.reconfigure(errors="replace")
        except (AttributeError, ValueError, OSError):
            pass


class Report:
    """收集每条检查的结果, 最后统一汇总; 关键项失败会让脚本以退出码 1 结束。"""

    def __init__(self) -> None:
        self.results: list[tuple[str, bool, str]] = []

    def step(self, name: str, ok: bool, detail: str = "") -> bool:
        self.results.append((name, ok, detail))
        print(f"  {'✓' if ok else '✗'} {name}" + (f"   <- {detail}" if detail and not ok else ""),
              flush=True)
        return ok

    def summary(self) -> int:
        bad = [r for r in self.results if not r[1]]
        print()
        print("=" * 70)
        if bad:
            print(f"结果: {len(self.results) - len(bad)}/{len(self.results)} 项通过, "
                  f"{len(bad)} 项失败")
            for name, _, detail in bad:
                print(f"  ✗ {name}" + (f"   <- {detail}" if detail else ""))
            print("\n排查: docker compose logs -f paddleocr-vl-vllm")
            return FAIL
        print(f"结果: 全部 {len(self.results)} 项通过 ✓")
        return OK


def preview(text: str, limit: int = 300) -> str:
    """把模型输出压成一行短预览(测试里打印用)。"""
    one = re.sub(r"\s+", " ", text or "").strip()
    return one if len(one) <= limit else one[:limit] + f" …(共 {len(one)} 字)"


# ==========================================================================
# HTTP(纯标准库)
# ==========================================================================
def http_json(url: str, payload: dict | None = None, headers: dict | None = None,
              timeout: float = 300.0) -> tuple[int, dict]:
    """GET(payload=None) / POST JSON。返回 (状态码, 解析后的 json)。

    连接层失败返回 (0, {"detail": ...}), 不抛异常 —— 测试脚本不该因为网络抖动崩掉。
    """
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="GET" if data is None else "POST")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read().decode("utf-8", "replace")
            try:
                return r.status, json.loads(body)
            except json.JSONDecodeError:
                return r.status, {"_raw": body}
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        try:
            return e.code, json.loads(body)
        except json.JSONDecodeError:
            return e.code, {"_raw": body}
    except (urllib.error.URLError, OSError, ValueError) as e:
        return 0, {"detail": f"连接失败: {e}"}
    except Exception as e:  # noqa: BLE001
        return 0, {"detail": f"请求异常: {type(e).__name__}: {e}"}


def http_text(url: str, timeout: float = 30.0) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")
    except (urllib.error.URLError, OSError, ValueError) as e:
        return 0, f"连接失败: {e}"


def chat_stream(url: str, payload: dict, headers: dict,
                timeout: float) -> tuple[int, dict]:
    """POST /v1/chat/completions + stream=true, 逐块读取 SSE。

    返回 (状态码, {"text": 拼接后的内容, "ttft": 首字延迟秒, "usage": {...}})
    """
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), method="POST")
    req.add_header("Content-Type", "application/json")
    for k, v in headers.items():
        req.add_header(k, v)

    t0 = time.perf_counter()
    ttft = None
    parts: list[str] = []
    usage: dict = {}
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            status = r.status
            for raw in r:                      # 逐行(SSE 以 \n 分隔)
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
                    usage = obj["usage"]
                for choice in obj.get("choices") or []:
                    piece = (choice.get("delta") or {}).get("content")
                    if piece:
                        if ttft is None:
                            ttft = time.perf_counter() - t0
                        parts.append(piece)
            return status, {"text": "".join(parts), "ttft": ttft, "usage": usage}
    except urllib.error.HTTPError as e:
        return e.code, {"text": "", "ttft": None, "usage": {},
                        "detail": e.read().decode("utf-8", "replace")}
    except (urllib.error.URLError, OSError, ValueError) as e:
        return 0, {"text": "", "ttft": None, "usage": {}, "detail": f"连接失败: {e}"}


# ==========================================================================
# 现画测试图: 5x7 点阵数字 -> PNG(纯标准库, 不依赖 PIL)
# ==========================================================================
_FONT_5X7 = {
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11111", "00010", "00100", "00010", "00001", "10001", "01110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "11110", "00001", "00001", "10001", "01110"),
    "6": ("00110", "01000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00010", "01100"),
    " ": None,
}


def _encode_png(width: int, height: int, rows: list[bytearray]) -> bytes:
    """把 RGB 行数据编码成 PNG(8bit truecolor)。"""
    raw = bytearray()
    for row in rows:
        raw.append(0)          # 每行前面的 filter type: 0 = None
        raw += row

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(bytes(raw), 6)) + chunk(b"IEND", b""))


def render_digits_png(text: str, scale: int = 26, margin: int = 44) -> bytes:
    """白底黑字画一段数字(仅数字与空格), 返回 PNG 字节。

    scale = 每个点阵像素放大成多少屏幕像素, 越大字越粗越大(OCR 越稳)。
    默认尺寸约 900x270, 高于模型 min_pixels(112896) 的要求。
    """
    unknown = sorted({c for c in text if c not in _FONT_5X7})
    if unknown:
        raise ValueError(f"现画图只支持数字和空格, 不支持: {unknown!r}")

    # 先排布: 每个字符占 glyph_w 列, 字间空 1 列, 空格占 3 列
    layout: list[tuple[tuple[str, ...], int]] = []
    col = 0
    for ch in text:
        glyph = _FONT_5X7[ch]
        if glyph is None:
            col += 3
            continue
        layout.append((glyph, col))
        col += 5 + 1

    width = col * scale + margin * 2
    height = 7 * scale + margin * 2
    rows = [bytearray(b"\xff" * (width * 3)) for _ in range(height)]

    for glyph, base_col in layout:
        for ry, bits in enumerate(glyph):
            for rx, bit in enumerate(bits):
                if bit != "1":
                    continue
                x0 = margin + (base_col + rx) * scale
                y0 = margin + ry * scale
                for y in range(y0, y0 + scale):
                    row = rows[y]
                    for x in range(x0, x0 + scale):
                        i = x * 3
                        row[i] = row[i + 1] = row[i + 2] = 0

    return _encode_png(width, height, rows)


# ==========================================================================
# PDF -> PNG(逐页渲染)
# ==========================================================================
def parse_pages(spec: str | None, total: int) -> list[int]:
    """解析页码表达式 -> 0 基索引列表(去重、保序、越界丢弃)。

    支持 "3" / "1-5" / "1,3,5-7" / "all" / None(全部)。
    语法与 unlimited_ocr_doc_parse_service/test_ocr.py 保持一致, 免得多套记法。
    """
    if spec is None or str(spec).strip().lower() in ("", "all", "*"):
        return list(range(total))
    out: list[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, _, b = part.partition("-")
            start = int(a) if a.strip() else 1
            end = int(b) if b.strip() else total
            out.extend(range(start - 1, end))
        else:
            out.append(int(part) - 1)
    seen: set[int] = set()
    uniq: list[int] = []
    for i in out:
        if 0 <= i < total and i not in seen:
            seen.add(i)
            uniq.append(i)
    return uniq


def render_pdf_pages(path: str, pages_spec: str | None = None,
                     dpi: int = DEFAULT_DPI) -> tuple[int, list[tuple[int, bytes]]]:
    """把 PDF 指定页渲染成 PNG。返回 (总页数, [(页码(1基), png字节), ...])。

    优先 PyMuPDF(主项目 paper-trans-to-html 用的就是它), 没装则退回 pypdfium2。
    dpi=192 时 A4 约 1587x2245、单页 PNG 几百 KB —— 够用且不至于把请求体撑爆。
    """
    zoom = max(0.5, float(dpi) / 72.0)
    try:
        import pymupdf  # PyMuPDF 1.24+ 的正名(fitz 是旧别名, 已弃用)
    except ImportError:
        return _render_pdf_pages_pdfium(path, pages_spec, zoom)

    with pymupdf.open(path) as doc:
        picks = parse_pages(pages_spec, doc.page_count)
        if not picks:
            raise ValueError(f"页码 {pages_spec!r} 没落在 1..{doc.page_count} 内")
        out: list[tuple[int, bytes]] = []
        for i in picks:
            pix = doc.load_page(i).get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
            out.append((i + 1, pix.tobytes("png")))
        return doc.page_count, out


def _render_pdf_pages_pdfium(path: str, pages_spec: str | None,
                             zoom: float) -> tuple[int, list[tuple[int, bytes]]]:
    """PyMuPDF 缺席时的退路: pypdfium2(它自带预编译 Pdfium, 不用 npm/poppler)。"""
    try:
        import pypdfium2 as pdfium
    except ImportError:
        raise SystemExit(
            "渲染 PDF 需要 PyMuPDF(或 pypdfium2), 二者装一个即可:\n"
            "    pip install PyMuPDF\n"
            "不想装第三方库就改用 --image 传 PNG/JPG。")

    import io
    pdf = pdfium.PdfDocument(path)
    total = len(pdf)
    picks = parse_pages(pages_spec, total)
    if not picks:
        raise ValueError(f"页码 {pages_spec!r} 没落在 1..{total} 内")
    out: list[tuple[int, bytes]] = []
    for i in picks:
        buf = io.BytesIO()
        pdf[i].render(scale=zoom).to_pil().save(buf, format="PNG")
        out.append((i + 1, buf.getvalue()))
    return total, out


def png_dimensions(blob: bytes) -> tuple[int, int] | None:
    """读 PNG 的 IHDR 拿宽高(非 PNG 返回 None) —— 用于提示"会被服务端缩放"。"""
    if len(blob) >= 24 and blob[:8] == b"\x89PNG\r\n\x1a\n":
        return (int.from_bytes(blob[16:20], "big"), int.from_bytes(blob[20:24], "big"))
    return None


# ==========================================================================
# 图片 / 请求体
# ==========================================================================
def bytes_to_data_uri(blob: bytes, mime: str = "image/png") -> str:
    return f"data:{mime};base64,{base64.b64encode(blob).decode()}"


def image_content(data_uri: str, prompt: str) -> list[dict]:
    """按 vLLM recipe 的格式拼多模态 content: 先图后文。"""
    return [
        {"type": "image_url", "image_url": {"url": data_uri}},
        {"type": "text", "text": prompt},
    ]


def digits_of(text: str) -> str:
    return re.sub(r"\D", "", text or "")


def digits_match(expected: str, got: str, threshold: float = 0.8) -> tuple[bool, float]:
    """比较期望数字与识别出的数字。

    完全包含 = 通过(模型常会多输出页码/年份之类)。否则用相似度兜底,
    容忍个别数字被认错(例如 8/0、1/7 混淆), 超过阈值也算通过。
    """
    exp = digits_of(expected)
    got_digits = digits_of(got)
    if not exp or not got_digits:
        return False, 0.0
    if exp in got_digits:
        return True, 1.0
    ratio = difflib.SequenceMatcher(None, exp, got_digits).ratio()
    return ratio >= threshold, ratio


class Ctx:
    """一次测试运行的所有上下文。"""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.url = args.url.rstrip("/")
        # /health 与 /metrics 挂在根路径, 不在 /v1 下
        self.root = self.url[:-3] if self.url.endswith("/v1") else self.url
        self.headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
        self.timeout = args.timeout
        self.model = args.model
        self.max_tokens = args.max_tokens

    def post_chat(self, messages: list[dict], stream: bool = False,
                  model: str | None = None) -> tuple[int, dict]:
        payload = {
            "model": model or self.model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
        }
        if stream:
            payload["stream"] = True
            return chat_stream(f"{self.url}/chat/completions", payload, self.headers, self.timeout)
        status, obj = http_json(f"{self.url}/chat/completions", payload, self.headers,
                                self.timeout)
        if status == 200:
            choices = obj.get("choices") or []
            obj["_text"] = ((choices[0].get("message") or {}).get("content") or "") if choices else ""
        return status, obj

    @staticmethod
    def error_of(status: int, obj: dict) -> str:
        if status == 0:
            return str(obj.get("detail", "连接失败"))
        err = obj.get("error")
        if isinstance(err, dict):
            return f"HTTP {status}: {err.get('message', '')}"
        return f"HTTP {status}: {preview(str(obj.get('detail') or obj.get('_raw') or obj), 200)}"

    def save(self, name: str, text: str) -> None:
        if not self.args.out_dir:
            return
        os.makedirs(self.args.out_dir, exist_ok=True)
        path = os.path.join(self.args.out_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"      (已保存 {path})")


# ==========================================================================
# 各项检查
# ==========================================================================
def wait_ready(ctx: Ctx, report: Report, wait_s: float) -> bool:
    """等 /health 返回 200。vLLM 要加载权重 + 抓 CUDA 图, 首次启动可能好几分钟。"""
    t0 = time.time()
    deadline = t0 + max(wait_s, 0)
    attempt = 0
    last = ""
    while True:
        attempt += 1
        status, body = http_text(f"{ctx.root}/health", timeout=10)
        if status == 200:
            if attempt > 1:
                print(f"    (第 {attempt} 次探测就绪, 耗时 {time.time() - t0:.0f}s)")
            return True
        last = f"HTTP {status} {preview(body, 120)}" if status else preview(body, 120)
        if time.time() >= deadline:
            break
        time.sleep(2.0)
    print(f"    /health 探测 {attempt} 次仍未就绪: {last}")
    return False


def check_health(ctx: Ctx, report: Report) -> None:
    status, body = http_text(f"{ctx.root}/health", timeout=10)
    report.step("GET /health 服务就绪", status == 200, f"HTTP {status} {preview(body, 120)}")


def check_models(ctx: Ctx, report: Report) -> bool:
    status, obj = http_json(f"{ctx.url}/models", headers=ctx.headers, timeout=30)
    ids = [m.get("id") for m in (obj.get("data") or [])] if status == 200 else []
    ok = status == 200 and ctx.model in ids
    detail = ""
    if not ok:
        detail = (f"HTTP {status}: {Ctx.error_of(status, obj)}" if status != 200
                  else f"列表里没有 {ctx.model}, 实际有: {ids}")
    if ids:
        print(f"      可用模型: {ids}")
    return report.step(f"GET /v1/models 含 {ctx.model}", ok, detail)


def check_text_only(ctx: Ctx, report: Report) -> str:
    """纯文本请求: 只为验证 HTTP 层 / 参数 / 鉴权没问题(内容不做断言)。"""
    status, obj = ctx.post_chat([{"role": "user", "content": "你好"}])
    if status != 200:
        report.step("纯文本请求", False, Ctx.error_of(status, obj))
        return ""
    text = obj.get("_text", "")
    usage = obj.get("usage") or {}
    report.step("纯文本请求返回 200", bool(text) or usage.get("completion_tokens", 0) > 0,
                f"响应非空校验失败: {obj}")
    print(f"      输出预览: {preview(text, 160) or '(空)'}")
    return text


@dataclass
class PageInput:
    """一张待送进模型的图片: 现画图 / 真实图片 / PDF 的某一页。"""
    name: str                  # 打印用, 如 "第 3 页"
    save_name: str             # 存盘用, 如 "page_003"
    uri: str                   # data URI
    blob: bytes                # 原始图片字节(算尺寸/存盘用)
    expected: str = ""         # 期望出现的数字; 空 = 只做结构性断言


def structural_ok(text: str) -> bool:
    """结构性判定: 有输出、且长度没夸张到像复读/跑飞。"""
    return len(text.strip()) >= 1 and len(text) < 20000


def check_pages_ocr(ctx: Ctx, report: Report,
                    pages: list[PageInput]) -> list[tuple[str, str]]:
    """核心用例: 逐页发 OCR 请求。

    每页都会打一行统计(字数/耗时/图片尺寸/请求体大小/tokens), 第一页额外给输出预览,
    其余页只在出错时打预览 —— 否则 50 页 PDF 会把屏幕刷爆。
    返回 [(页名, 输出), ...], 供 --expect 断言与存盘使用。
    """
    outputs: list[tuple[str, str]] = []
    bad_struct: list[str] = []
    bad_digits: list[str] = []
    t_all = time.perf_counter()

    for idx, p in enumerate(pages):
        t0 = time.perf_counter()
        status, obj = ctx.post_chat([
            {"role": "user", "content": image_content(p.uri, PROMPTS["ocr"])}
        ])
        dt = time.perf_counter() - t0
        if status != 200:
            detail = Ctx.error_of(status, obj)
            bad_struct.append(f"{p.name}: {detail}")
            print(f"      {p.name} ✗ {detail}")
            continue

        text = obj.get("_text", "")
        usage = obj.get("usage") or {}
        outputs.append((p.name, text))
        ctx.save(f"{p.save_name}.txt", text)

        marks: list[str] = []
        if not structural_ok(text):
            bad_struct.append(f"{p.name}: 输出异常({len(text)} 字)")
            marks.append("✗结构")
        if p.expected:
            ok_d, ratio = digits_match(p.expected, text)
            if not ok_d:
                bad_digits.append(
                    f"{p.name}: 期望 {p.expected} / 实际 {digits_of(text) or '无数字'}"
                    f"(相似度 {ratio:.2f})")
            marks.append(f"数字{'✓' if ok_d else '✗'}({ratio:.2f})")

        size = png_dimensions(p.blob)
        print(f"      {p.name} {' '.join(marks) or '✓'}  {len(text)} 字 / {dt * 1000:.0f} ms"
              f" / {len(p.blob) / 1024:.0f} KB"
              + (f" / {size[0]}x{size[1]}" if size else "")
              + f" / tokens {usage.get('prompt_tokens')}+{usage.get('completion_tokens')}")
        if idx == 0 or not structural_ok(text):
            print(f"        输出预览: {preview(text) or '(空)'}")

    total_chars = sum(len(t) for _, t in outputs)
    avg = total_chars / len(outputs) if outputs else 0
    print(f"      汇总: {len(pages)} 页 -> 成功 {len(outputs)}, 失败 {len(pages) - len(outputs)},"
          f" 共 {total_chars} 字(平均 {avg:.0f} 字/页),"
          f" 总耗时 {time.perf_counter() - t_all:.1f}s")

    report.step(f"逐页 OCR({len(pages)} 页全部有输出)", not bad_struct, "; ".join(bad_struct[:3]))
    if any(p.expected for p in pages):
        report.step("现画图数字断言", not bad_digits, "; ".join(bad_digits[:3]))
    return outputs


def check_expect(ctx: Ctx, report: Report, outputs: list[tuple[str, str]],
                 expect: str) -> None:
    """--expect 断言: 期望内容是否出现在某一页的输出里。

    比对前把空白全去掉(OCR 输出里空格/换行到处乱插), 其余原样子串匹配。
    """
    key = re.sub(r"\s+", "", expect)
    if not key:
        return
    hits = [name for name, text in outputs if key in re.sub(r"\s+", "", text)]
    print(f"      命中页: {hits or '无'}")
    report.step(f"内容断言 {expect!r} 命中", bool(hits),
                f"{len(outputs)} 页输出里都没找到 {expect!r}")


def check_prompts(ctx: Ctx, report: Report, page: PageInput) -> None:
    """三种任务提示词是否都能正常返回(表格/公式/图表; 内容不做断言)。"""
    bad: list[str] = []
    for task in ("table", "formula", "chart"):
        status, obj = ctx.post_chat([
            {"role": "user", "content": image_content(page.uri, PROMPTS[task])}
        ])
        if status != 200:
            bad.append(f"{PROMPTS[task]} -> {Ctx.error_of(status, obj)}")
        else:
            print(f"      {PROMPTS[task]:22s} 输出 {len(obj.get('_text', ''))} 字")
    report.step("Table/Formula/Chart 提示词", not bad, "; ".join(bad))


def check_multi_image(ctx: Ctx, report: Report, pages: list[PageInput]) -> None:
    """一次请求带两张图(PDF 就是前两页): 验证多图多模态可用。"""
    if len(pages) < 2:
        print("  · 单请求多图: 已跳过(当前只有一张图, 可 --pages 1-2 或 --pages all)")
        return
    pair = pages[:2]
    content: list[dict] = [{"type": "image_url", "image_url": {"url": p.uri}} for p in pair]
    content.append({"type": "text", "text": PROMPTS["ocr"]})
    status, obj = ctx.post_chat([{"role": "user", "content": content}])
    if status != 200:
        report.step("单请求多图", False, Ctx.error_of(status, obj))
        return

    text = obj.get("_text", "")
    ctx.save("multi_image.txt", text)
    hits = []
    for p in pair:
        if p.expected:
            if digits_match(p.expected, text, 0.75)[0]:
                hits.append(p.name)
        elif structural_ok(text):
            hits.append(p.name)
    print(f"      送入 {[p.name for p in pair]}, 命中 {len(hits)}/{len(pair)}"
          f" | 输出预览: {preview(text)}")
    report.step(f"单请求多图({pair[0].name} + {pair[1].name})", len(hits) == len(pair),
                f"只认出 {hits}, 原始输出: {preview(text, 200)}")


def check_stream(ctx: Ctx, report: Report, page: PageInput) -> None:
    status, obj = ctx.post_chat([
        {"role": "user", "content": image_content(page.uri, PROMPTS["ocr"])}
    ], stream=True)
    if status != 200:
        report.step("流式输出(stream=true)", False, Ctx.error_of(status, obj))
        return
    text = obj.get("text", "")
    ttft = obj.get("ttft")
    if page.expected:
        ok, ratio = digits_match(page.expected, text)
        extra = f" | 数字相似度 {ratio:.2f}"
    else:
        ok, extra = len(text.strip()) >= 5, ""
    print(f"      TTFT(首字延迟) = " + (f"{ttft * 1000:.0f} ms" if ttft else "未测到")
          + f" | 总长 {len(text)} 字{extra}")
    report.step("流式输出(stream=true)", bool(text) and ttft is not None and ok,
                f"流式内容不完整或识别不对: {preview(text, 200)}")


def check_bad_model(ctx: Ctx, report: Report) -> None:
    """故意用不存在的模型名: 期望 4xx。返回 200 说明 --served-model-name 没生效。"""
    status, obj = http_json(f"{ctx.url}/chat/completions",
                            {"model": "__no_such_model__",
                             "messages": [{"role": "user", "content": "hi"}],
                             "max_tokens": 4},
                            ctx.headers, timeout=60)
    report.step("错误模型名返回 4xx", 400 <= status < 500,
                f"实际 HTTP {status}(期望 4xx): {preview(str(obj), 160)}")


def check_metrics(ctx: Ctx, report: Report) -> None:
    """顺手抓一次 /metrics(拿不到只提示, 不算失败; 压测脚本会用它看服务端状态)。"""
    status, body = http_text(f"{ctx.root}/metrics", timeout=15)
    if status != 200:
        print(f"  · GET /metrics 不可用(HTTP {status}), 不影响功能, 压测时少一个观测维度")
        return
    keys = re.findall(r"^vllm:(\w+)", body, flags=re.M)
    interesting = [k for k in keys if k in ("num_requests_running", "num_requests_waiting",
                                            "kv_cache_usage_perc", "gpu_cache_usage_perc")]
    print(f"  ✓ GET /metrics 可读, 指标 {len(set(keys))} 个; 关注: {sorted(set(interesting))}")


# ==========================================================================
# main
# ==========================================================================
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PaddleOCR-VL-1.6 (vLLM) 服务部署后功能自检",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n"
               "  python test_paddleocr_vl_service.py --wait-ready 300\n"
               "  python test_paddleocr_vl_service.py --pdf 论文.pdf --pages 1-3 --out-dir ./_out\n"
               "  python test_paddleocr_vl_service.py --pdf 扫描件.pdf --pages all --dpi 150")
    p.add_argument("--url", default=DEFAULT_URL, help=f"服务地址(默认 {DEFAULT_URL})")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"期望的模型名/别名(默认 {DEFAULT_MODEL})")
    p.add_argument("--api-key", default=os.environ.get("PADDLEOCR_VL_API_KEY", "EMPTY"),
                   help="服务端设了 --api-key 时用它; 没设时 vLLM 会忽略该头(默认 EMPTY)")
    p.add_argument("--timeout", type=float, default=300.0, help="单请求超时秒数(默认 300)")
    p.add_argument("--max-tokens", type=int, default=1024, help="生成长度上限(默认 1024)")
    p.add_argument("--wait-ready", type=float, default=60.0,
                   help="/health 就绪等待秒数(默认 60; 0 = 不等, 直接测)")

    src = p.add_argument_group("输入(不指定就用脚本现画的数字图)")
    src.add_argument("--pdf", default=None,
                     help="用 PDF 测试: 逐页渲染后每页各发一次 OCR 请求(最接近真实用法)")
    src.add_argument("--pages", default="1",
                     help="PDF 页码, 如 1 / 1-5 / 1,3,5-7 / all(默认 1; all = 全部渲染)")
    src.add_argument("--dpi", type=int, default=DEFAULT_DPI,
                     help=f"PDF 渲染 DPI, 96~192 较稳(默认 {DEFAULT_DPI})")
    src.add_argument("--image", default=None,
                     help="用单张真实图片测试(PNG/JPG; 与 --pdf 二选一)")
    src.add_argument("--expect", default=None,
                     help="断言某段内容出现在输出里(如 --expect \"总 结\"; 比对时忽略空白)")
    src.add_argument("--text", default=DEFAULT_TEXT,
                     help=f"现画图里的数字(默认 {DEFAULT_TEXT!r})")
    src.add_argument("--scale", type=int, default=26, help="现画图的字号缩放(默认 26)")

    p.add_argument("--dump-dir", default=None, help="把实际送入模型的图片存到该目录(便于核对)")
    p.add_argument("--out-dir", default=None, help="把模型原始输出按页存到该目录")
    return p.parse_args(argv)


def build_inputs(args: argparse.Namespace) -> tuple[list[PageInput], str]:
    """按 --pdf / --image / 默认, 组装待测图片列表。返回 (列表, 给人看的模式说明)。"""
    if args.pdf:
        if not os.path.isfile(args.pdf):
            print(f"错误: PDF 不存在: {args.pdf}")
            sys.exit(ENV)
        try:
            total, rendered = render_pdf_pages(args.pdf, args.pages, args.dpi)
        except SystemExit:
            raise                     # “没装渲染库”的提示已经打过, 直接退出
        except Exception as e:        # noqa: BLE001 - 坏 PDF / 加密 PDF 等, 给结论就好
            print(f"错误: 渲染 PDF 失败: {type(e).__name__}: {e}")
            sys.exit(ENV)

        pages: list[PageInput] = []
        oversized: list[int] = []
        for no, blob in rendered:
            size = png_dimensions(blob)
            if size and size[0] * size[1] > MODEL_MAX_PIXELS:
                oversized.append(no)
            pages.append(PageInput(name=f"第 {no} 页", save_name=f"page_{no:03d}",
                                   uri=bytes_to_data_uri(blob), blob=blob))
        desc = f"  输入: {args.pdf} 共 {total} 页, 本次渲染 {len(pages)} 页(页码 {args.pages}, dpi={args.dpi})"
        if oversized:
            desc += (f"\n  提示: 第 {oversized[:3]}{' 等' if len(oversized) > 3 else ''} 页像素超过模型"
                     f" max_pixels≈{MODEL_MAX_PIXELS // 10000} 万, 服务端会自动缩小(细节有损);"
                     f" 想更准请送裁剪后的区域图(或降 --dpi 减少上传量)")
        return pages, desc

    if args.image:
        if not os.path.isfile(args.image):
            print(f"错误: 图片不存在: {args.image}")
            sys.exit(ENV)
        with open(args.image, "rb") as f:
            blob = f.read()
        mime = mimetypes.guess_type(args.image)[0] or "image/png"
        size = png_dimensions(blob)
        return ([PageInput(name="图片", save_name="image",
                           uri=bytes_to_data_uri(blob, mime), blob=blob)],
                f"  输入: {args.image}({len(blob) / 1024:.0f} KB"
                + (f", {size[0]}x{size[1]}" if size else "") + "), 内容无法断言, 只做结构性检查")

    first, second = args.text, args.text[::-1]     # 第二张用倒序, 便于区分哪张图的结果
    pages = [PageInput(name=f"现画图{i + 1}", save_name=f"drawn_{i + 1}",
                       uri=bytes_to_data_uri(blob), blob=blob, expected=text)
             for i, (blob, text) in enumerate(zip(
                 [render_digits_png(first, args.scale), render_digits_png(second, args.scale)],
                 [first, second]))]
    return pages, f"  输入: 现画数字图 {first!r} 与 {second!r}(内容已知, 可断言识别结果)"


def main(argv: list[str] | None = None) -> int:
    setup_stdio()
    args = parse_args(argv)
    if args.pdf and args.image:
        print("错误: --pdf 与 --image 只能给一个")
        return ENV
    ctx = Ctx(args)
    report = Report()

    print("=" * 70)
    print("PaddleOCR-VL-1.6 (vLLM) 服务自检")
    print(f"  地址: {ctx.url}")
    print(f"  期望模型: {ctx.model}    单请求超时: {args.timeout:.0f}s")
    print("=" * 70)

    try:
        pages, desc = build_inputs(args)
    except ValueError as e:           # parse_pages / 渲染参数写错
        print(f"错误: {e}")
        return ENV
    print(desc)

    if args.dump_dir:
        os.makedirs(args.dump_dir, exist_ok=True)
        for page in pages:
            path = os.path.join(args.dump_dir, f"{page.save_name}.png")
            with open(path, "wb") as f:
                f.write(page.blob)
        print(f"  已把 {len(pages)} 张实际送入模型的图写到 {args.dump_dir}")

    print("\n-- 就绪探测 --")
    t0 = time.perf_counter()
    if not wait_ready(ctx, report, args.wait_ready):
        print("\n服务未就绪, 后续检查跳过。请查看: docker compose logs -f paddleocr-vl-vllm")
        return ENV
    print(f"    服务就绪, 用了 {time.perf_counter() - t0:.1f}s")

    print("\n-- 检查项 --")
    check_health(ctx, report)
    check_models(ctx, report)
    check_text_only(ctx, report)

    outputs = check_pages_ocr(ctx, report, pages)
    if args.expect:
        check_expect(ctx, report, outputs, args.expect)
    check_prompts(ctx, report, pages[0])
    check_multi_image(ctx, report, pages)
    check_stream(ctx, report, pages[0])
    check_bad_model(ctx, report)
    check_metrics(ctx, report)

    if args.out_dir and len(outputs) > 1 and os.path.isdir(args.out_dir):
        combined = "\n\n".join(f"<!-- {name} -->\n{text}" for name, text in outputs)
        path = os.path.join(args.out_dir, "all_pages.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(combined)
        print(f"  (已把 {len(outputs)} 页输出合并写入 {path})")

    return report.summary()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n已中断")
        sys.exit(ENV)
