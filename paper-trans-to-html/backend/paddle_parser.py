"""PaddleOCR-VL 推理服务客户端 · 整页 OCR

参考 `../paddle_ocr_doc_parse_service`：把 PaddleOCR-VL-1.6 当成**外部 HTTP 服务**
接入(vLLM 提供的 OpenAI 兼容接口；默认 http://127.0.0.1:8080/v1)，本进程只做客户端：

  1) 服务端只吃**图片**、不吃 PDF，所以这里先用 PyMuPDF 把每页渲染成 PNG
     (`render_page_png`)，再以 base64 data URI 塞进 `POST /v1/chat/completions`；
  2) 它是**指令式**模型：content 里先图后文，提示词决定任务 ——
     `OCR:`(默认，整页 → Markdown/HTML) / `Table Recognition:` /
     `Formula Recognition:` / `Chart Recognition:`(见 `PROMPTS`)；
  3) 就绪探测用 `/health`(vLLM 引擎起来后才返回 200) + `/v1/models`；
     模型名由服务端 `--served-model-name` 决定(默认 `PaddleOCR-VL-1.6-0.9B`)。

对外接口与 `surya_parser` 保持一致( `check_server` / `available` / `PaddleClient` )，
方便 `pdf_parser` 把两个 OCR 后端按同一套逻辑编排。

⚠️ 模型侧 `max_pixels`≈100 万(约 1000x1000)：整页高 DPI 送进去会被**服务端自己缩小**，
   细节必然有损 —— `dpi` 别一味调大；单页输出长度用 `max_tokens` 控制。
"""
from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request

# 默认地址与 paddle_ocr_doc_parse_service 的端口保持一致(见其 docker-compose.yml)
DEFAULT_URL = "http://127.0.0.1:8080/v1"
# 服务端 compose 里 --served-model-name 挂的别名
DEFAULT_MODEL = "PaddleOCR-VL-1.6-0.9B"
DEFAULT_DPI = 192
DEFAULT_MAX_TOKENS = 4096    # 单页生成上限(整页 Markdown 通常够用)
# 整页识别用哪套提示词(其它任务见 PROMPTS)。`OCR:` 是官方任务串，实测最稳。
#
# ⚠️ 改这句话之前先看这组实测(真论文 7 页，同一份 PDF，只看公式落地)：
#   `OCR:`                          anchor_missed=18  font_used=56  残留 58 行/276 字
#   `OCR: …transcribe… verbatim…`   anchor_missed= 8  font_used=55  残留 59 行/278 字
#   `OCR: …formulas as LaTeX…`      anchor_missed= 8  font_used=58  残留 56 行/273 字
# 后两种把 anchor_missed 砍掉一半(OCR 不再把段落念串，于是也没了那些"配不上"的公式)，
# 但**逐行 diff 残留清单**后，多覆盖的只有 2 行 —— 而且是正文里的斜体 `i`(`i-th`)，
# 不是公式；代价却是 LaTeX 保真度下降(丢 `\mathbf`、把上标的 `′` 读成 `r`)。
# 结论：**收益不成立**，默认保持 `OCR:`。想换可以在 config.json 里调 `paddle_prompt`。
DEFAULT_PROMPT = "OCR:"      # 整页识别；其它任务见 PROMPTS

PROBE_TIMEOUT = 1.5          # auto 模式探测服务的超时(秒)
CHECK_TIMEOUT = 10.0         # 显式检查 / 列模型单次请求的超时(秒)
REQUEST_TIMEOUT = 300.0      # 单页 OCR 请求超时(秒)
PAGE_TIMEOUT = 600.0         # 整份 PDF 逐页 OCR 的兜底超时(秒)

# PaddleOCR-VL 的任务提示词(见 vLLM 官方 recipe 与模型卡)
PROMPTS = {
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
    "chart": "Chart Recognition:",
}


class PaddleUnavailable(RuntimeError):
    """PaddleOCR-VL 服务不可用，或 OCR 任务失败/超时。"""


def normalize_url(url: str | None = None) -> str:
    """补全默认地址，并去掉结尾多余的 '/'。"""
    u = (url or "").strip() or DEFAULT_URL
    return u.rstrip("/")


def _root(url: str) -> str:
    """服务根地址(`/v1` 之外)。`/health`、`/metrics` 挂在根路径上，不在 /v1 下。"""
    return url[:-3] if url.endswith("/v1") else url


def _get_json(url: str, timeout: float) -> dict:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data if isinstance(data, dict) else {}


def _models(base: str, timeout: float) -> list[str]:
    """GET /v1/models 取模型别名列表；失败返回空列表，不影响主流程。"""
    try:
        data = _get_json(base + "/models", timeout)
    except Exception:
        return []
    return [str(m.get("id")) for m in (data.get("data") or [])
            if isinstance(m, dict) and m.get("id")]


def check_server(url: str | None = None, timeout: float = CHECK_TIMEOUT,
                 model: str | None = None) -> tuple[bool, list[str], str]:
    """探测服务就绪状态。返回 (是否就绪, 模型名列表, 说明文本)。

    vLLM 引擎(含权重加载)起来后 `/health` 才返回 200；连接失败/超时/未就绪
    都算不可用。就绪时顺手读 `/v1/models`，方便核对模型别名。
    """
    base = normalize_url(url)
    health_url = _root(base) + "/health"
    try:
        req = urllib.request.Request(health_url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp.read()
    except urllib.error.HTTPError as exc:
        return False, [], f"PaddleOCR-VL 服务未就绪({health_url} 返回 HTTP {exc.code})"
    except Exception as exc:  # 连接失败/超时都算不可用
        return False, [], f"无法连接 PaddleOCR-VL 服务 {health_url}：{exc}"

    names = _models(base, timeout)
    want = (model or "").strip() or DEFAULT_MODEL
    if names and want not in names:
        return True, names, (f"服务就绪，但模型列表未见 {want}：{names}；"
                             "请确认服务端 --served-model-name 是否包含它。")
    return True, names, f"服务就绪（模型 {names or [want]}）"


def available(url: str | None = None, timeout: float = PROBE_TIMEOUT
              ) -> tuple[bool, str]:
    """轻量探测(供 auto 模式使用)：只关心「服务是否就绪」。"""
    ok, _names, msg = check_server(url, timeout=timeout)
    return ok, msg


# =====================================================================
#  页面渲染(服务端只吃图片，PDF → PNG 在本地做)
# =====================================================================
def render_page_png(page, dpi: int = DEFAULT_DPI) -> bytes:
    """PyMuPDF 页面 → PNG 字节(整页渲染，作为模型的输入)。

    dpi 越大识别越准、上传与推理越慢；96~192 是比较稳的区间。**不需要 Pillow**：
    `pix.tobytes("png")` 直接给出 PNG。
    """
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:  # pragma: no cover - 项目本就依赖 PyMuPDF
        raise PaddleUnavailable(f"缺少 PyMuPDF：pip install PyMuPDF ({exc})") from exc

    zoom = max(0.5, float(dpi) / 72.0)
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    return pix.tobytes("png")


def _data_uri(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


# =====================================================================
#  客户端
# =====================================================================
class PaddleClient:
    """PaddleOCR-VL 推理服务客户端(OpenAI 兼容的 /v1/chat/completions)。

    只连接外部服务，不负责启动/停止服务端进程；用完调用 `close()` 释放引用即可
    (这里没有需要释放的连接，`close()` 仅为与 `SuryaClient` 保持同构)。
    """

    def __init__(self, url: str | None = None, *, model: str | None = None,
                 api_key: str | None = None,
                 timeout: float = REQUEST_TIMEOUT):
        self.url = normalize_url(url or os.environ.get("PADDLE_OCR_URL"))
        self.model = (model or os.environ.get("PADDLE_OCR_MODEL")
                      or DEFAULT_MODEL).strip()
        env_key = os.environ.get("PADDLE_OCR_API_KEY") or ""
        self.api_key = (api_key if api_key is not None else env_key).strip()
        self.timeout = max(5.0, float(timeout))

    # ---------- 体检 ----------
    def check(self, timeout: float = CHECK_TIMEOUT) -> tuple[bool, str]:
        ok, _names, msg = check_server(self.url, timeout=timeout, model=self.model)
        return ok, msg

    # ---------- 任务 ----------
    def ocr_image(self, png: bytes, *, prompt: str | None = None,
                  max_tokens: int | None = None,
                  timeout: float | None = None) -> str:
        """把一张整页 PNG 送进模型，返回识别出的文本(Markdown/HTML)。"""
        payload = {
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": _data_uri(png)}},
                    {"type": "text", "text": (prompt or DEFAULT_PROMPT).strip()
                     or DEFAULT_PROMPT},
                ],
            }],
            "temperature": 0.0,
            "max_tokens": int(max_tokens or DEFAULT_MAX_TOKENS),
        }
        req = urllib.request.Request(
            self.url + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"), method="POST")
        req.add_header("Content-Type", "application/json")
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")
        try:
            with urllib.request.urlopen(
                    req, timeout=float(timeout or self.timeout)) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            raise PaddleUnavailable(
                f"OCR 请求失败(HTTP {exc.code})：{detail}") from exc
        except Exception as exc:
            raise PaddleUnavailable(f"OCR 请求失败：{exc}") from exc

        choices = (data or {}).get("choices") or []
        if not choices:
            raise PaddleUnavailable(f"OCR 响应里没有 choices：{str(data)[:300]}")
        msg = choices[0].get("message") or {}
        text = msg.get("content")
        if isinstance(text, list):     # 少数实现会把 content 拆成多段
            text = "".join(str(p.get("text") or "") for p in text
                           if isinstance(p, dict))
        return str(text or "")

    def ocr_pdf(self, pdf_path, *, dpi: int | None = None,
                max_tokens: int | None = None, prompt: str | None = None,
                pages: str | None = None,
                timeout: float = PAGE_TIMEOUT, on_progress=None) -> list[dict]:
        """一步到位：本地逐页渲染 + 逐页 OCR，返回 `pages=[{page, text}, ...]`。

        `pages` 支持 "3" / "1-5" / "1,3,5-7" / "all"(None 或空 = 全部)。
        页码从 1 开始。
        """
        import fitz  # PyMuPDF

        picks = _parse_pages(pages)
        deadline = time.time() + max(1.0, float(timeout))
        doc = fitz.open(str(pdf_path))
        out: list[dict] = []
        try:
            for pno in range(doc.page_count):
                if picks is not None and pno not in picks:
                    continue
                remain = deadline - time.time()
                if remain <= 0:
                    raise PaddleUnavailable(
                        f"整份解析超时(>{timeout:.0f}s，已到第 {pno + 1} 页)")
                page = doc.load_page(pno)
                if page.rotation % 360 != 0:
                    page.set_rotation(0)
                png = render_page_png(page, int(dpi or DEFAULT_DPI))
                text = self.ocr_image(png, prompt=prompt, max_tokens=max_tokens,
                                      timeout=min(self.timeout, remain))
                out.append({"page": pno + 1, "text": text})
                if on_progress:
                    try:
                        on_progress({"page": pno + 1, "total": doc.page_count})
                    except Exception:      # 回调出错不影响主流程
                        pass
        finally:
            doc.close()
        return out

    def close(self) -> None:
        """无需释放的服务端连接；保留此方法以便与 SuryaClient 同构调用。"""
        return None


def _parse_pages(spec: str | None) -> set[int] | None:
    """页码表达式 → 0 基索引集合；None/空/"all" 表示全部。

    支持 "3" / "1-5" / "1,3,5-7" / "all"(语法与 doc_parse_service 的测试脚本一致)。
    """
    if spec is None or str(spec).strip().lower() in ("", "all", "*"):
        return None
    out: set[int] = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, _, b = part.partition("-")
            start = int(a) if a.strip() else 1
            end = int(b) if b.strip() else 1 << 30
            out.update(range(start - 1, end))
        else:
            out.add(int(part) - 1)
    return out
