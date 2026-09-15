"""入口：`parse_pdf` 的后端选择与回退链。

见包文档 `backend/pdf_parser/__init__.py`。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
from .. import paddle_parser
from .. import surya_parser

from .backend_paddle import (_parse_with_paddle)
from .backend_pymupdf import (_parse_with_pymupdf)
from .backend_surya import (_parse_with_surya)
from .options import (client_ready, load_options)


# =====================================================================
#  入口
# =====================================================================
def parse_pdf(pdf_path: Path, images_dir: Optional[Path] = None, *,
              backend: Optional[str] = None, options: Optional[dict] = None,
              config_file: Optional[Path] = None) -> dict:
    """解析 PDF → 版式数据(即 doc.json 的 pages 结构)。

    后端选择与回退规则见模块 docstring。返回值额外带：
      * parser      实际使用的后端("paddle" / "surya" / "pymupdf")
      * parser_info 服务地址、DPI、OCR/版面统计(视后端而定)
      * toc         PDF 自带书签目录 `[{level,title,page}]`（可能为空）
      * warnings    回退/配置问题说明(列表，可能为空)
    """
    opts, warnings = load_options(config_file=config_file, overrides=options, backend=backend)
    if images_dir is not None:
        # 各分支都要 images_dir.mkdir()，str 会 AttributeError；统一成 Path
        images_dir = Path(images_dir)

    mode = opts["backend"]
    order = ["paddle", "surya", "pymupdf"]
    if mode == "pymupdf":
        chain = ["pymupdf"]
    else:
        # auto 从链首开始；显式指定则从该后端开始，仍可往下回退(除非 fallback=false)
        chain = order[order.index(mode):] if mode in order else list(order)
        if not opts["fallback"]:
            chain = chain[:1]

    for name in chain:
        if name == "paddle":
            if mode == "auto":
                use, pmsg = paddle_parser.available(opts["paddle_url"])
                if not use:
                    warnings.append(f"未发现 PaddleOCR-VL 推理服务 "
                                    f"{opts['paddle_url']}，尝试下一后端：{pmsg}")
                    continue
            try:
                result = _parse_with_paddle(pdf_path, images_dir, opts)
                nf = (result.get("parser_info") or {}).get("ocr_failed_pages") or 0
                if nf:
                    warnings.append(f"PaddleOCR-VL 有 {nf} 页 OCR 失败，"
                                    "这些页已退回 PyMuPDF 本地抽取")
                result["warnings"] = warnings
                return result
            except (paddle_parser.PaddleUnavailable, OSError) as exc:
                if not opts["fallback"]:
                    raise
                warnings.append(f"PaddleOCR-VL 解析失败，回退下一后端：{exc}")
                continue

        if name == "surya":
            if mode in ("auto", "paddle"):
                use, smsg = surya_parser.available(opts["surya_url"])
                if not use:
                    warnings.append(f"未发现 Surya 推理服务 {opts['surya_url']}，"
                                    f"尝试下一后端：{smsg}")
                    continue
            # 服务在跑 ≠ 能解析：客户端还要 surya-ocr + Pillow。先自检，
            # 免得每次都白渲染一页、再失败回退（Docker 里最常见的就是这个）。
            ready, why = client_ready()
            if not ready:
                if not opts["fallback"]:
                    raise surya_parser.SuryaUnavailable(why)
                warnings.append(f"{why}，尝试下一后端")
                continue
            try:
                result = _parse_with_surya(pdf_path, images_dir, opts)
                result["warnings"] = warnings
                return result
            except surya_parser.SuryaUnavailable as exc:
                if not opts["fallback"]:
                    raise
                warnings.append(f"Surya 解析失败，回退下一后端：{exc}")
                continue

        # pymupdf：链条最后一个，永远兜底
        result = _parse_with_pymupdf(pdf_path, images_dir, opts)
        result["warnings"] = warnings
        return result

    # 只在“显式指定且关闭 fallback”且该后端失败时到达(上层已抛错，这里兜底)
    raise paddle_parser.PaddleUnavailable("所有解析后端均不可用")
