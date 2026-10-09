"""入口：`parse_pdf`。

见包文档 `backend/pdf_parser/__init__.py`。

两个解析分支：

  * `backend` = `pymupdf` / `auto` / `detection_service_group`：文字版式走 PyMuPDF
    本地抽取（`backend_pymupdf`），`auto`/`detection_service_group` 时再调
    `backend_detection_service_group.apply_formula_boxes` —— 全页请求
    detection-service-group（`formula_table_service_group` 服务组），公式写入
    **有公式空位的页**的 `pages[].formula_boxes`；Figure 检测裁剪图落盘到
    `images_dir` 并写入 `pages[].images`。
  * `backend` = `surya`：整页 OCR 走 Surya 2（`backend_surya._parse_with_surya`），
    **不再使用 PyMuPDF 的版式数据**：文字/公式（`<math>`→LaTeX）/插图裁图/阅读顺序
    全部来自 Surya 的版面 block，前端按 `DOC.parser === "surya"` 走独立的块级渲染
    （见 `frontend/js/reader2.js::buildSuryaItem`）。Surya 不可用时直接报错
    （`backend/surya_parser.SuryaUnavailable`，经转换接口映成页面上的失败提示），
    **不会悄悄回退 PyMuPDF**。

对外签名与返回结构**保持不变**(仍是 `parser` / `parser_info` / `toc` /
`warnings`)，所以 `converter.py` / `app.py` / `tools/*` 都不用改。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .backend_detection_service_group import (apply_formula_boxes)
from .backend_pymupdf import (_parse_with_pymupdf)
from .backend_surya import (_parse_with_surya)
from .options import (load_options)


# =====================================================================
#  公式后处理挂载点（detection-service-group）
# =====================================================================
def _formula_postprocess(result: dict, pdf_path: Path, images_dir: Optional[Path],
                         opts: dict, warnings: list[str]) -> dict:
    """PyMuPDF 解析**之后**的公式框和 Figure 图片增强钩子。

    契约（改这里之前先读）：

      * **PyMuPDF 是主干**：输入是 `_parse_with_pymupdf()` 的文字版式 + 本次的
        `opts`；拿不到服务组仍保留文字版式，Figure 图片不回退到本地提取；
      * **只做增量增强**：块框 / 阅读顺序 / 行框仍以 PyMuPDF 为准，公式以**覆盖层**
        渲染；Figure 使用 detection-service-group 的裁图，不使用 PyMuPDF 图片提取；
      * 现成可用的锚点：公式被擦掉的地方在正文里就是**一段等长空格**
        (`pages[]._formula_slots`，内部字段，见 `backend_pymupdf`)；
      * 增强内容记进 `result["parser_info"]`，失败原因追加到 `warnings`
        (会随 doc.json 一起落盘，排障时看得到)。

    服务组的 `recognized_detections[*].formula` 写入 `pages[].formula_boxes`；
    Figure 检测框的 `crop` 按 `bbox_pdf` 落盘并写入 `pages[].images`，由阅读器图片层显示。

    没配上/没拉取到的页保持等长空白不动 —— 那种页 `formula_action` 仍是 `space`。

    触发条件：`backend` 为 `auto`/`detection_service_group`（`pymupdf` = 明确不要外部服务；
    `surya` 走自己的整页 OCR 分支，不经过这里）且 `detection_service_group_enable` 为真。

    统计键(`parser_info`)：公式框 `formula_boxes` / `formula_box_pages` /
    `formula_box_failed` / `formula_box_ms`，图片 `figure_images` /
    `figure_image_pages` / `figure_image_failed`。
    """
    backend = str(opts.get("backend") or "").lower()
    if backend not in ("auto", "detection_service_group"):
        return result                     # pymupdf = 不要外部服务；surya = 待重写
    if not opts.get("detection_service_group_enable"):
        return result
    return apply_formula_boxes(result, pdf_path, opts, warnings,
                               images_dir=images_dir)


def pop_private(result: dict) -> dict:
    """摘掉只给内部用的字段(页级 `_` 前缀键) —— **不能进 doc.json**。

    当前有 `_formula_slots`(公式空位；detection-service-group 后处理用它挑「哪些页要送
    检测」，见 `backend_pymupdf`)—— 体积不小，且前端与接口层都不该看见；
    必须在公式后处理**之后**调用(后处理正是它的使用者)。
    """
    for pg in result.get("pages") or []:
        for key in [k for k in pg if k.startswith("_")]:
            pg.pop(key, None)
    return result


# =====================================================================
#  入口
# =====================================================================
def parse_pdf(pdf_path: Path, images_dir: Optional[Path] = None, *,
              backend: Optional[str] = None, options: Optional[dict] = None,
              config_file: Optional[Path] = None) -> dict:
    """解析 PDF → 版式数据(即 doc.json 的 pages 结构)。

    `backend` / `options` / `config_file` 三个参数为**兼容旧调用方保留**
    (`converter.reparse(..., backend=…)`、`tools/selftest.py --backend …`)：

      * `surya`：整页 OCR 走 Surya 2 —— 文字、公式（`<math>`→LaTeX）、插图都来自
        Surya，**不使用 PyMuPDF 版式**；失败直接抛 `SuryaUnavailable`
        （不会悄悄回退，免得看起来像解析成功了）；
      * `pymupdf`：纯本地 PyMuPDF 版式，不请求 detection-service-group，
        因此不会提取或显示 Figure 图片；
      * `auto` / `detection_service_group`：PyMuPDF 版式 + detection-service-group
        补公式框和 Figure 图片（见 `_formula_postprocess`）。

    返回值：
      * parser      `"pymupdf"` / `"surya"`
      * parser_info 解析统计(公式→空格的条数 / Surya 块与图片条数等)
      * toc         PDF 自带书签目录 `[{level,title,page}]`（可能为空）
      * warnings    说明性信息(列表，可能为空)
    """
    opts, warnings = load_options(config_file=config_file, overrides=options,
                                  backend=backend)
    if images_dir is not None:
        # `_parse_with_pymupdf` / `_parse_with_surya` 里都要 images_dir.mkdir()，
        # str 会 AttributeError
        images_dir = Path(images_dir)

    if opts["backend"] == "surya":
        # Surya 是**独立解析链**：文字 / 公式 / 插图全部用它自己的数据，
        # 前端按 `parser === "surya"` 走块级渲染（见 backend_surya 模块文档）。
        warnings.append("解析后端 'surya'：文字版式、公式（<math>→LaTeX）与插图都由 "
                        "Surya 2 整页 OCR 提供；不使用 PyMuPDF 版式，"
                        "也不跑 detection-service-group")
        result = _parse_with_surya(pdf_path, images_dir, opts)
        result["warnings"] = warnings
        return result

    if opts["backend"] == "pymupdf":
        warnings.append("解析后端 'pymupdf' 不调用 detection-service-group；"
                        "不再使用 PyMuPDF 图片提取，Figure 图片不会显示")
    elif opts["backend"] == "detection_service_group":
        # 版式还是 PyMuPDF；detection_service_group 只决定后面要不要做公式框增强
        # (见 `_formula_postprocess`)。
        warnings.append("解析后端 'detection_service_group'：文字版式由 PyMuPDF 本地抽取，"
                        "公式框与 Figure 图片由 detection-service-group 识别")
    if (opts["backend"] in ("auto", "detection_service_group")
            and not opts.get("detection_service_group_enable")):
        warnings.append("detection-service-group 检测已关闭；Figure 图片与公式增强不会显示")

    result = _parse_with_pymupdf(pdf_path, images_dir, opts)
    result = _formula_postprocess(result, pdf_path, images_dir, opts, warnings)
    # 公式最后是怎么落地的：
    #   "boxes" = 页面上叠了公式框覆盖层（pages[].formula_boxes，前端按 bbox_norm
    #             在页面上渲染 KaTeX，见 frontend/js/r2-math.js）；
    #   "space" = 公式字形已擦成等长空白（服务组不可用 / 一条都没返回时）。
    # "surya"（公式来自 Surya 的 <math>）走的是函数开头那条独立分支，不会到这里。
    # 旧的 "inline"（LaTeX 内联进正文文本）随 PaddleOCR 配对链路一起移除了，
    # 但旧 doc.json 里的这个值前端仍认（兼容）。
    if any(pg.get("formula_boxes") for pg in result.get("pages") or []):
        result["formula_action"] = "boxes"
    pop_private(result)          # `_formula_slots` 只给公式后处理用
    result["warnings"] = warnings
    return result
