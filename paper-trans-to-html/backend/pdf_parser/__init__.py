"""模块一 · 文档解析

把 PDF 解析为与原文版式一致的布局数据：
  - 文字块 texts：绝对坐标 bbox、多行(lines)、行内样式 run(spans)；
  - 图片   images：绝对坐标 bbox + 落盘文件名。

三种解析后端(见 `parse_pdf`)，对外输出**同一套** pages 结构，前端无需改动：

  1. paddle(推荐，GPU)：接 PaddleOCR-VL 推理服务(OpenAI 兼容的
     /v1/chat/completions，见 `paddle_parser.py` 与
     `../paddle_ocr_doc_parse_service`)。服务端只吃图片，所以这里先用 PyMuPDF
     把每页渲染成 PNG 再逐页送进模型，拿回一页的 Markdown/HTML 文本：
       * 有**可见文字层**的电子版 PDF 仍以 PyMuPDF 的精确行框/字体样式为准
         (OCR 文本带 LaTeX 转义等噪声)，OCR 只用来补插图；
       * 纯扫描件用 OCR 文本按页边距合成行框(近似版式)；
       * 若 OCR 输出带「标签 [x0,y0,x1,y1]文本」这种坐标格式(见
         `parse_ocr_blocks`)，则按标签与坐标还原**真实版式**；否则整页纯文本。
  2. surya：接 Surya 2 推理服务(OpenAI 兼容接口，见 `surya_parser.py`
     与 `doc_parse_service/`)做版面分析 + 整页 OCR：
       * 版面 block(label/html)决定分块、阅读顺序；公式块标签是 `Equation`，
         其 `<math>` 里是 **LaTeX 源码**(不是 MathML)，解析时抽成 `latex` 字段
         (见 `surya_parser.latex_from_html`)；表格 <table> 的 html 原样保留；
       * 图片类 block(Picture/Figure/Diagram/ChemicalBlock)按 polygon 裁剪成
         PNG —— 它们一定 skipped=True/html=""，只能按 label 判断，不能用
         skipped / 空 html 过滤，否则会把要提取的图全部丢掉；
       * 有文本层的 PDF 仍用 PyMuPDF 的精确行框/字体样式补齐 lines/runs，
         纯扫描件则用 OCR 文本按行高近似还原行框。
  3. pymupdf(兜底)：本地 PyMuPDF 抽取。图片抽取做三层保障，尽量“不遗漏”：
       ① 内嵌点阵图：page.get_image_info(xrefs=True) + doc.extract_image(xref)，
          并按 bbox 去重(同一位置 RGB 与软遮罩灰图并存时取更清晰的图)；
       ② 矢量绘制图表：page.get_drawings() 把较大的填充/描边形状聚类成
          “图形区域”，栅格化后落盘(避免纯矢量图在 HTML 中完全丢失)；
       ③ 过滤噪声：跳过整页背景图、过小的贴图、与文字高度重叠的区域。

三种后端最后都会做一次「去重影」：插图里已经烘焙了文字(Surya 裁剪图、
矢量图栅格化图，或扫描件/图片自带的 OCR 隐形文字层)时，不再把同一批文字按
文字层叠一遍 —— 否则会和图内字形错开成上下两层“重影”。详见
`_suppress_ghost_text`；可用 drop_text_in_images 关掉。

公式渲染：三个后端产出的公式**统一成块的 `latex` 字段** —— Paddle 从 OCR
文本里抽(`extract_latex`/`looks_like_latex`)，Surya 从 `<math>` 标签里抽
(`latex_from_html`)，前端用 MathJax 排版；PyMuPDF 没有数学语义，保持文字。

公式的**位置**有两条互补的判据(见 `_inject_ocr_math`)：
  ① **字体名**：电子版 PDF 的数学排版用独立的数学字体族(CM/AMS/Symbol)，与正文
     截然分开 —— 据此可以**不依赖 OCR** 先判出「哪些字形是公式」，粒度是 span 级
     (`_font_math_index`)；
  ② **OCR 文字锚点**：用公式前后的文字在本地文本里定位(`_find_anchor_pair`)。
两者结合：字体候选区负责定位、OCR 退化为「验证 + 补漏」；任一判据不可用(扫描件
没有字体名 / 数学与正文同体)时退回纯锚点法。

公式有**三种形态**，但落地只有**一套规则**(见「公式块」那一节)：
  * 一行之内 → 把那段字形换成 `\(latex\)`(行内替换)；
  * 跨行/跨块 → 把那些字形行从原块里**摘掉**，另合成一个 `formula` 块；
  * 独立成段 → 同上(只是覆盖层按 display 模式渲染)。
三种形态的**原字形一律从数据里删掉**，公式块只带几何 + `latex` —— 于是「公式覆盖掉
原来的字形」是结构上的事实，不依赖前端 CSS 隐藏，KaTeX 没渲染成也不会露出乱码碎片；
`block.text` / `page.text` / 句子切分 / 翻译 / 复制也都不会再被公式字形污染。

定位质量在 parser_info 里可查：`math_inline` / `math_inline_split` / `math_display`
是落地条数；`math_font_lines` / `math_font_used` 是字体判据的覆盖情况；
`math_leftover_lines` 是**落地之后仍有数学字形的行数**(含表格里的数字与斜体标识符，
属正常保留)；`math_anchor_missed` 是「OCR 给了 LaTeX 但没定位上」的条数。

⚠️ `math_anchor_missed` **不是**待修的 bug 清单：实测它主要由 OCR 侧的问题构成 ——
OCR 在一段普通文字上误报出公式(`O(npT)`/`top-k`/`n < T`，本地根本没有这段)，或把
一整段念串导致锚点在本地不存在。"再往下压"要动的是**定位之外**的东西(单独重识别 /
换 OCR 提示词)，定位逻辑这条路上已经到顶了。

后端选择：parse_pdf(backend=...) > 环境变量 PDF_PARSER_BACKEND >
config.json 的 parser.backend > 默认 auto。auto 按
**PaddleOCR-VL > Surya > PyMuPDF** 的优先级选：先看 PaddleOCR-VL 服务是否就绪
(`/health` 返回 200)；不可用再看 Surya —— 要**同时**满足① 服务连得上
(`/v1/models` 通)、② 本机装了客户端依赖 surya-ocr + Pillow，缺依赖时不必等失败，
直接跳过；都不可用才回退 PyMuPDF。显式指定 `paddle` / `surya` 时也按同一顺序
往下回退(可用 `fallback=false` 关闭回退，此时失败直接抛错)。每次回退都会记在
结果的 warnings 里，并在 `/api/config` 的 parser 状态中体现实际生效的后端。
PaddleOCR-VL 客户端只需要 PyMuPDF(渲染页面 + 读版式)，不依赖 surya-ocr / Pillow。

页面带 rotation 时先归一化为 0，保证文字/点阵图/矢量区域坐标系一致。
"""

# ---------------------------------------------------------------------
#  本包的模块划分(从原来的单文件 pdf_parser.py 拆出，代码逐行原样搬运、
#  行为完全不变；对外接口也不变，仍是 `from . import pdf_parser` /
#  `pdf_parser.parse_pdf(...)` / `pdf_parser.status(...)`)：
#    options.py          解析配置 / 后端可用性 / status()
#    text_layer.py       文字层抽取
#    images.py           插图抽取 + 去重影 + 挖洞
#    backend_pymupdf.py  后端一 PyMuPDF
#    backend_surya.py    后端二 Surya
#    ocr_blocks.py       OCR 文本 → 版式块
#    math_text.py        公式文本工具
#    math_font.py        公式定位·底层(字体名判据)
#    math_locate.py      公式定位(锚点 + 视觉行 + 按字体定位)
#    math_inline.py      行内公式落地
#    math_block.py       公式块落地 + 注入调度
#    backend_paddle.py   后端三 Paddle
#    entry.py            入口 parse_pdf
#  下面仍把全部内部名字再导出一次，保证 `pdf_parser.<任何原名字>` 照旧可用
#  (新代码建议直接从对应子模块 import)。
# ---------------------------------------------------------------------

from .text_layer import (_MONO, _SANS, _SERIF, _family, _is_super,
                         _join_lines, _line_baseline, _parse_text_blocks,
                         _text_line_rects, _union_line_rect)
from .images import (_IMG_OK, _carve_images, _cluster_rects, _covered_ratio,
                     _extract_raster, _extract_vector_figures, _image_rect,
                     _invisible_text_rects, _merge_rects, _overlap_area,
                     _render_region_png, _suppress_ghost_text)
from .options import (CONFIG_FILE, DEFAULT_OPTIONS, _ENV_KEYS, _to_bool,
                      client_ready, load_options, status)
from .backend_pymupdf import (_local_page, _parse_with_pymupdf, _pdf_outline,
                              _strip_font_keys)
from .backend_surya import (_merge_surya_page, _page_lines,
                            _parse_with_surya, _synth_lines)
from .ocr_blocks import (_LATEX_CMD_RE, _OCR_BLOCK_RE, _OCR_IMAGE_LABELS,
                         _OCR_KIND, _OCR_MARGIN, _OCR_MATH_LABELS,
                         _OCR_MATH_SEG, _STRONG_MATH_RE,
                         _drop_lines_in_rects, _ocr_block_lines,
                         _synth_ocr_blocks, _text_layer_invisible,
                         extract_latex, looks_like_latex, parse_ocr_blocks)
from .math_text import (_ANCHOR_LENS, _ANCHOR_MIN_TOTAL, _ANCHOR_RANK_GAP,
                        _ANCHOR_RANK_MIN, _ANCHOR_SPAN_MAX, _DISPLAY_MATH_RE,
                        _GLYPH_CLS, _GLYPH_RE, _LIG_EXPAND,
                        _LINE_MATH_TOKEN_RE, _NON_GLYPH_PREFIX_RE,
                        _NORM_CHAR_RE, _SPAN_MAX_LINES, _STYLE_KEYS,
                        _all_pos, _anchor_of, _apply_replacements,
                        _bag_ratio, _body_run, _count_chars, _gap_ok,
                        _has_glyphs, _lines_text, _looks_math_body,
                        _math_glyphs, _math_ish, _meta_at, _norm_keep,
                        _norm_keep_idx, _ocr_math_lines, _page_index,
                        _plain_math, _same_style, _split_line_math, _tighten)
from .math_font import (_LINE_MATH_PURE, _MATH_FONT_STRONG, _MATH_FONT_WEAK,
                        _SYMBOL_FONT, _font_base, _font_math_index,
                        _font_math_level, _font_math_line, _has_symbol_font,
                        _line_math_spans, _mask_out, _retighten,
                        _weak_math_enabled, _widen_cut, _widen_with_cand)
from .math_locate import (_ANCHOR_X_PAD, _COL_GAP, _ROW_EXTEND_PASS,
                          _ROW_MATH_RATIO, _ROW_OVERLAP, _display_job,
                          _display_job_font, _display_line_cands,
                          _extend_rows, _find_anchor_pair, _job_glyphs,
                          _job_rank, _math_rows_between, _row_of, _row_span,
                          _x_window)
from .math_inline import (_inline_job, _ocr_line_replacements)
from .math_block import (_TAG_LINE_RE, _accumulate_stats, _detach_lines,
                         _formula_block, _inject_ocr_math, _line_geom,
                         _materialize_inline_span, _prefix_width,
                         _retighten_block, _slice_runs)
from .backend_paddle import (_merge_paddle_page, _parse_with_paddle)
from .entry import parse_pdf
