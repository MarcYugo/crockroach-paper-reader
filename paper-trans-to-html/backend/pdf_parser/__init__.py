"""模块一 · 文档解析

把 PDF 解析为与原文版式一致的布局数据：
  - 文字块 texts：绝对坐标 bbox；PyMuPDF 分支带多行(lines：几何 + **行文本**)，
    Surya 分支是块级数据（一行 `runs` + 块内流式排版，见下）；
  - 图片   images：绝对坐标 bbox + 落盘文件名。

⚠️ **PyMuPDF 分支：一行只有一个 run**（`text_layer._merge_line_runs`）：行内不再有
碎片 span，一块文字就是「块 `text` + 每行几何 + 每行文本」。字距造成的粘连由
`text_layer._insert_gap_spaces` 在解析期用 span 几何补空格解决；代价是行内
粗体/斜体/上标与颜色退化成行级样式（真正的上标会与正文同号显示）。
公式字形被擦成空格，公式内容由 **detection-service-group** 的「公式框 + LaTeX」覆盖层
在页面上还原（见下）。

⚠️ **一个块 = 恰好一句**（`sentences.cut_blocks_by_sentence`）：一段话被公式空位切成
好几块、两块的**行在同一视觉行上左右相接**（实测 `p1b45` + `p1b47`），前端又是**按块**
切句的 —— 不重切的话点一下选中的「句子」就是半句碎片。现在解析侧把块按句子重切：
行链（同视觉行横向相接，分栏走廊除外）→ 续句（同栏 + 左边距 + 字面证据）→ 切句
（口径与前端 `reader2.splitSentences` 一致）→ 一个句子一块（端点落在行内部时按
**逐字符几何** `_CHAR_X` 把行拆成两段，几何精确到字符边界）。
整行都是公式字形的行作为**乘客**挂在邻近句子单元上（几何留给 `_absorb_orphan_gaps`）；
没被拆/合的块**沿用原 id**，其余用 `p{页}s{序号}`。
统计：`sentence_units` / `sentence_merged` / `sentence_line_splits` /
`sentence_tail_units`（末尾没有句末标点的单元，即「合并不了的碎片」）。
⚠️ 重切**只改分组、不改字符**，所以必须排在 `_blank_math_spans` **之前**
（擦除/slots 都按当时的块结构算偏移，排在前面就不需要任何偏移重映射）。

⚠️ **文字版式后端有两个**：

  * **PyMuPDF 本地抽取**（`backend_pymupdf.py`，`backend` = `auto`/`detection_service_group`/
    `pymupdf`）：Figure 图片与公式增强由 **detection-service-group**
    （`formula_table_service_group` 服务组）提供(见 `backend_detection_service_group.py`，
    客户端 `../detection_service_group.py`)；新流程是「**PyMuPDF 提取文字版式，
    服务组检测公式并提供 Figure 裁图**」，不使用 PyMuPDF 的图片提取结果，
    也不是旧的「OCR 整页解析 → 配对 → 内联」。
  * **Surya 2 整页 OCR**（`backend_surya.py` + 客户端 `../surya_parser.py`，
    `backend` = `surya`）：文字 / 公式（`<math>`→LaTeX）/ 插图裁图 / 阅读顺序全部来自
    Surya 的版面 block，**不使用 PyMuPDF 版式**；前端按 `doc.parser === "surya"` 走
    独立的块级流式渲染（`frontend/js/reader2.js::buildSuryaItem`）。Surya 不可用时
    直接报错，不回退 PyMuPDF。

PyMuPDF 后端做什么：
  * 文字层：PyMuPDF 的精确行框/字体样式 → texts(blocks → lines → runs)；
  * 图片：服务组检测 `Figure` 类并回传框内裁图，后端按 PDF 坐标落盘到
    `pages[].images`，前端沿用现有图片接口展示。检测裁图缺失时不回退 PyMuPDF；
  * 去重影(`_suppress_ghost_text`)：Figure 裁图里已经「烘焙」了文字时，不再把同一批
    文字按文字层叠一遍 —— 否则会和图内字形错开成上下两层「重影」。
    可用 `drop_text_in_images` 关掉；
  * 公式：PyMuPDF 没有数学语义、也拿不到 LaTeX，于是按**字体名**判出公式字形后把它们
    换成**空格**(`math_font._blank_math_spans`)，块结构与行框一个不动。每个空位的
    **原宽由行框宽度反推**(`math_font._line_gap_widths`：行框宽 − 保留正文宽)，空格
    **个数也由那个原宽反推**(`round(宽 / 0.25em)`) —— 不再等于公式的字形数；原宽随
    行级 `gaps` 下发，前端只把它撑回去，公式后面的正文因此落回原 x。正文不再被公式
    碎片污染；被擦空的行/块由 `backend_pymupdf._blank_math_and_prune`
    一并清掉。擦除**只把字形换成空格**，不记框；屏幕上那块地方先显示为空白，
    再由下面的公式框覆盖层画上真公式。
    `parser_info` 里 `formula_spans` / `formula_chars` / `formula_lines` 是被擦掉的
    run 数、被擦掉的字形数与行数（**实际写下的空格数看 `formula_spaces`** —— 它由
    宽度反推，与字形数不再相等），`formula_weak_pages` 是用了「弱数学字体」判据
    的页数(整篇 CM 系时自动关掉该判据，否则整页正文会被误擦)。

    公式**内容**是另一回事：`entry._formula_postprocess` 会（`backend` 为
    auto/detection_service_group 且 `detection_service_group_enable` 为真时）调
    `backend_detection_service_group.apply_formula_boxes` —— 全页渲染 PNG
    送 detection-service-group（yolov13 检测 → pp-formulanet-plus-l 识别，见服务组仓库
    `formula_table_service_group/`），Figure 裁图写入 `pages[].images`，公式框只对**有空格位
    的页**写进 `pages[].formula_boxes`：

        [{"bbox_norm": [x中心, y中心, 宽, 高],   # 归一化(YOLO 口径)
          "latex": "\\frac{…}", "score": 0.93, "class_name": "…"}, …]

    前端(`frontend/js/r2-math.js`)按 `bbox_norm × 当前页面显示宽高` 在页面上覆盖
    KaTeX —— **位置来自公式框、内容来自服务组**，本地空格只负责排版宽度。
    配上几条看 `parser_info.formula_boxes`；`formula_action` 随之变 `"boxes"`。

    ⚠️ 本流程**不改文本、不改版式**（只是增量字段 + 前端覆盖层），所以它排在
    所有按字符数寻址的东西（空格位、块 canonical、译文缓存键）之后。

    ⚠️ 内部字段：`pages[]._formula_slots`（空格位，只用来挑哪些页要送检测）
    只在 `parse_pdf` 内部存活 —— `entry.pop_private` 在返回前摘掉，**不进 doc.json**。

页面带 rotation 时先归一化为 0，保证文字与服务组检测框坐标系一致。

对外接口：`parse_pdf(...)`(见 `entry.py`)与 `status(...)`(见 `options.py`)。
`parse_pdf` 的签名与返回结构**保持不变** —— `converter.py`、`app.py` 与 `tools/*`
都不用改：`backend` / `options` / `config_file` 参数仍在；`pymupdf`/`auto`/
`detection_service_group` 走 PyMuPDF 版式（`detection_service_group` 决定是否补公式框
和 Figure 图片），`surya` 走独立整页 OCR 分支（文字/公式/插图都用 Surya 数据）。
顶层 `formula_action` 与页级 `formula_boxes` 是**增量**字段(旧调用方忽略即可)；
都会被 `converter` 一起写进 doc.json → `/reparse`。`formula_action` 取值：
`"boxes"`（服务组公式框覆盖层）/ `"space"`（公式擦成空格）/ `"surya"`（Surya 的
`<math>` LaTeX，前端行内覆盖层 + 独立公式块）/ 旧数据里的 `"inline"`。

配置契约：`config.json` 的 `parser` 段、`/api/settings/parser` 与网页
「⚙ 设置 → 存储 / OCR」面板用的是 `detection_service_group_*` / `surya_*` 那些键；
`options.status()` 返回 `detection_service_group_ready` /
`detection_service_group_enabled` / `detection_service_group_url` /
`detection_service_group_message` / `surya_*` —— `detection_service_group_ready` 与
`surya_ready` 都是**真探测**（`GET /health` / `GET /models`；`surya_ready` 还要求
本机 surya-ocr 客户端依赖就绪，细分字段 `surya_service_ready` / `surya_client_ready`）。
旧的 `paddle_*` 配置项**不再被读取**（键与 compose 里的 `PADDLE_OCR_*` 环境变量
已保留，仅作记录/回退参考）；旧配置里的 `backend="paddle"`
会被映射成 `auto` 并记一条 warning；旧名 `ft_group_*`/`FT_GROUP_*`/`ftgroup`
同样不再读取，但会被检测到并提醒（后端取值 `ftgroup` 会就地映射成新名）。

# ---------------------------------------------------------------------
#  本包的模块划分(从原来的单文件 pdf_parser.py 拆出)：
#    options.py          解析配置 / 环境变量覆盖 / status()
#    text_layer.py       文字层抽取（相邻 run 补间距 + 每行合并成单个 run）
#    sentences.py        按句子重切文字块（每个块 = 恰好一句）
#    images.py           插图检测区域去重影 + 挖洞
#    math_text.py        公式文本工具
#    math_font.py        公式定位·底层(字体名判据 / 字形→等长空格 / 碎片分组)
#    backend_pymupdf.py  版式后端 · PyMuPDF（公式字形→等长空格）
#    backend_detection_service_group.py  图片/公式增强 · detection-service-group
#    backend_surya.py    版式后端 · Surya 2 整页 OCR（块级数据，前端独立渲染）
#    entry.py            入口 parse_pdf(+ 公式后处理挂载点 / pop_private)
#  下面仍把内部名字再导出一次，保证 `pdf_parser.<原名字>` 照旧可用
#  (新代码建议直接从对应子模块 import)。
# ---------------------------------------------------------------------
"""

from .sentences import cut_blocks_by_sentence, split_sentences
from .text_layer import (_CHAR_X, _GAP_SPACE_MIN, _MONO, _SANS, _SERIF, _family,
                         _insert_gap_spaces, _is_super, _join_lines,
                         _line_baseline, _merge_line_runs, _parse_text_blocks,
                         _row_top, _span_char_x, _span_text, _text_line_rects,
                         _union_line_rect)
from .images import (_FIG_GAP, _FIG_MIN_H, _FIG_MIN_PARTS, _FIG_MIN_W, _FIG_PAD,
                     _IMG_OK, _carve_images, _cluster_rects, _covered_ratio,
                     _extract_raster, _extract_vector_figures, _image_rect,
                     _invisible_text_rects, _merge_rects, _overlap_area,
                     _render_region_png, _suppress_ghost_text)
from .options import (CONFIG_FILE, DEFAULT_IMAGE_LABELS, DEFAULT_OPTIONS,
                      PARSER_MODES, _ENV_KEYS, _to_bool, load_options, status)
from .backend_pymupdf import (_blank_math_and_prune, _local_page,
                              _parse_with_pymupdf, _pdf_outline,
                              _strip_font_keys)
from .backend_detection_service_group import apply_formula_boxes
from .backend_surya import _parse_with_surya
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
from .math_font import (_CM_TEXT_RE, _LINE_MATH_PURE, _MATH_FONT_STRONG,
                        _MATH_FONT_WEAK, _MERGE_GAP_CHARS, _SYMBOL_FONT,
                        _accumulate_stats, _blank_math_level, _blank_math_spans,
                        _cm_text_math_on, _font_base, _font_math_index,
                        _font_math_level, _font_math_line, _group_math_pieces,
                        _has_symbol_font, _line_math_spans, _line_text,
                        _mask_out, _retighten, _weak_math_enabled, _widen_cut,
                        _widen_with_cand)
from .entry import parse_pdf, pop_private
