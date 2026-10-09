"""【已清空】OCR 后处理 · PaddleOCR-VL（旧的「配对 + 内联」公式落地方式）。

原实现：对**有公式空位的页**渲染 PNG 送 PaddleOCR-VL（`collect_ocr`），把整页
Markdown 里的 LaTeX 按前后文配到空格位上（`formula_match`），再**内联进正文文本**
（`apply_formula_ocr` → `backend_pymupdf._inline_formula_latex`：行内 `\\(latex\\)`、
整行 `$$latex$$`，前端 `renderInlineMath` 出 KaTeX）。

**为什么清空**：公式落地方式改成「**覆盖层**」—— detection-service-group
（`formula_table_service_group` 服务组）直接返回公式框（`bbox_norm`）与 LaTeX，
写进 `pages[].formula_boxes`，前端按 `bbox_norm × 页面显示宽高` 在页面上覆盖
KaTeX（见 `backend_detection_service_group.py` 与 `frontend/js/r2-math.js`）。
配对、内联与前后文窗口全部不再需要。

旧码归档：`history_versions/paddle-ocr-cleared-20261003/backend/pdf_parser/backend_paddle.py`。
"""
__all__ = []
