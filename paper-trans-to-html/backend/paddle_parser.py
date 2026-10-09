"""【已清空】PaddleOCR-VL 推理服务客户端（旧的「整页 OCR → 公式配对」链路）。

原实现：把 PaddleOCR-VL-1.6 当外部 HTTP 服务（vLLM 的 OpenAI 兼容
`/v1/chat/completions`）调用，把整页 Markdown 里的公式与本地空格位做前后文配对
（配对于 `pdf_parser/formula_match.py`，编排于 `pdf_parser/backend_paddle.py`）。

**为什么清空**：公式内容改由 **detection-service-group**
（`formula_table_service_group` 服务组）以「公式框 + LaTeX」的形式直接给出 ——
不再需要「整页 OCR + 文本配对」，也就不需要这个客户端。新客户端见
`detection_service_group.py`（页面渲染 `render_page_png` 也搬去了那里）。

旧码归档：`history_versions/paddle-ocr-cleared-20261003/backend/paddle_parser.py`。
"""
__all__ = []
