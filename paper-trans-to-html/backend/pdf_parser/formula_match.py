"""【已清空】公式配对（空格位 × PaddleOCR-VL Markdown，前后文带权 LCS 对齐）。

原实现：`line_offsets` / `context_at` 给空格位取前后文窗口，`ocr_formulas` 从整页
Markdown 扫公式，`_align` 做带权 LCS 单调对齐，`match_slots` / `apply_to_page`
供 `backend_paddle` 调用。

**为什么清空**：新流程里公式的「框 + LaTeX」由 detection-service-group 直接给出
（见 `backend_detection_service_group.py`），本地不再做配对；`backend_pymupdf` 也不再
需要前后文窗口（空格位只剩「哪些页有公式」这一个用途）。

旧码归档：`history_versions/paddle-ocr-cleared-20261003/backend/pdf_parser/formula_match.py`。
"""
__all__ = []
