"""【已清空】公式配对自检工具（旧的「空格位 × OCR LaTeX」配对流程）。

配对链路已整体移除（见 `backend/pdf_parser/formula_match.py` 与
`backend/pdf_parser/backend_paddle.py` 的说明）；公式改由 detection-service-group 识别
公式框 + LaTeX，落地方式见 `backend/pdf_parser/backend_detection_service_group.py`。

旧码归档：`history_versions/paddle-ocr-cleared-20261003/tools/check_formula_match.py`。
保留本文件是为了让旧文档/脚本里的路径有一个明确去处；运行只会打印这条说明。
"""
from __future__ import annotations

__all__ = []


def main() -> None:
    print("公式配对链路已移除（改用 detection-service-group 的公式框 + LaTeX）。")
    print("旧工具归档于 history_versions/paddle-ocr-cleared-20261003/tools/check_formula_match.py")


if __name__ == "__main__":
    main()
