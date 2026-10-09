#!/usr/bin/env python3
"""校验本地模型目录是否符合 PaddleOCR 推理要求。

用法：
    python3 scripts/check_model.py ./models/PP-FormulaNet_plus-L

PP-FormulaNet_plus-L 是静态图（Paddle Inference）模型，目录里必须有：
    inference.json / inference.pdiparams / inference.yml
config.json 与 README.md 是模型仓库附带文件，不影响推理。
"""

from __future__ import annotations

import os
import sys

REQUIRED = ["inference.json", "inference.pdiparams", "inference.yml"]
OPTIONAL = ["config.json", "README.md"]


def main() -> int:
    if len(sys.argv) > 1:
        model_dir = sys.argv[1]
    else:
        model_dir = os.getenv("MODEL_DIR", "/models/PP-FormulaNet_plus-L")

    print(f"模型目录: {model_dir}")
    if not os.path.isdir(model_dir):
        print("[错误] 目录不存在", file=sys.stderr)
        return 1

    names = set(os.listdir(model_dir))
    print("目录内容:")
    for name in sorted(names):
        path = os.path.join(model_dir, name)
        if os.path.isfile(path):
            print(f"  - {name:24s} {os.path.getsize(path) / 1024 / 1024:10.2f} MB")
        else:
            print(f"  - {name}/")

    missing = [name for name in REQUIRED if name not in names]
    if missing:
        print(f"[错误] 缺少必需文件: {missing}", file=sys.stderr)
        return 1

    print(f"[OK] 必需文件齐全: {REQUIRED}")
    for name in OPTIONAL:
        if name not in names:
            print(f"[提示] 可选文件 {name} 不存在（不影响推理）")

    for name in REQUIRED:
        path = os.path.join(model_dir, name)
        if not os.access(path, os.R_OK):
            print(f"[错误] 无读取权限: {path}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
