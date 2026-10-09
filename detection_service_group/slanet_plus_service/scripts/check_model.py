#!/usr/bin/env python3
"""校验本地模型目录是否符合 PaddleOCR 推理要求。

用法：
    python3 scripts/check_model.py ./models/SLANet_plus

SLANet_plus 是静态图（Paddle Inference）模型，目录里必须有：
    inference.json / inference.pdiparams / inference.yml
config.json 与 README.md 是模型仓库附带文件，不影响推理。

额外检查：inference.yml / config.json 里声明的 Global.model_name 必须与
服务使用的 MODEL_NAME（默认 SLANet_plus）一致，否则 PaddleOCR 会报
"Model name mismatch，please input the correct model dir."。
"""

from __future__ import annotations

import os
import re
import sys

MODEL_NAME = "SLANet_plus"
REQUIRED = ["inference.json", "inference.pdiparams", "inference.yml"]
OPTIONAL = ["config.json", "README.md"]
_NAME_PATTERN = re.compile(r'model_name"?\s*[:=]\s*"?([A-Za-z0-9_.\-]+)')


def _declared_model_name(model_dir: str) -> str | None:
    for name in ("inference.yml", "config.json"):
        path = os.path.join(model_dir, name)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as fh:
                match = _NAME_PATTERN.search(fh.read())
        except OSError:
            continue
        if match:
            return match.group(1)
    return None


def main() -> int:
    if len(sys.argv) > 1:
        model_dir = sys.argv[1]
    else:
        model_dir = os.getenv("MODEL_DIR", "/models/SLANet_plus")

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

    declared = _declared_model_name(model_dir)
    if declared is None:
        print("[提示] 未能从 inference.yml / config.json 读到 model_name")
    elif declared != MODEL_NAME:
        print(
            f"[错误] 模型声明的 model_name={declared!r} 与服务使用的 {MODEL_NAME!r} 不一致，"
            f"请把 MODEL_NAME 设为 {declared!r}",
            file=sys.stderr,
        )
        return 1
    else:
        print(f"[OK] model_name 一致: {declared}")

    for name in REQUIRED:
        path = os.path.join(model_dir, name)
        if not os.access(path, os.R_OK):
            print(f"[错误] 无读取权限: {path}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
