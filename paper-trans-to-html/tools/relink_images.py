"""把某篇论文的插图，用原始 PDF 重新解析并“挂回”同一个 doc_id

用途：插图以前是磁盘文件（`data/docs/<id>/images/`），可能因为清理目录 / 换机器 /
容器换卷而丢失，而版式数据（`documents`）、笔记、译文都还在库里 —— 这时不用
重新上传（那样会生成新论文、笔记译文全丢），直接用这个工具把插图补回去。

原理：解析器产出的插图文件名是**确定性**的（`p{页码:03d}_i{序号}.png`），
和 `doc.json` 里记录的文件名一一对应，所以按同一个 doc_id 重新解析就能对上。

用法：
    python tools/relink_images.py --list                 # 看哪些论文缺插图
    python tools/relink_images.py <doc_id> 原PDF路径      # 补图（只写库里缺的、且是这篇论文需要的）
    python tools/relink_images.py <doc_id> 原PDF路径 --dry-run   # 只看会补什么，不写库
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from backend import pdf_parser                     # noqa: E402
from backend.storage import build as build_storage  # noqa: E402

DATA_DIR = BASE_DIR / "data"
CONFIG_FILE = BASE_DIR / "config.json"


def _doc_images(doc: dict) -> list[str]:
    """版式数据里记录的全部插图文件名（去重保序）。"""
    names: list[str] = []
    for pg in doc.get("pages") or []:
        for im in pg.get("images") or []:
            f = im.get("file")
            if f and f not in names:
                names.append(f)
    return names


def _list_docs(store) -> int:
    rows = store.list_records()
    if not rows:
        print("库里还没有论文。")
        return 0
    print(f"共 {len(rows)} 篇：\n")
    print(f"{'doc_id':34s} {'图(有/共)':>10s}  {'标题':44s} 归属")
    for r in rows:
        doc_id = r.get("id") or r.get("_id")
        try:
            doc = store.read_doc(doc_id)
            names = _doc_images(doc)
        except FileNotFoundError:
            print(f"{doc_id:34s} {'版式数据缺失':>10s}")
            continue
        have = sum(1 for n in names if store.get_image(doc_id, n))
        flag = "" if have == len(names) else "   ← 缺图，可用原 PDF 补"
        print(f"{doc_id:34s} {f'{have}/{len(names)}':>10s}  "
              f"{str(r.get('title'))[:44]:44s} {r.get('owner') or '-'}{flag}")
    print("\n补图：python tools/relink_images.py <doc_id> <原PDF路径>")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="用原 PDF 给已存在的论文补回插图")
    ap.add_argument("doc_id", nargs="?", help="论文 id（--list 时不用给）")
    ap.add_argument("pdf", nargs="?", help="原始 PDF 路径")
    ap.add_argument("--list", action="store_true", help="列出所有论文及插图完整情况")
    ap.add_argument("--dry-run", action="store_true", help="只解析对比，不写入存储")
    args = ap.parse_args()

    store = build_storage(DATA_DIR, CONFIG_FILE)[0]
    backend = getattr(store, "backend_name", "?")

    if args.list:
        print(f"存储后端：{backend}\n")
        return _list_docs(store)

    if not args.doc_id or not args.pdf:
        ap.print_help()
        return 2

    pdf = Path(args.pdf).expanduser()
    if not pdf.exists():
        print(f"❌ 找不到 PDF：{pdf}")
        return 1

    try:
        doc = store.read_doc(args.doc_id)
    except FileNotFoundError:
        print(f"❌ 库里没有这篇论文的版式数据：{args.doc_id}")
        print("   （如果它本来就没上传成功，直接重新上传 PDF 更简单）")
        return 1

    need = _doc_images(doc)
    have_before = [n for n in need if store.get_image(args.doc_id, n)]
    print(f"存储后端  ：{backend}")
    print(f"论文      ：{doc.get('title')}  （{doc.get('num_pages')} 页）")
    print(f"记录插图  ：{len(need)} 张，其中已存在 {len(have_before)} 张")
    if not need:
        print("这篇论文的版式数据里没有插图，无需处理。")
        return 0

    parsed_pages = doc.get("num_pages")
    with tempfile.TemporaryDirectory(prefix="relink-img-") as staging:
        tmp = Path(staging)
        print(f"重新解析  ：{pdf.name} → 临时目录")
        res = pdf_parser.parse_pdf(pdf, tmp)
        produced = sorted(p.name for p in tmp.iterdir() if p.is_file())
        print(f"解析产出  ：{len(produced)} 张图（后端 {res.get('parser')}，{res.get('num_pages')} 页）")
        if parsed_pages and res.get("num_pages") != parsed_pages:
            print(f"⚠️ 页数不一致：这篇论文是 {parsed_pages} 页，PDF 解析出 "
                  f"{res.get('num_pages')} 页 —— 确认是同一份 PDF、且解析设置一致。")

        hit = [n for n in need if n in produced]
        missing = [n for n in need if n not in produced]
        extra = [n for n in produced if n not in need]

        if args.dry_run:
            print(f"\n[dry-run] 会写入 {len(hit)} 张：{', '.join(hit[:8])}"
                  + (" …" if len(hit) > 8 else ""))
            if missing:
                print(f"[dry-run] 对不上、补不了 {len(missing)} 张：{', '.join(missing[:8])}")
            if extra:
                print(f"[dry-run] PDF 多出 {len(extra)} 张（论文里没引用，忽略）")
            return 0

        for name in hit:
            store.put_image(args.doc_id, name, (tmp / name).read_bytes())
        print(f"\n✅ 已写入 {len(hit)} 张插图（doc_id 不变，笔记/译文/进度都保留）")

    after = [n for n in need if store.get_image(args.doc_id, n)]
    print(f"核对      ：现在这篇论文 {len(after)}/{len(need)} 张图可读")
    if missing:
        print(f"⚠️ 仍有 {len(missing)} 张对不上（文件名/解析设置不一致）：{', '.join(missing[:8])}"
              + (" …" if len(missing) > 8 else ""))
    if extra:
        print(f"提示      ：PDF 多解析出 {len(extra)} 张，论文里没有引用，已忽略")
    return 0 if not missing else 3


if __name__ == "__main__":
    raise SystemExit(main())
