"""模块二 · 文档转换

把「文档解析」得到的版式数据保存为论文文档（doc + 插图），
并写入一条「论文阅读记录」元数据。

插图存哪里由存储后端决定（`store.images_in_db`）：
  - MongoDB（默认）：像素数据进 GridFS（`images` 桶）——服务重启/换容器/换机器都不丢；
  - 本地 JSON 回退：仍是 `data/docs/<id>/images/` 下的文件。
解析器（`pdf_parser`）只管把图写到目录里，所以它在两种情况下都只需一个目录：
入库时先落到临时目录，转完再整批进库。
"""
from __future__ import annotations

import datetime
import hashlib
import tempfile
from pathlib import Path

from . import pdf_parser
from .records import Store


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def convert(store: Store, doc_id: str, pdf_path: Path, source_name: str,
            owner: str = "") -> dict:
    """转换一份 PDF → 版式文档 + 阅读记录，返回记录元数据(meta)。

    `owner` = 归属账号：论文按账号隔离，列表/打开都会按它过滤。
    解析后端由 pdf_parser 决定(surya 推理服务优先，不可用时回退 PyMuPDF)，
    实际使用的后端与告警会记到 doc.json 里，便于排障。
    """
    images_dir = store.images_dir(doc_id)
    if getattr(store, "images_in_db", False):
        # 图片要进数据库：先解析到临时目录（解析器不用改），再整批入库。
        # 入库失败会抛出去，临时目录自动清理；外面还会 store.delete() 收尾。
        with tempfile.TemporaryDirectory(prefix="paper-img-") as staging:
            parsed = pdf_parser.parse_pdf(pdf_path, Path(staging))
            store.put_images(doc_id, staging)
    else:
        parsed = pdf_parser.parse_pdf(pdf_path, images_dir)

    if not parsed["pages"]:
        raise ValueError("未能从 PDF 解析出任何页面")

    if not any(pg.get("texts") or pg.get("images") for pg in parsed["pages"]):
        msg = "未能从 PDF 解析出任何文字或图片"
        for warn in parsed.get("warnings") or []:
            msg += f"；{warn}"
        raise ValueError(msg)

    title = (parsed.get("pdf_title") or "").strip() or Path(source_name).stem
    toc = parsed.get("toc") or []        # PDF 自带书签：有就直接当目录用（零成本、最准）
    doc = {
        "id": doc_id,
        "title": title,
        "source": source_name,
        "num_pages": parsed["num_pages"],
        "page_w": parsed["page_w"],
        "page_h": parsed["page_h"],
        "parser": parsed.get("parser"),
        "parser_info": parsed.get("parser_info"),
        "warnings": parsed.get("warnings") or [],
        "pages": parsed["pages"],
    }
    store.save_doc(doc_id, doc)

    meta = {
        "id": doc_id,
        "owner": owner,              # 归属账号：论文按账号隔离
        "title": title,
        "source": source_name,
        "num_pages": doc["num_pages"],
        "page_w": doc["page_w"],
        "parser": doc["parser"],
        "created_at": _now(),
        "last_read_at": None,
        "status": "unread",          # unread / reading / done
        "progress": 0.0,
        "last_page": 0,
        "highlights": 0,
        "notes": 0,
        # 侧栏目录：优先用 PDF 自带书签；没有时留空，由前端按需调 LLM 提取
        "toc": toc,
        "toc_source": "pdf" if toc else "",
    }
    store.ensure_record(meta)
    return meta


# =====================================================================
#  原地重解析版式（解析器升级后用，笔记 / 译文 / 进度都保留）
# =====================================================================
def _hash_text(text: str) -> str:
    """与 `app._hash_text` 同一口径 —— 译文缓存的键。"""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _block_texts(pages) -> dict[str, str]:
    out: dict[str, str] = {}
    for pg in pages or []:
        for t in pg.get("texts") or []:
            tid = t.get("id")
            if tid:
                out[tid] = t.get("text") or ""
    return out


def _latex_count(pages) -> int:
    return sum(1 for pg in pages or [] for t in pg.get("texts") or []
               if t.get("latex"))


def _remap_translations(store, doc_id: str, old_pages, new_pages) -> int:
    """把按「旧块文本」缓存的译文迁到新块文本上（块 id 不变、文本因补公式变了）。

    译文缓存的键是块文本的 sha1；解析升级后块文本会变（比如注入了 `\\(…\\)`），
    不迁移的话那些块就显得「没译过」。只迁「同 id、文本确实变了、新键还没有译文」
    的那些；迁不动的保持原样，最坏就是那几个块重新翻译一次。
    """
    try:
        cache = store.get_translations(doc_id)
    except Exception:
        return 0
    if not cache:
        return 0
    old = _block_texts(old_pages)
    updates: dict[str, str] = {}
    for tid, new_text in _block_texts(new_pages).items():
        if not new_text:
            continue
        new_key = _hash_text(new_text)
        if new_key in cache:
            continue
        old_text = old.get(tid)
        if old_text and old_text != new_text:
            zh = cache.get(_hash_text(old_text))
            if zh:
                updates[new_key] = zh
    if updates:
        store.set_translations(doc_id, updates)
    return len(updates)


def reparse(store: Store, doc_id: str, pdf_path: Path,
            backend: str | None = None) -> dict:
    """用原 PDF **原地**重解析一篇论文的版式数据，返回对比摘要。

    只换 `pages` / `parser` / `parser_info` / `warnings` 与插图；笔记、高亮、译文、
    阅读进度、分享状态一个不动。解析失败会抛 ValueError，且**不改动**库里已有的数据
    （先解析到临时目录，成功才入库）。
    """
    try:
        old = store.read_doc(doc_id)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"库里没有这篇论文的版式数据：{doc_id}") from exc
    old_pages = old.get("pages") or []

    with tempfile.TemporaryDirectory(prefix="paper-reparse-") as staging:
        parsed = pdf_parser.parse_pdf(pdf_path, Path(staging), backend=backend)
        if not parsed.get("pages"):
            msg = "未能从 PDF 解析出任何页面"
            for warn in parsed.get("warnings") or []:
                msg += f"；{warn}"
            raise ValueError(msg)
        images = sorted(p.name for p in Path(staging).iterdir() if p.is_file())
        store.put_images(doc_id, staging)          # 覆盖同名插图，两种后端都支持

    new_pages = parsed["pages"]
    new_doc = dict(old)
    new_doc.update({
        "parser": parsed.get("parser"),
        "parser_info": parsed.get("parser_info"),
        "warnings": parsed.get("warnings") or [],
        "pages": new_pages,
        "num_pages": parsed.get("num_pages"),
        "page_w": parsed.get("page_w"),
        "page_h": parsed.get("page_h"),
    })
    store.save_doc(doc_id, new_doc)
    store.update_record(doc_id, parser=parsed.get("parser"),
                        num_pages=parsed.get("num_pages"),
                        page_w=parsed.get("page_w"), page_h=parsed.get("page_h"))
    moved = _remap_translations(store, doc_id, old_pages, new_pages)

    old_texts = _block_texts(old_pages)
    new_texts = _block_texts(new_pages)
    return {
        "parser": parsed.get("parser"),
        "parser_info": parsed.get("parser_info") or {},
        "warnings": parsed.get("warnings") or [],
        "pages": [len(old_pages), len(new_pages)],
        "blocks": [len(old_texts), len(new_texts)],
        "changed": sum(1 for k, v in new_texts.items()
                       if k in old_texts and old_texts[k] != v),
        "added": sum(1 for k in new_texts if k not in old_texts),
        "gone": sum(1 for k in old_texts if k not in new_texts),
        "latex_blocks": [_latex_count(old_pages), _latex_count(new_pages)],
        "images": len(images),
        "translations_moved": moved,
    }
