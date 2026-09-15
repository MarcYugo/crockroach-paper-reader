"""生成一份样例英文论文 PDF 到 samples/sample.pdf，用于本地快速验证。

依赖: PyMuPDF (已列入 requirements.txt)
运行: python tools/make_sample_pdf.py
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import fitz  # PyMuPDF

OUT = Path(__file__).resolve().parent.parent / "samples" / "sample.pdf"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()

    # 第一页：标题 / 摘要 / 正文 / 图
    page = doc.new_page(width=612, height=792)
    x0, w = 56, 500
    x1 = x0 + w
    y = 60

    def text(s, size, bold=True, color=(0, 0, 0), gap=8):
        """自增 y 插入文本块。PyMuPDF 内建字体短名: tiro=Times-Roman, tibo=Times-Bold"""
        nonlocal y
        fname = "tibo" if bold else "tiro"
        page.insert_textbox(
            fitz.Rect(x0, y, x1, y + size * 2.0),
            s, fontname=fname, fontsize=size, color=color)
        y += size * 1.5 + gap

    text("Attention Is All You Need for Paper Readers", 20)
    y += 4
    text("John Doe, Jane Smith  and  Bo Zhang", 11, bold=False)
    text("Department of Computer Science, Example University", 10, bold=False)
    y += 8

    text("Abstract", 13)
    text("We present a simple approach to convert academic PDF documents into "
         "interactive HTML pages while preserving the original layout of text and "
         "figures. Our method extracts positioned text blocks and embedded images "
         "from each page, then reconstructs them in a web view with faithful fonts, "
         "sizes and colors. The resulting reader supports paragraph-level machine "
         "translation from English into Chinese, text highlighting and note taking.", 10.5, bold=False, gap=12)

    text("1  Introduction", 13)
    text("Reading a paper on a screen usually means opening a PDF viewer. However, "
         "PDF viewers provide little support for bilingual reading, annotations and "
         "personal notes. In this work we build a lightweight HTML reader that keeps "
         "the original visual layout while adding translation, highlighting and "
         "note features directly in the document.", 10.5, bold=False, gap=10)
    text("2  Related Work", 13)
    text("Several tools convert PDF into text or Markdown, but they usually destroy "
         "the original layout. Rendering pages as plain images keeps the layout yet "
         "loses selectable text. We combine precise text extraction with absolute "
         "positioning so that the web page looks like the original paper.", 10.5, bold=False, gap=12)

    # ---------- 插图(真实点阵图，用于验证图片抽取) ----------
    text("Figure 1.  Architecture of the proposed system.", 11, bold=False, gap=2)
    tmpdoc = fitz.open()
    tp = tmpdoc.new_page(width=520, height=240)
    tp.draw_rect(fitz.Rect(0, 0, 520, 240), color=None, fill=(0.93, 0.94, 0.97))
    tp.draw_circle(fitz.Point(110, 120), 62, color=(0.15, 0.35, 0.85), width=3)
    tp.draw_rect(fitz.Rect(180, 60, 470, 180), color=(0.1, 0.55, 0.3), width=2,
                 fill=(0.86, 1.0, 0.9))
    tp.insert_textbox(fitz.Rect(20, 15, 500, 70),
                      "Sample embedded figure (raster)", fontsize=16)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp_path = f.name
    tp.get_pixmap(matrix=fitz.Matrix(1, 1)).save(tmp_path)
    tmpdoc.close()
    img_rect = fitz.Rect(x0, y + 6, x0 + 400, y + 196)
    page.insert_image(img_rect, filename=tmp_path)
    Path(tmp_path).unlink(missing_ok=True)
    y = img_rect.y1 + 30

    text("3  Conclusion", 13)
    text("We show that a faithful, interactive HTML paper reader is feasible with "
         "modest engineering effort. Future work includes better handling of "
         "vector figures and multi-column layouts.", 10.5, bold=False, gap=8)

    # 第二页：参考文献，便于验证多页滚动
    page2 = doc.new_page(width=612, height=792)
    page2.insert_textbox(fitz.Rect(56, 60, 556, 120), "References", fontname="tiro", fontsize=14)
    refs = [
        "Vaswani, A., et al. Attention is all you need. NeurIPS 2017.",
        "Vasiliev, Y. Natural Language Processing with Transformers. 2020.",
        "Smith, J. Interactive scholarly reading tools. CHI 2021.",
        "Rahman, M. Preserving layout in document conversion. DocEng 2019.",
        "Li, Q. Bilingual interfaces for academic reading. ACL 2022.",
    ]
    refy = 92
    for r in refs:
        page2.insert_textbox(fitz.Rect(56, refy, 556, refy + 20), r,
                             fontname="tiro", fontsize=10.5)
        refy += 24

    doc.set_metadata({"title": "A Sample Paper for HTML Conversion"})
    doc.save(OUT)
    doc.close()
    print(f"已生成样例 PDF：{OUT}")


if __name__ == "__main__":
    main()
