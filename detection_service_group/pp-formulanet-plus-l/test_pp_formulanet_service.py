"""PP-FormulaNet_plus-L 服务调用样例：输入 PDF，输出公式 LaTeX。

依赖：
    pip install requests pymupdf

用法：
    python test_pp_formulanet_service.py paper.pdf                  # 每页整页识别
    python test_pp_formulanet_service.py paper.pdf --page 3         # 只处理第 3 页（页码从 1 开始）
    python test_pp_formulanet_service.py paper.pdf --page 3 --rect 72,300,520,360
    python test_pp_formulanet_service.py paper.pdf --auto           # 自动取页面内嵌图片区域

重要前提：
    PP-FormulaNet 是“单公式”识别模型，不会自己在页面上找公式：
      * 整页直传 = 把整页当一个公式，只适合“一页一公式”的截图
      * 论文正文页请用 --rect 圈出公式区域（阅读器里框选得到的坐标，单位是 PDF 坐标）
      * 要“整页自动找出所有公式”需要版面检测模型（PaddleOCR 的
        FormulaRecognitionPipeline）；当前服务只挂了公式识别模型，所以由客户端裁剪

服务地址默认 http://127.0.0.1:9001，远程部署时改下面的 SERVICE_URL。
"""

import argparse

import pymupdf  # PyMuPDF
import requests

SERVICE_URL = "http://127.0.0.1:9001"
PDF_PATH = "paper.pdf"  # 没传命令行参数时用这个文件
DPI = 200  # 页面渲染精度，公式偏小时调大到 300
TIMEOUT = 120  # 首次请求包含预热，给宽松一点


def check_health() -> None:
    """确认服务起来了、模型加载完了。"""
    resp = requests.get(f"{SERVICE_URL}/health", timeout=10)
    resp.raise_for_status()
    print("服务状态:", resp.json())


def recognize(name: str, image: bytes) -> str:
    """把一张图片的字节流发给服务，返回 LaTeX 字符串。"""
    resp = requests.post(
        f"{SERVICE_URL}/v1/formula/recognition",
        files={"files": (name, image, "image/png")},
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def render_pdf(pdf_path: str, page_no=None, rect=None, auto=False):
    """用 PyMuPDF 把 PDF 渲染成 [(文件名, png 字节流), ...]。

    服务端只接收图片，所以 PDF 必须在客户端先渲染/裁剪：
      rect=(x0, y0, x1, y1) 只渲染这块区域；auto=True 取页面内嵌图片区域。
    """
    doc = pymupdf.open(pdf_path)
    try:
        indexes = [page_no - 1] if page_no else list(range(doc.page_count))
        items = []
        for index in indexes:
            page = doc[index]
            if rect:
                areas = [("rect", pymupdf.Rect(*rect))]
            elif auto:
                # type == 1 是图片块（扫描页、以图片形式嵌入的公式）
                areas = [
                    (f"img{i + 1}", pymupdf.Rect(block["bbox"]))
                    for i, block in enumerate(page.get_text("dict")["blocks"])
                    if block["type"] == 1
                ]
            else:
                areas = [("full", None)]
            for tag, clip in areas:
                pix = page.get_pixmap(dpi=DPI, clip=clip)
                items.append((f"p{index + 1}_{tag}.png", pix.tobytes("png")))
        return items
    finally:
        doc.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="PDF 公式识别样例")
    parser.add_argument("pdf", nargs="?", default=PDF_PATH, help="PDF 文件路径")
    parser.add_argument("--page", type=int, help="只处理第几页（从 1 开始）")
    parser.add_argument("--rect", help="只处理该页的矩形区域，格式 x0,y0,x1,y1（PDF 坐标）")
    parser.add_argument("--auto", action="store_true", help="自动取页面内嵌图片作为公式区域")
    args = parser.parse_args()

    check_health()

    rect = [float(v) for v in args.rect.split(",")] if args.rect else None
    images = render_pdf(args.pdf, args.page, rect, args.auto)
    print(f"\n共 {len(images)} 个待识别区域（{args.pdf}）")

    for name, data in images:
        try:
            latex = recognize(name, data)
        except requests.HTTPError as exc:  # 单个区域失败不中断后面的
            latex = f"[失败 {exc.response.status_code}] {exc.response.text}"
        print(f"\n[{name}]\n  {latex}")

    print(f"\n完成：{len(images)} 个区域已输出 LaTeX")


if __name__ == "__main__":
    main()
