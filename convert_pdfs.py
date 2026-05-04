"""
Script chuyển đổi PDF (scan) sang Markdown bằng PyMuPDF + OpenAI Vision API.
Chạy: python convert_pdfs.py
"""

import base64
import os
import time

import fitz  # pymupdf
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def pdf_to_markdown_via_vision(pdf_path: str, max_pages: int | None = None) -> str:
    """
    Chuyển đổi từng trang PDF thành ảnh PNG, gửi lên OpenAI Vision để OCR → Markdown.

    Args:
        pdf_path: Đường dẫn file PDF.
        max_pages: Giới hạn số trang (None = tất cả).

    Returns:
        Chuỗi Markdown đầy đủ của document.
    """
    doc = fitz.open(pdf_path)
    pages_to_process = len(doc) if max_pages is None else min(max_pages, len(doc))
    print(f"  Tổng {len(doc)} trang, xử lý {pages_to_process} trang...")

    full_text: list[str] = []

    for page_num in range(pages_to_process):
        page = doc[page_num]

        # Render trang thành ảnh PNG ở 150 DPI (đủ rõ cho OCR)
        mat = fitz.Matrix(150 / 72, 150 / 72)
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        img_b64 = base64.b64encode(img_bytes).decode()

        print(f"  Trang {page_num + 1}/{pages_to_process}...", end=" ", flush=True)
        t0 = time.perf_counter()

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Chuyển đổi toàn bộ nội dung trang này sang định dạng Markdown. "
                                "Giữ nguyên cấu trúc: tiêu đề (dùng #, ##, ###), bảng (dùng |), "
                                "danh sách, và tất cả nội dung văn bản. "
                                "Trả về CHỈ nội dung Markdown, không thêm giải thích hay code block."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_b64}",
                                "detail": "high",
                            },
                        },
                    ],
                }
            ],
            max_tokens=4096,
        )

        page_md = response.choices[0].message.content or ""
        elapsed = time.perf_counter() - t0
        full_text.append(f"<!-- Trang {page_num + 1} -->\n\n{page_md}")
        print(f"OK ({len(page_md):,} chars, {elapsed:.1f}s)")

    doc.close()
    return "\n\n---\n\n".join(full_text)


def convert_all_pdfs(data_dir: str = "data") -> None:
    """Chuyển đổi tất cả file PDF trong thư mục data/ sang Markdown."""
    pdf_files = sorted(f for f in os.listdir(data_dir) if f.lower().endswith(".pdf"))

    if not pdf_files:
        print("Không tìm thấy file PDF nào trong data/")
        return

    print(f"Tìm thấy {len(pdf_files)} file PDF: {pdf_files}\n")

    for pdf_file in pdf_files:
        pdf_path = os.path.join(data_dir, pdf_file)
        out_name = os.path.splitext(pdf_file)[0] + ".md"
        out_path = os.path.join(data_dir, out_name)

        print(f"{'=' * 60}")
        print(f"Đang chuyển đổi: {pdf_file}")
        print(f"  Output: {out_path}")

        try:
            md_content = pdf_to_markdown_via_vision(pdf_path)

            with open(out_path, "w", encoding="utf-8") as f:
                f.write(f"# {os.path.splitext(pdf_file)[0]}\n\n")
                f.write(md_content)

            size_kb = os.path.getsize(out_path) / 1024
            print(f"  Xong! File size: {size_kb:.1f} KB\n")

        except Exception as e:
            print(f"  LỖI: {e}\n")

    print("Hoàn thành chuyển đổi tất cả PDF!")


if __name__ == "__main__":
    convert_all_pdfs()
