from markitdown import MarkItDown
import os

data_dir = "data"
md = MarkItDown()

for fname in os.listdir(data_dir):
    if fname.endswith(".pdf"):
        pdf_path = os.path.join(data_dir, fname)
        md_path = pdf_path.replace(".pdf", ".md")
        existing = open(md_path, encoding="utf-8").read().strip() if os.path.exists(md_path) else ""
        if existing:
            print(f"Skip (already converted): {md_path}")
            continue
        print(f"Converting: {pdf_path} ...")
        result = md.convert(pdf_path)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(result.text_content)
        print(f"  -> Saved {len(result.text_content)} chars to {md_path}")
