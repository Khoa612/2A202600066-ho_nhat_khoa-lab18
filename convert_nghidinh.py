"""Convert Nghi dinh PDF with retry on rate limit."""
import base64, os, time
import fitz
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

pdf_path = "data/Nghi_dinh_so_13-2023_ve_bao_ve_du_lieu_ca_nhan_508ee.pdf"
out_path = "data/Nghi_dinh_so_13-2023_ve_bao_ve_du_lieu_ca_nhan_508ee.md"

doc = fitz.open(pdf_path)
total = len(doc)
print(f"Total pages: {total}")

full_text = []

for page_num in range(total):
    page = doc[page_num]
    mat = fitz.Matrix(120 / 72, 120 / 72)  # Lower DPI to reduce tokens
    pix = page.get_pixmap(matrix=mat)
    img_bytes = pix.tobytes("png")
    img_b64 = base64.b64encode(img_bytes).decode()

    print(f"Page {page_num + 1}/{total}...", end=" ", flush=True)
    t0 = time.perf_counter()

    for attempt in range(5):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Convert page to Markdown. Keep structure: headers (#), tables (|), lists. Return ONLY Markdown content."},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}", "detail": "low"}},
                    ],
                }],
                max_tokens=2048,
            )
            page_md = response.choices[0].message.content or ""
            elapsed = time.perf_counter() - t0
            full_text.append(f"<!-- Page {page_num + 1} -->\n\n{page_md}")
            print(f"OK ({len(page_md):,} chars, {elapsed:.1f}s)")
            break
        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e):
                wait = 30 * (attempt + 1)
                print(f"Rate limited, waiting {wait}s...", end=" ", flush=True)
                time.sleep(wait)
            else:
                print(f"Error: {e}")
                break

doc.close()

content = "\n\n---\n\n".join(full_text)
with open(out_path, "w", encoding="utf-8") as f:
    f.write(f"# Nghi dinh so 13-2023 ve bao ve du lieu ca nhan\n\n")
    f.write(content)

print(f"\nDone! Saved {len(content):,} chars to {out_path}")
