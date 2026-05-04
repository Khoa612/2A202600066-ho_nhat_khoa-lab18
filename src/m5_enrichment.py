"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import os
import sys
import json
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


def _get_openai_client():
    from openai import OpenAI
    return OpenAI(api_key=OPENAI_API_KEY)


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.

    Args:
        text: Raw chunk text.

    Returns:
        Summary string (2-3 câu).
    """
    if not text.strip():
        return ""
    # Extractive summarization: take first 2 sentences, always stays within original length
    sentences = [s.strip() for s in text.replace(".\n", ". ").split(". ") if s.strip()]
    if not sentences:
        return text[:len(text) // 2] if text else ""
    # Take at most 2 sentences but cap at original length
    summary = ". ".join(sentences[:2])
    if len(sentences) > 2:
        summary += "."
    # Ensure summary does not exceed original length
    if len(summary) > len(text):
        summary = text[: len(text) // 2]
    return summary


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Index cả questions lẫn chunk → query match tốt hơn (bridge vocabulary gap).

    Args:
        text: Raw chunk text.
        n_questions: Số câu hỏi cần generate.

    Returns:
        List of question strings.
    """
    if not text.strip():
        return []
    try:
        client = _get_openai_client()
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": f"Dua tren doan van, tao {n_questions} cau hoi ma doan van co the tra loi. Tra ve moi cau hoi tren 1 dong. Chi tra ve cac cau hoi, khong giai thich.",
                },
                {"role": "user", "content": text},
            ],
            max_tokens=200,
        )
        raw = resp.choices[0].message.content.strip().split("\n")
        questions = [q.strip().lstrip("0123456789.-) ") for q in raw if q.strip()]
        return [q for q in questions if q]
    except Exception:
        # Fallback: generate basic question from text
        words = text.split()[:10]
        return [f"Thong tin ve: {' '.join(words)}?"]


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    Anthropic benchmark: giảm 49% retrieval failure (alone).

    Args:
        text: Raw chunk text.
        document_title: Tên document gốc.

    Returns:
        Text với context prepended.
    """
    if not text.strip():
        return text
    try:
        client = _get_openai_client()
        doc_info = f"Tai lieu: {document_title}\n\n" if document_title else ""
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Viet 1 cau ngan mo ta doan van nay nam o dau trong tai lieu va noi ve chu de gi. Chi tra ve 1 cau, khong giai thich them.",
                },
                {"role": "user", "content": f"{doc_info}Doan van:\n{text}"},
            ],
            max_tokens=80,
        )
        context_sentence = resp.choices[0].message.content.strip()
        return f"{context_sentence}\n\n{text}"
    except Exception:
        # Fallback: prepend document title
        prefix = f"[{document_title}] " if document_title else ""
        return f"{prefix}{text}"


# ─── Technique 4: Auto Metadata Extraction ──────────────


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, date_range, category.

    Args:
        text: Raw chunk text.

    Returns:
        Dict with extracted metadata fields.
    """
    if not text.strip():
        return {}
    try:
        client = _get_openai_client()
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": 'Trich xuat metadata tu doan van. Tra ve JSON: {"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance|legal", "language": "vi|en"}. Chi tra ve JSON, khong giai thich.',
                },
                {"role": "user", "content": text},
            ],
            max_tokens=150,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content.strip()
        return json.loads(raw)
    except Exception:
        return {"topic": "", "entities": [], "category": "unknown", "language": "vi"}


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks.

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: List of methods to apply. Default: ["contextual", "hyqa", "metadata"]
                 Options: "summary", "hyqa", "contextual", "metadata", "full"

    Returns:
        List of EnrichedChunk objects.
    """
    if methods is None:
        methods = ["contextual", "hyqa", "metadata"]

    use_summary = "summary" in methods or "full" in methods
    use_hyqa = "hyqa" in methods or "full" in methods
    use_contextual = "contextual" in methods or "full" in methods
    use_metadata = "metadata" in methods or "full" in methods

    enriched = []
    for chunk in chunks:
        text = chunk.get("text", "")
        meta = chunk.get("metadata", {})
        doc_title = meta.get("source", "")

        summary = summarize_chunk(text) if use_summary else ""
        questions = generate_hypothesis_questions(text) if use_hyqa else []
        enriched_text = contextual_prepend(text, doc_title) if use_contextual else text
        auto_meta = extract_metadata(text) if use_metadata else {}

        enriched.append(EnrichedChunk(
            original_text=text,
            enriched_text=enriched_text,
            summary=summary,
            hypothesis_questions=questions,
            auto_metadata={**meta, **auto_meta},
            method="+".join(methods),
        ))

    return enriched


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhan vien chinh thuc duoc nghi phep nam 12 ngay lam viec moi nam. So ngay nghi phep tang them 1 ngay cho moi 5 nam tham nien cong tac."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "So tay nhan vien VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")
