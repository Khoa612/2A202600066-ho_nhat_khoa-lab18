"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import os
import sys
import glob
import re
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE,
                    SEMANTIC_THRESHOLD)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load all markdown/text files from data/. (Đã implement sẵn)"""
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})
    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.

    Args:
        text: Input text.
        threshold: Cosine similarity threshold. Dưới threshold → tách chunk mới.
        metadata: Metadata gắn vào mỗi chunk.

    Returns:
        List of Chunk objects grouped by semantic similarity.
    """
    metadata = metadata or {}

    # Split text into sentences (support Vietnamese with no space before period)
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n\n', text) if s.strip()]
    if not sentences:
        return []
    if len(sentences) == 1:
        return [Chunk(text=sentences[0], metadata={**metadata, "chunk_index": 0, "strategy": "semantic"})]

    from sentence_transformers import SentenceTransformer
    import numpy as np

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(sentences, show_progress_bar=False)

    def cosine_sim(a, b):
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    chunks = []
    current_group = [sentences[0]]

    for i in range(1, len(sentences)):
        sim = cosine_sim(embeddings[i - 1], embeddings[i])
        if sim < threshold:
            chunk_text = " ".join(current_group)
            chunks.append(Chunk(
                text=chunk_text,
                metadata={**metadata, "chunk_index": len(chunks), "strategy": "semantic"}
            ))
            current_group = []
        current_group.append(sentences[i])

    if current_group:
        chunk_text = " ".join(current_group)
        chunks.append(Chunk(
            text=chunk_text,
            metadata={**metadata, "chunk_index": len(chunks), "strategy": "semantic"}
        ))

    return chunks


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Args:
        text: Input text.
        parent_size: Chars per parent chunk.
        child_size: Chars per child chunk.
        metadata: Metadata gắn vào mỗi chunk.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent.
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    # Build parent chunks by accumulating paragraphs up to parent_size
    parents: list[Chunk] = []
    children: list[Chunk] = []

    current_parts: list[str] = []
    current_len = 0

    def flush_parent(parts: list[str], parent_idx: int) -> Chunk:
        parent_text = "\n\n".join(parts)
        pid = f"parent_{parent_idx}"
        return Chunk(
            text=parent_text,
            metadata={**metadata, "chunk_type": "parent", "chunk_id": pid, "parent_id": pid, "chunk_index": parent_idx},
            parent_id=None,
        )

    for para in paragraphs:
        if current_len + len(para) > parent_size and current_parts:
            parents.append(flush_parent(current_parts, len(parents)))
            current_parts = []
            current_len = 0
        current_parts.append(para)
        current_len += len(para)

    if current_parts:
        parents.append(flush_parent(current_parts, len(parents)))

    # Build child chunks from each parent using sliding window
    for parent in parents:
        pid = parent.metadata["chunk_id"]
        parent_text = parent.text
        start = 0
        child_idx = 0
        while start < len(parent_text):
            end = min(start + child_size, len(parent_text))
            child_text = parent_text[start:end].strip()
            if child_text:
                children.append(Chunk(
                    text=child_text,
                    metadata={**metadata, "chunk_type": "child", "chunk_index": child_idx, "parent_id": pid},
                    parent_id=pid,
                ))
                child_idx += 1
            start += child_size

    return parents, children


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.

    Args:
        text: Markdown text.
        metadata: Metadata gắn vào mỗi chunk.

    Returns:
        List of Chunk objects, mỗi chunk = 1 section (header + content).
    """
    metadata = metadata or {}

    # Split by markdown headers (level 1-3)
    sections = re.split(r'(^#{1,3}\s+.+$)', text, flags=re.MULTILINE)

    chunks: list[Chunk] = []
    current_header = ""
    current_content = ""

    for part in sections:
        if re.match(r'^#{1,3}\s+', part):
            # Save previous section if it has content
            if current_content.strip():
                chunk_text = f"{current_header}\n{current_content}".strip() if current_header else current_content.strip()
                chunks.append(Chunk(
                    text=chunk_text,
                    metadata={**metadata, "section": current_header.strip(), "strategy": "structure",
                               "chunk_index": len(chunks)}
                ))
            current_header = part.strip()
            current_content = ""
        else:
            current_content += part

    # Flush last section
    if current_content.strip() or current_header:
        chunk_text = f"{current_header}\n{current_content}".strip() if current_header else current_content.strip()
        if chunk_text:
            chunks.append(Chunk(
                text=chunk_text,
                metadata={**metadata, "section": current_header.strip(), "strategy": "structure",
                           "chunk_index": len(chunks)}
            ))

    # If no headers found, fall back to basic chunks
    if not chunks and text.strip():
        chunks = [Chunk(
            text=text.strip(),
            metadata={**metadata, "section": "", "strategy": "structure", "chunk_index": 0}
        )]

    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.

    Returns:
        {"basic": {...}, "semantic": {...}, "hierarchical": {...}, "structure": {...}}
    """
    results: dict[str, dict] = {}

    for strategy_name in ["basic", "semantic", "hierarchical", "structure"]:
        all_chunks: list[Chunk] = []
        for doc in documents:
            text = doc["text"]
            meta = doc.get("metadata", {})
            if strategy_name == "basic":
                chunks = chunk_basic(text, metadata=meta)
            elif strategy_name == "semantic":
                chunks = chunk_semantic(text, metadata=meta)
            elif strategy_name == "hierarchical":
                parents, children = chunk_hierarchical(text, metadata=meta)
                all_chunks.extend(parents)
                # Report stats on children for hierarchical
                child_lengths = [len(c.text) for c in children] if children else [0]
                results["hierarchical"] = {
                    "count": len(children),
                    "parent_count": len(parents),
                    "avg_length": sum(child_lengths) / len(child_lengths),
                    "min_length": min(child_lengths),
                    "max_length": max(child_lengths),
                }
                continue
            elif strategy_name == "structure":
                chunks = chunk_structure_aware(text, metadata=meta)
            else:
                chunks = []
            all_chunks.extend(chunks)

        if strategy_name != "hierarchical":
            lengths = [len(c.text) for c in all_chunks] if all_chunks else [0]
            results[strategy_name] = {
                "count": len(all_chunks),
                "avg_length": sum(lengths) / len(lengths),
                "min_length": min(lengths),
                "max_length": max(lengths),
            }

    # Print comparison table
    print(f"\n{'Strategy':<15} | {'Chunks':>6} | {'Avg Len':>8} | {'Min':>6} | {'Max':>6}")
    print("-" * 55)
    for name, stats in results.items():
        print(f"{name:<15} | {stats['count']:>6} | {stats['avg_length']:>8.0f} | {stats['min_length']:>6} | {stats['max_length']:>6}")

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
