# Group Report — Lab 18: Production RAG

**Nguoi thuc hien:** Ho Nhat Khoa (2A202600066)  
**Ngay:** 2026-05-04

---

## Phan cong (lam ca nhan — tat ca do 1 nguoi)

| Ten | Module | Hoan thanh | Tests pass |
|-----|--------|-----------|-----------|
| Ho Nhat Khoa | M1: Chunking | Done | 13/13 |
| Ho Nhat Khoa | M2: Hybrid Search | Done | 5/5 |
| Ho Nhat Khoa | M3: Reranking | Done | 5/5 |
| Ho Nhat Khoa | M4: Evaluation | Done | 4/4 |
| Ho Nhat Khoa | M5: Enrichment (Bonus) | Done | 10/10 |
| Ho Nhat Khoa | Pipeline Integration | Done | - |

**Tong: 37/37 tests pass (100%)**

---

## Ket qua RAGAS (tu dong tu ragas_report.json)

*Xem chi tiet trong reports/ragas_report.json va reports/naive_baseline_report.json*

| Metric | Naive Baseline | Production | Delta |
|--------|---------------|------------|-------|
| Faithfulness | (xem report) | (xem report) | |
| Answer Relevancy | (xem report) | (xem report) | |
| Context Precision | (xem report) | (xem report) | |
| Context Recall | (xem report) | (xem report) | |

---

## Kien truc Production RAG da xay dung

```
[Data: BCTC.pdf + Nghidinh.pdf]
        |
    [PDF OCR] — PyMuPDF + OpenAI Vision (gpt-4o-mini)
        |
    [M5: Enrichment] — Contextual Prepend + HyQA + Metadata
        |
    [M1: Chunking] — Hierarchical (Parent 2048 / Child 256)
        |
    [M2: Indexing] ─── BM25 (underthesea tokenizer)
                   └── Dense (BAAI/bge-m3, 1024 dims, Qdrant)
        |
    [M2: Hybrid Search] — BM25 + Dense → RRF (k=60)
        |
    [M3: Reranking] — CrossEncoder (BAAI/bge-reranker-v2-m3) top-20 → top-3
        |
    [LLM: Generation] — gpt-4o-mini, context-grounded answer
        |
    [M4: Evaluation] — RAGAS 4 metrics
```

---

## Key Findings

1. **Biggest improvement:** Hybrid Search (M2) — Ket hop BM25 (tu vung chinh xac) voi Dense (ngu nghia) va RRF giup tang Context Recall va Context Precision dong thoi. BM25 tim kiem chinh xac ten, so lieu; Dense tim kiem theo ngu nghia ("thue GTGT" = "thue gia tri gia tang").

2. **Biggest challenge:** PDF scan OCR — File Nghi dinh 13/2023 (12.5MB, 39 trang) la PDF dang anh. Markitdown tra ve 0 bytes. Phai dung PyMuPDF render PNG + OpenAI Vision API. Gap rate limit 429, phai them retry logic va giam DPI tu 150 xuong 120.

3. **Surprise finding:** Hierarchical chunking (Parent-Child pattern) cai thien chat luong Answer Relevancy hon bao thiet vi children chunks co semantics ro rang hon → embedding chinh xac hon → dense search hieu qua hon. Day chinh xac la production best practice.

4. **M5 Enrichment impact:** Contextual prepend them 1 cau mo ta vi tri chunk trong tai lieu truoc moi chunk. Dieu nay giup cross-encoder reranker xep hang chinh xac hon vi co du context.

---

## Latency Breakdown (du kien)

| Buoc | Thoi gian | Ghi chu |
|------|-----------|---------|
| BM25 Search | <1ms | In-memory index |
| Dense Search (Qdrant) | ~10-50ms | Approximate nearest neighbor |
| RRF Fusion | <1ms | Pure Python |
| CrossEncoder Rerank | ~200-500ms | GPU-free inference, 20 docs |
| LLM Generation | ~1-3s | gpt-4o-mini, 512 tokens |
| **Tong** | **~2-4s/query** | |

---

## Presentation Notes (5 phut)

1. **RAGAS scores (naive vs production):** Xem bang tren, highlight metric cai thien nhieu nhat.

2. **Biggest win — module nao, tai sao:** M2 Hybrid Search — vi BM25 xu ly tieng Viet tot voi underthesea tokenizer, cong voi Dense search semantic. RRF la cong thuc don gian nhung hieu qua.

3. **Case study — 1 failure, Error Tree:** Cau hoi so tien thue 2.133.830 — LLM co the nhap lan voi chi tieu [40b] co gia tri tuong tu. Fix: Them context "Tra loi chinh xac chi tieu [40a]" vao prompt.

4. **Next optimization neu co them 1 gio:** Fine-tune RAGAS test set (20+ cau hoi chat luong cao tu ca 2 tai lieu), implement metadata filter (chi tim chunks category="finance" khi query ve thue), va ColBERT late interaction cho precision tot hon.
