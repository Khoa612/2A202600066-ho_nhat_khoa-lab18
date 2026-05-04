# Individual Reflection — Lab 18

**Ten:** Ho Nhat Khoa  
**MSSV:** 2A202600066  
**Module phu trach:** M1 + M2 + M3 + M4 + M5 + Pipeline 

---

## 1. Dong gop ky thuat

- **Module da implement:** Tat ca (M1, M2, M3, M4, M5, Pipeline)
- **Cac ham/class chinh da viet:**
  - M1: `chunk_semantic()`, `chunk_hierarchical()`, `chunk_structure_aware()`, `compare_strategies()`
  - M2: `segment_vietnamese()`, `BM25Search.index()`, `BM25Search.search()`, `DenseSearch.index()`, `DenseSearch.search()`, `reciprocal_rank_fusion()`
  - M3: `CrossEncoderReranker._load_model()`, `CrossEncoderReranker.rerank()`, `benchmark_reranker()`
  - M4: `evaluate_ragas()`, `failure_analysis()`
  - M5: `summarize_chunk()`, `generate_hypothesis_questions()`, `contextual_prepend()`, `extract_metadata()`, `enrich_chunks()`
  - Pipeline: `run_query()` voi LLM generation (gpt-4o-mini)
- **So tests pass:** 37/37 (100%)

## 2. Kien thuc hoc duoc

- **Khai niem moi nhat:** Reciprocal Rank Fusion (RRF) - cach ket hop BM25 va dense search de tang recall ma khong can retrain. Cong thuc score(d) = Sum(1/(k+rank)) don gian nhung hieu qua.
- **Dieu bat ngo nhat:** Hierarchical chunking (Parent-Child) la pattern rat thuc te - index child chunks nho cho embedding chinh xac, nhung tra ve parent chunk lon cho LLM du context. Day la mot trong nhung pattern hay nhat trong Production RAG.
- **Ket noi voi bai giang:** 
  - Slide Chunking strategies → M1 (Semantic, Hierarchical, Structure-Aware)
  - Slide Hybrid Search → M2 (BM25 + Dense + RRF)
  - Slide Reranking → M3 (Cross-encoder vs Bi-encoder)
  - Slide RAGAS Evaluation → M4 (4 metrics: Faithfulness, AR, CP, CR)
  - Slide Contextual Retrieval (Anthropic) → M5 (contextual_prepend giam 49% retrieval failure)

## 3. Kho khan & Cach giai quyet

- **Kho khan 1:** PDF Nghi dinh so 13-2023 la dang scan (image-based PDF), markitdown tra ve 0 bytes. Giai quyet: Dung PyMuPDF render tung trang thanh PNG + OpenAI Vision API OCR.
- **Kho khan 2:** Rate limit 429 khi convert 39 trang PDF voi gpt-4o-mini. Giai quyet: Viet script voi retry logic, giam DPI tu 150 xuong 120, dung `detail: "low"` trong Vision API.
- **Kho khan 3:** Test `test_hierarchical_valid_parent_ids` fail vi parent chunks khong co `parent_id` trong metadata. Giai quyet: Doc ki test file, them key `"parent_id": pid` vao metadata cua parent chunk.
- **Kho khan 4:** `summarize_chunk()` voi OpenAI tao ra summary dai hon original text (vi LLM them giai thich). Giai quyet: Dung extractive summarization thay vi generative, lay 2 cau dau dam bao `len(summary) <= len(original)`.
- **Thoi gian debug:** ~30 phut cho cac loi tren

## 4. Neu lam lai

- **Se lam khac:** Nen tao test set co nhieu cau hoi hon (20+ cau) tu ca 2 tai lieu de RAGAS scores co y nghia hon.
- **Module muon thu tiep:** Muon implement ColBERT late interaction retrieval (M2 variant) de so sanh voi BM25 + Dense, vi ColBERT co the cai thien context_precision dang ke.

## 5. Tu danh gia

| Tieu chi | Tu cham (1-5) |
|----------|---------------|
| Hieu bai giang | 5 |
| Code quality | 4 |
| Teamwork | 4 |
| Problem solving | 5 |

**Ghi chu:** Lam toan bo bai lab (tat ca 5 module + pipeline) nen co the hieu sau hon tung module va tuong tac giua cac module.
