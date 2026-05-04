# Failure Analysis — Lab 18: Production RAG

**Nguoi thuc hien:** Ho Nhat Khoa (2A202600066) — lam ca nhan toan bo bai lab  
**Module phu trach:** M1 + M2 + M3 + M4 + M5 + Pipeline

---

## RAGAS Scores (sau khi chay pipeline)

| Metric | Naive Baseline | Production | Delta |
|--------|---------------|------------|-------|
| Faithfulness | (xem ragas_report.json) | (xem ragas_report.json) | |
| Answer Relevancy | | | |
| Context Precision | | | |
| Context Recall | | | |

*Luu y: Gia tri chinh xac trong reports/ragas_report.json va reports/naive_baseline_report.json*

---

## Error Tree Diagnostic Framework

```
Output sai?
├── YES: Context co chua cau tra loi?
│   ├── NO: → Loi retrieval
│   │   ├── BM25 tra ve ket qua? → Kiem tra segment_vietnamese()
│   │   ├── Dense tra ve ket qua? → Kiem tra embedding + Qdrant
│   │   └── Fix: Cai thien chunking (M1) hoac tang BM25 weight (M2)
│   └── YES: Context co relevant khong?
│       ├── NO: → Qua nhieu noise → Fix: Tang reranking (M3)
│       └── YES: LLM tao answer sai → Fix: Cai thien prompt template
└── NO: Thanh cong
```

---

## Bottom-5 Failures (Phan tich gia dinh)

Do test set co 8 cau hoi, duoi day la phan tich 5 cau co the gap van de:

### #1: Cau hoi ve ten cong ty
- **Question:** "Ten cong ty nop thue la gi?"
- **Expected:** "CONG TY CO PHAN DHA SURFACES"
- **Worst metric du kien:** Context Recall — neu chunking cat ngay phan tieu de (structured data)
- **Error Tree:** Output sai → Context chua du thong tin (cau tra loi nam trong bang, co the bi cut khi chunk)
- **Root cause:** Structure-aware chunking chua xu ly tot cac bang HTML/Markdown trong PDF scan
- **Suggested fix:** Dung `chunk_structure_aware()` thay vi `chunk_hierarchical()` cho tai lieu co bieu mau

### #2: Cau hoi so tien thue
- **Question:** "Thue gia tri gia tang phai nop doi voi hoat dong san xuat kinh doanh la bao nhieu?"
- **Expected:** "2.133.830"
- **Worst metric du kien:** Faithfulness — LLM co the nhap lan nhieu con so trong bang
- **Error Tree:** Output sai → Context dung → LLM hallucinate con so → Giam nhiet do (temperature)
- **Root cause:** Bieu mau thue co nhieu con so tuong tu, LLM kho phan biet [40a] va [40b]
- **Suggested fix:** Them metadata filter cho category="finance" + prompt "Tra loi CHINH XAC con so, khong lam tron"

### #3: Cau hoi ngay ban hanh Nghi dinh
- **Question:** "Nghi dinh 13/2023/ND-CP ban hanh ngay nao?"
- **Expected:** "ngay 17 thang 4 nam 2023"
- **Worst metric du kien:** Context Precision — nhieu chunks cua Nghi dinh co the duoc tra ve
- **Error Tree:** Output sai → Context co nhieu chunks → Reranker khong chon dung chunk header
- **Root cause:** Thong tin ngay thang nam trong header page 1, nhung reranker co the uu tien chunks co nhieu tu trung lap voi query
- **Suggested fix:** Tang weight cho chunks chua structured metadata (section = "QUYET DINH")

### #4: Cau hoi ve co quan ban hanh
- **Question:** "Co quan nao ban hanh Nghi dinh so 13/2023/ND-CP?"
- **Expected:** "Chinh phu"
- **Worst metric du kien:** Answer Relevancy — LLM co the tra loi qua dai (neu ca tinh Chinh phu + Bo truong)
- **Error Tree:** Output dung nhung qua verbose → Cai thien prompt de yeu cau tra loi ngan gon
- **Root cause:** Prompt chua gioi han do dai cau tra loi
- **Suggested fix:** Them vao system prompt: "Tra loi ngan gon, toi da 1-2 cau"

### #5: Cau hoi ve chu de Nghi dinh
- **Question:** "Nghi dinh 13/2023/ND-CP quy dinh ve van de gi?"
- **Expected:** "Bao ve du lieu ca nhan"
- **Worst metric du kien:** Answer Relevancy — cau tra loi co the dung nhung khong match ground truth
- **Error Tree:** Output dung (noi dung ve bao ve du lieu ca nhan) → Nhung cach dien dat khac voi ground truth → Context Recall cao nhung Answer Relevancy thap
- **Root cause:** Khong phai loi thuc su ma la mismatch giua cach LLM dien dat va ground truth string
- **Suggested fix:** Su dung LLM-as-judge thay vi exact match cho evaluation; hoac normalize ground truth

---

## Case Study (cho presentation)

**Question chon phan tich:** "Thue gia tri gia tang phai nop doi voi hoat dong san xuat kinh doanh la bao nhieu?"

**Error Tree walkthrough:**
1. **Output dung?** → KIEM TRA: LLM co tra ve "2.133.830" khong?
2. **Context co thong tin?** → Chunk tu BCTC.md phai chua dong "[40a] 2.133.830"
3. **Retrieval OK?** → BM25 + Dense deu tim duoc chunk bieu mau thue
4. **Reranking OK?** → Cross-encoder xep chunk co so tien len dau
5. **LLM generation?** → Prompt co ro rang yeu cau tra loi chinh xac so?

**Ket qua du kien:** Van de chinh la con so "2.133.830" bi lap lai trong nhieu o cua bieu mau ([40a] va [40b] deu = 2.133.830), khien LLM kho phan biet hay bi confuse.

**Fix o buoc:** B5 - Cai thien prompt: "Tim gia tri tai chi tieu [40a] — Thue GTGT phai nop doi voi hoat dong san xuat kinh doanh"

**Neu co them 1 gio, se optimize:**
- Tang so luong cau hoi test len 20+ de RAGAS scores co y nghia thong ke
- Implement ColBERT late interaction cho M2 de cai thien Context Precision
- Fine-tune reranker voi du lieu tieng Viet (BCTC domain)
- Them metadata filtering: query "thue" chi tim trong category="finance" chunks
