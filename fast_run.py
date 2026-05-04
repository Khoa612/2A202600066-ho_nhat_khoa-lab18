"""Fast pipeline runner — generates reports without M5 enrichment (M5 scored by tests)."""
import json, os, time, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.m1_chunking import load_documents, chunk_hierarchical
from src.m2_search import HybridSearch
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import load_test_set, evaluate_ragas, failure_analysis, save_report
from naive_baseline import main as run_baseline
from config import RERANK_TOP_K
from openai import OpenAI

os.makedirs("reports", exist_ok=True)

# ── Step 1: Baseline ─────────────────────────────────────
print("\n[Step 1] Running Naive Baseline...")
run_baseline()
if os.path.exists("naive_baseline_report.json"):
    os.replace("naive_baseline_report.json", "reports/naive_baseline_report.json")

# ── Step 2: Production Pipeline (M1+M2+M3+M4, skip slow M5 API) ──
print("\n[Step 2] Production Pipeline...")
docs = load_documents()
all_chunks = []
for doc in docs:
    parents, children = chunk_hierarchical(doc["text"], metadata=doc["metadata"])
    for child in children:
        all_chunks.append({"text": child.text, "metadata": {**child.metadata, "parent_id": child.parent_id}})
print(f"  {len(all_chunks)} child chunks from {len(docs)} documents")

search = HybridSearch()
print("  Indexing (BM25 + Dense)...")
search.index(all_chunks)

reranker = CrossEncoderReranker()

# ── Step 3: Evaluate ─────────────────────────────────────
print("\n[Step 3] Evaluating queries...")
test_set = load_test_set()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
questions, answers, all_contexts, ground_truths = [], [], [], []

for i, item in enumerate(test_set):
    results = search.search(item["question"])
    docs_for_rerank = [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in results]
    reranked = reranker.rerank(item["question"], docs_for_rerank, top_k=RERANK_TOP_K)
    contexts = [r.text for r in reranked] if reranked else [r.text for r in results[:3]]

    try:
        ctx_str = "\n\n".join(contexts)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Tra loi CHI dua tren context duoc cung cap. Neu khong co thong tin → tra loi 'Khong tim thay thong tin lien quan.' Tra loi ngan gon, chinh xac."},
                {"role": "user", "content": f"Context:\n{ctx_str}\n\nCau hoi: {item['question']}"},
            ],
            max_tokens=256,
        )
        answer = resp.choices[0].message.content.strip()
    except Exception:
        answer = contexts[0] if contexts else "Khong tim thay."

    questions.append(item["question"])
    answers.append(answer)
    all_contexts.append(contexts)
    ground_truths.append(item["ground_truth"])
    print(f"  [{i+1}/{len(test_set)}] {item['question'][:55]}...")

print("\n[Step 4] Running RAGAS evaluation...")
results = evaluate_ragas(questions, answers, all_contexts, ground_truths)
failures = failure_analysis(results.get("per_question", []))
save_report(results, failures)
if os.path.exists("ragas_report.json"):
    os.replace("ragas_report.json", "reports/ragas_report.json")

# ── Summary ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("PRODUCTION RAG SCORES")
print("=" * 60)
for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
    s = results.get(m, 0)
    print(f"  {'V' if isinstance(s, float) and s >= 0.75 else 'X'} {m}: {s}")

naive_path = "reports/naive_baseline_report.json"
prod_path = "reports/ragas_report.json"
if os.path.exists(naive_path) and os.path.exists(prod_path):
    with open(naive_path) as f:
        naive = json.load(f)
    with open(prod_path) as f:
        prod = json.load(f)
    print(f"\n{'Metric':<25} {'Naive':>8} {'Prod':>8} {'Delta':>8}")
    print("-" * 55)
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        n = naive.get("aggregate", {}).get(m, 0) or 0
        p = prod.get("aggregate", {}).get(m, 0) or 0
        print(f"  {m:<23} {n:>8.4f} {p:>8.4f} {p-n:>+8.4f}")
