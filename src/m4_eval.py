"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os
import sys
import json
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH

# Patch langchain compatibility for RAGAS 0.4.x with langchain 1.x
try:
    import langchain
    if not hasattr(langchain, "debug"):
        langchain.debug = False
except Exception:
    pass


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    from datasets import Dataset

    # Patch embed_query for langchain-openai 1.2+ compatibility with RAGAS 0.4
    try:
        import langchain_openai
        if not hasattr(langchain_openai.OpenAIEmbeddings, "embed_query"):
            import asyncio

            def _sync_embed_query(self, text: str) -> list[float]:
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        import concurrent.futures
                        with concurrent.futures.ThreadPoolExecutor() as pool:
                            future = pool.submit(asyncio.run, self.aembed_query(text))
                            return future.result()
                    return loop.run_until_complete(self.aembed_query(text))
                except Exception:
                    return self.embed_documents([text])[0]

            langchain_openai.OpenAIEmbeddings.embed_query = _sync_embed_query
    except Exception:
        pass

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    try:
        from ragas import RunConfig
        run_cfg = RunConfig(max_workers=4, max_wait=180, timeout=120)
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            run_config=run_cfg,
        )
    except TypeError:
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        )

    df = result.to_pandas()

    import math

    def _safe_float(val, default=0.0):
        try:
            v = float(val)
            return default if math.isnan(v) else v
        except (TypeError, ValueError):
            return default

    per_question = []
    for i, (_, row) in enumerate(df.iterrows()):
        per_question.append(EvalResult(
            question=questions[i] if i < len(questions) else row.get("question", ""),
            answer=answers[i] if i < len(answers) else row.get("answer", ""),
            contexts=contexts[i] if i < len(contexts) else row.get("contexts", []),
            ground_truth=ground_truths[i] if i < len(ground_truths) else row.get("ground_truth", ""),
            faithfulness=_safe_float(row.get("faithfulness")),
            answer_relevancy=_safe_float(row.get("answer_relevancy")),
            context_precision=_safe_float(row.get("context_precision")),
            context_recall=_safe_float(row.get("context_recall")),
        ))

    # Compute aggregates from per-question results (prefer RAGAS aggregate if available)
    if per_question:
        def _mean(vals):
            valid = [v for v in vals if v > 0 or v == 0]
            return sum(valid) / len(valid) if valid else 0.0
        agg = {
            "faithfulness": _mean([r.faithfulness for r in per_question]),
            "answer_relevancy": _mean([r.answer_relevancy for r in per_question]),
            "context_precision": _mean([r.context_precision for r in per_question]),
            "context_recall": _mean([r.context_recall for r in per_question]),
        }
    else:
        agg = {"faithfulness": 0.0, "answer_relevancy": 0.0,
               "context_precision": 0.0, "context_recall": 0.0}

    return {**agg, "per_question": per_question}


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    if not eval_results:
        return []

    # Compute composite score for each result (exclude 0-valued answer_relevancy from NaN)
    scored = []
    for r in eval_results:
        vals = [r.faithfulness, r.context_precision, r.context_recall]
        if r.answer_relevancy > 0:
            vals.append(r.answer_relevancy)
        avg_score = sum(vals) / len(vals) if vals else 0.0
        scored.append((avg_score, r))

    # Sort ascending (worst first), take bottom_n
    scored.sort(key=lambda x: x[0])
    bottom = scored[:bottom_n]

    failures = []
    for avg_score, r in bottom:
        # Find worst metric (exclude answer_relevancy=0 from NaN)
        metrics = {
            "faithfulness": r.faithfulness,
            "context_precision": r.context_precision,
            "context_recall": r.context_recall,
        }
        if r.answer_relevancy > 0:
            metrics["answer_relevancy"] = r.answer_relevancy
        worst_metric = min(metrics, key=lambda m: metrics[m])
        worst_score = metrics[worst_metric]

        # Diagnostic mapping
        if worst_metric == "faithfulness" or r.faithfulness < 0.85:
            diagnosis = "LLM hallucinating — answer not grounded in context"
            suggested_fix = "Tighten prompt: add 'Answer ONLY based on context', lower temperature"
        elif worst_metric == "context_recall" or r.context_recall < 0.75:
            diagnosis = "Missing relevant chunks — retrieval not finding answer"
            suggested_fix = "Improve chunking (smaller chunks) or add BM25 to hybrid search"
        elif worst_metric == "context_precision" or r.context_precision < 0.75:
            diagnosis = "Too many irrelevant chunks in context"
            suggested_fix = "Add reranking (bge-reranker) or metadata filtering"
        else:
            diagnosis = "Answer does not match question format"
            suggested_fix = "Improve prompt template to better match answer style"

        failures.append({
            "question": r.question,
            "worst_metric": worst_metric,
            "score": worst_score,
            "avg_score": avg_score,
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix,
        })

    return failures


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
