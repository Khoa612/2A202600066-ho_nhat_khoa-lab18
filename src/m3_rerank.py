"""Module 3: Reranking — Cross-encoder top-20 → top-3 + latency benchmark."""

import os, sys, time
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None
        self._model_type = None

    def _load_model(self):
        if self._model is None:
            try:
                from FlagEmbedding import FlagReranker
                self._model = FlagReranker(self.model_name, use_fp16=True)
                self._model_type = "flag"
            except (ImportError, Exception):
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
                self._model_type = "cross_encoder"
        return self._model

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank documents: top-20 → top-k."""
        if not documents:
            return []

        model = self._load_model()
        pairs = [(query, doc["text"]) for doc in documents]

        if self._model_type == "flag":
            scores = model.compute_score(pairs, normalize=True)
        else:
            scores = model.predict(pairs)

        # Ensure scores is a list
        if not isinstance(scores, list):
            try:
                scores = scores.tolist()
            except AttributeError:
                scores = list(scores)

        scored_docs = list(zip(scores, documents))
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        return [
            RerankResult(
                text=doc["text"],
                original_score=float(doc.get("score", 0.0)),
                rerank_score=float(score),
                metadata=doc.get("metadata", {}),
                rank=i + 1,
            )
            for i, (score, doc) in enumerate(scored_docs[:top_k])
        ]


class FlashrankReranker:
    """Lightweight alternative (<5ms). Optional."""
    def __init__(self):
        self._model = None

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        try:
            from flashrank import Ranker, RerankRequest
            if self._model is None:
                self._model = Ranker()
            passages = [{"text": d["text"]} for d in documents]
            request = RerankRequest(query=query, passages=passages)
            results = self._model.rerank(request)
            return [
                RerankResult(
                    text=r["text"],
                    original_score=float(documents[i].get("score", 0.0)),
                    rerank_score=float(r.get("score", 0.0)),
                    metadata=documents[i].get("metadata", {}),
                    rank=i + 1,
                )
                for i, r in enumerate(results[:top_k])
            ]
        except (ImportError, Exception):
            return []


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    """Benchmark latency over n_runs."""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        times.append((time.perf_counter() - start) * 1000)  # ms

    avg_ms = sum(times) / len(times)
    return {
        "avg_ms": avg_ms,
        "min_ms": min(times),
        "max_ms": max(times),
    }


if __name__ == "__main__":
    query = "Nhan vien duoc nghi phep bao nhieu ngay?"
    docs = [
        {"text": "Nhan vien duoc nghi 12 ngay/nam.", "score": 0.8, "metadata": {}},
        {"text": "Mat khau thay doi moi 90 ngay.", "score": 0.7, "metadata": {}},
        {"text": "Thoi gian thu viec la 60 ngay.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    for r in reranker.rerank(query, docs):
        print(f"[{r.rank}] {r.rerank_score:.4f} | {r.text}")
