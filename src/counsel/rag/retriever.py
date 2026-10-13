"""하이브리드 검색기: BM25 + BGE-M3 dense → RRF 결합 → (선택) 메타 필터 → (선택) 리랭크.

python -m counsel.rag.retriever 은행 "적금 만기 연장하면 금리는 어떻게 되나요?"
"""
from __future__ import annotations

import copy
import sys
import time
from dataclasses import dataclass, field

import lancedb
import numpy as np
import pandas as pd

from .bm25 import BM25, tokenize, tokenize_many
from .index import index_dir

RRF_K = 60


@dataclass
class Hit:
    collection: str
    id: str
    score: float
    text: str
    meta: dict = field(default_factory=dict)


class _Collection:
    def __init__(self, df: pd.DataFrame):
        self.vectors = np.stack(df.pop("vector").to_numpy()).astype(np.float32)
        self.tokens = [list(t) for t in df.pop("tokens")]
        self.bm25 = BM25(self.tokens)
        self.df = df.reset_index(drop=True)

    def extend(self, df: pd.DataFrame, vectors: np.ndarray, tokens: list[list[str]]) -> None:
        self.df = pd.concat([self.df, df], ignore_index=True)
        self.vectors = np.vstack([self.vectors, vectors.astype(np.float32)])
        self.tokens += tokens
        self.bm25 = BM25(self.tokens)


def make_reranker(kind: bool | str | None):
    """True/"bge" → bge-reranker(CPU), "llm" → LLM 리랭커(llm.roles.reranker), False/None → 없음."""
    if kind == "llm":
        from .llm_rerank import LLMReranker
        return LLMReranker()
    if kind:
        from .models import Reranker
        return Reranker()
    return None


class Retriever:
    def __init__(self, domain: str, collections=("qa", "case"), rerank: bool | str = True, root=None):
        from .models import Embedder

        db = lancedb.connect(str((root or index_dir(domain)) / "lancedb"))
        self.cols = {c: _Collection(db.open_table(c).to_pandas()) for c in collections}
        self.embedder = Embedder()
        self.reranker = make_reranker(rerank)

    def fork(self, rerank: bool = True) -> "Retriever":
        """임베더·리랭커는 공유하고 컬렉션만 따로 가진 복사본. 여기에 add_docs 로 합성사례를 넣어도 원본은 그대로라
        (대화형 앱에서 교훈 구성을 바꿔 가며 쓸 때). rerank=False 면 리랭크 없이 검색한다."""
        new = copy.copy(self)
        new.cols = {}
        for name, col in self.cols.items():
            c = copy.copy(col)
            c.tokens = list(col.tokens)          # extend 가 이 리스트를 제자리에서 늘린다
            new.cols[name] = c
        if not rerank:
            new.reranker = None
        return new

    def add_docs(self, collection: str, rows: list[dict]) -> None:
        """메모리 상의 컬렉션에 문서 추가 (합성사례 등). rows 는 id·text 와 메타 필드."""
        if not rows:
            return
        texts = [r["text"] for r in rows]
        self.cols[collection].extend(pd.DataFrame(rows), self.embedder.encode(texts), tokenize_many(texts))

    def search(self, query: str, collection: str = "qa", k: int = 5, mode: str = "hybrid",
               filters: dict | None = None, rerank: bool | None = None, candidates: int = 30) -> list[Hit]:
        order, scores = self.candidates(query, collection, mode, filters, candidates)
        if self.reranker is not None and rerank is not False and order:
            order, scores = self.rerank(query, collection, order)
        return self.hits(collection, order[:k], scores[:k])

    def candidates(self, query: str, collection: str = "qa", mode: str = "hybrid",
                   filters: dict | None = None, n: int = 30) -> tuple[list[int], list[float]]:
        """BM25·dense 후보를 RRF 로 결합한 (행 번호, 점수). Kiwi·임베딩을 쓰므로 한 스레드에서 호출."""
        col = self.cols[collection]
        mask = np.ones(len(col.df), dtype=bool)
        for key, val in (filters or {}).items():
            mask &= (col.df[key] == val).to_numpy()
        if not mask.any():
            return [], []

        ranked: list[np.ndarray] = []
        if mode in ("hybrid", "bm25"):
            s = col.bm25.scores(tokenize(query))
            s[~mask] = -np.inf
            top = np.argsort(-s)[:n]
            ranked.append(top[s[top] > 0])
        if mode in ("hybrid", "dense"):
            s = col.vectors @ self.embedder.encode([query])[0]
            s[~mask] = -np.inf
            ranked.append(np.argsort(-s)[:n])
        # Reciprocal Rank Fusion
        fused: dict[int, float] = {}
        for r in ranked:
            for rank, i in enumerate(r):
                fused[int(i)] = fused.get(int(i), 0.0) + 1.0 / (RRF_K + rank + 1)
        order = sorted(fused, key=fused.get, reverse=True)[:n]
        return order, [fused[i] for i in order]

    def rerank(self, query: str, collection: str, order: list[int]) -> tuple[list[int], list[float]]:
        """후보를 리랭커 점수로 재정렬. 동점은 기존(RRF) 순서 유지. LLM 리랭커는 스레드 병렬 호출 가능."""
        rs = self.reranker.score(query, self.cols[collection].df.text.iloc[order].tolist())
        pairs = sorted(zip(order, rs), key=lambda x: -x[1])
        return [p[0] for p in pairs], [float(p[1]) for p in pairs]

    def hits(self, collection: str, order: list[int], scores: list[float]) -> list[Hit]:
        df = self.cols[collection].df
        out = []
        for i, sc in zip(order, scores):
            row = df.iloc[i].to_dict()
            out.append(Hit(collection, row.pop("id"), float(sc), row.pop("text"), row))
        return out


if __name__ == "__main__":
    domain, query = sys.argv[1], " ".join(sys.argv[2:]) or "적금 만기 연장하면 금리는 어떻게 되나요?"
    t0 = time.perf_counter()
    r = Retriever(domain)
    print(f"로드 {time.perf_counter() - t0:.1f}s")
    for coll in r.cols:
        t0 = time.perf_counter()
        hits = r.search(query, coll, k=3)
        print(f"\n[{coll}] {time.perf_counter() - t0:.2f}s")
        for h in hits:
            print(f"  {h.score:7.3f} {h.id} [{h.meta.get('topic')}] {h.text[:90]!r}")
