"""Kiwi 형태소 기반 BM25. 금융 고유 용어(부수거래금리감면 등)는 키워드 일치가 중요해 dense 와 병행."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from kiwipiepy import Kiwi

# 명사·동사/형용사 어간·외국어·숫자·어근만 색인
_KEEP = {"NNG", "NNP", "NNB", "NR", "SL", "SN", "SH", "XR", "VV", "VA", "MAG"}
_kiwi: Kiwi | None = None


def _get_kiwi() -> Kiwi:
    global _kiwi
    if _kiwi is None:
        _kiwi = Kiwi()
    return _kiwi


def tokenize(text: str) -> list[str]:
    return [t.form for t in _get_kiwi().tokenize(text or "") if t.tag in _KEEP]


def tokenize_many(texts: list[str]) -> list[list[str]]:
    return [[t.form for t in toks if t.tag in _KEEP] for toks in _get_kiwi().tokenize(texts)]


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.2, b: float = 0.75):
        self.n = len(docs)
        lens = np.array([len(d) for d in docs], dtype=np.float32)
        self.norm = k1 * (1 - b + b * lens / max(lens.mean(), 1e-6))
        self.k1 = k1
        postings: dict[str, dict[int, int]] = defaultdict(dict)
        for i, doc in enumerate(docs):
            for tok in doc:
                postings[tok][i] = postings[tok].get(i, 0) + 1
        self.index = {}
        for tok, p in postings.items():
            ids = np.fromiter(p.keys(), dtype=np.int32, count=len(p))
            tf = np.fromiter(p.values(), dtype=np.float32, count=len(p))
            idf = np.log(1 + (self.n - len(p) + 0.5) / (len(p) + 0.5))
            self.index[tok] = (ids, tf, idf)

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        s = np.zeros(self.n, dtype=np.float32)
        for tok in set(query_tokens):
            if tok in self.index:
                ids, tf, idf = self.index[tok]
                s[ids] += idf * tf * (self.k1 + 1) / (tf + self.norm[ids])
        return s
