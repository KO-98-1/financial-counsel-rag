"""BGE-M3 dense 임베딩과 bge-reranker-v2-m3 리랭커 (CPU)."""
from __future__ import annotations

import os

import numpy as np
import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

EMBED_MODEL = "BAAI/bge-m3"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

torch.set_num_threads(os.cpu_count() or 4)


class Embedder:
    """BGE-M3 dense: [CLS] 은닉 상태를 L2 정규화한 1024차원 벡터."""

    def __init__(self, name: str = EMBED_MODEL, max_length: int = 512):
        # bge-m3 는 .bin 가중치만 있어 transformers 가 Hub 에 safetensors 자동 변환을 요청하는
        # 백그라운드 스레드를 띄우고, 이 스레드가 프로세스 종료를 막는다 → 로컬 캐시만 사용
        # (모델 다운로드는 scripts/overnight_index.py 등 별도 단계에서 수행)
        self.tok = AutoTokenizer.from_pretrained(name, local_files_only=True)
        self.model = AutoModel.from_pretrained(name, local_files_only=True).eval()
        self.max_length = max_length
        self.dim = self.model.config.hidden_size

    @torch.inference_mode()
    def encode(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        order = np.argsort([len(t) for t in texts])  # 길이순 배치로 패딩 낭비 감소
        out = np.empty((len(texts), self.dim), dtype=np.float32)
        for i in range(0, len(texts), batch_size):
            idx = order[i:i + batch_size]
            enc = self.tok([texts[j] for j in idx], padding=True, truncation=True,
                           max_length=self.max_length, return_tensors="pt")
            cls = self.model(**enc).last_hidden_state[:, 0]
            out[idx] = torch.nn.functional.normalize(cls, dim=-1).float().numpy()
        return out


class Reranker:
    """질문-문서 쌍의 관련도 점수 (높을수록 관련)."""

    def __init__(self, name: str = RERANK_MODEL, max_length: int = 512):
        self.tok = AutoTokenizer.from_pretrained(name, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(name, local_files_only=True).eval()
        self.max_length = max_length

    @torch.inference_mode()
    def score(self, query: str, docs: list[str], batch_size: int = 16) -> np.ndarray:
        scores = []
        for i in range(0, len(docs), batch_size):
            enc = self.tok([[query, d] for d in docs[i:i + batch_size]], padding=True, truncation=True,
                           max_length=self.max_length, return_tensors="pt")
            scores.append(self.model(**enc).logits.view(-1).float().numpy())
        return np.concatenate(scores) if scores else np.zeros(0, dtype=np.float32)
