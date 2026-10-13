"""2단계 인덱스 구축: processed parquet (train) → 문서 → 형태소 토큰 + BGE-M3 임베딩 → LanceDB.

python -m counsel.rag.index 은행            # 전체
python -m counsel.rag.index 은행 --limit 200  # 동작 확인용 소량

컬렉션
  qa    : Training QA 1건 = 문서 1개 (질문 + 모범답변). 답변 근거의 1차 검색 단위
  case  : Training 대화 1건 = 문서 1개 (대화 요약). 유사 사례 맥락
  대화 전문은 임베딩하지 않고 source_id 로 dialogs.parquet 에서 펼친다.

임베딩은 청크(.npy)로 저장해 중단돼도 이어서 진행한다.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import lancedb
import numpy as np
import pandas as pd
import pyarrow as pa

from ..config import load_config, path
from ..ingest import DOMAIN_KEYS
from .bm25 import tokenize_many

CHUNK = 1024


def index_dir(domain: str) -> Path:
    return path("data/index") / DOMAIN_KEYS[domain]


def build_docs(domain: str) -> dict[str, pd.DataFrame]:
    src = path(load_config()["data"]["processed_dir"]) / DOMAIN_KEYS[domain]
    qa = pd.read_parquet(src / "qa.parquet")
    qa = qa[qa.split == "train"].reset_index(drop=True)
    qa_docs = pd.DataFrame({
        "id": qa.qa_id, "source_id": qa.source_id, "topic": qa.topic, "qa_topic": qa.qa_topic,
        "situation": qa.situation, "purpose": qa.purpose, "core_terms": qa.core_terms.fillna(""),
        "age": qa.age, "client_type": qa.client_type,
        "question": qa.question, "follow_up": qa.follow_up.fillna(""), "answer": qa.output,
        "text": qa.question + "\n" + qa.output,
    })
    dl = pd.read_parquet(src / "dialogs.parquet", columns=[
        "source_id", "split", "topic", "age", "gender", "client_type", "summary", "has_qa"])
    dl = dl[(dl.split == "train") & dl.summary.notna()].reset_index(drop=True)
    case_docs = pd.DataFrame({
        "id": dl.source_id, "source_id": dl.source_id, "topic": dl.topic, "age": dl.age,
        "gender": dl.gender, "client_type": dl.client_type, "summary": dl.summary, "text": dl.summary,
    })
    return {"qa": qa_docs, "case": case_docs}


def embed_resumable(texts: list[str], chunk_dir: Path, log=print) -> np.ndarray:
    from .models import Embedder

    chunk_dir.mkdir(parents=True, exist_ok=True)
    n_chunks = (len(texts) + CHUNK - 1) // CHUNK
    todo = [i for i in range(n_chunks) if not (chunk_dir / f"{i:05d}.npy").exists()]
    if todo:
        emb = Embedder()
        t0, done = time.perf_counter(), 0
        for i in todo:
            part = texts[i * CHUNK:(i + 1) * CHUNK]
            np.save(chunk_dir / f"{i:05d}.npy", emb.encode(part))
            done += len(part)
            rate = done / (time.perf_counter() - t0)
            remain = sum(len(texts[j * CHUNK:(j + 1) * CHUNK]) for j in todo[todo.index(i) + 1:])
            log(f"  {chunk_dir.name}: chunk {i + 1}/{n_chunks}  {rate:.1f} docs/s  남은 시간 약 {remain / rate / 60:.0f}분")
    return np.concatenate([np.load(chunk_dir / f"{i:05d}.npy") for i in range(n_chunks)])


def build(domain: str, limit: int | None = None, log=print) -> dict:
    out = index_dir(domain) if limit is None else index_dir(domain) / f"_sample{limit}"
    out.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(out / "lancedb"))
    summary = {}
    for name, docs in build_docs(domain).items():
        if limit:
            docs = docs.head(limit)
        t0 = time.perf_counter()
        tok_path = out / f"{name}_tokens.parquet"
        if tok_path.exists() and len(pd.read_parquet(tok_path)) == len(docs):
            tokens = pd.read_parquet(tok_path).tokens.tolist()
        else:
            tokens = tokenize_many(docs.text.tolist())
            pd.DataFrame({"id": docs.id, "tokens": tokens}).to_parquet(tok_path, index=False)
        log(f"[{name}] 형태소 토큰 {len(tokens):,}건 ({time.perf_counter() - t0:.0f}s)")

        vecs = embed_resumable(docs.text.tolist(), out / "_emb" / name, log)
        table = pa.Table.from_pandas(docs.assign(tokens=tokens), preserve_index=False)
        table = table.append_column("vector", pa.FixedSizeListArray.from_arrays(
            pa.array(vecs.ravel(), type=pa.float32()), vecs.shape[1]))
        db.create_table(name, data=table, mode="overwrite")
        summary[name] = {"docs": len(docs), "dim": int(vecs.shape[1]), "seconds": round(time.perf_counter() - t0)}
        log(f"[{name}] LanceDB 저장 완료: {summary[name]}")
    (out / "index_meta.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("domain")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    print(build(a.domain, a.limit))
