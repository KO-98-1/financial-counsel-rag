"""검색 단독 평가: Validation 시나리오의 첫 질문으로 검색해 설정별 품질을 비교한다.

python -m counsel.rag.eval 은행 --n 200

정답 문서가 따로 없으므로 다음 대리 지표를 쓴다.
  topic@1 / topic_hit@5 / topic_p@5 : 검색 문서의 상담 주제가 질문의 주제와 같은가
  qa_topic@1                        : 세부 주제 일치
  term_hit@5                        : 핵심 금융 용어(core_terms)가 상위 5개 문서에 등장하는가
  ref_sim@1 / ref_sim_max@5         : 모범답안 임베딩과 검색 문서 벡터의 코사인 유사도
또한 Validation 과 Training 의 근접 중복(코사인 ≥ 0.97) 비율을 함께 보고한다.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np

from ..config import load_config, path
from ..ingest import DOMAIN_KEYS
from .retriever import Retriever, make_reranker

# 이름 → (검색 모드, 정답 주제 필터 여부, 리랭커 종류)
# 정답 주제 필터는 라우터의 주제 분류가 완벽할 때의 상한선
CONFIGS = {
    "bm25": ("bm25", False, None),
    "dense": ("dense", False, None),
    "hybrid": ("hybrid", False, None),
    "hybrid+topic필터(정답주제)": ("hybrid", True, None),
    "hybrid+rerank(bge)": ("hybrid", False, "bge"),   # CPU 질의당 ~19초, 기본 제외
    "hybrid+rerank(llm)": ("hybrid", False, "llm"),
    "hybrid+rerank(llm)+topic필터(정답주제)": ("hybrid", True, "llm"),
}
DEFAULT_CONFIGS = ["bm25", "dense", "hybrid", "hybrid+topic필터(정답주제)", "hybrid+rerank(llm)"]


def load_cards(domain: str, n: int, seed: int) -> list[dict]:
    d = path(load_config()["data"]["processed_dir"]) / DOMAIN_KEYS[domain]
    cards = []
    for name in ("scenarios_dev.jsonl", "scenarios_holdout.jsonl"):
        with open(d / name, encoding="utf-8") as f:
            cards += [json.loads(line) for line in f]
    random.Random(seed).shuffle(cards)
    return cards[:n]


def run(domain: str, n: int, configs: list[str] = DEFAULT_CONFIGS, workers: int = 4) -> dict:
    seed = load_config()["simulation"]["seed"]
    cards = load_cards(domain, n, seed)
    r = Retriever(domain, collections=("qa",), rerank=False)
    rerankers = {}
    col = r.cols["qa"]
    pos = {v: i for i, v in enumerate(col.df.id)}
    ref = r.embedder.encode([c["judge_only"]["reference_answer"] or "" for c in cards])
    val_full = r.embedder.encode([(c["opening"] or "") + "\n" + (c["judge_only"]["reference_answer"] or "")
                                  for c in cards])
    near_dup = float(((val_full @ col.vectors.T).max(axis=1) >= 0.97).mean())

    out_dir = path("runs/retrieval_eval") / f"{DOMAIN_KEYS[domain]}_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics, per_query = {}, []
    for name in configs:
        mode, topic_filter, rr_kind = CONFIGS[name]
        if rr_kind and rr_kind not in rerankers:
            rerankers[rr_kind] = make_reranker(rr_kind)
        r.reranker = rerankers.get(rr_kind)
        rows, t0 = [], time.perf_counter()
        # 후보 추출은 한 스레드에서(Kiwi·임베딩), LLM 리랭크만 병렬
        cands = [r.candidates(c["opening"], "qa", mode, {"topic": c["topic"]} if topic_filter else None)
                 for c in cards]
        if r.reranker is None:
            ranked = [(o[:5], s[:5]) for o, s in cands]
        else:
            def _rerank(args):
                c, (order, _) = args
                return r.rerank(c["opening"], "qa", order) if order else ([], [])
            with ThreadPoolExecutor(workers if rr_kind == "llm" else 1) as ex:
                ranked = [(o[:5], s[:5]) for o, s in ex.map(_rerank, zip(cards, cands))]
        for c, rv, (order, scores) in zip(cards, ref, ranked):
            hits = r.hits("qa", order, scores)
            topics = [h.meta["topic"] for h in hits]
            vecs = col.vectors[[pos[h.id] for h in hits]] if hits else np.zeros((0, ref.shape[1]))
            sims = vecs @ rv
            term = c["judge_only"].get("core_terms")
            term = term if isinstance(term, str) and term else None  # 결측값은 NaN(float)으로 들어옴
            rows.append({
                "topic@1": float(bool(topics) and topics[0] == c["topic"]),
                "topic_hit@5": float(c["topic"] in topics),
                "topic_p@5": float(np.mean([t == c["topic"] for t in topics])) if topics else 0.0,
                "qa_topic@1": float(bool(hits) and hits[0].meta["qa_topic"] == c["qa_topic"]),
                "term_hit@5": float(any(term in h.text for h in hits)) if term else None,
                "ref_sim@1": float(sims[0]) if len(sims) else 0.0,
                "ref_sim_max@5": float(sims.max()) if len(sims) else 0.0,
            })
            per_query.append({"config": name, "scenario_id": c["scenario_id"], "query": c["opening"],
                              "topic": c["topic"], "hits": [[h.id, h.meta["topic"], round(h.score, 4)] for h in hits],
                              **rows[-1]})
        agg = {k: round(float(np.mean([x[k] for x in rows if x[k] is not None])), 4) for k in rows[0]}
        agg["sec_per_query"] = round((time.perf_counter() - t0) / len(cards), 3)
        if rr_kind == "llm":
            agg["rerank_parse_fail"] = r.reranker.parse_failures
            r.reranker.parse_failures = 0
        metrics[name] = agg
        print(name, agg, flush=True)

    result = {"domain": domain, "n_queries": len(cards), "near_dup_rate_val_vs_train": round(near_dup, 4),
              "metrics": metrics}
    (out_dir / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(out_dir / "per_query.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in per_query)
    keys = list(dict.fromkeys(k for m in metrics.values() for k in m))
    md = [f"# 검색 평가 — {domain} (질문 {len(cards)}개)\n",
          f"Validation↔Training 근접 중복률(코사인≥0.97): {near_dup:.1%}\n",
          "| 설정 | " + " | ".join(keys) + " |", "|---" * (len(keys) + 1) + "|"]
    md += [f"| {name} | " + " | ".join(str(m.get(k, "")) for k in keys) + " |" for name, m in metrics.items()]
    (out_dir / "metrics.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"저장: {out_dir}")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("domain")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--configs", default=",".join(DEFAULT_CONFIGS),
                    help="쉼표 구분. 가능: " + ", ".join(CONFIGS))
    ap.add_argument("--workers", type=int, default=4, help="LLM 리랭크 동시 호출 수")
    a = ap.parse_args()
    run(a.domain, a.n, a.configs.split(","), a.workers)
