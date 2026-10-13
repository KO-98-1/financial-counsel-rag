"""1단계 데이터 파이프라인: AI Hub zip → 정제 → parquet + 시나리오 카드 + 통계.

python -m counsel.ingest            # config.yaml 의 data.domains 전부
python -m counsel.ingest 은행        # 특정 도메인만

산출물 (data/processed/<domain_key>/):
  dialogs.parquet          대화 단위 (train/val), RAG 의 case/dialog 컬렉션 원천
  qa.parquet               QA 단위 (train/val), RAG 의 qa 컬렉션 원천 + 시나리오 원천
  scenarios_dev.jsonl      개선용 시나리오 풀 (Validation, 피드백 수집 허용)
  scenarios_holdout.jsonl  고정 holdout 벤치마크 (Validation, KB 반영 금지), 앞에서부터 N개 사용
  stats.json / stats.md    데이터 통계 리포트
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from .config import load_config, path

DOMAIN_KEYS = {"은행": "bank", "보험": "insurance", "증권": "securities"}
SPEAKERS = {"TX": "상담사", "RX": "고객"}

# --- 마스킹 정제 -----------------------------------------------------------
# 원본 마스킹 기호: ●(숫자·금액·날짜) ▲(날짜·기관) *(숫자) ★◆(기관명) ■(이름·비밀번호) ○(지역 등)
# 숫자형 마스킹 덩어리: 기호로 시작해 기호로 끝남 (사이 공백·구분자 허용).
# 반복을 중첩하면 긴 덩어리에서 백트래킹이 폭발하므로 단일 반복으로만 구성
_M = r"[●▲*](?:[●▲*\s,.]*[●▲*])?"
_MASK_RULES = [
    (re.compile(rf"{_M}\s*년\s*{_M}\s*월\s*{_M}\s*일"), "[날짜]"),
    (re.compile(rf"{_M}\s*월\s*{_M}\s*일"), "[날짜]"),
    (re.compile(rf"{_M}\s*(?:만|천|억)?\s*(?:원|달러|엔|유로)"), "[금액]"),
    (re.compile(rf"{_M}\s*(?:만|억)"), "[금액]"),
    (re.compile(rf"{_M}\s*%"), "[비율]"),
    (re.compile(rf"{_M}\s*년"), "[연도]"),
    (re.compile(rf"{_M}\s*월"), "[월]"),
    (re.compile(rf"{_M}\s*일"), "[일]"),
    (re.compile(rf"{_M}\s*시(?:\s*{_M}\s*분)?"), "[시각]"),
    (re.compile(rf"{_M}\s*(?:지점|점)"), "[지점]"),
    (re.compile(r"[★◆▲](?:[★◆▲\s]*[★◆▲])?"), "[기관]"),
    (re.compile(r"■(?:[■\s]*■)?"), "[개인정보]"),
    (re.compile(r"●(?:[●O\s,.-]*[●O])?|\*{2,}|○+"), "[비공개]"),
]
_MASK_CHARS = "●▲★◆■○"


def clean_mask(text: str | None) -> str | None:
    if not text:
        return text
    for pat, rep in _MASK_RULES:
        text = pat.sub(rep, text)
    return text


_TURN_SPLIT = re.compile(r"(?:^|\s)(TX|RX)\s+")


def split_turns(content: str) -> list[dict]:
    """'TX ...\\nRX ...' 형식을 발화 목록으로. 한 줄에 여러 발화가 붙은 경우도 분리."""
    parts = _TURN_SPLIT.split(content)
    turns = []
    for spk, text in zip(parts[1::2], parts[2::2]):
        text = text.strip()
        if text:
            turns.append({"spk": SPEAKERS[spk], "text": text})
    return turns


def _norm(s: str | None) -> str:
    return re.sub(r"[\W_]+", "", s or "")


# --- 원본 읽기 ---------------------------------------------------------------

def _zip_path(raw_root: Path, split: str, kind: str, domain: str) -> Path:
    split_dir = {"train": "Training", "val": "Validation"}[split]
    kind_dir = {"source": "01.원천데이터", "label": "02.라벨링데이터"}[kind]
    prefix = {("train", "source"): "TS", ("train", "label"): "TL",
              ("val", "source"): "VS", ("val", "label"): "VL"}[(split, kind)]
    return raw_root / split_dir / kind_dir / f"{prefix}_{domain}.zip"


def _iter_json(zpath: Path):
    with zipfile.ZipFile(zpath) as z:
        for name in z.namelist():
            if name.endswith(".json"):
                yield name, json.loads(z.read(name).decode("utf-8"))


def _dialog_row(src: dict, consulting: dict, split: str, domain: str, has_qa: bool) -> dict:
    raw = src.get("consulting_content") or ""
    clean = clean_mask(raw)
    turns = split_turns(clean)
    return {
        "source_id": src["source_id"], "split": split, "domain": domain,
        "institution": src.get("source_institution"), "date": str(src.get("source_date") or ""),
        "gender": src.get("client_gender"), "age": src.get("client_age"),
        "client": src.get("consulting_client"), "client_type": src.get("consulting_client_type") or "개인",
        "topic": consulting.get("consulting_topic"), "summary": clean_mask(consulting.get("consulting_summary")),
        "content": clean, "content_raw": raw,
        "turns": json.dumps(turns, ensure_ascii=False), "n_turns": len(turns),
        "n_chars": len(raw), "n_masks": sum(raw.count(c) for c in _MASK_CHARS), "has_qa": has_qa,
    }


def load_split(raw_root: Path, domain: str, split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    dialogs: dict[str, dict] = {}
    qa_rows = []
    for _, d in _iter_json(_zip_path(raw_root, split, "label", domain)):
        src, consulting = d["source"], d.get("consulting") or {}
        sid = src["source_id"]
        if sid not in dialogs:
            dialogs[sid] = _dialog_row(src, consulting, split, domain, has_qa=True)
        for q in d.get("qa_data") or []:
            inp = q.get("input") or {}
            qa_rows.append({
                "qa_id": q["qa_id"], "source_id": sid, "split": split, "domain": domain,
                "topic": consulting.get("consulting_topic"), "qa_topic": q.get("qa_topic"),
                "task_category": q.get("task_category"), "situation": q.get("consulting_situation"),
                "purpose": q.get("consulting_purpose"), "core_terms": q.get("core_financial_terms"),
                "instruction": q.get("instruction"),
                "question": clean_mask(inp.get("question")), "answer": clean_mask(inp.get("answer")),
                "follow_up": clean_mask(inp.get("follow_up_question")), "output": clean_mask(q.get("output")),
                "gender": src.get("client_gender"), "age": src.get("client_age"),
                "client": src.get("consulting_client"), "client_type": src.get("consulting_client_type") or "개인",
            })
    # 라벨이 없는 원천 대화도 RAG 대화 컬렉션에는 포함
    for _, d in _iter_json(_zip_path(raw_root, split, "source", domain)):
        src = d["source"]
        if src["source_id"] not in dialogs:
            dialogs[src["source_id"]] = _dialog_row(src, d.get("consulting") or {}, split, domain, has_qa=False)
    return pd.DataFrame(dialogs.values()), pd.DataFrame(qa_rows)


# --- 시나리오 카드 -----------------------------------------------------------

def _rng(seed: int, key: str) -> random.Random:
    return random.Random(int(hashlib.md5(f"{seed}-{key}".encode()).hexdigest(), 16))


def _persona(row, seed: int, temperament_dist: dict) -> dict:
    r = _rng(seed, row.qa_id)
    if row.situation == "민원응대" and r.random() < 0.7:
        temperament = "불만"
    else:
        temperament = r.choices(list(temperament_dist), weights=list(temperament_dist.values()))[0]
    age = row.age or ""
    if age.startswith(("70", "80")):
        literacy = r.choices(["낮음", "보통"], weights=[0.6, 0.4])[0]
    elif age.startswith("60"):
        literacy = r.choices(["낮음", "보통"], weights=[0.3, 0.7])[0]
    else:
        literacy = r.choices(["보통", "높음"], weights=[0.5, 0.5])[0]
    if temperament == "디지털미숙":
        literacy = "낮음"
    return {"gender": row.gender, "age": row.age, "client": row.client, "client_type": row.client_type,
            "temperament": temperament, "digital_literacy": literacy}


def build_scenarios(dialogs_val: pd.DataFrame, qa_val: pd.DataFrame, sim: dict,
                    leaked_qa_ids: set) -> tuple[list, list]:
    seed = sim["seed"]
    # 대화 단위로 holdout 풀을 떼어 같은 대화의 QA가 양쪽에 섞이지 않게 함 (주제별 층화)
    holdout_sids = set()
    for topic, g in dialogs_val[dialogs_val.has_qa].groupby("topic"):
        sids = sorted(g.source_id)
        _rng(seed, f"pool-{topic}").shuffle(sids)
        holdout_sids.update(sids[: max(1, round(len(sids) * sim["holdout_pool_ratio"]))])

    customer_lines = {
        row.source_id: [t["text"] for t in json.loads(row.turns) if t["spk"] == "고객"]
        for row in dialogs_val.itertuples()
    }
    summaries = dict(zip(dialogs_val.source_id, dialogs_val.summary))
    dev, holdout_pool = [], []
    for row in qa_val.itertuples():
        if row.qa_id in leaked_qa_ids:  # Training 과 겹치는 시나리오는 평가에서 제외
            continue
        card = {
            "scenario_id": f"S-{row.qa_id}", "qa_id": row.qa_id, "source_id": row.source_id,
            "pool": "holdout" if row.source_id in holdout_sids else "dev",
            "domain": row.domain, "topic": row.topic, "qa_topic": row.qa_topic, "situation": row.situation,
            "persona": _persona(row, seed, sim["temperament_dist"]),
            # 고객 에이전트가 보는 정보
            "goal": row.purpose, "opening": row.question,
            "follow_ups": [row.follow_up] if row.follow_up else [],
            "background": customer_lines.get(row.source_id, []),
            "max_turns": sim["max_turns"],
            # 심판만 보는 정보
            "judge_only": {"instruction": row.instruction, "reference_answer": row.output,
                           "original_answer": row.answer, "dialog_summary": summaries.get(row.source_id),
                           "core_terms": row.core_terms, "task_category": row.task_category},
        }
        (holdout_pool if card["pool"] == "holdout" else dev).append(card)

    # 고정 holdout 벤치마크: 주제별 최소 할당 + 비례, 대화당 1건, 주제 라운드로빈 순서
    # → 앞에서부터 N개만 써도 주제가 고르게 섞임
    by_topic = defaultdict(list)
    seen_sids = set()
    pool = sorted(holdout_pool, key=lambda c: c["scenario_id"])
    _rng(seed, "benchmark").shuffle(pool)
    for c in pool:
        if c["source_id"] not in seen_sids:
            seen_sids.add(c["source_id"])
            by_topic[c["topic"]].append(c)
    size = sim["holdout_benchmark_size"]
    topics = sorted(by_topic, key=lambda t: -len(by_topic[t]))
    total = sum(len(v) for v in by_topic.values())
    min_each = max(1, size // (len(topics) * 2))
    quota = {t: min(len(by_topic[t]), max(min_each, round(size * len(by_topic[t]) / total))) for t in topics}
    while sum(quota.values()) > size:  # 반올림 초과분은 큰 주제에서 차감
        t = max(quota, key=lambda k: quota[k] - min_each)
        quota[t] -= 1
    benchmark = []
    queues = {t: by_topic[t][: quota[t]] for t in topics}
    while any(queues.values()):
        for t in topics:
            if queues[t]:
                benchmark.append(queues[t].pop(0))
    for i, c in enumerate(benchmark):
        c["benchmark_rank"] = i
    return dev, benchmark


def leaked_val_qa_ids(dialogs: pd.DataFrame, qa: pd.DataFrame) -> set:
    """질문 또는 대화 원문이 Training 과 정확히 일치하는 Validation QA."""
    tr_q = set(qa[qa.split == "train"].question.map(_norm))
    tr_c = set(dialogs[dialogs.split == "train"].content_raw.map(_norm))
    leaked_sids = set(dialogs[(dialogs.split == "val") & dialogs.content_raw.map(_norm).isin(tr_c)].source_id)
    va = qa[qa.split == "val"]
    return set(va[va.question.map(_norm).isin(tr_q) | va.source_id.isin(leaked_sids)].qa_id)


# --- 통계 ---------------------------------------------------------------------

def _pct(s: pd.Series, qs=(0.1, 0.5, 0.9)) -> dict:
    return {f"p{int(q * 100)}": int(s.quantile(q)) for q in qs}


def build_stats(dialogs: pd.DataFrame, qa: pd.DataFrame, dev: list, benchmark: list) -> dict:
    tr_q = set(qa[qa.split == "train"].question.map(_norm))
    va_q = qa[qa.split == "val"].question.map(_norm)
    tr_c = set(dialogs[dialogs.split == "train"].content_raw.map(_norm))
    va_c = dialogs[dialogs.split == "val"].content_raw.map(_norm)
    vc = lambda s: {str(k): int(v) for k, v in s.value_counts().items()}
    return {
        "dialogs": vc(dialogs.split), "dialogs_with_qa": vc(dialogs[dialogs.has_qa].split), "qa": vc(qa.split),
        "topic_qa_train": vc(qa[qa.split == "train"].topic), "topic_qa_val": vc(qa[qa.split == "val"].topic),
        "situation": vc(qa.situation), "task_category": vc(qa.task_category),
        "age": vc(dialogs.age), "gender": vc(dialogs.gender), "client": vc(dialogs.client),
        "client_type": vc(dialogs.client_type),
        "dialog_chars": _pct(dialogs.n_chars), "dialog_turns": _pct(dialogs.n_turns),
        "qa_per_dialog": _pct(qa.groupby("source_id").size()),
        "dialogs_with_mask_pct": round(float((dialogs.n_masks > 0).mean() * 100), 1),
        "leftover_raw_mask_in_clean": int(dialogs.content.str.count(f"[{_MASK_CHARS}]").sum()),
        "leak_check": {"val_questions_exact_in_train": int(va_q.isin(tr_q).sum()),
                       "val_dialogs_exact_in_train": int(va_c.isin(tr_c).sum()),
                       "val_qa_excluded_from_scenarios": len(leaked_val_qa_ids(dialogs, qa))},
        "scenarios": {"dev_pool": len(dev), "holdout_benchmark": len(benchmark),
                      "benchmark_topics": dict(Counter(c["topic"] for c in benchmark)),
                      "temperament_dev": dict(Counter(c["persona"]["temperament"] for c in dev)),
                      "literacy_dev": dict(Counter(c["persona"]["digital_literacy"] for c in dev))},
    }


def stats_markdown(domain: str, st: dict) -> str:
    def table(title: str, d: dict) -> str:
        rows = "\n".join(f"| {k} | {v:,} |" if isinstance(v, int) else f"| {k} | {v} |" for k, v in d.items())
        return f"### {title}\n| 항목 | 값 |\n|---|---|\n{rows}\n"
    sc = st["scenarios"]
    return "\n".join([
        f"# 데이터 통계 — {domain}\n",
        table("대화 수", st["dialogs"]), table("QA가 있는 대화 수", st["dialogs_with_qa"]), table("QA 수", st["qa"]),
        table("주제별 QA (train)", st["topic_qa_train"]), table("주제별 QA (val)", st["topic_qa_val"]),
        table("상담 상황", st["situation"]), table("연령대", st["age"]), table("고객 유형", st["client_type"]),
        table("대화 글자 수 분위", st["dialog_chars"]), table("대화 턴 수 분위", st["dialog_turns"]),
        table("마스킹", {"마스킹 포함 대화 비율(%)": st["dialogs_with_mask_pct"],
                         "정제 후 남은 마스킹 기호": st["leftover_raw_mask_in_clean"]}),
        table("누수 점검 (정확 일치)", st["leak_check"]),
        table("시나리오", {"dev 풀": sc["dev_pool"], "holdout 벤치마크": sc["holdout_benchmark"]}),
        table("holdout 벤치마크 주제 분포", sc["benchmark_topics"]),
        table("dev 고객 성격 분포", sc["temperament_dev"]), table("dev 디지털 이해도 분포", sc["literacy_dev"]),
    ])


# --- 실행 ---------------------------------------------------------------------

def run(domain: str) -> dict:
    cfg = load_config()
    raw_root = path(cfg["data"]["raw_root"])
    out = path(cfg["data"]["processed_dir"]) / DOMAIN_KEYS[domain]
    out.mkdir(parents=True, exist_ok=True)

    parts = [load_split(raw_root, domain, s) for s in ("train", "val")]
    dialogs = pd.concat([p[0] for p in parts], ignore_index=True)
    qa = pd.concat([p[1] for p in parts], ignore_index=True)
    dialogs.to_parquet(out / "dialogs.parquet", index=False)
    qa.to_parquet(out / "qa.parquet", index=False)

    dev, benchmark = build_scenarios(dialogs[dialogs.split == "val"], qa[qa.split == "val"], cfg["simulation"],
                                     leaked_val_qa_ids(dialogs, qa))
    for name, cards in (("scenarios_dev.jsonl", dev), ("scenarios_holdout.jsonl", benchmark)):
        with open(out / name, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(c, ensure_ascii=False) + "\n" for c in cards)

    st = build_stats(dialogs, qa, dev, benchmark)
    (out / "stats.json").write_text(json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "stats.md").write_text(stats_markdown(domain, st), encoding="utf-8")
    return st


if __name__ == "__main__":
    for dom in sys.argv[1:] or load_config()["data"]["domains"]:
        print(f"[{dom}] 처리 중...")
        print(json.dumps(run(dom), ensure_ascii=False, indent=1))
