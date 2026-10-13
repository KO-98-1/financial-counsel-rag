"""앱 지표 화면용 데이터 로더 — runs/sim·data/feedback·runs/app_sessions 를 표로 만든다 (streamlit 비의존).

라운드 지표는 holdout(고정 벤치마크 100건) 기준이다. 라운드 폴더의 metrics.json·manifest.json·compare_*/compare.json 을
그대로 읽을 뿐 다시 계산하지 않는다.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pandas as pd

from .config import path
from .ingest import DOMAIN_KEYS

DIMS = {"accuracy": "정확성", "grounding": "근거·수치정책", "resolution": "문제해결", "service": "고객응대", "efficiency": "효율"}
OUTCOMES = ["해결", "이관", "미해결", "포기", "턴초과"]
STATUS_KO = {"active": "적용 중", "rolled_back": "롤백", "vetoed": "거부(사람)", "retired": "퇴출", "rejected": "게이트 탈락"}
SESSION_DIR = os.environ.get("COUNSEL_SESSION_DIR", "runs/app_sessions")      # 시연·시험 때는 환경변수로 따로 둔다
_ROUND_RE = re.compile(r"r\d{3}")


_MD_SPECIAL = re.compile(r"([\\`*_~$#<>\[\]|])")


def md(text) -> str:
    """고객·모델 문장을 마크다운 서식으로 읽지 않게 이스케이프한다. 예: '3~5일 … 7~10일' 의 ~ 가 취소선이 되는 것을 막는다."""
    return _MD_SPECIAL.sub(r"\\\1", str(text)).replace("\n", "  \n")


def sim_dir(domain: str) -> Path:
    return path("runs/sim") / DOMAIN_KEYS[domain]


def _read_json(p: Path) -> dict:
    with open(p, encoding="utf-8") as f:
        return json.load(f)


# --- 라운드 -------------------------------------------------------------------------

def list_rounds(domain: str) -> list[str]:
    """채점까지 끝난(metrics.json 이 있는) 정식 라운드(rNNN), 오름차순. pilot 등은 제외."""
    d = sim_dir(domain)
    if not d.exists():
        return []
    return sorted(p.name for p in d.iterdir() if _ROUND_RE.fullmatch(p.name) and (p / "metrics.json").exists())


def load_metrics(domain: str, rid: str) -> dict:
    return _read_json(sim_dir(domain) / rid / "metrics.json")


def load_manifest(domain: str, rid: str) -> dict:
    return _read_json(sim_dir(domain) / rid / "manifest.json")


def round_summary(domain: str, pool: str = "holdout") -> pd.DataFrame:
    """라운드별 한 행: 합격률·종합·5항목·평균 턴·통화 결과·부적절이관 태그·적용 피드백 수."""
    rows = []
    for rid in list_rounds(domain):
        m = load_metrics(domain, rid)
        p = (m.get("pools") or {}).get(pool)
        if not p:
            continue
        out = p.get("outcomes") or {}
        rows.append({"라운드": rid, "대화 수": p["n"], "합격률": p["pass_rate"], "종합": p["overall"],
                     **{label: p[k] for k, label in DIMS.items()}, "평균 턴": p["avg_turns"],
                     **{o: out.get(o, 0) for o in OUTCOMES},
                     "부적절이관": (p.get("failures") or {}).get("부적절이관", 0),
                     "적용 피드백": len(load_manifest(domain, rid).get("feedback_active") or [])})
    return pd.DataFrame(rows)


def compare_table(domain: str) -> pd.DataFrame:
    """짝비교(compare_<a>_<b>/compare.json): 같은 시나리오끼리 종합점수 차이와 유의성."""
    rows = []
    for p in sorted(sim_dir(domain).glob("compare_r*_r*/compare.json")):
        try:
            c = _read_json(p)
            o, ps = c["overall"], c["pass"]
            rows.append({"비교": f"{c['a']} → {c['b']}", "대화 수": c["n"], "종합 변화": o["delta"],
                         "95% 신뢰구간": f"{o['ci95'][0]:+.2f} ~ {o['ci95'][1]:+.2f}", "p(종합)": o["p_normal"],
                         "합격률": f"{ps['a']:.0%} → {ps['b']:.0%}", "p(합격률)": ps["mcnemar_p"]})
        except (KeyError, TypeError, ValueError):
            continue
    return pd.DataFrame(rows)


def topic_table(domain: str, rid: str, pool: str = "holdout") -> pd.DataFrame:
    """metrics.json 의 주제별 표(holdout). 대화 수가 적은 주제는 숫자 해석에 주의."""
    t = load_metrics(domain, rid).get("topics") or {}
    rows = [{"주제": k, "대화 수": v["n"], "합격률": v["pass_rate"], "종합": v["overall"]} for k, v in t.items()]
    return pd.DataFrame(rows).sort_values("대화 수", ascending=False, ignore_index=True) if rows else pd.DataFrame()


def failure_counts(domain: str, rid: str, pool: str = "holdout") -> pd.Series:
    f = ((load_metrics(domain, rid).get("pools") or {}).get(pool) or {}).get("failures") or {}
    return pd.Series(f, dtype=int).sort_values(ascending=False)


def latest_retrieval_eval(domain: str) -> tuple[str, str] | None:
    """가장 최근 검색 평가의 (폴더명, metrics.md 본문). 없으면 None."""
    d = path("runs/retrieval_eval")
    found = sorted(p for p in d.glob(f"{DOMAIN_KEYS[domain]}_*/metrics.md")) if d.exists() else []
    return (found[-1].parent.name, found[-1].read_text(encoding="utf-8")) if found else None


# --- 교훈·합성사례 ---------------------------------------------------------------------

def latest_measured(domain: str) -> tuple[str | None, list[str], float | None]:
    """가장 최근 라운드와 그때 상담사에게 적용된 피드백 id, 그 라운드의 holdout 합격률."""
    rounds = list_rounds(domain)
    if not rounds:
        return None, [], None
    rid = rounds[-1]
    pool = (load_metrics(domain, rid).get("pools") or {}).get("holdout") or {}
    return rid, list(load_manifest(domain, rid).get("feedback_active") or []), pool.get("pass_rate")


def feedback_items(domain: str) -> list[dict]:
    from .feedback import Store
    return Store(domain).items


def store_version(domain: str) -> int:
    """피드백 저장소 파일의 수정 시각 — 바뀌면 앱이 상담사를 다시 만든다."""
    p = path("data/feedback") / DOMAIN_KEYS[domain] / "items.jsonl"
    return p.stat().st_mtime_ns if p.exists() else 0


def lesson_table(domain: str) -> pd.DataFrame:
    rows = []
    for i in feedback_items(domain):
        if i["kind"] == "lesson":
            scope = "범용" if i["scope"] == "global" else "주제"
            topics, text = ", ".join(i.get("topics") or []), f"{i['trigger']} → {i['text']}"
        else:
            scope, topics, text = "합성사례", i.get("topic") or "", f"Q: {i['question']}  A: {i['answer']}"
        rows.append({"ID": i["id"], "종류": "교훈" if i["kind"] == "lesson" else "합성사례", "범위": scope,
                     "상태": STATUS_KO.get(i["status"], i["status"]), "출처": i["source_round"],
                     "지지 대화": len({s["scenario_id"] for s in i.get("support") or []}), "주제": topics, "내용": text,
                     "이력": " / ".join(_event(h) for h in i.get("history") or [] if not h["action"].startswith("→"))})
    return pd.DataFrame(rows)


def _event(h: dict) -> str:
    """이력 한 줄을 읽기 좋게: 'active→rolled_back' → '적용 중→롤백'. 처음 채택('→active')은 건너뛴다."""
    action = "→".join(STATUS_KO.get(a, "수정" if a == "revised" else a) for a in h["action"].split("→"))
    return action + (f"({h['note'][:200]})" if h.get("note") else "")


# --- 내 상담 기록 (이 앱에서 직접 한 상담) -----------------------------------------------------

def load_sessions() -> pd.DataFrame:
    """runs/app_sessions/*.jsonl 의 턴 기록을 한 행씩. 세션 = 파일 하나."""
    rows = []
    d = path(SESSION_DIR)
    for p in sorted(d.glob("*.jsonl")) if d.exists() else []:
        with open(p, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                t, route = r.get("timings") or {}, r.get("route") or {}
                rows.append({"세션": p.stem, "시각": r.get("ts"), "턴": r.get("turn"), "고객": r.get("customer"),
                             "상담사": r.get("reply"), "주제": route.get("topic"),
                             "검색": bool(route.get("search")) and not r.get("evidence_carried"),
                             "근거 수": len(r.get("evidence") or []), "인용 수": len(r.get("cited") or []),
                             "교훈 주입": len(r.get("lessons") or []), "이관": r.get("transfer"),
                             "분석(초)": t.get("route"), "검색(초)": t.get("retrieve"), "답변(초)": t.get("respond"),
                             "구성": (r.get("mode") or {}).get("label", "")})
    df = pd.DataFrame(rows)
    if not df.empty:
        df["응답(초)"] = df[["분석(초)", "검색(초)", "답변(초)"]].sum(axis=1, min_count=1)
    return df
