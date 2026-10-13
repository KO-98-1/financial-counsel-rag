"""4단계: 시뮬레이션 라운드 — 고객 에이전트 ↔ 상담사 대화 → 심판 채점 → 리포트.

python -m counsel.simulate 은행 --round r001                 # config 의 dev_size·holdout_size
python -m counsel.simulate 은행 --round pilot --dev 3 --holdout 0
python -m counsel.simulate 은행 --round r001 --report-only   # 채점된 것만으로 리포트 다시 생성

같은 --round 로 다시 실행하면 이어서 진행한다(대화는 턴 단위 체크포인트, 채점은 대화 단위).
결과: runs/sim/<domain>/<round>/ — manifest.json, dialogs/*.jsonl, judgments/*.json, metrics.json, report.md
holdout 대화·채점은 측정 전용이며 지식 저장소(교훈·합성사례)로 되돌리지 않는다.
"""
from __future__ import annotations

import argparse
import json
import random
import threading
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np

from .config import load_config, path
from .counselor import Counselor
from .customer import Customer
from .ingest import DOMAIN_KEYS
from .judge import DIMENSIONS, judge

_print_lock = threading.Lock()


def log(msg: str) -> None:
    with _print_lock:
        print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


# --- 시나리오 선택 --------------------------------------------------------------

def _read_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def select_scenarios(domain: str, round_id: str, dev_n: int, holdout_n: int) -> list[dict]:
    d = path(load_config()["data"]["processed_dir"]) / DOMAIN_KEYS[domain]
    seed = load_config()["simulation"]["seed"]
    dev = _read_jsonl(d / "scenarios_dev.jsonl")
    # dev 는 라운드마다 새로 뽑아 피드백을 넓게 모으고, holdout 은 고정 벤치마크 앞에서부터
    dev_sel = random.Random(f"{seed}-dev-{round_id}").sample(dev, min(dev_n, len(dev)))
    hold = sorted(_read_jsonl(d / "scenarios_holdout.jsonl"), key=lambda c: c.get("benchmark_rank", 0))
    return dev_sel + hold[:holdout_n]


# --- 대화 실행 (턴 단위 체크포인트) -------------------------------------------------

def _append(p: Path, ev: dict) -> None:
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(ev, ensure_ascii=False) + "\n")


def _history(events: list[dict]) -> list[dict]:
    return [{"speaker": e["type"], "text": e["text"] if e["type"] == "customer" else e["reply"]}
            for e in events if e["type"] in ("customer", "counselor") and (e.get("text") or e.get("reply"))]


def run_dialog(card: dict, counselor: Counselor, p: Path) -> dict:
    """대화를 끝까지(또는 이어서) 진행하고 end 이벤트를 반환."""
    events = _read_jsonl(p)
    for e in events:
        if e["type"] == "end":
            return e
    max_turns = card.get("max_turns") or load_config()["simulation"]["max_turns"]
    customer = Customer(card)
    t_start = time.perf_counter() - sum(e.get("seconds", 0) for e in events)
    last_evidence = None  # 재개 시에는 다음 턴에서 새로 검색

    if not events:
        ev = {"type": "customer", "turn": 0, "text": customer.opening().text, "end": None, "seconds": 0}
        _append(p, ev)
        events.append(ev)

    while True:
        last = events[-1]
        n_counselor = sum(e["type"] == "counselor" for e in events)
        outcome = None
        if last["type"] == "customer":
            if last.get("end"):
                outcome = last["end"]
            else:
                t0 = time.perf_counter()
                res = counselor.reply(card["persona"], _history(events), last_evidence)
                last_evidence = res.evidence
                ev = {"type": "counselor", "turn": n_counselor + 1, **res.record(),
                      "seconds": round(time.perf_counter() - t0, 2)}
        else:  # counselor
            if last.get("transfer"):
                outcome = "이관"
            elif n_counselor >= max_turns:
                outcome = "턴초과"
            else:
                t0 = time.perf_counter()
                c = customer.next(_history(events))
                ev = {"type": "customer", "turn": n_counselor, "text": c.text, "end": c.end, "raw": c.raw,
                      "seconds": round(time.perf_counter() - t0, 2)}
        if outcome:
            end = {"type": "end", "outcome": outcome, "counselor_turns": n_counselor,
                   "seconds": round(time.perf_counter() - t_start, 1)}
            _append(p, end)
            return end
        _append(p, ev)
        events.append(ev)


# --- 라운드 ---------------------------------------------------------------------

class Round:
    def __init__(self, domain: str, round_id: str):
        self.domain = domain
        self.dir = path("runs/sim") / DOMAIN_KEYS[domain] / round_id
        self.round_id = round_id

    @property
    def manifest_path(self) -> Path:
        return self.dir / "manifest.json"

    def prepare(self, dev_n: int, holdout_n: int) -> list[dict]:
        if self.manifest_path.exists():  # 재개: 처음 고른 시나리오를 그대로 (holdout 만 벤치마크 순서대로 확장 가능)
            m = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            have = {c["scenario_id"] for c in m["scenarios"]}
            if holdout_n > m["n_holdout"]:
                extra = [c for c in select_scenarios(self.domain, self.round_id, 0, holdout_n)
                         if c["scenario_id"] not in have]
                m["scenarios"] += extra
                m["n_holdout"] += len(extra)
                m.setdefault("extended", []).append({"at": datetime.now().isoformat(timespec="seconds"),
                                                     "holdout_added": len(extra)})
                self.manifest_path.write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
                log(f"holdout {len(extra)}건 추가 → {m['n_holdout']}건")
            return m["scenarios"]
        cards = select_scenarios(self.domain, self.round_id, dev_n, holdout_n)
        conf = load_config()
        from .feedback import Store
        feedback_active = [i["id"] for i in Store(self.domain).active()]  # 이 라운드 내내 쓸 스냅샷
        self.dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(json.dumps({
            "round": self.round_id, "domain": self.domain, "created": datetime.now().isoformat(timespec="seconds"),
            "llm_roles": conf["llm"]["roles"], "counselor": conf["counselor"],
            "n_dev": sum(c["pool"] == "dev" for c in cards), "n_holdout": sum(c["pool"] == "holdout" for c in cards),
            "feedback_active": feedback_active, "scenarios": cards}, ensure_ascii=False, indent=1), encoding="utf-8")
        return cards

    def rejudge(self) -> None:
        """기존 채점을 judgments_<모델>_<시각>/ 으로 옮겨 두고, 다음 run 에서 현재 심판으로 다시 채점하게 한다."""
        jd = self.dir / "judgments"
        if not jd.exists() or not any(jd.iterdir()):
            return
        olds = [json.loads(p.read_text(encoding="utf-8")).get("judge_model", "unknown") for p in list(jd.glob("*.json"))[:1]]
        dest = self.dir / f"judgments_{olds[0] if olds else 'unknown'}_{datetime.now():%Y%m%d_%H%M%S}"
        jd.rename(dest)
        m = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        m.setdefault("rejudged", []).append({"at": datetime.now().isoformat(timespec="seconds"), "moved_to": dest.name,
                                             "judge": load_config()["llm"]["roles"]["judge"]})
        m["llm_roles"]["judge"] = load_config()["llm"]["roles"]["judge"]
        self.manifest_path.write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
        log(f"기존 채점 → {dest.name}, 현재 심판으로 다시 채점합니다")

    def run(self, cards: list[dict], workers: int) -> None:
        (self.dir / "dialogs").mkdir(parents=True, exist_ok=True)
        (self.dir / "judgments").mkdir(parents=True, exist_ok=True)
        log("상담사 로드 중…")
        m = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if "feedback_active" not in m:  # 피드백 기능 이전에 만든 라운드(기준선) → 피드백 없음
            m["feedback_active"] = []
            self.manifest_path.write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
        counselor = Counselor(self.domain, feedback_ids=m["feedback_active"])
        log(f"적용 피드백 {len(counselor.feedback_ids)}개 (라운드 시작 시 스냅샷): {counselor.feedback_ids}")
        docs = {c: dict(zip(col.df.id, col.df.text)) for c, col in counselor.r.cols.items()}
        doc_text = lambda coll, i: docs[coll].get(i, "")
        done = Counter()

        def work(card: dict) -> None:
            sid = card["scenario_id"]
            jp = self.dir / "judgments" / f"{sid}.json"
            try:
                end = run_dialog(card, counselor, self.dir / "dialogs" / f"{sid}.jsonl")
                if not jp.exists():
                    events = _read_jsonl(self.dir / "dialogs" / f"{sid}.jsonl")
                    t0 = time.perf_counter()
                    j = judge(card, events, end["outcome"], doc_text)
                    j.update({"scenario_id": sid, "pool": card["pool"], "topic": card["topic"],
                              "judge_model": load_config()["llm"]["roles"]["judge"]["model"],
                              "situation": card["situation"], "outcome": end["outcome"],
                              "counselor_turns": end["counselor_turns"], "judge_seconds": round(time.perf_counter() - t0, 1)})
                    jp.write_text(json.dumps(j, ensure_ascii=False, indent=1), encoding="utf-8")
                j = json.loads(jp.read_text(encoding="utf-8"))
                with _print_lock:
                    done["ok"] += 1
                log(f"✔ {done['ok']}/{len(cards)} {sid} [{card['pool']}] {end['outcome']} "
                    f"턴{end['counselor_turns']} 종합 {j['overall']} {'합격' if j['pass'] else '불합격'}")
            except Exception as e:  # 한 대화의 실패가 라운드를 멈추지 않게 — 다음 실행에서 이어서
                with _print_lock:
                    done["fail"] += 1
                log(f"✘ {sid} {type(e).__name__}: {str(e)[:300]}")
                with open(self.dir / "errors.log", "a", encoding="utf-8") as f:
                    f.write(f"[{datetime.now().isoformat(timespec='seconds')}] {sid}\n{traceback.format_exc()}\n")

        log(f"라운드 {self.round_id}: 대화 {len(cards)}개, 동시 {workers}개")
        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(work, cards))
        log(f"완료 {done['ok']} / 실패 {done['fail']}")

    # --- 리포트 ---
    def report(self, cards: list[dict]) -> dict:
        js = {}
        for c in cards:
            jp = self.dir / "judgments" / f"{c['scenario_id']}.json"
            if jp.exists():
                js[c["scenario_id"]] = json.loads(jp.read_text(encoding="utf-8"))
        turns_lat = defaultdict(list)
        cite_rate = []
        for c in cards:
            for e in _read_jsonl(self.dir / "dialogs" / f"{c['scenario_id']}.jsonl"):
                if e["type"] == "counselor":
                    for k, v in (e.get("timings") or {}).items():
                        turns_lat[k].append(v)
                    turns_lat["turn_total"].append(e.get("seconds", 0))
                    cite_rate.append(float(bool(e.get("cited"))))

        def agg(items: list[dict]) -> dict:
            if not items:
                return {"n": 0}
            return {
                "n": len(items),
                "pass_rate": round(float(np.mean([j["pass"] for j in items])), 3),
                "overall": round(float(np.mean([j["overall"] for j in items])), 3),
                **{k: round(float(np.mean([j["scores"][k] for j in items])), 3) for k in DIMENSIONS},
                "avg_turns": round(float(np.mean([j["counselor_turns"] for j in items])), 2),
                "outcomes": dict(Counter(j["outcome"] for j in items)),
                "failures": dict(Counter(f["type"] for j in items for f in j["failures"]).most_common()),
            }

        by_pool = {pool: agg([j for j in js.values() if j["pool"] == pool]) for pool in ("dev", "holdout")}
        by_topic = {}
        for t in sorted({j["topic"] for j in js.values()}):
            items = [j for j in js.values() if j["topic"] == t]
            by_topic[t] = {"n": len(items), "pass_rate": round(float(np.mean([j["pass"] for j in items])), 3),
                           "overall": round(float(np.mean([j["overall"] for j in items])), 3)}
        metrics = {
            "round": self.round_id, "domain": self.domain, "generated": datetime.now().isoformat(timespec="seconds"),
            "scenarios": len(cards), "judged": len(js), "pools": by_pool, "topics": by_topic,
            "latency_s": {k: round(float(np.median(v)), 2) for k, v in turns_lat.items()},
            "cited_turn_rate": round(float(np.mean(cite_rate)), 3) if cite_rate else None,
        }
        (self.dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")
        self._write_md(metrics, js)
        return metrics

    def _write_md(self, m: dict, js: dict) -> None:
        dims = list(DIMENSIONS)
        md = [f"# 시뮬레이션 리포트 — {m['domain']} {m['round']}\n",
              f"생성 {m['generated']} · 시나리오 {m['scenarios']} · 채점 {m['judged']}\n",
              "## 풀별 성적\n",
              "| 풀 | n | 합격률 | 종합 | " + " | ".join(dims) + " | 평균 턴 |",
              "|---" * (5 + len(dims)) + "|"]
        for pool, a in m["pools"].items():
            if a["n"]:
                md.append(f"| {pool} | {a['n']} | {a['pass_rate']:.0%} | {a['overall']} | "
                          + " | ".join(str(a[k]) for k in dims) + f" | {a['avg_turns']} |")
        md += ["\n## 통화 결과·실패 유형\n"]
        for pool, a in m["pools"].items():
            if a["n"]:
                md.append(f"- **{pool}** 결과: {a['outcomes']} · 실패 태그: {a['failures'] or '없음'}")
        md += ["\n## 주제별 (dev+holdout)\n", "| 주제 | n | 합격률 | 종합 |", "|---|---|---|---|"]
        md += [f"| {t} | {v['n']} | {v['pass_rate']:.0%} | {v['overall']} |" for t, v in m["topics"].items()]
        md += ["\n## 지연(중앙값, 초)\n", ", ".join(f"{k} {v}" for k, v in m["latency_s"].items()),
               f"\n근거 인용 턴 비율: {m['cited_turn_rate']}\n",
               "\n## 불합격 대화\n"]
        for sid, j in sorted(js.items(), key=lambda x: x[1]["overall"]):
            if not j["pass"]:
                tags = ", ".join(f"T{f['turn']} {f['type']}" for f in j["failures"])
                md.append(f"- `{sid}` [{j['pool']}·{j['topic']}] 종합 {j['overall']} · {j['outcome']} · {tags}\n  - {j['summary']}")
        (self.dir / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def main() -> None:
    sim = load_config()["simulation"]
    ap = argparse.ArgumentParser()
    ap.add_argument("domain")
    ap.add_argument("--round", required=True, help="라운드 이름 (같은 이름으로 다시 실행하면 이어서 진행)")
    ap.add_argument("--dev", type=int, default=sim["dev_size"])
    ap.add_argument("--holdout", type=int, default=sim["holdout_size"])
    ap.add_argument("--workers", type=int, default=sim.get("workers", 3))
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--rejudge", action="store_true", help="기존 채점을 보관하고 현재 심판 설정으로 다시 채점")
    a = ap.parse_args()

    r = Round(a.domain, a.round)
    cards = r.prepare(a.dev, a.holdout)
    if a.rejudge:
        r.rejudge()
    if not a.report_only:
        r.run(cards, a.workers)
    m = r.report(cards)
    log(f"리포트: {r.dir / 'report.md'}")
    for pool, v in m["pools"].items():
        if v["n"]:
            log(f"  {pool}: n={v['n']} 합격률 {v['pass_rate']:.0%} 종합 {v['overall']}")


if __name__ == "__main__":
    main()
