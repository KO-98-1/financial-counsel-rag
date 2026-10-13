"""5단계: 피드백 루프 연속 실행 — 직전 라운드 triage → 라운드 실행 → holdout 회귀 롤백 검사 → triage → 추이 리포트.

python -m counsel.loop 은행 --prev r001 --rounds r002 r003 r004
python -m counsel.loop 은행 --prev r001 --rounds r002 r003 r004 --summary-only

같은 명령을 다시 실행하면 끝난 단계(채점 완료 대화, triage.json, rollback.json)는 건너뛰고 이어서 진행한다.
추이: runs/sim/<domain>/loop_summary.md
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter

from .config import load_config, path
from .feedback import Store, rollback_check, triage
from .ingest import DOMAIN_KEYS
from .judge import DIMENSIONS
from .simulate import Round, log


def _load(p):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def write_summary(domain: str, rounds: list[str]) -> str:
    base = path("runs/sim") / DOMAIN_KEYS[domain]
    dims = list(DIMENSIONS)
    md = [f"# 피드백 루프 추이 — {domain}\n",
          "holdout(고정 벤치마크)은 측정 전용, dev 는 라운드마다 새로 뽑아 피드백 수집.\n",
          "## holdout\n",
          "| 라운드 | n | 합격률 | 종합 | " + " | ".join(dims) + " | 평균 턴 | 적용 피드백 | 채택(다음 라운드용) | 롤백 |",
          "|---" * (8 + len(dims)) + "|"]
    dev_rows, tag_rows = [], []
    for rid in rounds:
        d = base / rid
        m, man = _load(d / "metrics.json"), _load(d / "manifest.json")
        if not m:
            continue
        tri, rb = _load(d / "triage.json"), _load(d / "rollback.json")
        h, v = m["pools"].get("holdout", {}), m["pools"].get("dev", {})
        n_fb = len((man or {}).get("feedback_active") or [])
        adopted = ", ".join(tri["adopted"]) if tri else "-"
        rolled = (", ".join(rb["rolled_back"]) or f"없음(Δ{rb['delta']:+})") if rb else "-"
        if h.get("n"):
            md.append(f"| {rid} | {h['n']} | {h['pass_rate']:.0%} | {h['overall']} | "
                      + " | ".join(str(h[k]) for k in dims) + f" | {h['avg_turns']} | {n_fb} | {adopted or '없음'} | {rolled} |")
            tag_rows.append(f"| {rid} | " + " | ".join(str(h["failures"].get(t, 0)) for t in
                                                       ("응대미흡", "근거미활용", "이관누락", "절차누락", "검색실패", "환각", "수치단정", "부적절이관", "기타")) + " |")
        if v.get("n"):
            dev_rows.append(f"| {rid} | {v['n']} | {v['pass_rate']:.0%} | {v['overall']} | {dict(Counter(v['outcomes']))} |")
    md += ["\n## holdout 실패 태그 수\n",
           "| 라운드 | 응대미흡 | 근거미활용 | 이관누락 | 절차누락 | 검색실패 | 환각 | 수치단정 | 부적절이관 | 기타 |",
           "|---" * 10 + "|"] + tag_rows
    md += ["\n## dev (라운드마다 다른 시나리오 — 추이 비교용 아님)\n", "| 라운드 | n | 합격률 | 종합 | 통화 결과 |",
           "|---|---|---|---|---|"] + dev_rows
    store = Store(domain)
    md += ["\n## 피드백 항목 상태\n"]
    for it in store.items:
        body = f"{it['trigger']} → {it['text']}" if it["kind"] == "lesson" else f"Q: {it['question']}"
        scope = it.get("scope") or "case"
        md.append(f"- `{it['id']}` {it['status']} · {scope} · src {it['source_round']} · "
                  f"지지 {len({s['scenario_id'] for s in it['support']})} — {body}")
    out = base / "loop_summary.md"
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    return str(out)


def main() -> int:
    sim = load_config()["simulation"]
    ap = argparse.ArgumentParser()
    ap.add_argument("domain")
    ap.add_argument("--prev", required=True, help="기준(직전) 라운드 — 채점이 끝나 있어야 함")
    ap.add_argument("--rounds", nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=sim.get("workers", 3))
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()
    all_rounds = [a.prev] + a.rounds
    if a.summary_only:
        print(write_summary(a.domain, all_rounds))
        return 0

    prev = a.prev
    pm = _load(Round(a.domain, prev).dir / "metrics.json")
    if not pm or pm["judged"] < pm["scenarios"]:
        log(f"기준 라운드 {prev} 채점이 끝나지 않았습니다 ({pm and pm['judged']}/{pm and pm['scenarios']})")
        return 1
    log(f"triage {prev} …")
    t = triage(a.domain, prev)
    log(f"  {prev} 채택 {t['adopted']} / 보강 {t['merged_into']}")

    for rid in a.rounds:
        r = Round(a.domain, rid)
        cards = r.prepare(sim["dev_size"], sim["holdout_size"])
        # 실행 중에 동시 실행 수를 바꿀 수 있게: runs/sim/<domain>/loop_workers 파일이 있으면 라운드 시작 때 그 값을 씀
        wf = r.dir.parent / "loop_workers"
        workers = int(wf.read_text(encoding="utf-8").strip()) if wf.exists() else a.workers
        if workers != a.workers:
            log(f"{wf.name} = {workers} → 동시 {workers}개로 실행")
        r.run(cards, workers)
        m = r.report(cards)
        h = m["pools"].get("holdout", {})
        log(f"{rid}: holdout 합격 {h.get('pass_rate', 0):.0%} 종합 {h.get('overall')} · dev 종합 {m['pools'].get('dev', {}).get('overall')}")
        if m["judged"] < len(cards):
            log(f"{rid} 미완료 ({m['judged']}/{len(cards)}) — 같은 명령으로 다시 실행하면 이어서 진행합니다")
            write_summary(a.domain, all_rounds)
            return 1
        if not (r.dir / "rollback.json").exists():
            rb = rollback_check(a.domain, rid, prev)
            log(f"  회귀 검사 Δ{rb['delta']:+} → 롤백 {rb['rolled_back'] or '없음'}")
        log(f"triage {rid} …")
        t = triage(a.domain, rid)
        log(f"  {rid} 채택 {t['adopted']} / 보강 {t['merged_into']}")
        log(f"추이: {write_summary(a.domain, all_rounds)}")
        prev = rid
        # 실행 중에 멈출 지점을 바꿀 수 있게: runs/sim/<domain>/loop_stop_after 에 라운드 이름을 적어 두면 그 라운드 뒤에서 멈춤
        stop = r.dir.parent / "loop_stop_after"
        if stop.exists() and stop.read_text(encoding="utf-8").strip() == rid:
            log(f"{stop.name} = {rid} → 여기서 멈춥니다")
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
