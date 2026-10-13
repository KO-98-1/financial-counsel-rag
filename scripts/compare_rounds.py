"""두 라운드의 holdout 짝비교 (같은 시나리오·같은 심판).

python scripts/compare_rounds.py 은행 r001 r002
python scripts/compare_rounds.py 은행 r001 r003 --ja judgments_claude-sonnet-5-5_20261008_152653   # 라운드의 채점 폴더를 지정(기본 judgments)
결과: runs/sim/<domain>/compare_<a>_<b>/compare.json, compare.md
"""
from __future__ import annotations

import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from counsel.config import path  # noqa: E402
from counsel.ingest import DOMAIN_KEYS  # noqa: E402
from counsel.judge import DIMENSIONS  # noqa: E402

DIMS = list(DIMENSIONS)


def load_round(domain: str, rid: str, jdir: str = "judgments") -> dict:
    d = path("runs/sim") / DOMAIN_KEYS[domain] / rid
    man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    js = {}
    for p in (d / jdir).glob("*.json"):
        j = json.loads(p.read_text(encoding="utf-8"))
        js[j["scenario_id"]] = j
    turns = {}
    for p in (d / "dialogs").glob("*.jsonl"):
        turns[p.stem] = [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]
    return {"dir": d, "manifest": man, "judgments": js, "dialogs": turns,
            "cards": {c["scenario_id"]: c for c in man["scenarios"]}}


def bootstrap_ci(xs: list[float], n=5000, seed=1) -> tuple[float, float]:
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(xs) for _ in xs) / len(xs) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def mcnemar_exact(b: int, c: int) -> float:
    """b: 기준 합격→비교 불합격, c: 기준 불합격→비교 합격. 양측 정확 이항 검정."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def paired_t_p(ds: list[float]) -> tuple[float, float]:
    n = len(ds)
    m = sum(ds) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in ds) / (n - 1)) if n > 1 else 0
    if sd == 0:
        return 0.0, 1.0
    t = m / (sd / math.sqrt(n))
    return t, math.erfc(abs(t) / math.sqrt(2))  # n=100 이라 정규 근사


def main() -> None:
    domain, ra, rb = sys.argv[1], sys.argv[2], sys.argv[3]
    opt = dict(zip(sys.argv[4::2], sys.argv[5::2]))  # --ja/--jb: 라운드 A/B 의 채점 폴더
    A, B = load_round(domain, ra, opt.get("--ja", "judgments")), load_round(domain, rb, opt.get("--jb", "judgments"))
    hold = sorted(s for s, c in A["cards"].items() if c["pool"] == "holdout"
                  and s in A["judgments"] and s in B["judgments"])
    ja = {s: A["judgments"][s] for s in hold}
    jb = {s: B["judgments"][s] for s in hold}
    deltas = [jb[s]["overall"] - ja[s]["overall"] for s in hold]
    lo, hi = bootstrap_ci(deltas)
    t, p_t = paired_t_p(deltas)
    b = sum(ja[s]["pass"] and not jb[s]["pass"] for s in hold)
    c = sum((not ja[s]["pass"]) and jb[s]["pass"] for s in hold)

    def mean(xs):
        return round(sum(xs) / len(xs), 3) if xs else None

    def group(key):
        out = {}
        for g in sorted({A["cards"][s][key] for s in hold}):
            ss = [s for s in hold if A["cards"][s][key] == g]
            out[g] = {"n": len(ss),
                      "pass_a": mean([ja[s]["pass"] for s in ss]), "pass_b": mean([jb[s]["pass"] for s in ss]),
                      "overall_a": mean([ja[s]["overall"] for s in ss]), "overall_b": mean([jb[s]["overall"] for s in ss])}
        return out

    # 교훈 주입 빈도 (비교 라운드의 상담사 턴)
    inj, n_turns, transfers = Counter(), 0, 0
    for sid, evs in B["dialogs"].items():
        for e in evs:
            if e["type"] == "counselor":
                n_turns += 1
                inj.update(e.get("lessons") or [])
                transfers += bool(e.get("transfer"))
    cited_syn = sum(1 for evs in B["dialogs"].values() for e in evs if e["type"] == "counselor"
                    for c_ in e.get("cited") or [] if str(c_).startswith("SYN-"))
    retrieved_syn = sum(1 for evs in B["dialogs"].values() for e in evs if e["type"] == "counselor"
                        for ev in e.get("evidence") or [] if str(ev.get("id", "")).startswith("SYN-"))

    def outcomes(js):
        return dict(Counter(j["outcome"] for j in js.values()))

    def tags(js):
        return dict(Counter(f["type"] for j in js.values() for f in j["failures"]).most_common())

    changes = sorted(hold, key=lambda s: jb[s]["overall"] - ja[s]["overall"])
    ex = lambda s: {"scenario_id": s, "topic": A["cards"][s]["topic"], "situation": A["cards"][s]["situation"],
                    "opening": A["cards"][s]["opening"], "a": {"overall": ja[s]["overall"], "pass": ja[s]["pass"],
                                                              "outcome": ja[s]["outcome"], "summary": ja[s]["summary"]},
                    "b": {"overall": jb[s]["overall"], "pass": jb[s]["pass"], "outcome": jb[s]["outcome"],
                          "summary": jb[s]["summary"]}}
    res = {
        "a": ra, "b": rb, "n": len(hold),
        "feedback_b": B["manifest"].get("feedback_active"),
        "overall": {"a": mean(list(o["overall"] for o in ja.values())), "b": mean(list(o["overall"] for o in jb.values())),
                    "delta": mean(deltas), "ci95": [round(lo, 3), round(hi, 3)], "t": round(t, 2), "p_normal": round(p_t, 4)},
        "pass": {"a": mean([j["pass"] for j in ja.values()]), "b": mean([j["pass"] for j in jb.values()]),
                 "pass_to_fail": b, "fail_to_pass": c, "mcnemar_p": round(mcnemar_exact(b, c), 4)},
        "per_scenario": {"improved": sum(d > 0 for d in deltas), "same": sum(d == 0 for d in deltas),
                         "worse": sum(d < 0 for d in deltas)},
        "deltas": [round(d, 2) for d in deltas],
        "dims": {d: {"a": mean([ja[s]["scores"][d] for s in hold]), "b": mean([jb[s]["scores"][d] for s in hold])} for d in DIMS},
        "avg_turns": {"a": mean([ja[s]["counselor_turns"] for s in hold]), "b": mean([jb[s]["counselor_turns"] for s in hold])},
        "outcomes": {"a": outcomes(ja), "b": outcomes(jb)},
        "failures": {"a": tags(ja), "b": tags(jb)},
        "by_situation": group("situation"), "by_topic": group("topic"),
        "lesson_injection": {"counselor_turns": n_turns, "by_lesson": dict(inj.most_common()),
                             "transfer_turns": transfers, "synthetic_retrieved": retrieved_syn, "synthetic_cited": cited_syn},
        "most_improved": [ex(s) for s in reversed(changes[-5:])], "most_worse": [ex(s) for s in changes[:5]],
    }
    out = path("runs/sim") / DOMAIN_KEYS[domain] / f"compare_{ra}_{rb}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "compare.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

    o, pz = res["overall"], res["pass"]
    md = [f"# holdout 짝비교 — {ra} → {rb} (n={res['n']}, 같은 시나리오·같은 심판)\n",
          f"- 종합점수 {o['a']} → {o['b']} (Δ {o['delta']:+.3f}, 95% CI [{o['ci95'][0]:+.3f}, {o['ci95'][1]:+.3f}], 짝 t={o['t']}, p≈{o['p_normal']})",
          f"- 합격률 {pz['a']:.0%} → {pz['b']:.0%} (불합격→합격 {pz['fail_to_pass']}건, 합격→불합격 {pz['pass_to_fail']}건, McNemar p={pz['mcnemar_p']})",
          f"- 시나리오별: 개선 {res['per_scenario']['improved']} · 동일 {res['per_scenario']['same']} · 악화 {res['per_scenario']['worse']}",
          f"- 평균 상담사 턴 {res['avg_turns']['a']} → {res['avg_turns']['b']}\n",
          "| 항목 | " + ra + " | " + rb + " | Δ |", "|---|---|---|---|"]
    md += [f"| {d} | {v['a']} | {v['b']} | {v['b'] - v['a']:+.3f} |" for d, v in res["dims"].items()]
    md += ["\n## 상황별\n", f"| 상황 | n | 합격 {ra} | 합격 {rb} | 종합 {ra} | 종합 {rb} |", "|---|---|---|---|---|---|"]
    md += [f"| {g} | {v['n']} | {v['pass_a']:.0%} | {v['pass_b']:.0%} | {v['overall_a']} | {v['overall_b']} |"
           for g, v in res["by_situation"].items()]
    md += ["\n## 통화 결과\n", f"- {ra}: {res['outcomes']['a']}", f"- {rb}: {res['outcomes']['b']}",
           "\n## 실패 태그\n", f"- {ra}: {res['failures']['a']}", f"- {rb}: {res['failures']['b']}",
           "\n## 교훈 주입 (" + rb + " 상담사 턴 " + str(n_turns) + "개)\n", f"- {res['lesson_injection']}"]
    (out / "compare.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
