"""교훈 전/후 A/B 비교: r001 dev 실패와 비슷한 성격의 새 질문(고정 고객 대본)으로
피드백 없는 상담사(A) vs r001 채택 피드백을 적용한 상담사(B)를 같은 조건에서 돌리고, 블라인드 쌍대 비교로 채점한다.

python scripts/ab_probe.py 은행 --repeats 2
결과: runs/ab_probe/<시각>/ — dialogs.jsonl, pairwise.jsonl, summary.json
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from counsel.config import load_config, path  # noqa: E402
from counsel.counselor import Counselor  # noqa: E402
from counsel.feedback import Store  # noqa: E402
from counsel.ingest import DOMAIN_KEYS  # noqa: E402
from counsel.llm import chat  # noqa: E402
from counsel.rag.retriever import Retriever  # noqa: E402

# 고정 고객 대본: r001 dev 실패 유형과 비슷한 성격의 새 질문 (dev·holdout 원문과 다름)
PROBES = [
    {"id": "P1", "target": "L0001·L0003 (직접 조회 요청·앱 경로 막힘)", "topic": "자동이체조회",
     "persona": {"age": "60~69세", "gender": "여", "client": "기존고객", "client_type": "개인"},
     "turns": ["통신요금 자동이체가 어느 계좌에서 나가는지 알고 싶어요.",
               "앱에 들어가 봤는데 자동이체 메뉴를 못 찾겠어요.",
               "그냥 지금 바로 확인해 주시면 안 돼요?"]},
    {"id": "P2", "target": "L0005 (불만·민원 응대)", "topic": "이자/연체금액",
     "persona": {"age": "50~59세", "gender": "남", "client": "기존고객", "client_type": "개인"},
     "turns": ["지난달에 대출 이자가 두 번 빠져나갔어요. 이게 말이 됩니까?",
               "아니 그러니까 왜 두 번 나갔냐고요. 설명을 해 보세요.",
               "그럼 잘못 나간 돈은 언제 돌려받을 수 있어요?"]},
    {"id": "P3", "target": "L0006·L0004 (대출 연장 서류·실행 정보)", "topic": "대출문의(만기/연장/조회등)",
     "persona": {"age": "40~49세", "gender": "남", "client": "기존고객", "client_type": "개인사업자"},
     "turns": ["사업자 대출 만기가 다음 달인데 연장하려면 서류를 뭘 준비해야 해요?",
               "소득 증빙은 어떤 걸로 내면 되나요?",
               "서류는 어디로 제출하면 되나요?"]},
    {"id": "P4", "target": "L0002 (상품별로 다른 조건)", "topic": "만기,연장/해지,수신",
     "persona": {"age": "30~39세", "gender": "여", "client": "기존고객", "client_type": "개인"},
     "turns": ["정기예금 만기되면 자동으로 재예치되나요? 금리는 그대로예요?",
               "제 상품은 어떻게 되는지 딱 잘라서 알려 주세요.",
               "그럼 어디서 확인하면 돼요?"]},
    {"id": "P5", "target": "K0001 (합성사례: 카드 자동이체 확인)", "topic": "자동이체조회",
     "persona": {"age": "40~49세", "gender": "여", "client": "기존고객", "client_type": "개인"},
     "turns": ["카드값 자동이체가 제 통장에 걸려 있는지 확인하려면 어떻게 해요?",
               "카드사에 물어봐야 하는 건가요, 은행에서 보는 건가요?",
               "알겠습니다. 카드사 번호는 어디서 봐요?"]},
    {"id": "P6", "target": "K0002 (합성사례: 대출 이자 부담)", "topic": "이자/연체금액",
     "persona": {"age": "30~39세", "gender": "남", "client": "기존고객", "client_type": "개인"},
     "turns": ["대출 이자가 너무 부담되는데 줄일 방법이 있나요?",
               "중간에 일부 갚으면 이자가 줄어드나요?",
               "얼마나 줄어드는지 미리 볼 수 있어요?"]},
    {"id": "P7", "target": "L0004·L0001 (착오송금 실행 정보)", "topic": "중계요청/착오송금",
     "persona": {"age": "70~79세", "gender": "남", "client": "기존고객", "client_type": "개인"},
     "turns": ["계좌번호를 잘못 눌러서 모르는 사람한테 돈을 보냈어요. 어떻게 해야 돼요?",
               "상대방이 안 돌려주면 어떻게 돼요?",
               "그 신청은 어디서 해요? 제가 휴대폰은 잘 못 써요."]},
    {"id": "P8", "target": "대조군 (해당 교훈 없음)", "topic": "환전문의",
     "persona": {"age": "20~29세", "gender": "여", "client": "기존고객", "client_type": "개인"},
     "turns": ["다음 주에 일본 가는데 엔화 환전은 어떻게 하는 게 좋아요?",
               "앱으로 신청하면 어디서 받아요?",
               "공항에서도 받을 수 있나요?"]},
]

PAIR_SYSTEM = "당신은 은행 콜센터 상담 품질 심사관입니다. 두 상담을 공정하게 비교하고 JSON 하나만 출력합니다."
PAIR_TASK = """같은 고객이 같은 말을 한 두 상담(상담 1, 상담 2)을 비교하세요. 고객 발화는 미리 정해진 대본이라 상담사 답과 어긋날 수 있으니 상담사 쪽만 평가합니다.

[상담사 운영 조건 — 지킨 것은 감점하지 않음]
- 계좌 조회·처리 권한 없음(안내형). 고객이 직접 조회를 요청하면 솔직히 말하고 대안을 제시하는 것이 바람직함.
- 금리·수수료·한도·기간 등 구체 수치는 말하지 않음. 다른 부서 이관 가능. 전화 구어체 2~4문장.

[고객] {persona}

[상담 1]
{d1}

[상담 2]
{d2}

[평가 항목 — 각 1~5점] accuracy(정확성), grounding(근거·수치 정책), resolution(문제 해결·실행 가능한 안내),
service(공감·눈높이·불만 대응), efficiency(반복 없이 진전)

출력(JSON 하나만):
{{"scores": {{"1": {{"accuracy": 0, "grounding": 0, "resolution": 0, "service": 0, "efficiency": 0}},
             "2": {{"accuracy": 0, "grounding": 0, "resolution": 0, "service": 0, "efficiency": 0}}}},
 "better": "1" | "2" | "tie",
 "key_difference": "두 상담의 가장 큰 차이 한두 문장",
 "reason": "판정 이유 한 문장"}}"""

_JSON_RE = re.compile(r"\{.*\}", re.S)
DIMS = ["accuracy", "grounding", "resolution", "service", "efficiency"]


def shared_copy(base: Retriever) -> Retriever:
    """임베더·리랭커는 공유하고 컬렉션만 복사 — B 쪽 합성사례 추가가 A 에 번지지 않게."""
    r = copy.copy(base)
    r.cols = {}
    for k, col in base.cols.items():
        c = copy.copy(col)
        c.tokens = list(col.tokens)
        r.cols[k] = c
    return r


def run_probe(c: Counselor, probe: dict) -> list[dict]:
    history, out, last_ev = [], [], None
    for text in probe["turns"]:
        history.append({"speaker": "customer", "text": text})
        res = c.reply(probe["persona"], history, last_ev)
        last_ev = res.evidence
        history.append({"speaker": "counselor", "text": res.reply})
        out.append({"customer": text, "counselor": res.reply, "lessons": res.lessons, "transfer": res.transfer,
                    "evidence": [e.id for e in res.evidence], "cited": res.cited})
        if res.transfer:
            break
    return out


def render(turns: list[dict]) -> str:
    lines = []
    for i, t in enumerate(turns, 1):
        lines.append(f"고객: {t['customer']}")
        lines.append(f"상담사(턴{i}): {t['counselor']}" + (f" [이관: {t['transfer']}]" if t["transfer"] else ""))
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("domain")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--feedback-from", default="r001", help="이 라운드에서 채택된 항목을 B 에 적용")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()

    out_dir = path("runs/ab_probe") / f"{DOMAIN_KEYS[a.domain]}_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)
    store = Store(a.domain)
    fb_ids = [i["id"] for i in store.items if i["source_round"] == a.feedback_from and i["status"] == "active"]
    print(f"B 에 적용할 피드백: {fb_ids}", flush=True)

    conf = load_config()["counselor"]
    base = Retriever(a.domain, collections=("qa", "case"), rerank=None if conf["rerank"] == "none" else conf["rerank"])
    cA = Counselor(a.domain, retriever=shared_copy(base), feedback_ids=[])
    cB = Counselor(a.domain, retriever=shared_copy(base), feedback_ids=fb_ids)
    cB._cand_lock = cA._cand_lock  # Kiwi·임베더 공유 → 잠금도 공유

    jobs = [(p, cond, rep) for p in PROBES for cond in ("A", "B") for rep in range(a.repeats)]
    lock = threading.Lock()
    results = {}

    def work(job):
        p, cond, rep = job
        turns = run_probe(cA if cond == "A" else cB, p)
        with lock:
            results[(p["id"], cond, rep)] = turns
            with open(out_dir / "dialogs.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({"probe": p["id"], "cond": cond, "rep": rep, "turns": turns}, ensure_ascii=False) + "\n")
        print(f"  {p['id']} {cond}#{rep} 완료 ({len(turns)}턴)", flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(work, jobs))

    # 블라인드 쌍대 비교 (반복마다 A/B 순서 무작위)
    rng = random.Random(7)
    pairs = []
    for p in PROBES:
        for rep in range(a.repeats):
            flip = rng.random() < 0.5
            first, second = ("B", "A") if flip else ("A", "B")
            persona = p["persona"]
            prompt = PAIR_TASK.format(persona=f"{persona['age']} {persona['gender']}, {persona['client_type']}",
                                      d1=render(results[(p["id"], first, rep)]), d2=render(results[(p["id"], second, rep)]))
            d = {}
            for _ in range(3):
                try:
                    d = json.loads(_JSON_RE.search(chat("judge", [{"role": "user", "content": prompt}],
                                                        system=PAIR_SYSTEM).text).group(0))
                    break
                except (AttributeError, json.JSONDecodeError):
                    continue
            m = {"1": first, "2": second}
            row = {"probe": p["id"], "target": p["target"], "rep": rep, "order": f"{first}{second}",
                   "better": m.get(str(d.get("better")), "tie"),
                   "scores": {m[k]: v for k, v in (d.get("scores") or {}).items() if k in m},
                   "key_difference": d.get("key_difference"), "reason": d.get("reason")}
            pairs.append(row)
            with open(out_dir / "pairwise.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"  {p['id']}#{rep} 우세: {row['better']} — {row['key_difference']}", flush=True)

    def mean(xs):
        return round(sum(xs) / len(xs), 3) if xs else None
    summary = {"feedback": fb_ids, "repeats": a.repeats, "n_pairs": len(pairs),
               "wins": {k: sum(r["better"] == k for r in pairs) for k in ("A", "B", "tie")},
               "dims": {cond: {d: mean([r["scores"][cond][d] for r in pairs if cond in r["scores"]]) for d in DIMS}
                        for cond in ("A", "B")},
               "by_probe": {p["id"]: {"target": p["target"],
                                      "B_wins": sum(r["better"] == "B" for r in pairs if r["probe"] == p["id"]),
                                      "A_wins": sum(r["better"] == "A" for r in pairs if r["probe"] == p["id"])}
                            for p in PROBES}}
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print(f"저장: {out_dir}")


if __name__ == "__main__":
    main()
