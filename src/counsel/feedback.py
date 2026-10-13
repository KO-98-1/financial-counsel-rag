"""5단계: 피드백 저장소 — dev 실패 → triage(교훈·합성사례 후보) → 정적 게이트 → 채택 / holdout 회귀 롤백 / 사람 거부권.

python -m counsel.feedback 은행 triage r001          # r001 dev 실패로 후보 생성·검사·채택 (다음 라운드부터 적용)
python -m counsel.feedback 은행 list [--all]
python -m counsel.feedback 은행 veto L0003 "사유"     # 사람 거부권: 즉시 비활성
python -m counsel.feedback 은행 restore L0003
python -m counsel.feedback 은행 rollback-check r002 r001   # r002 holdout 이 r001 보다 나빠지면(다중 기준) r001 채택분 롤백
python -m counsel.feedback 은행 revise L0001 --text "새 지침" [--trigger "새 상황"]   # 교훈 수정(이전 문구는 이력에 보관)
python -m counsel.feedback 은행 retire L0001 "사유"   # 교훈 퇴출(범용 자리 확보)

항목은 data/feedback/<domain>/items.jsonl 에 이력과 함께 남는다(삭제하지 않고 status 만 바꿈).
holdout 대화·채점은 절대 입력으로 쓰지 않는다 — triage 는 dev 풀만 읽고, 근거 시나리오가 dev 인지 다시 확인한다.
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

from .config import load_config, path
from .ingest import DOMAIN_KEYS
from .llm import chat

_JSON_RE = re.compile(r"\{.*\}", re.S)
_DIGIT_RE = re.compile(r"\d")
_SUPPORT_RE = re.compile(r"(S-[\w-]+?)(?::T?(\d+))?$")

POLICY = """- 계좌 조회·처리 권한 없음(안내형). 고객이 직접 조회를 요청하면 솔직히 말하고 대안 제시.
- 금리·수수료·한도·기간 등 구체 수치는 말하지 않음.
- 다른 부서·기관으로 이관 가능(이관하면 통화 종료).
- 전화 상담 구어체 2~4문장."""

TRIAGE_SYSTEM = ("당신은 은행 콜센터 AI 상담사의 품질 개선 담당입니다. 심판 평가에서 나온 실패를 분석해 "
                 "다음 라운드에 재사용할 개선 항목을 만듭니다. JSON 하나만 출력합니다.")

TRIAGE_TASK = """아래는 이번 라운드 dev 상담에서 심판이 지적한 실패들입니다. 다음 라운드 상담사를 개선할 항목을 만드세요.

[상담사 운영 조건 — 항목이 이와 충돌하면 안 됩니다]
{policy}

[만들 항목]
1) lessons — 상담 방식 지침
 - scope "global": 주제와 무관하게 늘 적용할 범용 지침(예: 같은 안내 반복 금지, 고객이 어려움을 호소하면 다른 경로 제시). 기존 범용 지침을 포함해 최대 {global_cap}개가 되도록 꼭 필요한 것만.
 - scope "topic": 특정 업무·상황에서만 필요한 지침. topics 에 아래 주제 목록의 이름을 그대로 적습니다.
 - trigger: 어떤 상황에서 적용하는지 한 문장. text: 무엇을 어떻게 하라는 지침 1~2문장.
 - 여러 대화에서 반복된 문제를 우선합니다. 한 시나리오에만 해당하는 내용, 숫자, 고객 개인정보는 넣지 않습니다.
2) cases — 지식 보강용 합성 QA. 검색실패·근거미활용·정확성 오류에서 심판이 올바른 안내 내용을 구체적으로 밝힌 경우만 만듭니다.
 - question: 고객이 실제로 물을 법한 질문 한 문장. answer: 올바른 업무 안내 2~4문장(숫자 없이, 일반 규정으로 쓸 수 있는 내용만).
 - 심판 노트에 근거가 없는 내용을 지어내지 않습니다.
3) revisions / retire — 기존 활성 교훈(L로 시작)을 고치거나 없앱니다.
 - 실패(특히 부적절이관, 같은 안내 반복, 근거미활용)가 기존 교훈의 문구 때문에 생겼다고 볼 근거가 있을 때만 만듭니다.
   예: 교훈이 "담당 부서 연결을 제안하라"고 해서 해결 가능한 문의를 불필요하게 이관한 경우 → 연결 제안의 조건을 좁혀 revisions 로 고칩니다.
 - revisions: id, trigger(고친 상황), text(고친 지침 1~2문장), reason. retire: id, reason(고쳐도 쓸모없거나 해로운 경우).
 - 근거 없이 고치지 않습니다. 해당 없으면 빈 목록.
support 에는 근거가 된 실패를 "시나리오ID:T턴" 형식으로 모두 적습니다.

[기존 활성 항목 — 같은 내용은 다시 만들지 않습니다]
{existing}

[주제 목록]
{topics}

[이번 라운드 dev 실패]
{failures}

출력(JSON 하나만):
{{"lessons": [{{"scope": "global", "topics": [], "trigger": "...", "text": "...", "support": ["S-...:T2"]}}],
  "cases": [{{"topic": "...", "question": "...", "answer": "...", "support": ["S-...:T3"]}}],
  "revisions": [{{"id": "L0001", "trigger": "...", "text": "...", "reason": "...", "support": ["S-...:T2"]}}],
  "retire": [{{"id": "L0002", "reason": "...", "support": ["S-...:T4"]}}]}}"""

GATE_SYSTEM = "당신은 AI 상담사 개선 항목의 검수자입니다. 엄격하게 검사하고 JSON 하나만 출력합니다."

GATE_TASK = """다음 개선 항목 후보를 하나씩 검사해 채택 여부를 정하세요.

[검사 기준]
- 중복: 기존 활성 항목과 사실상 같은 내용이면 거절하고 duplicate_of 에 기존 id 를 적습니다.
- 충돌: 기존 항목이나 상담사 운영 조건과 모순되면 거절합니다.
- 일반성: 특정 시나리오·고객에게만 맞는 내용이면 거절합니다.
- 정확성(case): 일반적인 은행 업무로서 맞는 내용이고 아래 '관련 기존 지식'과 모순되지 않아야 합니다. 확신할 수 없으면 거절합니다.
- 실효성: 모호해서 상담사 행동을 바꾸지 못하면 거절합니다.

[상담사 운영 조건]
{policy}

[기존 활성 항목]
{existing}

[후보]
{candidates}

출력(JSON 하나만): {{"verdicts": [{{"cand": "c1", "accept": true, "reason": "...", "duplicate_of": null}}]}}"""


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _read_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _conf() -> dict:
    return {"global_cap": 5, "topic_k": 3, "min_support_lesson": 2, "min_support_case": 1,
            "rollback_margin": 0.10, "lesson_min_sim": 0.50,
            "rollback_pass_drop": 0.05, "rollback_resolution_drop": 0.20,
            "rollback_misxfer_rise": 0.15, "rollback_transfer_rise": 0.20, **(load_config().get("feedback") or {})}


def _parse_json(text: str) -> dict:
    m = _JSON_RE.search(text or "")
    if not m:
        raise ValueError(f"JSON 없음: {text[:200]!r}")
    return json.loads(m.group(0))


# --- 저장소 ---------------------------------------------------------------------

class Store:
    def __init__(self, domain: str):
        self.domain = domain
        self.path = path("data/feedback") / DOMAIN_KEYS[domain] / "items.jsonl"
        self.items = _read_jsonl(self.path)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(i, ensure_ascii=False) + "\n" for i in self.items)
        tmp.replace(self.path)

    def active(self, kind: str | None = None) -> list[dict]:
        return [i for i in self.items if i["status"] == "active" and (kind is None or i["kind"] == kind)]

    def get(self, item_id: str) -> dict:
        return next(i for i in self.items if i["id"] == item_id)

    def new_id(self, kind: str) -> str:
        prefix = "L" if kind == "lesson" else "K"
        n = max([int(i["id"][1:]) for i in self.items if i["id"].startswith(prefix)] or [0])
        return f"{prefix}{n + 1:04d}"

    def set_status(self, item_id: str, status: str, by: str, note: str = "") -> dict:
        it = self.get(item_id)
        it["history"].append({"at": _now(), "action": f"{it['status']}→{status}", "by": by, "note": note})
        it["status"] = status
        self.save()
        return it


def revise_lesson(store: Store, item_id: str, trigger: str | None, text: str, by: str, note: str = "") -> dict:
    """교훈 문구 수정. 이전 문구는 history 에 남긴다."""
    it = store.get(item_id)
    assert it["kind"] == "lesson", f"{item_id} 는 교훈이 아닙니다"
    it["history"].append({"at": _now(), "action": "revised", "by": by, "note": note,
                          "before": {"trigger": it["trigger"], "text": it["text"]}})
    if trigger:
        it["trigger"] = trigger
    it["text"] = text
    store.save()
    return it


def render_item(i: dict) -> str:
    if i["kind"] == "lesson":
        where = "범용" if i["scope"] == "global" else f"주제: {', '.join(i.get('topics') or [])}"
        return f"[{i['id']}] ({where}) 상황: {i['trigger']} → 지침: {i['text']}"
    return f"[{i['id']}] (합성사례·{i.get('topic')}) Q: {i['question']} A: {i['answer']}"


def case_doc(i: dict) -> dict:
    """합성사례 → qa 컬렉션 문서 행."""
    return {"id": f"SYN-{i['id']}", "topic": i.get("topic"), "qa_topic": i.get("topic"), "situation": None,
            "question": i["question"], "answer": i["answer"], "text": f"{i['question']}\n{i['answer']}"}


# --- triage ---------------------------------------------------------------------

def _round_dir(domain: str, round_id: str) -> Path:
    return path("runs/sim") / DOMAIN_KEYS[domain] / round_id


def _dev_failures(rdir: Path) -> tuple[str, set[str]]:
    manifest = json.loads((rdir / "manifest.json").read_text(encoding="utf-8"))
    cards = {c["scenario_id"]: c for c in manifest["scenarios"] if c["pool"] == "dev"}
    blocks = []
    for sid, c in cards.items():
        jp = rdir / "judgments" / f"{sid}.json"
        if not jp.exists():
            continue
        j = json.loads(jp.read_text(encoding="utf-8"))
        assert j["pool"] == "dev"
        if not j["failures"]:
            continue
        p = c["persona"]
        head = (f"## {sid} ({c['topic']} · {c['situation']} · 고객 {p.get('age')} {p.get('temperament')} 디지털 {p.get('digital_literacy')}"
                f" · 결과 {j['outcome']} · {'합격' if j['pass'] else '불합격'} · 종합 {j['overall']})\n첫 질문: {c['opening'][:120]}")
        lines = [f"- T{f['turn']} {f['type']}: {f['note']}" for f in j["failures"]]
        blocks.append(head + "\n" + "\n".join(lines) + f"\n총평: {j['summary']}")
    return "\n\n".join(blocks), set(cards)


def _support(raw: list, dev_ids: set[str]) -> list[dict]:
    out = []
    for s in raw or []:
        m = _SUPPORT_RE.match(re.sub(r"\s+", "", str(s)))
        if m and m.group(1) in dev_ids:  # holdout·존재하지 않는 시나리오는 근거로 인정하지 않음
            out.append({"scenario_id": m.group(1), "turn": int(m.group(2)) if m.group(2) else None})
    return out


def triage(domain: str, round_id: str, retriever=None) -> dict:
    """round_id 의 dev 실패로 후보 생성 → 규칙 검사 → LLM 게이트 → 채택. 결과는 라운드 폴더 triage.json·feedback.md."""
    rdir = _round_dir(domain, round_id)
    out_path = rdir / "triage.json"
    if out_path.exists():
        return json.loads(out_path.read_text(encoding="utf-8"))
    conf = _conf()
    store = Store(domain)
    failures, dev_ids = _dev_failures(rdir)
    if retriever is None:
        from .rag.retriever import Retriever
        retriever = Retriever(domain, collections=("qa",), rerank=False)
    topics = sorted(retriever.cols["qa"].df.topic.dropna().unique())
    existing = "\n".join(render_item(i) for i in store.active()) or "(없음)"

    # 1) 후보 생성
    prompt = TRIAGE_TASK.format(policy=POLICY, global_cap=conf["global_cap"], existing=existing,
                                topics="\n".join(f"  {t}" for t in topics), failures=failures or "(없음)")
    raw = {}
    for _ in range(3):
        try:
            raw = _parse_json(chat("triage", [{"role": "user", "content": prompt}], system=TRIAGE_SYSTEM).text)
            break
        except (ValueError, json.JSONDecodeError):
            continue
    cands = []
    for x in raw.get("lessons") or []:
        cands.append({"kind": "lesson", "scope": "global" if x.get("scope") == "global" else "topic",
                      "topics": [t for t in (x.get("topics") or []) if t in topics],
                      "trigger": str(x.get("trigger") or "").strip(), "text": str(x.get("text") or "").strip(),
                      "support": _support(x.get("support"), dev_ids)})
    for x in raw.get("cases") or []:
        cands.append({"kind": "case", "topic": x.get("topic") if x.get("topic") in topics else None,
                      "question": str(x.get("question") or "").strip(), "answer": str(x.get("answer") or "").strip(),
                      "support": _support(x.get("support"), dev_ids)})

    # 2) 규칙 검사: 지지 대화 수, 숫자, 빈 내용
    for i, c in enumerate(cands, 1):
        c["cand"] = f"c{i}"
        n_sup = len({s["scenario_id"] for s in c["support"]})
        need = conf["min_support_lesson"] if c["kind"] == "lesson" else conf["min_support_case"]
        body = (c.get("trigger", "") + c.get("text", "")) if c["kind"] == "lesson" else (c["question"] + c["answer"])
        if not body.strip() or (c["kind"] == "lesson" and not c["text"]) or (c["kind"] == "case" and not c["answer"]):
            c["rule"] = "내용 없음"
        elif _DIGIT_RE.search(body):
            c["rule"] = "숫자 포함"
        elif n_sup < need:
            c["rule"] = f"지지 대화 {n_sup}건 < {need}"
        else:
            c["rule"] = None

    # 3) LLM 게이트 (규칙 통과분만)
    passed = [c for c in cands if c["rule"] is None]
    verdicts = {}
    if passed:
        lines = []
        for c in passed:
            if c["kind"] == "lesson":
                where = "범용" if c["scope"] == "global" else f"주제: {', '.join(c['topics'])}"
                lines.append(f"{c['cand']} [교훈·{where}] 상황: {c['trigger']} → 지침: {c['text']} (지지 대화 {len({s['scenario_id'] for s in c['support']})}건)")
            else:
                order, _ = retriever.candidates(c["question"], "qa", "hybrid", None, 3)
                rel = " / ".join(retriever.cols["qa"].df.text.iloc[order].str.replace("\n", " ").str[:200])
                lines.append(f"{c['cand']} [합성사례·{c['topic']}] Q: {c['question']} A: {c['answer']}\n    관련 기존 지식: {rel}")
        gp = GATE_TASK.format(policy=POLICY, existing=existing, candidates="\n".join(lines))
        for _ in range(3):
            try:
                d = _parse_json(chat("gate", [{"role": "user", "content": gp}], system=GATE_SYSTEM).text)
                verdicts = {v.get("cand"): v for v in d.get("verdicts") or []}
                break
            except (ValueError, json.JSONDecodeError):
                continue

    # 4) 채택 / 중복은 기존 항목 지지 보강
    adopted, merged = [], []
    for c in passed:
        v = verdicts.get(c["cand"]) or {"accept": False, "reason": "게이트 판정 없음"}
        c["gate"] = {"accept": bool(v.get("accept")), "reason": v.get("reason"), "duplicate_of": v.get("duplicate_of")}
        dup = v.get("duplicate_of")
        if dup and any(i["id"] == dup for i in store.active()):
            it = store.get(dup)
            it["support"] += c["support"]
            it["history"].append({"at": _now(), "action": "support+", "by": f"triage:{round_id}", "note": c["cand"]})
            merged.append(dup)
            continue
        if not c["gate"]["accept"]:
            continue
        item = {k: c[k] for k in c if k not in ("cand", "rule", "gate")}
        item.update({"id": store.new_id(c["kind"]), "source_round": round_id, "status": "active",
                     "gate": c["gate"], "created": _now(),
                     "history": [{"at": _now(), "action": "→active", "by": f"triage:{round_id}", "note": c["cand"]}]})
        store.items.append(item)
        adopted.append(item["id"])
        c["adopted_as"] = item["id"]

    # 4b) 기존 교훈 수정·퇴출: 활성 교훈이고 dev 지지 대화가 min_support_lesson 건 이상일 때만
    revised, retired = [], []
    for x in raw.get("revisions") or []:
        it = next((i for i in store.active("lesson") if i["id"] == x.get("id")), None)
        sup = _support(x.get("support"), dev_ids)
        txt = str(x.get("text") or "").strip()
        if it and txt and not _DIGIT_RE.search(txt) and len({s["scenario_id"] for s in sup}) >= conf["min_support_lesson"]:
            revise_lesson(store, it["id"], str(x.get("trigger") or "").strip() or None, txt, f"triage:{round_id}",
                          str(x.get("reason") or ""))
            it["support"] += sup
            revised.append(it["id"])
    for x in raw.get("retire") or []:
        it = next((i for i in store.active("lesson") if i["id"] == x.get("id")), None)
        sup = _support(x.get("support"), dev_ids)
        if it and it["id"] not in revised and len({s["scenario_id"] for s in sup}) >= conf["min_support_lesson"]:
            it["history"].append({"at": _now(), "action": "active→retired", "by": f"triage:{round_id}",
                                  "note": str(x.get("reason") or "")})
            it["status"] = "retired"
            retired.append(it["id"])

    # 5) 범용 지침 상한: 넘치면 이번에 들어온 것 중 지지가 적은 것부터 거절
    globals_ = [i for i in store.active("lesson") if i["scope"] == "global"]
    overflow = len(globals_) - conf["global_cap"]
    if overflow > 0:
        new_g = sorted([i for i in globals_ if i["id"] in adopted],
                       key=lambda i: len({s["scenario_id"] for s in i["support"]}))
        for it in new_g[:overflow]:
            it["status"] = "rejected"
            it["history"].append({"at": _now(), "action": "active→rejected", "by": "gate", "note": "범용 지침 상한 초과"})
            adopted.remove(it["id"])
    store.save()

    result = {"round": round_id, "at": _now(), "n_candidates": len(cands), "adopted": adopted,
              "merged_into": sorted(set(merged)), "revised": revised, "retired": retired, "candidates": cands,
              "active_after": [i["id"] for i in store.active()]}
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    _write_feedback_md(rdir, result, store)
    return result


def _write_feedback_md(rdir: Path, r: dict, store: Store) -> None:
    md = [f"# 피드백 — {r['round']} dev 실패 기반\n",
          f"후보 {r['n_candidates']}개 → 채택 {len(r['adopted'])}개, 기존 항목 보강 {len(r['merged_into'])}개, "
          f"수정 {r.get('revised') or '없음'}, 퇴출 {r.get('retired') or '없음'}. "
          "채택 항목은 다음 라운드부터 적용됩니다.\n",
          "거부하려면: `uv run python -m counsel.feedback 은행 veto <ID> \"사유\"`\n", "## 채택\n"]
    md += [f"- {render_item(store.get(i))}" for i in r["adopted"]] or ["- (없음)"]
    md += ["\n## 거절된 후보\n"]
    for c in r["candidates"]:
        if c.get("adopted_as"):
            continue
        why = c["rule"] or (f"중복 → {c['gate']['duplicate_of']}" if (c.get("gate") or {}).get("duplicate_of")
                            else (c.get("gate") or {}).get("reason"))
        desc = (f"{c['trigger']} → {c['text']}" if c["kind"] == "lesson" else f"Q: {c['question']} A: {c['answer']}")
        md.append(f"- {c['cand']} [{c['kind']}] {desc}\n  - 사유: {why}")
    md += ["\n## 현재 활성 항목\n"] + [f"- {render_item(i)}" for i in store.active()]
    (rdir / "feedback.md").write_text("\n".join(md) + "\n", encoding="utf-8")


# --- holdout 회귀 롤백 -------------------------------------------------------------

def rollback_triggers(cur: dict, prev: dict, conf: dict) -> tuple[list[str], dict]:
    """holdout 지표 비교 → 발동한 롤백 기준. 종합점수만 보면 r002 의 과잉 이관(합격 -6%p, 문제해결 -0.31)을 놓쳤다."""
    def rate(m, key):
        return m["outcomes"].get(key, 0) / m["n"]

    def tag(m, key):
        return m["failures"].get(key, 0) / m["n"]

    d = {"overall": round(cur["overall"] - prev["overall"], 3),
         "pass_rate": round(cur["pass_rate"] - prev["pass_rate"], 3),
         "resolution": round(cur["resolution"] - prev["resolution"], 3),
         "misxfer_per_dialog": round(tag(cur, "부적절이관") - tag(prev, "부적절이관"), 3),
         "transfer_rate": round(rate(cur, "이관") - rate(prev, "이관"), 3)}
    hit = []
    if d["overall"] < -conf["rollback_margin"]:
        hit.append(f"종합 {d['overall']:+} < -{conf['rollback_margin']}")
    if d["pass_rate"] <= -conf["rollback_pass_drop"]:
        hit.append(f"합격률 {d['pass_rate']:+.0%}p ≤ -{conf['rollback_pass_drop']:.0%}p")
    if d["resolution"] <= -conf["rollback_resolution_drop"]:
        hit.append(f"문제해결 {d['resolution']:+} ≤ -{conf['rollback_resolution_drop']}")
    if d["misxfer_per_dialog"] >= conf["rollback_misxfer_rise"]:
        hit.append(f"부적절이관 태그 대화당 {d['misxfer_per_dialog']:+} ≥ +{conf['rollback_misxfer_rise']}")
    if d["transfer_rate"] >= conf["rollback_transfer_rise"]:
        hit.append(f"이관 비율 {d['transfer_rate']:+.0%}p ≥ +{conf['rollback_transfer_rise']:.0%}p")
    return hit, d


def rollback_check(domain: str, round_id: str, prev_round: str) -> dict:
    """round_id 의 holdout 이 prev_round 보다 다중 기준(종합·합격률·문제해결·부적절이관·이관 비율) 중 하나라도 나빠지면
    prev_round 에서 채택된(=round_id 에 처음 적용된) 항목을 롤백."""
    conf = _conf()
    cur = json.loads((_round_dir(domain, round_id) / "metrics.json").read_text(encoding="utf-8"))["pools"]["holdout"]
    prev = json.loads((_round_dir(domain, prev_round) / "metrics.json").read_text(encoding="utf-8"))["pools"]["holdout"]
    triggers, delta = rollback_triggers(cur, prev, conf)
    store = Store(domain)
    rolled = []
    if triggers:
        for it in store.active():
            if it["source_round"] == prev_round:
                store.set_status(it["id"], "rolled_back", "auto-rollback", f"{round_id} vs {prev_round}: " + "; ".join(triggers))
                rolled.append(it["id"])
    res = {"round": round_id, "prev": prev_round, "holdout_overall": cur["overall"], "prev_overall": prev["overall"],
           "delta": delta["overall"], "deltas": delta, "triggers": triggers, "margin": conf["rollback_margin"],
           "rolled_back": rolled, "at": _now()}
    (_round_dir(domain, round_id) / "rollback.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return res


# --- CLI ------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("domain")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("triage"); t.add_argument("round")
    ls = sub.add_parser("list"); ls.add_argument("--all", action="store_true")
    v = sub.add_parser("veto"); v.add_argument("id"); v.add_argument("note", nargs="?", default="")
    rs = sub.add_parser("restore"); rs.add_argument("id"); rs.add_argument("note", nargs="?", default="")
    rb = sub.add_parser("rollback-check"); rb.add_argument("round"); rb.add_argument("prev")
    rv = sub.add_parser("revise"); rv.add_argument("id"); rv.add_argument("--text", required=True); rv.add_argument("--trigger")
    rt = sub.add_parser("retire"); rt.add_argument("id"); rt.add_argument("note", nargs="?", default="")
    a = ap.parse_args()

    if a.cmd == "triage":
        r = triage(a.domain, a.round)
        print(f"후보 {r['n_candidates']} → 채택 {r['adopted']} / 보강 {r['merged_into']}")
        print(f"상세: {_round_dir(a.domain, a.round) / 'feedback.md'}")
    elif a.cmd == "list":
        store = Store(a.domain)
        for i in (store.items if a.all else store.active()):
            print(f"{i['status']:>11}  src={i['source_round']}  지지={len({s['scenario_id'] for s in i['support']})}  {render_item(i)}")
    elif a.cmd in ("veto", "restore"):
        store = Store(a.domain)
        it = store.set_status(a.id, "vetoed" if a.cmd == "veto" else "active", "human", a.note)
        print(f"{it['id']} → {it['status']}")
    elif a.cmd == "revise":
        it = revise_lesson(Store(a.domain), a.id, a.trigger, a.text, "human")
        print(render_item(it))
    elif a.cmd == "retire":
        it = Store(a.domain).set_status(a.id, "retired", "human", a.note)
        print(f"{it['id']} → {it['status']}")
    elif a.cmd == "rollback-check":
        print(json.dumps(rollback_check(a.domain, a.round, a.prev), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
