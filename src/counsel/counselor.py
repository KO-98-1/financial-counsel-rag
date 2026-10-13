"""3단계 상담 에이전트: router(질의 재작성) → 검색(QA·상담사례) → LLM 리랭크 → 상담사 응답.

python -m counsel.counselor 은행                      # 직접 고객이 되어 대화 (빈 줄로 종료)
python -m counsel.counselor 은행 --scenario 0         # dev 시나리오 카드의 opening·follow_ups 를 고객 발화로 재생
python -m counsel.counselor 은행 --say "적금 만기 연장하면 금리는?"

상담사는 매 호출마다 전체 대화 이력을 받는 무상태 구조라 시뮬레이션에서 턴 단위 체크포인트·재개가 쉽다.
codex(GPT-6 Luna)는 시스템 프롬프트보다 사용자 메시지에 끌려가는 경향이 있어 형식 지시는 마지막 메시지에도 넣는다.
"""
from __future__ import annotations

import argparse
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime

import numpy as np

from .config import load_config, path
from .ingest import DOMAIN_KEYS
from .llm import chat
from .rag.retriever import Hit, Retriever

Turn = dict  # {"speaker": "customer" | "counselor", "text": str}

ROUTER_SYSTEM = "당신은 은행 고객상담 시스템의 검색 질의 생성기입니다. 고객에게 답하지 말고 JSON 한 줄만 출력합니다."

ROUTER_TASK = """아래 상담 대화의 마지막 고객 발화를 보고 지식 검색 질의를 만드세요. 고객에게 답변하지 마세요.
- search: 고객이 새로운 정보나 안내를 요구하면 true. 인사·감사·맞장구, 본인확인 응답(이름·생년월일·"네 맞아요")처럼 새 검색이 필요 없으면 false.
- query: 대화 맥락을 반영해 혼자 읽어도 뜻이 통하는 한 문장 질문. 상품·업무·상황을 명시하고 개인정보는 넣지 않습니다.
- topic: 아래 목록의 한 줄을 글자 그대로 (해당 없으면 기타)
{topics}

<대화>
{dialog}
</대화>

출력(JSON 한 줄만): {{"search": true, "query": "...", "topic": "..."}}"""

COUNSELOR_SYSTEM = """당신은 {domain} 고객센터 전화 상담사입니다.

[고객 정보]
{profile}

[응답 방식]
- 전화 상담 구어체로 2~4문장. 한 번에 한 가지씩 확인하거나 안내하고, 필요하면 다음에 무엇을 할지 묻습니다.
- 목록·마크다운·이모지를 쓰지 않습니다.
- 고객 정보(연령·고객 유형)에 맞춰 설명 수준과 안내 채널(앱·영업점·전화)을 고릅니다.

[근거 사용]
- 고객 발화마다 <참고자료>가 붙습니다. [Q*]는 과거 상담의 질문·답변, [C*]는 과거 상담 사례 요약입니다. 업무 절차와 안내 방향은 참고자료를 우선 따릅니다.
- 참고자료에 없는 내용도 일반적인 은행 업무 지식이면 보충해도 됩니다.
- 금리·수수료·한도·기간(예: 만기 전 며칠)·횟수·금액 같은 구체적 수치는 말하지 않습니다. 참고자료의 수치는 다른 고객의 개별 사례라 일반 규정이 아닙니다. 수치가 필요하면 상품·가입 조건에 따라 다르다고 하고 정확히 확인하는 방법을 안내합니다.
- [금액], [날짜], [기관], [개인정보], [비공개] 같은 표기는 가려진 값입니다. 그대로 읽지 말고 일반적인 표현으로 바꿉니다.

[권한]
- 지금은 계좌 조회·처리 시스템에 접근할 수 없습니다. 조회 결과나 처리 완료를 지어내지 말고 필요한 절차(본인 확인, 준비물, 채널)를 안내합니다.
- 이 제약을 먼저 꺼내지는 않고, 고객이 정확한 내용을 확인할 수 있는 방법(앱 메뉴, 인터넷뱅킹, 영업점, 상품 설명서)을 안내합니다. 다만 고객이 직접 조회·처리를 요청하면 지금 이 상담에서는 직접 조회해 드리기 어렵다고 짧고 솔직하게 말한 뒤 대안을 제시합니다.
- 다른 부서·기관(카드사, 대출·외환 담당 부서 등)의 처리가 필요하거나 고객이 연결을 원하면 이관할 수 있습니다. 이관하면 통화가 그쪽으로 넘어가므로, 이관 전에 필요한 안내를 마치고 연결한다고 말합니다. 이관할 수 없는 업무는 영업점 방문을 안내합니다.

[출력 형식]
상담사 발화만 출력하고, 마지막 줄에 실제로 참고한 자료 번호를 적습니다. 예: [근거] Q1, C2  (참고하지 않았으면 [근거] 없음)
이관할 때는 그 앞줄에 대상을 적습니다. 예: [이관] 카드사 고객센터"""

TURN_REMINDER = ("(상담사로서 위 고객 발화에 답하세요. 구어체 2~4문장, 금리·기간·한도 등 구체적 수치 언급 금지, "
                 "마지막 줄에 [근거] 표기, 이관 시 그 앞줄에 [이관] 대상.)")

_JSON_RE = re.compile(r"\{.*\}", re.S)
_CITE_RE = re.compile(r"\n?\s*\[근거\]\s*(.*)\s*$")
_TRANSFER_RE = re.compile(r"\n?[ \t]*\[이관\][ \t]*([^\n]*)")


@dataclass
class TurnResult:
    reply: str
    cited: list[str]                       # 상담사가 [근거] 로 밝힌 문서 id
    route: dict                            # {"search", "query", "topic"}
    evidence: list[Hit] = field(default_factory=list)
    evidence_carried: bool = False         # 검색 생략 턴에서 직전 근거를 재사용했는가
    transfer: str | None = None            # [이관] 대상 (있으면 통화 종료)
    timings: dict = field(default_factory=dict)
    raw: str = ""
    lessons: list[str] = field(default_factory=list)  # 이 턴에 주입된 교훈 id

    def record(self) -> dict:
        d = asdict(self)
        d["evidence"] = [{"id": h.id, "collection": h.collection, "score": round(h.score, 3),
                          "topic": h.meta.get("topic")} for h in self.evidence]
        return d


def profile_text(persona: dict) -> str:
    """상담사가 아는 고객 정보: 연령·성별·신규/기존·개인/개인사업자 (성격·디지털 이해도는 모름)."""
    gender = {"남": "남성", "여": "여성"}.get(persona.get("gender"), persona.get("gender") or "")
    return ", ".join(x for x in (persona.get("age"), gender, persona.get("client"), persona.get("client_type")) if x)


def _dialog_text(history: list[Turn], last_n: int = 8) -> str:
    names = {"customer": "고객", "counselor": "상담사"}
    return "\n".join(f"{names[t['speaker']]}: {t['text']}" for t in history[-last_n:])


def _lesson_block(lessons: list[dict]) -> str:
    if not lessons:
        return ""
    lines = [f"- 상황: {l['trigger']} → {l['text']}" for l in lessons]
    return ("<상담 지침> (과거 상담 평가에서 나온 지침입니다. 지금 상황에 해당하면 따릅니다)\n"
            + "\n".join(lines) + "\n</상담 지침>\n\n")


def _evidence_block(hits: list[Hit], max_chars: int = 600) -> tuple[str, dict[str, str]]:
    labels, lines, n = {}, [], {"qa": 0, "case": 0}
    for h in hits:
        n[h.collection] += 1
        label = f"{'Q' if h.collection == 'qa' else 'C'}{n[h.collection]}"
        labels[label] = h.id
        lines.append(f"[{label}] ({h.meta.get('topic')}) {h.text[:max_chars]}")
    return "<참고자료>\n" + ("\n\n".join(lines) or "(관련 자료 없음)") + "\n</참고자료>", labels


class Counselor:
    def __init__(self, domain: str, retriever: Retriever | None = None, use_feedback: bool = True,
                 feedback_ids: list[str] | None = None):
        """feedback_ids 를 주면 그 항목만(상태 무관) 쓴다 — 라운드는 시작 시점 스냅샷을 고정해 재개해도 같은 조건.
        None 이면 현재 활성 항목 전부(대화 시험용)."""
        conf = load_config()["counselor"]
        self.domain = domain
        self.k = {"qa": conf["k_qa"], "case": conf["k_case"]}
        self.n_cand = conf["candidates"]
        rerank = None if conf["rerank"] == "none" else conf["rerank"]
        self.r = retriever or Retriever(domain, collections=("qa", "case"), rerank=rerank)
        self.topics = sorted(self.r.cols["qa"].df.topic.dropna().unique())
        self._cand_lock = threading.Lock()
        self.lessons_global: list[dict] = []
        self.lessons_topic: list[dict] = []
        self.feedback_ids: list[str] = []
        if use_feedback:
            self._load_feedback(feedback_ids)

    def _load_feedback(self, ids: list[str] | None = None) -> None:
        """교훈(범용·주제별)과 합성사례를 불러온다. 합성사례는 qa 컬렉션에 메모리로 추가."""
        from .feedback import Store, _conf, case_doc
        fc = _conf()
        store = Store(self.domain)
        items = store.active() if ids is None else [store.get(i) for i in ids]
        cases = [i for i in items if i["kind"] == "case"]
        if cases:
            self.r.add_docs("qa", [case_doc(i) for i in cases])
        lessons = [i for i in items if i["kind"] == "lesson"]
        self.lessons_global = [i for i in lessons if i["scope"] == "global"][:fc["global_cap"]]
        self.lessons_topic = [i for i in lessons if i["scope"] != "global"]
        if self.lessons_topic:
            self.lesson_vecs = self.r.embedder.encode([f"{i['trigger']} {i['text']}" for i in self.lessons_topic])
        self.topic_k, self.lesson_min_sim = fc["topic_k"], fc["lesson_min_sim"]
        self.feedback_ids = [i["id"] for i in self.lessons_global + self.lessons_topic + cases]

    def select_lessons(self, route: dict, history: list[Turn]) -> list[dict]:
        """범용 지침 전부 + 주제별 교훈: 라우터 주제가 교훈의 topics 에 있는 것만 후보로 하고, 그중 이번 턴과 가까운
        topic_k 개(유사도 ≥ lesson_min_sim)를 주입한다. 주제가 안 맞으면 유사도가 높아도 넣지 않는다(r002 의 L0006 과잉 주입 교훈)."""
        picked = list(self.lessons_global)
        idx = [i for i, l in enumerate(self.lessons_topic) if route["topic"] in (l.get("topics") or [])]
        if idx:
            with self._cand_lock:
                q = self.r.embedder.encode([f"{route['query']} {history[-1]['text']}"])[0]
            sims = self.lesson_vecs[idx] @ q
            picked += [self.lessons_topic[idx[j]] for j in np.argsort(-sims)[:self.topic_k] if sims[j] >= self.lesson_min_sim]
        return picked

    # --- 단계 ---
    def route(self, history: list[Turn]) -> dict:
        last = history[-1]["text"]
        prompt = ROUTER_TASK.format(topics="\n".join(f"  {t}" for t in self.topics), dialog=_dialog_text(history))
        res = chat("router", [{"role": "user", "content": prompt}], system=ROUTER_SYSTEM)
        try:
            d = json.loads(_JSON_RE.search(res.text).group(0))
        except (AttributeError, json.JSONDecodeError):
            d = {}
        query = d.get("query") if isinstance(d.get("query"), str) and d.get("query").strip() else last
        return {"search": d.get("search") is not False, "query": query,
                "topic": self._match_topic(d.get("topic")), "parsed": bool(d)}

    def _match_topic(self, s) -> str:
        """주제명에 쉼표·괄호가 있어 모델이 일부만 쓰는 경우가 있어 공백 무시 + 유일 부분 일치 허용."""
        if not isinstance(s, str) or not s.strip():
            return "기타"
        norm = lambda x: re.sub(r"\s+", "", x)
        if s in self.topics:
            return s
        hits = [t for t in self.topics if norm(s) in norm(t) or norm(t) in norm(s)]
        return hits[0] if len(hits) == 1 else "기타"

    def retrieve(self, query: str) -> list[Hit]:
        with self._cand_lock:  # 여러 대화를 병렬로 돌릴 때 Kiwi·임베딩은 한 번에 하나씩
            cands = {c: self.r.candidates(query, c, "hybrid", None, self.n_cand) for c in self.k}
        if self.r.reranker is not None:
            # 후보 추출은 이 스레드에서(Kiwi·임베딩), 리랭크(LLM 호출)만 컬렉션별 병렬
            with ThreadPoolExecutor(len(cands)) as ex:
                futs = {c: ex.submit(self.r.rerank, query, c, o) for c, (o, _) in cands.items() if o}
                cands = {c: futs[c].result() if c in futs else cands[c] for c in cands}
        hits = []
        for c, (order, scores) in cands.items():
            hits += self.r.hits(c, order[:self.k[c]], scores[:self.k[c]])
        return hits

    def respond(self, persona: dict, history: list[Turn], evidence: list[Hit],
                lessons: list[dict] | None = None) -> tuple[str, list[str], str | None, str]:
        block, labels = _evidence_block(evidence)
        block = _lesson_block(lessons or []) + block
        messages = [{"role": "user" if t["speaker"] == "customer" else "assistant", "content": t["text"]}
                    for t in history[:-1]]
        messages.append({"role": "user", "content": f"{block}\n\n<고객 발화>\n{history[-1]['text']}\n</고객 발화>\n{TURN_REMINDER}"})
        system = COUNSELOR_SYSTEM.format(domain=self.domain, profile=profile_text(persona))
        raw = chat("counselor", messages, system=system).text
        m = _CITE_RE.search(raw)
        cited = [labels[x] for x in re.findall(r"[QC]\d+", m.group(1)) if x in labels] if m else []
        reply = raw[:m.start()].strip() if m else raw.strip()
        t = _TRANSFER_RE.search(reply)
        transfer = (t.group(1).strip() or "담당 부서") if t else None
        if t:
            reply = (reply[:t.start()] + reply[t.end():]).strip()
        return reply, list(dict.fromkeys(cited)), transfer, raw

    # --- 한 턴 ---
    def reply(self, persona: dict, history: list[Turn], last_evidence: list[Hit] | None = None,
              on_stage=None) -> TurnResult:
        """history 의 마지막은 고객 발화. last_evidence 는 직전 턴 근거(검색 생략 시 재사용).
        on_stage(단계) 는 각 단계 시작 때 부르는 선택 콜백 — 단계는 "route"·"retrieve"·"respond" (앱의 진행 표시용)."""
        notify = on_stage or (lambda stage: None)
        t = {}
        notify("route")
        t0 = time.perf_counter()
        route = self.route(history)
        t["route"] = round(time.perf_counter() - t0, 2)
        carried = False
        if route["search"] or not last_evidence:
            notify("retrieve")
            t0 = time.perf_counter()
            evidence = self.retrieve(route["query"])
            t["retrieve"] = round(time.perf_counter() - t0, 2)
        else:
            evidence, carried = last_evidence, True
        lessons = self.select_lessons(route, history)
        notify("respond")
        t0 = time.perf_counter()
        reply, cited, transfer, raw = self.respond(persona, history, evidence, lessons)
        t["respond"] = round(time.perf_counter() - t0, 2)
        return TurnResult(reply, cited, route, evidence, carried, transfer, t, raw, [l["id"] for l in lessons])


# --- CLI ---------------------------------------------------------------------

def _load_card(domain: str, which: str) -> dict:
    d = path(load_config()["data"]["processed_dir"]) / DOMAIN_KEYS[domain]
    with open(d / "scenarios_dev.jsonl", encoding="utf-8") as f:
        cards = [json.loads(line) for line in f]
    if which.isdigit():
        return cards[int(which)]
    return next(c for c in cards if c["scenario_id"] == which)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("domain")
    ap.add_argument("--scenario", help="dev 시나리오 번호 또는 scenario_id — opening·follow_ups 를 재생")
    ap.add_argument("--say", action="append", help="고객 발화 (여러 번 지정 가능)")
    a = ap.parse_args()

    card = _load_card(a.domain, a.scenario) if a.scenario else None
    persona = card["persona"] if card else {"age": "40~49세", "gender": "남", "client": "기존고객", "client_type": "개인"}
    script = ([card["opening"]] + card["follow_ups"]) if card else (a.say or [])

    out = path("runs/counselor_demo") / f"{DOMAIN_KEYS[a.domain]}_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    print("모델 로드 중…", flush=True)
    c = Counselor(a.domain)
    print(f"고객 정보: {profile_text(persona)}" + (f" | 시나리오 {card['scenario_id']} ({card['topic']})" if card else ""))

    history: list[Turn] = []
    last_evidence = None
    i = 0
    while True:
        if i < len(script):
            text = script[i]
            print(f"\n고객> {text}")
        elif script:
            break
        else:
            text = input("\n고객> ").strip()
            if not text:
                break
        i += 1
        history.append({"speaker": "customer", "text": text})
        res = c.reply(persona, history, last_evidence)
        last_evidence = res.evidence
        history.append({"speaker": "counselor", "text": res.reply})
        r = res.route
        print(f"  [router] search={r['search']} topic={r['topic']} query={r['query']!r}"
              + (" (근거 재사용)" if res.evidence_carried else ""))
        print(f"  [근거] {', '.join(h.id for h in res.evidence)}  인용={res.cited}  시간={res.timings}")
        print(f"상담사> {res.reply}" + (f"\n  [이관] {res.transfer}" if res.transfer else ""), flush=True)
        with open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps({"turn": i, "customer": text, "persona": persona,
                                "scenario_id": card["scenario_id"] if card else None, **res.record()},
                               ensure_ascii=False) + "\n")
        if res.transfer:
            break
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
