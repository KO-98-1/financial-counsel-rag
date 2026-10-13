"""심판 에이전트: 대화 단위 루브릭 채점 + 실패 태그 (config 의 llm.roles.judge, 상담사와 다른 모델).

턴별 검색 근거를 함께 보여 줘 검색 문제(검색실패)와 생성 문제(근거미활용·환각 등)를 구분하게 한다.
"""
from __future__ import annotations

import json
import re

from .llm import chat

DIMENSIONS = {
    "accuracy": "정확성 — 안내 내용이 은행 업무상 맞는가. 잘못된 정보·모범답안과 모순되는 안내가 없는가.",
    "grounding": "근거·정책 준수 — 참고자료를 적절히 활용했는가. 조회 결과·처리 완료를 지어내지 않았는가. 금리·수수료·한도·기간 같은 구체 수치를 단정하지 않았는가.",
    "resolution": "문제 해결 — 고객의 목적과 꼭 물어볼 것에 대해 실행 가능한 안내(무엇을, 어디서, 어떻게)를 했는가. 이관이 필요한 경우 적절히 이관했는가.",
    "service": "고객 응대 — 전화 상담 구어체, 공감, 고객 눈높이(연령·디지털 이해도·성격)에 맞춘 설명, 불만 대응.",
    "efficiency": "효율 — 불필요한 반복·장황함 없이 적절한 턴 안에 진행했는가.",
}
FAILURE_TYPES = {
    "검색실패": "그 턴의 참고자료에 필요한 내용이 없었다 (검색 문제)",
    "근거미활용": "참고자료에 답이 있었는데 쓰지 않거나 다르게 말했다 (생성 문제)",
    "환각": "근거 없는 사실·조회 결과·처리 완료를 지어냈다",
    "수치단정": "금리·수수료·한도·기간 등 구체 수치를 일반 규정처럼 단정했다",
    "절차누락": "본인확인·준비물·채널 등 필요한 절차 안내가 빠졌다",
    "부적절이관": "직접 안내할 수 있는데 이관했거나, 엉뚱한 곳으로 이관했다",
    "이관누락": "이관·연결이 필요하거나 고객이 원했는데 하지 않았다",
    "응대미흡": "톤·공감·눈높이 문제, 고객 질문 회피",
    "기타": "위에 없는 문제",
}

SYSTEM = """당신은 은행 고객센터 AI 상담사의 통화를 평가하는 품질 심사관입니다. 공정하고 엄격하게, 근거를 들어 채점합니다.

[상담사 운영 조건 — 이 조건을 지킨 것은 감점하지 않습니다]
- 상담사는 계좌 조회·처리 시스템에 접근할 수 없습니다(안내형). 조회·처리 대신 절차와 확인 방법을 안내하는 것이 정상입니다.
- 상담사는 금리·수수료·한도·기간 같은 구체 수치를 말하지 않도록 지시받았습니다. 모범답안에 수치가 있어도, 상담사가 수치 대신 확인 방법을 안내했다면 감점하지 않습니다. 반대로 수치를 일반 규정처럼 단정하면 감점합니다.
- 상담사는 다른 부서·기관으로 이관할 수 있고, 이관하면 통화가 끝납니다. 이관은 상담사가 직접 줄 수 있는 안내(절차·채널·준비물·확인 방법)를 먼저 제공한 뒤, 계좌별 조회·처리가 꼭 필요하거나 고객이 원할 때만 적절합니다. 안내할 수 있는 내용을 주지 않고 이관하거나 첫 응답부터 이관하면 문제를 해결한 것으로 보지 않습니다(부적절이관).
- 모범답안은 실제 상담사의 답변을 다듬은 하나의 예시입니다. 표현이 달라도 핵심 안내가 맞으면 인정합니다.

[채점 항목 — 각 1~5점]
{dimensions}
5=흠잡을 데 없음, 4=사소한 아쉬움, 3=보통(중요한 빈틈 하나), 2=중대한 문제, 1=전혀 못함

[합격 기준]
고객 목적에 대해 실질적으로 도움이 되는 안내를 했고, 고객에게 해가 될 수 있는 잘못된 정보(환각·틀린 절차·수치 단정)가 없으면 합격.
고객 질문에 답할 수 있었는데 안내 없이 이관으로 넘긴 경우는 불합격.

[실패 유형 태그]
{failure_types}

[출력]
JSON 하나만 출력합니다(코드블록 없이). 간결하게 씁니다.
- failures: 중요한 것부터 최대 4개. note 는 1~2문장으로 "무엇이 문제였고 어떻게 했어야 하는지".
- strengths, summary: 각각 한 문장.
{{"scores": {{"accuracy": 1-5, "grounding": 1-5, "resolution": 1-5, "service": 1-5, "efficiency": 1-5}},
 "pass": true/false,
 "failures": [{{"turn": 상담사 턴 번호, "type": "태그", "note": "..."}}],
 "strengths": "...",
 "summary": "..."}}"""

TASK = """[시나리오]
- 상담 주제: {topic} / 상황: {situation}
- 고객: {persona}
- 고객 목적: {goal}
- 고객이 꼭 물어볼 것: {follow_ups}

[심판 전용 참고]
- 상담 지시: {instruction}
- 모범답안: {reference_answer}
- 실제 통화 요약: {dialog_summary}

[통화 기록 — 상담사 턴마다 그 턴에 검색된 참고자료와 상담사가 인용한 자료를 함께 보여 줍니다]
{transcript}

[통화 결과] {outcome}

위 통화를 채점해 JSON 하나만 출력하세요."""

_JSON_RE = re.compile(r"\{.*\}", re.S)


def render_transcript(events: list[dict], doc_text, cited_chars: int = 300, other_chars: int = 110) -> str:
    """events: simulate 의 대화 이벤트. doc_text(collection, id) → 문서 본문.
    인용된 문서는 길게, 나머지는 앞부분만 (검색실패/근거미활용 구분에 필요한 만큼만 보여 심판 입력을 줄임)."""
    lines, n = [], 0
    for ev in events:
        if ev["type"] == "customer":
            lines.append(f"고객: {ev['text']}" + (f"  [종료:{ev['end']}]" if ev.get("end") else ""))
        elif ev["type"] == "counselor":
            n += 1
            ev_lines = []
            labels = {"qa": 0, "case": 0}
            for e in ev["evidence"]:
                labels[e["collection"]] += 1
                tag = f"{'Q' if e['collection'] == 'qa' else 'C'}{labels[e['collection']]}"
                is_cited = e["id"] in ev.get("cited", [])
                cited = " ★인용" if is_cited else ""
                body = doc_text(e["collection"], e["id"]).replace("\n", " / ")[:cited_chars if is_cited else other_chars]
                ev_lines.append(f"    [{tag}]{cited} {body}")
            route = ev.get("route") or {}
            lines.append(f"\n--- 상담사 턴 {n} (검색 질의: {route.get('query', '')!r}"
                         + (", 직전 근거 재사용" if ev.get("evidence_carried") else "") + ")")
            lines.append("  참고자료:\n" + ("\n".join(ev_lines) or "    (없음)"))
            lines.append(f"상담사: {ev['reply']}" + (f"  [이관: {ev['transfer']}]" if ev.get("transfer") else ""))
    return "\n".join(lines)


def _validate(d: dict) -> dict:
    scores = d.get("scores") or {}
    out = {"scores": {}, "pass": bool(d.get("pass")), "failures": [],
           "strengths": str(d.get("strengths") or ""), "summary": str(d.get("summary") or "")}
    for k in DIMENSIONS:
        v = scores.get(k)
        if not isinstance(v, (int, float)) or not 1 <= v <= 5:
            raise ValueError(f"점수 누락/범위 오류: {k}={v!r}")
        out["scores"][k] = float(v)
    for f in d.get("failures") or []:
        if isinstance(f, dict):
            t = f.get("type") if f.get("type") in FAILURE_TYPES else "기타"
            out["failures"].append({"turn": f.get("turn"), "type": t, "note": str(f.get("note") or "")})
    return out


def judge(card: dict, events: list[dict], outcome: str, doc_text, retries: int = 2) -> dict:
    p = card["persona"]
    ref = card["judge_only"]
    system = SYSTEM.format(dimensions="\n".join(f"- {k}: {v}" for k, v in DIMENSIONS.items()),
                           failure_types="\n".join(f"- {k}: {v}" for k, v in FAILURE_TYPES.items()))
    prompt = TASK.format(
        topic=card["topic"], situation=card["situation"],
        persona=f"{p.get('age')} {p.get('gender')}, {p.get('client')}, {p.get('client_type')}, "
                f"성격 {p.get('temperament')}, 디지털 이해도 {p.get('digital_literacy')}",
        goal=card.get("goal") or "", follow_ups=" / ".join(card.get("follow_ups") or []) or "(없음)",
        instruction=ref.get("instruction") or "", reference_answer=ref.get("reference_answer") or "",
        dialog_summary=ref.get("dialog_summary") or "",
        transcript=render_transcript(events, doc_text), outcome=outcome)
    last_err = None
    for _ in range(retries + 1):
        res = chat("judge", [{"role": "user", "content": prompt}], system=system)
        try:
            d = json.loads(_JSON_RE.search(res.text).group(0))
            out = _validate(d)
            out["overall"] = round(sum(out["scores"].values()) / len(out["scores"]), 3)
            return out
        except (AttributeError, json.JSONDecodeError, ValueError) as e:
            last_err = f"{e}: {res.text[:300]}"
    raise RuntimeError(f"심판 출력 파싱 실패: {last_err}")
