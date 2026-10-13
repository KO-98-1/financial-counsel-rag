"""고객 에이전트: 시나리오 카드의 목적·페르소나로 상담사와 통화하는 고객 (config 의 llm.roles.customer).

목표 중심 즉흥 — 첫 발화는 카드의 opening 그대로, follow_ups 는 대화 중 반드시 묻고,
background(실제 통화의 고객 발화)는 세부 사정·본인확인 정보 참고용. 끝낼 때 [종료:해결|미해결|포기].
judge_only 필드는 보지 않는다.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .llm import chat

TEMPERAMENT = {
    "보통": "차분하고 협조적으로 말합니다.",
    "급함": "바쁘고 성격이 급합니다. 말이 짧고, 설명이 길어지면 요점만 말해 달라고 재촉합니다.",
    "불만": "이미 불편을 겪어 기분이 좋지 않습니다. 불만을 드러내고, 성의 없는 답에는 쉽게 납득하지 않습니다.",
    "디지털미숙": "앱·인터넷뱅킹이 서툽니다. 앱으로 하라는 안내에는 어떻게 하는지 되묻거나 어렵다고 말합니다.",
}
LITERACY = {
    "낮음": "앱·인터넷뱅킹을 거의 못 씁니다. 전화나 영업점 처리를 선호하고 금융 용어를 잘 모릅니다.",
    "보통": "앱으로 간단한 조회·이체 정도는 합니다.",
    "높음": "앱·인터넷뱅킹에 익숙합니다.",
}

SYSTEM = """당신은 {domain} 고객센터에 전화한 고객입니다. 상담사와 통화하며 아래 목적을 해결하려 합니다.
당신은 고객이지 상담사가 아닙니다. 상담사처럼 안내하거나 질문에 답해 주지 마세요.

[나]
- {age} {gender}, {client}, {client_type}
- 성격: {temperament_desc}
- 디지털 이해도: {literacy_desc}

[목적]
{goal}

[통화 중 꼭 물어볼 것]
{follow_ups}

[참고 — 실제 통화에서 내가 했던 말들]
{background}
(세부 사정과 본인확인 정보는 여기서 가져옵니다. 이름·번호 같은 개인정보나 [금액], [날짜] 같은 가려진 값은 그럴듯한 가짜 값으로 말합니다.)

[행동 규칙]
- 전화 통화 말투로 한 번에 1~3문장만 말합니다.
- 상담사가 질문에 제대로 답하지 않거나 모호하게 넘기면 다시 묻습니다. 성격에 맞게 반응합니다.
- 상담사가 본인확인 정보를 물으면 답합니다.
- 꼭 물어볼 것을 다 물었고 답이 충분하면 감사 인사로 통화를 끝냅니다. 상담사가 연결·이관을 안내하면 받아들이거나 거절합니다.
- 같은 질문을 두 번 넘게 했는데도 해결이 안 되면 포기하고 끝냅니다.

[출력 형식]
고객의 다음 말만 출력합니다. 통화를 끝낼 때는 마지막 줄에 다음 중 하나를 적습니다.
[종료:해결] 목적을 이뤘거나 필요한 안내를 충분히 받음
[종료:미해결] 끝내지만 원하는 답을 못 받음
[종료:포기] 답답해서 그만둠"""

TASK = """<통화 기록>
{dialog}
</통화 기록>

위 통화에서 고객(당신)의 다음 말만 출력하세요. 통화를 끝낼 때는 마지막 줄에 [종료:해결] / [종료:미해결] / [종료:포기] 중 하나를 적으세요."""

_END_RE = re.compile(r"\[\s*종료\s*[:：]\s*(해결|미해결|포기)\s*\]")
_PREFIX_RE = re.compile(r"^\s*(고객|나)\s*[:：]\s*")


@dataclass
class CustomerTurn:
    text: str
    end: str | None     # 해결 | 미해결 | 포기 | None
    raw: str


def system_prompt(card: dict) -> str:
    p = card["persona"]
    gender = {"남": "남성", "여": "여성"}.get(p.get("gender"), p.get("gender") or "")
    follow = "\n".join(f"- {q}" for q in card.get("follow_ups") or []) or "- (없음)"
    bg = "\n".join(f"- {line}" for line in (card.get("background") or [])[:20]) or "- (없음)"
    return SYSTEM.format(domain=card.get("domain", "은행"), age=p.get("age") or "", gender=gender,
                         client=p.get("client") or "", client_type=p.get("client_type") or "",
                         temperament_desc=TEMPERAMENT.get(p.get("temperament"), TEMPERAMENT["보통"]),
                         literacy_desc=LITERACY.get(p.get("digital_literacy"), LITERACY["보통"]),
                         goal=card.get("goal") or card["opening"], follow_ups=follow, background=bg)


def dialog_text(history: list[dict]) -> str:
    names = {"customer": "고객(나)", "counselor": "상담사"}
    return "\n".join(f"{names[t['speaker']]}: {t['text']}" for t in history)


def parse(raw: str) -> CustomerTurn:
    m = _END_RE.search(raw)
    end = m.group(1) if m else None
    text = _END_RE.sub("", raw).strip()
    text = _PREFIX_RE.sub("", text).strip()
    return CustomerTurn(text, end, raw)


class Customer:
    def __init__(self, card: dict):
        self.card = card
        self.system = system_prompt(card)

    def opening(self) -> CustomerTurn:
        return CustomerTurn(self.card["opening"], None, self.card["opening"])

    def next(self, history: list[dict]) -> CustomerTurn:
        """history 의 마지막은 상담사 발화. 빈 발화는 한 번 다시 묻고, 그래도 비면 미해결 종료."""
        msg = [{"role": "user", "content": TASK.format(dialog=dialog_text(history))}]
        for _ in range(2):
            turn = parse(chat("customer", msg, system=self.system).text)
            if turn.text or turn.end:
                return turn
        return CustomerTurn("", "미해결", turn.raw)
