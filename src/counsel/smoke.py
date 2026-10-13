"""역할별 LLM 연결 확인: python -m counsel.smoke"""
from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor

from .config import load_config
from .llm import chat

CASES = {
    "counselor": ("당신은 은행 고객센터 상담사입니다. 두 문장 이내로 답하세요.",
                  "체크카드를 잃어버렸어요. 분실 신고는 어떻게 하나요?"),
    "customer": ("당신은 60대 은행 고객입니다. 상담사에게 할 첫 질문 한 문장만 말하세요.",
                 "주제: 예금 만기 연장"),
    "judge": ("상담 답변을 채점합니다. JSON 한 줄만 출력: {\"score\": 1~5, \"reason\": \"...\"}",
              "고객: 체크카드 분실 신고 방법?\n상담사: 앱 또는 고객센터에서 즉시 분실 신고 가능합니다."),
    "router": ("고객 발화의 은행 상담 주제를 한 단어로 답하세요.", "환율 우대 받으려면 어떻게 해요?"),
}


def run(role: str):
    system, user = CASES[role]
    try:
        r = chat(role, [{"role": "user", "content": user}], system=system, max_retries=0)
        return role, r, None
    except Exception as e:  # 연결 확인이 목적이라 모든 실패를 보고
        return role, None, e


def main(roles: list[str]) -> int:
    cfg = load_config()["llm"]["roles"]
    with ThreadPoolExecutor(len(roles)) as ex:
        results = list(ex.map(run, roles))
    failed = 0
    for role, r, err in results:
        c = cfg[role]
        print(f"\n[{role}] {c['provider']} / {c['model']}")
        if err:
            failed += 1
            print(f"  실패: {err}")
        else:
            print(f"  {r.latency_s}s  in={r.input_tokens} out={r.output_tokens} cost={r.cost_usd}")
            print(f"  → {r.text[:300]}")
    return failed


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or list(CASES)))
