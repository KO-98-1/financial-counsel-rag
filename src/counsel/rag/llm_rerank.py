"""LLM 리랭커: 후보 문서 목록을 LLM 에 보여 주고 관련도 순 번호를 받는다 (config 의 llm.roles.reranker).

bge-reranker 와 같은 score(query, docs) 인터페이스라 Retriever 에 그대로 끼운다.
선택된 문서는 순위대로 높은 점수, 나머지는 0점 → 안정 정렬로 RRF 순서를 유지한다.
"""
from __future__ import annotations

import json
import re

import numpy as np

from ..llm import chat

SYSTEM = """당신은 은행 고객상담 지식 검색 결과를 재정렬합니다.
고객 질문에 답할 때 직접 근거가 되는 후보 문서를 관련도 높은 순으로 고르세요.
- 질문의 상품·업무·상황(예: 대출 만기 연장, 적금 해지, 해외 송금)이 일치하는 문서를 우선합니다.
- 단어만 겹치고 업무가 다른 문서는 제외합니다.
- 출력은 JSON 한 줄만: {{"ranking": [번호, ...]}} — 관련도 높은 순, 최대 {top}개, 관련 문서가 없으면 빈 목록."""

_JSON_RE = re.compile(r"\{.*\}", re.S)


def parse_ranking(text: str, n: int) -> list[int]:
    """LLM 출력에서 1-based 번호 목록을 꺼내 0-based 로 (범위 밖·중복 제거)."""
    nums = None
    m = _JSON_RE.search(text or "")
    if m:
        try:
            nums = json.loads(m.group(0)).get("ranking")
        except (json.JSONDecodeError, AttributeError):
            nums = None
    if not isinstance(nums, list):
        nums = re.findall(r"\d+", text or "")
    out: list[int] = []
    for x in nums:
        try:
            i = int(x) - 1
        except (TypeError, ValueError):
            continue
        if 0 <= i < n and i not in out:
            out.append(i)
    return out


class LLMReranker:
    def __init__(self, role: str = "reranker", top: int = 10, max_chars: int = 600):
        self.role, self.top, self.max_chars = role, top, max_chars
        self.parse_failures = 0

    def score(self, query: str, docs: list[str]) -> np.ndarray:
        if not docs:
            return np.zeros(0, dtype=np.float32)
        body = "\n\n".join(f"[{i + 1}] {d[:self.max_chars]}" for i, d in enumerate(docs))
        prompt = f"<질문>\n{query}\n</질문>\n\n<후보 {len(docs)}건>\n{body}\n</후보>"
        res = chat(self.role, [{"role": "user", "content": prompt}], system=SYSTEM.format(top=self.top))
        ranking = parse_ranking(res.text, len(docs))[:self.top]
        if not ranking and "[]" not in res.text:
            self.parse_failures += 1
        scores = np.zeros(len(docs), dtype=np.float32)
        for rank, i in enumerate(ranking):
            scores[i] = len(ranking) - rank
        return scores
