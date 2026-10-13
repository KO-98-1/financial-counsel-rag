"""금융 상담 RAG 앱 (Streamlit) — 직접 고객이 되어 상담사와 대화하고, 라운드 지표를 본다.

run_app.bat 더블클릭, 또는 프로젝트 루트에서
  .venv\\Scripts\\python.exe -m streamlit run src\\counsel\\app.py

상담 화면은 python -m counsel.counselor 와 같은 Counselor 를 쓴다(router → QA·사례 검색 → LLM 리랭크 → 응답).
대화는 runs/app_sessions/ 에 턴 단위로 남고, 지표 화면의 '내 상담 기록'에서 집계된다.
"""
from __future__ import annotations

import json
import traceback
from datetime import datetime

import streamlit as st

from counsel import dashboard as dash
from counsel.app_metrics import metrics_page
from counsel.config import load_config, path
from counsel.counselor import Counselor, TurnResult
from counsel.ingest import DOMAIN_KEYS
from counsel.llm import set_interactive
from counsel.rag.retriever import Hit, Retriever

DOMAIN = load_config()["data"]["domains"][0]
AGES = ["20~29세", "30~39세", "40~49세", "50~59세", "60~69세", "70~79세", "80세이상"]
GENDERS, CLIENTS, CLIENT_TYPES = ["남", "여"], ["기존고객", "신규고객"], ["개인", "개인사업자"]
STAGES = {"route": "질문을 분석하는 중…", "retrieve": "근거를 검색하고 재정렬하는 중…", "respond": "답변을 쓰는 중…"}
EXAMPLES = ["대출 연장을 해야 하는데 어떻게 하면 되나요?", "자동이체가 안 빠져나갔어요. 왜 그런가요?",
            "해외여행 가는데 환전은 어떻게 하나요?", "앱에서 거래내역 조회가 안 돼요."]
CUSTOMER, COUNSELOR = "🙋", "🎧"


# --- 상담 엔진 (프로세스당 한 번 로드) ----------------------------------------------------------

@st.cache_resource(show_spinner="상담 엔진을 불러오는 중입니다. 처음 한 번만 20초~1분 걸립니다.")
def _retriever(domain: str) -> Retriever:
    kind = load_config()["counselor"]["rerank"]
    return Retriever(domain, collections=("qa", "case"), rerank=None if kind == "none" else kind)


@st.cache_resource(show_spinner=False, max_entries=4)
def _counselor(domain: str, lessons: str, rerank: bool, store_version: int, measured: tuple[str, ...]) -> Counselor:
    """교훈 구성 × 리랭크 조합마다 하나. 임베더·리랭커는 공유하고, 저장소 파일이 바뀌면(store_version) 새로 만든다."""
    r = _retriever(domain).fork(rerank)
    if lessons == "none":
        return Counselor(domain, retriever=r, use_feedback=False)
    return Counselor(domain, retriever=r, feedback_ids=list(measured) if lessons == "measured" else None)


# --- 상태 -----------------------------------------------------------------------------

def _init_state() -> None:
    s = st.session_state
    s.setdefault("log", [])               # 화면에 보이는 턴(고객·상담사)
    s.setdefault("history", [])           # 상담사에게 주는 대화 이력
    s.setdefault("last_evidence", None)   # 검색을 생략한 턴이 재사용하는 직전 근거
    s.setdefault("ended", None)           # 이관 대상 (있으면 통화 종료)
    s.setdefault("failed", None)          # 마지막 오류: {"text", "error"}
    s.setdefault("sess_file", None)


def _reset() -> None:
    for k in ("log", "history", "last_evidence", "ended", "failed", "sess_file", "pending"):
        st.session_state.pop(k, None)


def _set_pending(text: str) -> None:
    st.session_state.pending = text


def _retry() -> None:
    st.session_state.pending = st.session_state.failed["text"]
    st.session_state.failed = None


# --- 사이드바 ---------------------------------------------------------------------------

def _sidebar(latest: str | None, measured: list[str], rate: float | None, active: list[dict]) -> dict:
    locked = bool(st.session_state.log)
    with st.sidebar:
        st.header("상담 설정")
        st.caption("상담사는 아래 고객 정보만 알고 시작합니다." + (" 대화 중에는 바꿀 수 없으니 새 상담에서 바꾸세요." if locked else ""))
        age = st.selectbox("연령대", AGES, index=2, disabled=locked)
        c1, c2 = st.columns(2)
        gender = c1.selectbox("성별", GENDERS, disabled=locked)
        client = c2.selectbox("고객 구분", CLIENTS, disabled=locked)
        client_type = st.selectbox("개인 / 개인사업자", CLIENT_TYPES, disabled=locked)

        unverified = [i["id"] for i in active if i["source_round"] == latest]
        labels = {"active": f"현재 활성 교훈 ({len(active)}개)", "none": "교훈 없음 (기준선)"}
        caps = {"active": (f"{latest}에서 새로 채택된 {len(unverified)}개는 아직 holdout 검증 전입니다." if unverified
                           else "저장소의 활성 항목 전부"),
                "none": "교훈·합성사례 없이. 첫 라운드와 같은 조건"}
        if latest:
            labels["measured"] = f"{latest} 측정 구성 ({len(measured)}개)"
            caps["measured"] = f"{latest} holdout에서 측정한 구성" + (f" (합격률 {rate:.0%})" if rate is not None else "")
        opts = [o for o in ("active", "measured", "none") if o in labels]
        lessons = st.radio("적용 교훈", opts, format_func=labels.get, captions=[caps[o] for o in opts], disabled=locked)
        rerank = st.toggle("LLM 리랭크", value=True, disabled=locked,
                           help="근거 후보를 LLM 이 다시 고릅니다(완성 구성). 끄면 응답이 빨라지지만 평가한 구성과 달라집니다.")
        st.button("새 상담 시작", on_click=_reset, type="primary", width="stretch")
        roles = load_config()["llm"]["roles"]
        st.caption(f"상담사 모델: {roles['counselor']['model']} ({roles['counselor']['provider']})")
        if st.session_state.sess_file:
            st.caption(f"기록: {st.session_state.sess_file}")
    label = f"{labels[lessons]} · " + ("LLM 리랭크" if rerank else "리랭크 없음")
    persona = {"age": age, "gender": gender, "client": client, "client_type": client_type}
    return {"persona": persona, "lessons": lessons, "rerank": rerank, "label": label}


# --- 표시 -------------------------------------------------------------------------------

def _labeled(evidence: list[Hit]) -> list[tuple[str, Hit]]:
    """상담사가 본 번호표(Q1.., C1..)를 그대로 다시 붙인다."""
    n, out = {"qa": 0, "case": 0}, []
    for h in evidence:
        n[h.collection] += 1
        out.append((f"{'Q' if h.collection == 'qa' else 'C'}{n[h.collection]}", h))
    return out


def _snippet(h: Hit, limit: int = 400) -> str:
    """카드에 보일 근거 본문. QA 는 '질문\\n답변' 이라 라벨을 붙이고, 길면 앞 limit 자만 보인다."""
    text = h.text
    if h.collection == "qa" and "\n" in text:
        q, a = text.split("\n", 1)
        text = f"질문: {q}\n답변: {a}"
    return text[:limit] + ("…" if len(text) > limit else "")


def _show(e: dict) -> None:
    if e["role"] == "customer":
        with st.chat_message("user", avatar=CUSTOMER):
            st.markdown(dash.md(e["text"]))
        return
    res: TurnResult = e["res"]
    with st.chat_message("assistant", avatar=COUNSELOR):
        st.markdown(dash.md(res.reply))
        if res.transfer:
            st.warning(f"📞 통화가 **{dash.md(res.transfer)}**(으)로 이관되었습니다. 이 상담은 여기서 끝납니다.")
        _details(e)


def _details(e: dict) -> None:
    res: TurnResult = e["res"]
    total = sum(res.timings.values())
    head = f"근거·진단 · 근거 {len(res.evidence)}건 중 {len(res.cited)}건 인용 · 교훈 {len(e['lessons'])}개 · {total:.0f}초"
    with st.expander(head):
        r = res.route
        if res.evidence_carried:                       # 검색을 건너뛴 턴의 query 는 고객 발화 그대로라 보여 주지 않는다
            st.markdown(f"**주제** {dash.md(r['topic'])} · 검색 생략 (직전 근거 재사용)")
        else:
            st.markdown(f"**검색 질의** {dash.md(r['query'])}  \n**주제** {dash.md(r['topic'])} · 검색 수행")
        names = (("route", "질문 분석"), ("retrieve", "검색·재정렬"), ("respond", "답변 작성"))
        st.caption(" · ".join(f"{label} {res.timings[k]:.1f}초" for k, label in names if k in res.timings))
        if e["lessons"]:
            st.markdown("**주입된 교훈**")
            for l in e["lessons"]:
                st.markdown(f"- `{l['id']}` ({l['scope']}) {dash.md(l['trigger'])} → {dash.md(l['text'])}")
        else:
            st.caption("주입된 교훈 없음")
        st.markdown("**근거 자료** (✓ = 답변에 인용, 본문은 앞부분만)")
        for label, h in _labeled(res.evidence):
            kind = "합성사례" if h.id.startswith("SYN-") else ("QA" if h.collection == "qa" else "상담사례")
            with st.container(border=True):
                st.markdown(("✓ " if h.id in res.cited else "") + f"**{label}** · {kind} · {dash.md(h.meta.get('topic'))}")
                st.caption(dash.md(_snippet(h)))


def _empty_state() -> None:
    with st.container(border=True):
        st.markdown("**이렇게 시작해 보세요**")
        cols = st.columns(2)
        for i, ex in enumerate(EXAMPLES):
            cols[i % 2].button(ex, key=f"ex{i}", on_click=_set_pending, args=(ex,), width="stretch")


# --- 한 턴 ----------------------------------------------------------------------------------

def _turn(counselor: Counselor, cfg: dict, text: str) -> None:
    s = st.session_state
    hist = [*s.history, {"speaker": "customer", "text": text}]
    with st.chat_message("user", avatar=CUSTOMER):
        st.markdown(dash.md(text))
    with st.chat_message("assistant", avatar=COUNSELOR):
        status = st.status(STAGES["route"], expanded=False)
        try:
            res = counselor.reply(cfg["persona"], hist, s.last_evidence, on_stage=lambda stage: status.update(label=STAGES[stage]))
        except Exception as e:                         # 사용량 한도·CLI 오류·타임아웃 등
            traceback.print_exc()
            s.failed = {"text": text, "error": str(e)[:400]}
            status.update(label="답변을 만들지 못했습니다", state="error")
            st.rerun()
        status.update(label=f"완료 · {sum(res.timings.values()):.0f}초", state="complete")
    pool = {l["id"]: l for l in counselor.lessons_global + counselor.lessons_topic}
    lessons = [{"id": i, "scope": "범용" if pool[i]["scope"] == "global" else "주제",
                "trigger": pool[i]["trigger"], "text": pool[i]["text"]} for i in res.lessons if i in pool]
    s.history = [*hist, {"speaker": "counselor", "text": res.reply}]
    s.last_evidence = res.evidence
    s.ended = res.transfer
    s.log += [{"role": "customer", "text": text}, {"role": "counselor", "res": res, "lessons": lessons}]
    if not s.sess_file:
        s.sess_file = f"{dash.SESSION_DIR}/{DOMAIN_KEYS[DOMAIN]}_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
    rec = {"ts": datetime.now().isoformat(timespec="seconds"), "turn": len(s.history) // 2, "customer": text,
           "persona": cfg["persona"], "mode": {"lessons": cfg["lessons"], "rerank": cfg["rerank"], "label": cfg["label"]},
           **res.record()}
    out = path(s.sess_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    st.rerun()


# --- 화면 -------------------------------------------------------------------------------

def chat_page() -> None:
    st.html("<style>.stMainBlockContainer{max-width:980px}</style>")      # 대화는 읽기 좋은 폭으로
    _init_state()
    s = st.session_state
    latest, measured, rate = dash.latest_measured(DOMAIN)
    active = [i for i in dash.feedback_items(DOMAIN) if i["status"] == "active"]
    cfg = _sidebar(latest, measured, rate, active)

    st.title("금융 상담 RAG")
    st.caption("고객이 되어 질문해 보세요. 상담사는 은행 상담 지식(QA·사례)을 검색해 답합니다. "
               "계좌 조회·처리는 하지 못하고, 금리·수수료 같은 구체 수치는 안내하지 않습니다.")
    _retriever(DOMAIN)
    counselor = _counselor(DOMAIN, cfg["lessons"], cfg["rerank"], dash.store_version(DOMAIN), tuple(measured))

    for e in s.log:
        _show(e)
    if not s.log and not s.get("pending"):
        _empty_state()
    if s.failed:
        st.error(f"답변을 만들지 못했습니다. {s.failed['error']}\n\n보낸 질문: {s.failed['text']}")
        st.button("같은 질문 다시 보내기", on_click=_retry)
    if s.ended:
        st.info("이관되어 상담이 끝났습니다. 왼쪽의 '새 상담 시작'으로 다시 시작하세요.")

    typed = st.chat_input("고객으로서 말해 보세요", disabled=bool(s.ended))
    text = typed or s.pop("pending", None)
    if text and not s.ended:
        s.failed = None
        _turn(counselor, cfg, text)


set_interactive()
st.set_page_config(page_title="금융 상담 RAG", page_icon="🏦", layout="wide")
st.navigation([st.Page(chat_page, title="상담", icon="💬", default=True),
               st.Page(metrics_page, title="지표", icon="📊", url_path="metrics")], position="top").run()
