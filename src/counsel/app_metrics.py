"""앱 지표 화면: 라운드 비교 · 수정·합성사례 · 내 상담 기록 (데이터는 dashboard.py)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from counsel import dashboard as dash
from counsel.config import load_config

_NUM = st.column_config.NumberColumn


def metrics_page() -> None:
    domain = load_config()["data"]["domains"][0]
    st.title("지표")
    t_rounds, t_lessons, t_mine = st.tabs(["라운드 비교", "수정·합성사례", "내 상담 기록"])
    with t_rounds:
        _rounds(domain)
    with t_lessons:
        _lessons(domain)
    with t_mine:
        _mine(domain)


# --- 라운드 비교 -----------------------------------------------------------------------

def _rounds(domain: str) -> None:
    sm = dash.round_summary(domain)
    if sm.empty:
        st.info("채점이 끝난 라운드가 없습니다. `python -m counsel.simulate 은행 --round r001` 로 라운드를 실행하세요.")
        return
    st.caption("holdout 100건(고정 벤치마크)을 같은 심판으로 채점한 값입니다. 라운드마다 달라진 것은 상담사에게 적용한 사례 기반 수정 구성입니다.")

    mode = st.radio("변화 비교 대상", ["기준선(첫 라운드)", "직전 라운드"], horizontal=True)
    last = sm.iloc[-1]
    ref = sm.iloc[0] if mode.startswith("기준선") else sm.iloc[max(len(sm) - 2, 0)]
    same = ref["라운드"] == last["라운드"]

    def diff(key: str, fmt: str, scale: float = 1.0, unit: str = "") -> str | None:
        return None if same else f"{(last[key] - ref[key]) * scale:{fmt}}{unit}"

    k = st.columns(5)
    k[0].metric(f"합격률 · {last['라운드']}", f"{last['합격률']:.0%}", diff("합격률", "+.0f", 100, "%p"), border=True)
    k[1].metric("종합점수", f"{last['종합']:.3f}", diff("종합", "+.3f"), border=True)
    k[2].metric("문제해결", f"{last['문제해결']:.2f}", diff("문제해결", "+.2f"), border=True)
    k[3].metric("해결", f"{int(last['해결'])}건", diff("해결", "+.0f"), border=True)
    k[4].metric("이관", f"{int(last['이관'])}건", diff("이관", "+.0f"), delta_color="off", border=True)
    if not same:
        st.caption(f"{ref['라운드']} 대비 {last['라운드']}의 변화입니다.")

    show = sm.assign(합격률=sm["합격률"] * 100)
    cfg = {"합격률": _NUM(format="%.0f%%"), "종합": _NUM(format="%.3f"), "평균 턴": _NUM(format="%.2f"),
           **{label: _NUM(format="%.2f") for label in dash.DIMS.values()}}
    st.dataframe(show, hide_index=True, column_config=cfg)

    a, b, c = st.columns(3)
    with a:
        st.markdown("**합격률 (%)**")
        st.bar_chart(show.set_index("라운드")[["합격률"]], height=260)
    with b:
        st.markdown("**통화 결과 (건)**")
        st.bar_chart(sm.set_index("라운드")[dash.OUTCOMES], stack=True, height=260,
                     color=["#008e65", "#e07b39", "#9aa7b5", "#d9534f", "#0f1721"])        # 해결·이관·미해결·포기·턴초과
    with c:
        st.markdown("**5항목 평균 (1~5점)**")
        long = sm.melt(id_vars="라운드", value_vars=list(dash.DIMS.values()), var_name="항목", value_name="점수")
        st.bar_chart(long, x="항목", y="점수", color="라운드", stack=False, horizontal=True, sort=False, height=260)

    cmp = dash.compare_table(domain)
    if not cmp.empty:
        st.subheader("라운드 간 차이는 유의한가")
        st.dataframe(cmp, hide_index=True, column_config={"종합 변화": _NUM(format="%+.3f"), "p(종합)": _NUM(format="%.2f"),
                                                          "p(합격률)": _NUM(format="%.2f")})
        sig = cmp[cmp["p(종합)"] < 0.05]
        if sig.empty:
            st.info("종합점수 차이는 모두 통계적으로 유의하지 않습니다 (p ≥ 0.05). 같은 100개 시나리오를 짝지어 비교한 값입니다.")
        else:
            st.success("종합점수 차이가 유의한 비교(p < 0.05): " + ", ".join(sig["비교"]))

    st.subheader("라운드 상세")
    rid = st.selectbox("라운드", list(reversed(sm["라운드"])))
    left, right = st.columns(2)
    with left:
        st.markdown("**주제별 (holdout)**")
        topics = dash.topic_table(domain, rid)
        st.dataframe(topics.assign(합격률=topics["합격률"] * 100) if not topics.empty else topics, hide_index=True,
                     column_config={"합격률": _NUM(format="%.0f%%"), "종합": _NUM(format="%.2f")})
    with right:
        st.markdown("**실패 태그 (holdout, 건)**")
        fc = dash.failure_counts(domain, rid)
        if not fc.empty:
            st.bar_chart(fc, horizontal=True, height=300)
    m = dash.load_metrics(domain, rid)
    lat = m.get("latency_s") or {}
    applied = dash.load_manifest(domain, rid).get("feedback_active") or []
    st.caption(f"적용된 피드백: {', '.join(applied) if applied else '없음 (기준선)'}")
    if lat:
        st.caption(f"턴당 시간(초): 분석 {lat.get('route')} · 검색 {lat.get('retrieve')} · 답변 {lat.get('respond')} · 합계 {lat.get('turn_total')}"
                   f" — 시뮬레이션은 대화 10개를 동시에 돌려 실제 상담보다 깁니다. 근거 인용 턴 {m.get('cited_turn_rate', 0):.1%}")

    ev = dash.latest_retrieval_eval(domain)
    if ev:
        with st.expander(f"검색 평가 ({ev[0]})"):
            st.markdown(ev[1])


# --- 사례 기반 수정·합성사례 ---------------------------------------------------------------------

def _lessons(domain: str) -> None:
    items = dash.lesson_table(domain)
    if items.empty:
        st.info("피드백 저장소가 비어 있습니다.")
        return
    st.caption("사례 기반 수정은 실패 사례에서 만든 지침으로, 상담사 프롬프트에 넣습니다(범용은 항상, 주제별은 주제가 맞을 때만). 합성사례는 지식 검색에 추가하는 QA입니다. "
               "롤백·거부·퇴출된 항목은 상담사에게 적용되지 않습니다.")
    states = list(items["상태"].unique())
    pick = st.multiselect("상태", states, default=["적용 중"] if "적용 중" in states else states)
    view = items[items["상태"].isin(pick)]
    st.caption(f"{len(view)}개 / 전체 {len(items)}개")
    for _, r in view.iterrows():
        with st.container(border=True):
            head = f"**{r['ID']}** · {r['범위']} · {r['상태']} · 출처 {r['출처']} · 지지 대화 {r['지지 대화']}건"
            st.markdown(head + (f" · 주제 {dash.md(r['주제'])}" if r["주제"] else ""))
            st.markdown(dash.md(r["내용"]))
            if r["이력"]:
                st.caption(dash.md(r["이력"]))


# --- 내 상담 기록 ----------------------------------------------------------------------

def _mine(domain: str) -> None:
    df = dash.load_sessions()
    if df.empty:
        st.info("아직 상담 기록이 없습니다. 상담 화면에서 대화를 시작하면 이곳에 쌓입니다. (runs/app_sessions)")
        return
    st.caption("이 앱에서 직접 한 상담의 집계입니다. 채점 지표가 아니라 동작 지표(시간·인용·이관)입니다.")
    n_sess = df["세션"].nunique()
    k = st.columns(6)
    k[0].metric("상담", f"{n_sess}회", border=True)
    k[1].metric("고객 발화", f"{len(df)}턴", border=True)
    k[2].metric("평균 응답", f"{df['응답(초)'].mean():.0f}초", border=True)
    k[3].metric("근거 인용 턴", f"{(df['인용 수'] > 0).mean():.0%}", border=True)
    k[4].metric("수정 적용 턴", f"{(df['수정 적용'] > 0).mean():.0%}", border=True)
    k[5].metric("이관으로 끝남", f"{df.loc[df['이관'].notna(), '세션'].nunique()}회", border=True)

    a, b = st.columns(2)
    with a:
        st.markdown("**단계별 평균 시간 (초)**")
        st.bar_chart(df[["분석(초)", "검색(초)", "답변(초)"]].mean(), horizontal=True, sort=False, height=220)
    with b:
        st.markdown("**주제 분포 (턴)**")
        st.bar_chart(df["주제"].value_counts(), horizontal=True, height=220)

    st.subheader("상담 다시 보기")
    sessions = sorted(df["세션"].unique(), reverse=True)

    def label(s: str) -> str:
        first = df[df["세션"] == s].iloc[0]
        return f"{str(first['시각'])[:16].replace('T', ' ')} · {str(first['고객'])[:24]}"

    pick = st.selectbox("상담", sessions, format_func=label)
    for _, r in df[df["세션"] == pick].iterrows():
        with st.chat_message("user", avatar="🙋"):
            st.markdown(dash.md(r["고객"]))
        with st.chat_message("assistant", avatar="🎧"):
            st.markdown(dash.md(r["상담사"]))
            sec = "" if pd.isna(r["응답(초)"]) else f" · {r['응답(초)']:.0f}초"
            st.caption(f"주제 {r['주제']} · 근거 {r['근거 수']}건 중 {r['인용 수']}건 인용 · 수정 {r['수정 적용']}개{sec}"
                       + (f" · 이관 → {r['이관']}" if pd.notna(r["이관"]) else "") + (f" · {r['구성']}" if r["구성"] else ""))
