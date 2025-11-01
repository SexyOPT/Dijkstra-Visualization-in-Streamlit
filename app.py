# app.py
import streamlit as st
import io
import json
import pandas as pd
import networkx as nx

from dijkstra import dijkstra_steps, reconstruct_path
from graph_utils import load_graph_from_json_str, draw_step, spring_layout_cached

st.set_page_config(page_title="Dijkstra Visualizer", layout="wide")

st.title("🧭 Dijkstra Visualizer (Streamlit)")
st.caption("JSON으로 그래프 업로드 → 시작/종료 선택 → NEXT로 단계별 탐색 로그/시각화")

# ---------------------------
# Session State 초기화
# ---------------------------
ss = st.session_state
ss.setdefault("G", None)
ss.setdefault("pos", None)
ss.setdefault("start", None)
ss.setdefault("goal", None)
ss.setdefault("steps", [])
ss.setdefault("idx", 0)

# ---------------------------
# 1) JSON 업로드
# ---------------------------
st.subheader("1) 그래프 JSON 업로드")
uploaded = st.file_uploader("그래프 JSON 파일 선택", type=["json"])

# 샘플 다운 링크
with st.expander("샘플 JSON 보기/복사"):
    st.code(json.dumps({
        "directed": False,
        "nodes": [{"id":"1","label":"1"},{"id":"2","label":"2"},{"id":"3","label":"3"},{"id":"4","label":"4"}],
        "edges": [
            {"source": "1", "target": "2", "weight": 1.5},
            {"source": "1", "target": "3", "weight": 2.1},
            {"source": "2", "target": "3", "weight": 0.9},
            {"source": "2", "target": "4", "weight": 2.0},
            {"source": "3", "target": "4", "weight": 1.2}
        ]
    }, indent=2), language="json")

colA, colB = st.columns([1, 1])

with colA:
    if uploaded is not None:
        raw = uploaded.read().decode("utf-8")
        try:
            G = load_graph_from_json_str(raw)
            ss.G = G
            ss.pos = spring_layout_cached(G, seed=7)
            st.success("그래프 로드 성공!")
            st.write(f"노드 수: {G.number_of_nodes()}, 간선 수: {G.number_of_edges()}")
        except Exception as e:
            st.error(f"JSON 파싱/검증 실패: {e}")
            ss.G = None
            ss.steps = []
            ss.idx = 0
    else:
        st.info("샘플로 진행하려면 위 expander의 JSON을 파일로 저장해 업로드하세요.")

with colB:
    if ss.G is not None:
        st.pyplot(draw_step(ss.G, ss.pos, {
            "iter": 0, "trying": [], "selected_node": None,
            "closed": set(), "dist": {}, "prev": {}
        }, start=None, goal=None, figsize=(6.5, 4.2)))

# ---------------------------
# 1.2) 시작/종료 노드 선택
# ---------------------------
st.subheader("2) 시작/종료 노드 선택")
if ss.G is not None:
    nodes = list(ss.G.nodes)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        ss.start = st.selectbox("Start (시작 노드)", nodes, index=0 if nodes else None)
    with col2:
        # goal은 선택 옵션(없으면 전체 수행)
        goal_label = "Goal (종료 노드; 선택)"
        goal_choices = ["(선택 안 함)"] + nodes
        goal_sel = st.selectbox(goal_label, goal_choices, index=0)
        ss.goal = None if goal_sel == "(선택 안 함)" else goal_sel
    with col3:
        if st.button("▶️ 준비/초기화"):
            try:
                ss.steps = dijkstra_steps(ss.G, ss.start, ss.goal)
                ss.idx = 0
                st.success(f"스텝 {len(ss.steps)}개 생성 완료!")
            except Exception as e:
                st.error(f"스텝 생성 실패: {e}")
                ss.steps = []
                ss.idx = 0

# ---------------------------
# 3) NEXT 버튼으로 Iter 진행 + 로그
# ---------------------------
st.subheader("3) Iter-by-Iter 시각화 & 로그")
if ss.G is not None and ss.steps:
    c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
    with c1:
        if st.button("⏮ Reset"):
            ss.idx = 0
    with c2:
        if st.button("◀ Prev", disabled=(ss.idx <= 0)):
            ss.idx = max(0, ss.idx - 1)
    with c3:
        if st.button("Next ▶", disabled=(ss.idx >= len(ss.steps) - 1)):
            ss.idx = min(len(ss.steps) - 1, ss.idx + 1)
    with c4:
        st.markdown(f"**현재 Iter:** {ss.steps[ss.idx]['iter']} / {len(ss.steps)}")

    step = ss.steps[ss.idx]

    # 좌측: 그래프, 우측: 로그
    vcol1, vcol2 = st.columns([5, 4])
    with vcol1:
        st.pyplot(draw_step(ss.G, ss.pos, step, ss.start, ss.goal, figsize=(8.5, 6)))

    with vcol2:
        st.markdown("**로그**")
        # trying 테이블 (i->j, perm, link, temp, selected, deleted)
        if step["trying"]:
            df = pd.DataFrame([{
                "i": t["i"],
                "j": t["j"],
                "Permanent": f"{step['dist'].get(t['i'], float('inf')):.4f}" if step['dist'] else t["perm_cost"],
                "Link Cost": f"{t['link_cost']:.4f}",
                "Temp Cost": f"{t['temp_cost']:.4f}",
                "Selected": t.get("selected", "N/A"),
                "Deleted": t.get("deleted", "NA")
            } for t in step["trying"]])

            st.dataframe(df, use_container_width=True, height=280)

            # 사용자가 제시한 예시 형태의 텍스트 로그도 같이 출력
            lines = []
            for _, r in df.iterrows():
                lines.append(
                    f"i {r['i']} -> j {r['j']} | Permanent : {r['Permanent']} | "
                    f"Link Cost: {r['Link Cost']} | Temp Cost {r['Temp Cost']} | "
                    f"Selected: {r['Selected']}| Deleted: {r['Deleted']}"
                )
            st.code("\n".join(lines))
        else:
            st.info("이번 Iter에서 완화 시도가 없었습니다.")

        # 상태 요약
        st.markdown("---")
        selected = step.get("selected_node", None)
        closed = sorted(list(step.get("closed", [])))
        st.write(f"**Selected Node:** {selected if selected is not None else 'N/A'}")
        st.write(f"**Closed Nodes:** {closed}")

        # goal 확정 시 최종 경로 및 총 비용
        if ss.goal is not None and ss.goal in step.get("closed", set()):
            path = reconstruct_path(step["prev"], ss.start, ss.goal)
            if path:
                st.success(f"최종 경로: {path} | 총 비용: {step['dist'][ss.goal]:.4f}")
            else:
                st.warning("goal까지의 경로가 없습니다.")

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; font-size: 0.8em;'>"
    "2025. NOV, Jeongmin Andy Eom | SCOA / GSL | Inha University"
    "</div>",
    unsafe_allow_html=True
)
