# graph_utils.py
from typing import Any, Dict, Tuple, List
import json
import networkx as nx
import matplotlib.pyplot as plt
from pydantic import BaseModel, ValidationError, field_validator

# ---------- JSON 검증 ----------
class Node(BaseModel):
    id: str
    label: str

class Edge(BaseModel):
    source: str
    target: str
    weight: float

class GraphPayload(BaseModel):
    directed: bool = False
    nodes: List[Node]
    edges: List[Edge]

    @field_validator("nodes")
    @classmethod
    def unique_ids(cls, v):
        ids = [n.id for n in v]
        if len(ids) != len(set(ids)):
            raise ValueError("node ids must be unique")
        return v

def load_graph_from_json_str(s: str) -> nx.Graph:
    data = json.loads(s)
    try:
        payload = GraphPayload(**data)
    except ValidationError as e:
        raise ValueError(f"Invalid JSON schema: {e}") from e

    G = nx.DiGraph() if payload.directed else nx.Graph()
    for n in payload.nodes:
        G.add_node(n.id, label=n.label)
    for e in payload.edges:
        G.add_edge(e.source, e.target, weight=float(e.weight))
    return G


# ---------- 시각화 ----------
def draw_step(
    G: nx.Graph,
    pos: Dict[Any, Tuple[float, float]],
    step: Dict[str, Any],
    start: Any,
    goal: Any,
    figsize=(6, 4)
):
    """
    색상 규칙:
      - 기본 노드: 흰색(투명 느낌) + 검은 테두리
      - 닫힌(확정) 노드: 빨간색
      - 이번에 선택된 노드: 노란색
      - 시도중인 링크(이번 iter의 trying): 파란색
      - 최종 경로(가능하면): 초록색 (goal이 확정된 이후 단계에서)
    """
    closed = step.get("closed", set())
    selected = step.get("selected_node", None)
    trying = step.get("trying", [])
    dist = step.get("dist", {})
    prev = step.get("prev", {})

    # 기본 그리기
    plt.figure(figsize=figsize)
    # 노드 색
    node_colors = []
    for n in G.nodes:
        if n == selected:
            node_colors.append("#FFD700")  # 노랑
        elif n in closed:
            node_colors.append("#FF6B6B")  # 빨강
        else:
            node_colors.append("#FFFFFF")  # 흰색(투명 느낌)
    # 노드 및 라벨
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, edgecolors="black")
    nx.draw_networkx_labels(G, pos, labels={n: G.nodes[n].get("label", n) for n in G.nodes})

    # 기본 간선(검은색)
    nx.draw_networkx_edges(G, pos, edge_color="black", width=1, alpha=0.6)

    # 시도중인 간선(파란색)
    trying_edges = [(t["i"], t["j"]) for t in trying if t.get("deleted") != "already_closed"]
    if len(trying_edges) > 0:
        nx.draw_networkx_edges(G, pos, edgelist=trying_edges, edge_color="#1f77b4", width=3)

    # 최종 경로(초록) — goal이 closed에 포함되어 있으면 prev로 복원
    if goal in closed and start in G and goal in G:
        # prev dict로 path 복원 시도
        path_edges = []
        node = goal
        ok = True
        while node != start:
            p = prev.get(node)
            if p is None:
                ok = False
                break
            path_edges.append((p, node))
            node = p
        if ok:
            nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color="#2ca02c", width=4)

    # 가중치 라벨
    edge_labels = {(u, v): f'{G[u][v].get("weight", 1)}' for u, v in G.edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.axis("off")
    plt.tight_layout()
    return plt


def spring_layout_cached(G: nx.Graph, seed: int = 42):
    # 레이아웃 고정(스텝마다 튀는 것 방지)
    return nx.spring_layout(G, seed=seed)
