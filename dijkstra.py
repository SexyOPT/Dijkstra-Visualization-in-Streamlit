# dijkstra.py
from typing import Any, Dict, List, Tuple, Iterator, Optional
import math
import networkx as nx

Step = Dict[str, Any]

def dijkstra_steps(G: nx.Graph, start: Any, goal: Optional[Any] = None) -> List[Step]:
    """
    다익스트라 과정을 '스텝' 단위로 기록해 반환.
    각 스텝에는 다음 정보가 담긴다:
      - iter: int (1부터)
      - trying: List[Dict]  # 이번 반복에서 시도한 (u->v) 완화 로깅
         * i, j, link_cost, perm_cost, temp_cost, selected, deleted
      - selected_node: Any  # 이번 반복에서 permanent로 확정한 노드
      - closed: set         # 지금까지 확정된 노드 집합
      - dist: dict          # 현재까지의 최단거리 추정
      - prev: dict          # 직전 노드
    """
    if start not in G:
        raise ValueError(f"start {start} not in graph")

    dist = {n: math.inf for n in G.nodes}
    prev = {n: None for n in G.nodes}
    dist[start] = 0.0

    closed = set()  # 확정(Permanent) 노드
    open_set = set(G.nodes)

    steps: List[Step] = []
    iteration = 0

    while open_set:
        iteration += 1

        # 아직 확정 안 된(open_set) 노드 중 dist가 최소인 노드 선택
        current = min(open_set, key=lambda n: dist[n])

        # 로깅용 컨테이너
        trying_logs = []

        # 만약 도달 불가(무한대)면 더 진행 의미 없음
        if math.isinf(dist[current]):
            # 남은 노드가 모두 분리 그래프
            steps.append({
                "iter": iteration,
                "trying": [],
                "selected_node": None,
                "closed": set(closed),
                "dist": dict(dist),
                "prev": dict(prev)
            })
            break

        # 이웃 완화(relaxation) 시도
        for nbr in G.neighbors(current):
            if nbr in closed:
                # 이미 확정된 이웃은 스킵 (deleted=N/A 대체)
                trying_logs.append({
                    "i": current,
                    "j": nbr,
                    "link_cost": G[current][nbr].get("weight", 1.0),
                    "perm_cost": dist[current],
                    "temp_cost": dist[current] + G[current][nbr].get("weight", 1.0),
                    "selected": "N/A",
                    "deleted": "already_closed"
                })
                continue

            w = G[current][nbr].get("weight", 1.0)
            new_cost = dist[current] + w

            # 로깅(완화 '시도' 자체를 기록)
            log_item = {
                "i": current,
                "j": nbr,
                "link_cost": w,
                "perm_cost": dist[current],
                "temp_cost": new_cost,
                "selected": "N/A",
                "deleted": "NA"
            }

            if new_cost < dist[nbr]:
                dist[nbr] = new_cost
                prev[nbr] = current
            trying_logs.append(log_item)

        # current 확정
        closed.add(current)
        open_set.remove(current)

        steps.append({
            "iter": iteration,
            "trying": trying_logs,
            "selected_node": current,
            "closed": set(closed),
            "dist": dict(dist),
            "prev": dict(prev)
        })

        # goal이 있고, 방금 goal을 확정했다면 종료
        if goal is not None and current == goal:
            break

    return steps


def reconstruct_path(prev: Dict[Any, Any], start: Any, goal: Any) -> List[Any]:
    path = []
    node = goal
    while node is not None:
        path.append(node)
        if node == start:
            break
        node = prev[node]
    if not path or path[-1] != start:
        return []  # 경로 없음
    return list(reversed(path))
