"""v4.2 topology-derived defense action mapping."""
from __future__ import annotations

from typing import Iterable

import networkx as nx

from src.defense.action import DefenseAction, DefenseType


def _node_effect(data: dict[str, object]) -> float:
    if bool(data.get("critical", False)):
        return 0.60
    node_type = str(data.get("type", "server"))
    return {
        "internet": 0.45,
        "gateway": 0.35,
        "vpn": 0.48,
        "web": 0.42,
        "api": 0.40,
        "admin": 0.55,
        "database": 0.62,
        "server": 0.38,
    }.get(node_type, 0.30)


def map_actions_to_topology(
    graph: nx.DiGraph,
    *,
    max_actions: int = 24,
    edge_risk_threshold: float = 0.55,
) -> tuple[list[DefenseAction], dict[str, object]]:
    """Generate bounded, deterministic actions from graph metadata."""
    actions: list[DefenseAction] = []
    node_rows = sorted(
        graph.nodes(data=True),
        key=lambda item: (
            not bool(item[1].get("critical", False)),
            -float(item[1].get("asset_value", 1.0)),
            str(item[0]),
        ),
    )
    for node_id, data in node_rows:
        node_type = str(data.get("type", "server"))
        cost = 1.0 if node_type in {"gateway", "internet"} else 2.0
        delay = 1 if node_type in {"gateway", "internet", "web", "api"} else 2
        actions.append(DefenseAction(
            id=f"T{len(actions) + 1:02d}",
            action_type=DefenseType.NODE_HARDENING,
            target=str(node_id),
            cost=cost,
            effect=_node_effect(data),
            name=f"Harden {node_id}",
            response_delay=delay,
        ))
    edge_rows = sorted(
        graph.edges(data=True),
        key=lambda item: (-float(item[2].get("risk", 0.0)), str(item[0]), str(item[1])),
    )
    for source, target, data in edge_rows:
        risk = float(data.get("risk", 0.0))
        if risk < float(edge_risk_threshold):
            continue
        actions.append(DefenseAction(
            id=f"T{len(actions) + 1:02d}",
            action_type=DefenseType.EDGE_RESTRICTION,
            target=(str(source), str(target)),
            cost=1.0 if risk >= 0.8 else 2.0,
            effect=round(min(0.72, 0.25 + 0.45 * risk), 6),
            name=f"Restrict {source} → {target}",
            response_delay=1,
        ))
    actions = actions[:max(1, int(max_actions))]
    report = {
        "topology_action_mapping": True,
        "generated_action_count": len(actions),
        "node_action_count": sum(action.action_type == DefenseType.NODE_HARDENING for action in actions),
        "edge_action_count": sum(action.action_type == DefenseType.EDGE_RESTRICTION for action in actions),
        "edge_risk_threshold": float(edge_risk_threshold),
        "generated_action_ids": tuple(action.id for action in actions),
    }
    return actions, report
