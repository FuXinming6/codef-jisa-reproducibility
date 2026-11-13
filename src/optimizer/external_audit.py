"""v4.0 external-topology validation and audit reporting."""
from __future__ import annotations

from statistics import mean
from typing import Any, Sequence

import networkx as nx

from src.defense.action import DefenseAction, DefenseType


def audit_topology_payload(payload: Any) -> dict[str, object]:
    """Validate the JSON shape without constructing a graph or raising errors."""
    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(payload, dict):
        return {"valid": False, "errors": ["topology must be a JSON object"], "warnings": []}
    nodes = payload.get("nodes")
    edges = payload.get("edges")
    if not isinstance(nodes, list):
        errors.append("nodes must be an array")
        nodes = []
    if not isinstance(edges, list):
        errors.append("edges must be an array")
        edges = []
    node_ids: list[str] = []
    for index, node in enumerate(nodes):
        if isinstance(node, str):
            node_id = node.strip()
        elif isinstance(node, dict):
            node_id = str(node.get("id", "")).strip()
        else:
            node_id = ""
        if not node_id:
            errors.append(f"nodes[{index}] has no non-empty id")
        node_ids.append(node_id)
    duplicates = sorted({item for item in node_ids if item and node_ids.count(item) > 1})
    if duplicates:
        errors.append(f"duplicate node ids: {', '.join(duplicates)}")
    node_set = set(node_ids)
    edge_risks: list[float] = []
    for index, edge in enumerate(edges):
        if isinstance(edge, dict):
            source, target, raw_risk = edge.get("source"), edge.get("target"), edge.get("risk")
        elif isinstance(edge, (list, tuple)) and len(edge) >= 3:
            source, target, raw_risk = edge[0], edge[1], edge[2]
        else:
            errors.append(f"edges[{index}] needs source, target and risk")
            continue
        source, target = str(source or "").strip(), str(target or "").strip()
        if source not in node_set or target not in node_set:
            errors.append(f"edges[{index}] references an unknown node")
        try:
            risk = float(raw_risk)
        except (TypeError, ValueError):
            errors.append(f"edges[{index}] risk is not numeric")
            continue
        if not 0 <= risk <= 1:
            errors.append(f"edges[{index}] risk must be between 0 and 1")
        edge_risks.append(risk)
    ingress_count = sum(
        1 for node in nodes
        if isinstance(node, dict) and str(node.get("type", "")) == "internet"
    )
    critical_count = sum(
        1 for node in nodes
        if isinstance(node, dict) and bool(node.get("critical", False))
    )
    if not ingress_count:
        warnings.append("no node is marked type=internet")
    if not critical_count:
        warnings.append("no node is marked critical=true")
    if len(nodes) > 200:
        errors.append("topology may contain at most 200 nodes")
    if len(edges) > 1000:
        errors.append("topology may contain at most 1000 edges")
    return {
        "valid": not errors,
        "name": str(payload.get("name", "imported-topology")),
        "node_count": len(nodes),
        "edge_count": len(edges),
        "ingress_count": ingress_count,
        "critical_count": critical_count,
        "risk_mean": round(mean(edge_risks), 6) if edge_risks else 0.0,
        "risk_max": round(max(edge_risks), 6) if edge_risks else 0.0,
        "errors": errors,
        "warnings": warnings,
    }


def audit_graph(graph: nx.DiGraph, actions: Sequence[DefenseAction] = ()) -> dict[str, object]:
    """Create a machine-readable audit report for a validated graph."""
    ingress = [node for node, data in graph.nodes(data=True) if data.get("type") == "internet"]
    critical = [node for node, data in graph.nodes(data=True) if data.get("critical")]
    risks = [float(data.get("risk", 0.0)) for _, _, data in graph.edges(data=True)]
    reachable_pairs = 0
    for source in ingress:
        for target in critical:
            if nx.has_path(graph, source, target):
                reachable_pairs += 1
    action_coverage: list[dict[str, object]] = []
    for action in actions:
        if action.action_type == DefenseType.EDGE_RESTRICTION:
            source, target = action.target  # type: ignore[misc]
            available = graph.has_edge(source, target)
        else:
            available = str(action.target) in graph
        action_coverage.append({"id": action.id, "available": available})
    warnings: list[str] = []
    if not ingress:
        warnings.append("缺少互联网入口节点")
    if not critical:
        warnings.append("缺少关键资产节点")
    if ingress and critical and reachable_pairs == 0:
        warnings.append("没有入口到关键资产的可达路径，需核对拓扑方向")
    unavailable = [item["id"] for item in action_coverage if not item["available"]]
    if unavailable:
        warnings.append(f"有 {len(unavailable)} 个防御动作不适用于当前拓扑")
    return {
        "valid": True,
        "name": graph.graph.get("name", "imported-topology"),
        "node_count": graph.number_of_nodes(),
        "edge_count": graph.number_of_edges(),
        "ingress_nodes": ingress,
        "critical_assets": critical,
        "reachable_ingress_critical_pairs": reachable_pairs,
        "risk_mean": round(mean(risks), 6) if risks else 0.0,
        "risk_max": round(max(risks), 6) if risks else 0.0,
        "action_coverage": action_coverage,
        "warnings": warnings,
    }
