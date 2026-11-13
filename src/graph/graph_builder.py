from __future__ import annotations

from pathlib import Path

import networkx as nx
import yaml


def build_network_from_config(path: str | Path) -> nx.DiGraph:
    """Build an attack graph from the project YAML scenario format."""
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "nodes" not in payload or "edges" not in payload:
        raise ValueError(f"Invalid network config: {config_path}")

    graph = nx.DiGraph(name=payload.get("name", config_path.stem))
    for node in payload["nodes"]:
        node_id = node["id"]
        graph.add_node(
            node_id,
            id=node_id,
            type=node["type"],
            asset_value=float(node.get("asset_value", 1.0)),
            exposure=float(node.get("exposure", 0.0)),
            critical=bool(node.get("critical", False)),
        )
    for edge in payload["edges"]:
        source, target = edge["source"], edge["target"]
        if source not in graph or target not in graph:
            raise ValueError(f"Edge {source!r} -> {target!r} references an unknown node")
        risk = float(edge["risk"])
        if not 0 <= risk <= 1:
            raise ValueError(f"Edge risk must be between 0 and 1: {source!r} -> {target!r}")
        graph.add_edge(source, target, risk=risk)
    return graph


def build_reference_network() -> nx.DiGraph:
    """Return the original 10-node deterministic v0.1 reference graph."""
    graph = nx.DiGraph(name="reference-enterprise")
    nodes = [
        ("Internet", "internet", 0.1, False),
        ("Remote-Access", "internet", 0.2, False),
        ("Gateway-01", "gateway", 0.4, False),
        ("Web-01", "web", 0.7, False),
        ("API-01", "api", 0.8, False),
        ("VPN-01", "vpn", 0.6, False),
        ("Office-01", "office", 0.5, False),
        ("File-01", "server", 0.65, False),
        ("Admin-01", "admin", 0.95, True),
        ("DB-01", "database", 1.0, True),
    ]
    for node_id, node_type, asset_value, critical in nodes:
        graph.add_node(
            node_id,
            id=node_id,
            type=node_type,
            asset_value=asset_value,
            exposure=1.0 if node_type == "internet" else 0.0,
            critical=critical,
        )

    edges = [
        ("Internet", "Gateway-01", 0.88),
        ("Gateway-01", "Web-01", 0.82),
        ("Web-01", "API-01", 0.80),
        ("API-01", "DB-01", 0.76),
        ("Internet", "VPN-01", 0.70),
        ("Remote-Access", "VPN-01", 0.86),
        ("VPN-01", "Office-01", 0.78),
        ("Office-01", "Admin-01", 0.73),
        ("Admin-01", "DB-01", 0.91),
        ("Office-01", "File-01", 0.66),
        ("File-01", "DB-01", 0.62),
        ("Web-01", "Admin-01", 0.42),
    ]
    for source, target, risk in edges:
        graph.add_edge(source, target, risk=risk)
    return graph


def ingress_nodes(graph: nx.DiGraph) -> list[str]:
    return [node for node, data in graph.nodes(data=True) if data.get("type") == "internet"]


def critical_assets(graph: nx.DiGraph) -> list[str]:
    return [node for node, data in graph.nodes(data=True) if data.get("critical", False)]
