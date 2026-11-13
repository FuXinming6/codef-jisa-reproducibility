from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable

import networkx as nx

from src.graph.path_search import find_attack_paths
from src.risk.path_risk import RiskPath, calculate_path_risk


@dataclass(frozen=True)
class NetworkRiskReport:
    score: float
    max_path_risk: float
    top_k_path_risk: float
    paths: tuple[RiskPath, ...]


def assess_network_risk(
    graph: nx.DiGraph,
    sources: Iterable[str] | None = None,
    top_k: int = 5,
    alpha: float = 0.5,
) -> NetworkRiskReport:
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")
    risk_paths = sorted(
        (calculate_path_risk(graph, path) for path in find_attack_paths(graph, sources=sources)),
        key=lambda item: (-item.risk, item.nodes),
    )[:top_k]
    if not risk_paths:
        return NetworkRiskReport(0.0, 0.0, 0.0, ())
    max_risk = risk_paths[0].risk
    top_k_risk = sum(path.risk for path in risk_paths)
    return NetworkRiskReport(alpha * max_risk + (1 - alpha) * top_k_risk, max_risk, top_k_risk, tuple(risk_paths))


def assess_fixed_path_risk(
    graph: nx.DiGraph,
    paths: Iterable[tuple[str, ...]],
    top_k: int = 5,
    alpha: float = 0.5,
) -> NetworkRiskReport:
    """Assess only a pre-selected set of paths for the static ablation."""
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")
    surviving = [
        calculate_path_risk(graph, path)
        for path in paths
        if all(graph.has_edge(source, target) for source, target in zip(path, path[1:]))
    ]
    risk_paths = sorted(surviving, key=lambda item: (-item.risk, item.nodes))[:top_k]
    if not risk_paths:
        return NetworkRiskReport(0.0, 0.0, 0.0, ())
    max_risk = risk_paths[0].risk
    top_k_risk = sum(path.risk for path in risk_paths)
    return NetworkRiskReport(alpha * max_risk + (1 - alpha) * top_k_risk, max_risk, top_k_risk, tuple(risk_paths))


def robust_risk(graph: nx.DiGraph, sources: Iterable[str], **kwargs: object) -> float:
    """Worst remaining network risk over independently available ingress points."""
    per_source = ingress_risk_vector(graph, sources, **kwargs)
    return max(per_source, default=0.0)


def ingress_risk_vector(graph: nx.DiGraph, sources: Iterable[str], **kwargs: object) -> tuple[float, ...]:
    """Return one risk score per ingress, in the supplied deterministic order."""
    return tuple(assess_network_risk(graph, sources=[source], **kwargs).score for source in sources)
