from __future__ import annotations

from dataclasses import dataclass
from math import prod

import networkx as nx


@dataclass(frozen=True)
class RiskPath:
    nodes: tuple[str, ...]
    probability: float
    risk: float


def calculate_path_risk(graph: nx.DiGraph, path: tuple[str, ...]) -> RiskPath:
    if len(path) < 2:
        raise ValueError("An attack path must contain at least one edge.")
    probability = prod(graph[source][target]["risk"] for source, target in zip(path, path[1:]))
    target_value = float(graph.nodes[path[-1]].get("asset_value", 1.0))
    return RiskPath(nodes=path, probability=probability, risk=probability * target_value)
