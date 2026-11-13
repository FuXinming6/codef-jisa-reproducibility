from __future__ import annotations

from random import Random

import networkx as nx


def perturb_edge_risks(graph: nx.DiGraph, seed: int, relative_jitter: float = 0.10) -> nx.DiGraph:
    """Return a reproducible, bounded variation of an abstract risk graph."""
    if not 0 <= relative_jitter <= 1:
        raise ValueError("relative_jitter must be between 0 and 1")
    rng = Random(seed)
    varied = graph.copy()
    for source, target, data in varied.edges(data=True):
        multiplier = 1 + rng.uniform(-relative_jitter, relative_jitter)
        data["risk"] = min(1.0, max(0.0, float(data["risk"]) * multiplier))
    return varied
