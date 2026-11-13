from __future__ import annotations

from collections.abc import Iterable

import networkx as nx

from src.graph.graph_builder import critical_assets, ingress_nodes


def find_attack_paths(
    graph: nx.DiGraph,
    sources: Iterable[str] | None = None,
    targets: Iterable[str] | None = None,
    max_hops: int = 6,
) -> list[tuple[str, ...]]:
    """Find simple ingress-to-critical-asset paths in the current graph state."""
    source_nodes = sorted(sources) if sources is not None else sorted(ingress_nodes(graph))
    target_nodes = sorted(targets) if targets is not None else sorted(critical_assets(graph))
    paths: list[tuple[str, ...]] = []
    for source in source_nodes:
        if source not in graph:
            continue
        for target in target_nodes:
            if target not in graph or source == target:
                continue
            paths.extend(tuple(path) for path in nx.all_simple_paths(graph, source, target, cutoff=max_hops))
    # NetworkX preserves insertion order, but callers may supply sets and
    # graph construction order can vary across loaders. A stable path order
    # makes equal-risk tie handling reproducible across processes.
    return sorted(paths)
