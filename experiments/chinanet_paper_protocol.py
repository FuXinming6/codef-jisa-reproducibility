"""Deterministic synthetic topologies and common settings for ChinaNet paper runs.

These are controlled simulator benchmarks, not measurements from a deployed
enterprise network. Every method receives the same graph, action catalog,
constraints, attacker profiles, budgets, path cap, and response pool.
"""
from __future__ import annotations

from math import ceil
from random import Random

import networkx as nx

from src.defense.action import DefenseAction, DefenseType
from src.defense.constraints import ActionGroupLimit, PortfolioConstraints


FAMILIES = ("er_dag", "hierarchical", "multipath")
DEFAULT_SEEDS = tuple(5201 + 37 * index for index in range(20))
DEFAULT_MAX_PATHS = 4
DEFAULT_RESPONSE_POOL = 2

_LAYERS = (
    ("internet", "Ingress"),
    ("gateway", "Gateway"),
    ("web", "Web"),
    ("api", "API"),
    ("business", "Service"),
    ("database", "Database"),
)
_LAYER_FRACTIONS = (0.08, 0.16, 0.19, 0.19, 0.19, 0.19)
_ASSET_VALUES = (0.15, 0.35, 0.55, 0.68, 0.76, 0.92)


def _layer_sizes(node_count: int) -> list[int]:
    sizes = [max(1, round(node_count * fraction)) for fraction in _LAYER_FRACTIONS]
    sizes[4] += node_count - sum(sizes)
    if sizes[4] < 1:
        raise ValueError("node_count is too small for the paper topology generator")
    return sizes


def build_topology(family: str, node_count: int, seed: int) -> nx.DiGraph:
    """Build a seeded directed attack graph with ingress-to-critical routes."""
    if family not in FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES}")
    if node_count < 30:
        raise ValueError("paper topology node_count must be at least 30")
    rng = Random(seed)
    graph = nx.DiGraph(name=f"paper-{family}-{node_count}-s{seed}")
    graph.graph.update(paper_family=family, paper_seed=seed, synthetic=True)
    layer_sizes = _layer_sizes(node_count)
    layers: list[list[str]] = []
    for layer_no, (size, (node_type, prefix), asset_value) in enumerate(zip(layer_sizes, _LAYERS, _ASSET_VALUES)):
        current: list[str] = []
        for local_no in range(size):
            node_id = f"{prefix}-{local_no + 1:03d}"
            current.append(node_id)
            graph.add_node(
                node_id,
                id=node_id,
                type=node_type,
                asset_value=asset_value + rng.uniform(-0.035, 0.035),
                exposure=1.0 if node_type == "internet" else 0.0,
                critical=False,
            )
        layers.append(current)
    critical_count = max(2, ceil(node_count * 0.04))
    for node_id in layers[-1][-critical_count:]:
        graph.nodes[node_id]["critical"] = True

    def add_edge(source: str, target: str, multiplier: float = 1.0) -> None:
        if source == target or graph.has_edge(source, target):
            return
        risk = rng.uniform(0.56, 0.86)
        if graph.nodes[target]["critical"]:
            risk = min(0.97, risk + 0.08)
        graph.add_edge(source, target, risk=min(0.98, risk * multiplier))

    if family in {"hierarchical", "multipath"}:
        for left, right in zip(layers, layers[1:]):
            # Give each next-zone node a predecessor so tiers are reachable.
            for target_no, target in enumerate(right):
                add_edge(left[target_no % len(left)], target)
            for source_no, source in enumerate(left):
                fanout = 1 if family == "hierarchical" else 2
                if family == "hierarchical" and rng.random() < 0.30:
                    fanout += 1
                for offset in range(fanout):
                    target_no = (source_no * (2 if family == "multipath" else 1) + offset + rng.randrange(len(right))) % len(right)
                    add_edge(source, right[target_no], 0.92 if family == "multipath" else 1.0)
            if family == "multipath":
                for source in left:
                    for _ in range(min(2, len(right))):
                        add_edge(source, rng.choice(right), 0.96)
        if family == "hierarchical":
            for stage in range(len(layers) - 2):
                for source in layers[stage]:
                    if rng.random() < 0.14:
                        add_edge(source, rng.choice(layers[stage + 2]), 0.82)
    else:
        ordered_nodes = [node for layer in layers for node in layer]
        for source_no in range(len(ordered_nodes) - 1):
            add_edge(ordered_nodes[source_no], ordered_nodes[source_no + 1])
        probability = min(0.075, 3.0 / node_count)
        for source_no, source in enumerate(ordered_nodes):
            for target_no in range(source_no + 2, len(ordered_nodes)):
                if rng.random() < probability:
                    add_edge(source, ordered_nodes[target_no], 0.94)
    return graph


def build_action_catalog(graph: nx.DiGraph) -> tuple[list[DefenseAction], PortfolioConstraints]:
    """Create the same-sized, topology-grounded defense catalog per instance."""
    centrality = nx.betweenness_centrality(graph, normalized=True)
    eligible_nodes = [
        node for node, data in graph.nodes(data=True)
        if data.get("type") != "internet" and not data.get("critical", False)
    ]
    ranked_nodes = sorted(eligible_nodes, key=lambda node: (-centrality.get(node, 0.0), node))
    hardening_targets = ranked_nodes[:4]
    isolation_targets = [node for node in ranked_nodes[4:] if node not in hardening_targets][:4]
    if len(isolation_targets) < 4:
        isolation_targets.extend(node for node in ranked_nodes if node not in hardening_targets and node not in isolation_targets)
    if len(hardening_targets) < 4 or len(isolation_targets) < 4:
        raise ValueError("topology does not contain enough internal defense targets")
    critical = {node for node, data in graph.nodes(data=True) if data.get("critical", False)}
    edge_targets = sorted(
        graph.edges(data=True),
        key=lambda edge: (not (edge[1] in critical), -float(edge[2].get("risk", 0.0)), -centrality.get(edge[0], 0.0), edge[0], edge[1]),
    )[:4]
    actions: list[DefenseAction] = []
    for index, node in enumerate(hardening_targets, start=1):
        actions.append(DefenseAction(f"D{index:02d}", DefenseType.NODE_HARDENING, node, 1.0, 0.28 + (index % 3) * 0.04, f"Harden {node}", 2))
    for index, (source, target, _) in enumerate(edge_targets, start=5):
        actions.append(DefenseAction(f"D{index:02d}", DefenseType.EDGE_RESTRICTION, (source, target), 2.0, 0.52 + (index % 3) * 0.04, f"Restrict {source} to {target}", 1))
    for index, node in enumerate(isolation_targets, start=9):
        actions.append(DefenseAction(f"D{index:02d}", DefenseType.NODE_ISOLATION, node, 3.0, 0.90, f"Isolate {node}", 3))
    isolation_ids = frozenset(action.id for action in actions if action.action_type == DefenseType.NODE_ISOLATION)
    constraints = PortfolioConstraints(group_limits=(ActionGroupLimit("single_isolation_window", isolation_ids, 1),))
    return actions, constraints
