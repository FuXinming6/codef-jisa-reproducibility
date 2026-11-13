"""v5.3 multi-scale guarded proxy search."""
from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import networkx as nx

from src.defense.action import DefenseAction
from src.defense.constraints import PortfolioConstraints
from src.optimizer.attacker_uncertainty import ATTACKER_PROFILES, AttackerUncertaintyResult, attacker_uncertainty_codef_search
from src.optimizer.multi_scale_proxy import MultiScaleProxyRegistry
from src.optimizer.safe_proxy import safe_proxy_candidate_order


def multiscale_codef_search(
    graph: nx.DiGraph,
    actions: list[DefenseAction],
    registry: MultiScaleProxyRegistry,
    initial_budget: float = 4.0,
    reserve_budget: float = 4.0,
    *,
    constraints: PortfolioConstraints | None = None,
    response_pool_size: int = 2,
    max_phase_one_paths: int | None = 4,
    max_action_candidates: int | None = 6,
    profiles: Sequence[str] = ATTACKER_PROFILES,
) -> AttackerUncertaintyResult:
    """Compare proxy-routed and full candidates, retaining the lower objective."""
    proxy = registry.for_graph(graph, actions)
    candidate_ids, gate = safe_proxy_candidate_order(graph, actions, proxy, top_k=max_action_candidates or 6)
    common = {
        "graph": graph,
        "profiles": profiles,
        "actions": actions,
        "initial_budget": initial_budget,
        "reserve_budget": reserve_budget,
        "constraints": constraints,
        "response_pool_size": response_pool_size,
        "max_phase_one_paths": max_phase_one_paths,
        "response_pool_mode": "adaptive_diverse",
        "max_action_candidates": max_action_candidates,
    }
    full = attacker_uncertainty_codef_search(**common, method_name="v5.3-Full-CoDef")
    routed = full
    if candidate_ids:
        routed = attacker_uncertainty_codef_search(
            **common,
            candidate_action_ids=candidate_ids,
            method_name="v5.3-ProxyRouted-CoDef",
        )
    selected = min(
        (full, routed),
        key=lambda result: (float(result.objective), float(result.worst_terminal_risk), tuple(action.id for action in result.initial_actions)),
    )
    diagnostics = dict(selected.diagnostics)
    diagnostics.update({
        "multiscale_proxy": True,
        "multiscale_selected": selected.method,
        "multiscale_quality_guard": True,
        "multiscale_full_objective": float(full.objective),
        "multiscale_routed_objective": float(routed.objective),
        "multiscale_checkpoint": proxy.checkpoint.name,
        "multiscale_gate": gate,
    })
    return replace(selected, method="v5.3-MultiScaleProxy-CoDef", diagnostics=diagnostics)
