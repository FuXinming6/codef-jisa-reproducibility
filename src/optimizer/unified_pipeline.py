"""v4.1 unified decision pipeline.

This is the mainline composition of the previously validated modules.  It
keeps each stage bounded and auditable instead of exposing five unrelated
solver buttons as if they were five different products.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import networkx as nx

from src.defense.action import DefenseAction
from src.defense.constraints import PortfolioConstraints
from src.optimizer.attacker_uncertainty import ATTACKER_PROFILES, AttackerUncertaintyResult
from src.optimizer.external_audit import audit_graph
from src.optimizer.pareto_policy import pareto_policy_search
from src.optimizer.synergy_pruning import analyze_action_synergy
from src.optimizer.topology_stress import _evaluate_stress_policy, stress_graphs
from src.optimizer.uncertainty_calibration import uncertainty_calibrated_search


def unified_pipeline_search(
    graph: nx.DiGraph,
    actions: list[DefenseAction],
    total_budget: float,
    *,
    profiles: Sequence[str] = ATTACKER_PROFILES,
    min_initial_budget: float = 2.0,
    constraints: PortfolioConstraints | None = None,
    response_pool_size: int = 4,
    max_phase_one_paths: int | None = 12,
    response_pool_mode: str = "adaptive_diverse",
    max_action_candidates: int | None = 6,
    candidate_action_ids: Sequence[str] | None = None,
) -> AttackerUncertaintyResult:
    """Run audit -> synergy -> calibration -> Pareto -> stress in one call."""
    audit = audit_graph(graph, actions)
    synergy = analyze_action_synergy(graph, actions)
    candidate_ids = synergy["candidate_action_ids"]
    if candidate_action_ids is not None:
        known_ids = {action.id for action in actions}
        routed_ids = tuple(action_id for action_id in candidate_action_ids if action_id in known_ids)
        if routed_ids:
            candidate_ids = routed_ids

    calibrated = uncertainty_calibrated_search(
        graph,
        actions,
        total_budget,
        profiles=profiles,
        min_initial_budget=min_initial_budget,
        constraints=constraints,
        response_pool_size=response_pool_size,
        max_phase_one_paths=max_phase_one_paths,
        response_pool_mode=response_pool_mode,
        max_action_candidates=max_action_candidates,
        candidate_action_ids=candidate_ids,
    )
    pareto = pareto_policy_search(
        graph,
        actions,
        total_budget,
        profiles=profiles,
        min_initial_budget=min_initial_budget,
        constraints=constraints,
        response_pool_size=response_pool_size,
        max_phase_one_paths=max_phase_one_paths,
        response_pool_mode=response_pool_mode,
        max_action_candidates=max_action_candidates,
        candidate_action_ids=candidate_ids,
    )

    stress_cases = stress_graphs(graph)
    policy_candidates = [calibrated, pareto]
    stress_scored: list[tuple[str, AttackerUncertaintyResult]] = []
    for source, seed in (("calibration", calibrated), ("pareto", pareto)):
        reserve = float(seed.diagnostics.get("selected_reserve_budget", total_budget))
        stress_scored.append((source, _evaluate_stress_policy(
            stress_cases,
            actions,
            frozenset(action.id for action in seed.initial_actions),
            reserve,
            profiles=profiles,
            constraints=constraints,
            response_pool_size=response_pool_size,
            max_phase_one_paths=max_phase_one_paths,
            response_pool_mode=response_pool_mode,
            baseline=False,
        )))
    selected_source, selected = min(
        stress_scored,
        key=lambda item: (item[1].objective, item[1].worst_terminal_risk, tuple(a.id for a in item[1].initial_actions)),
    )
    diagnostics = dict(selected.diagnostics)
    diagnostics.update({
        "unified_pipeline": True,
        "unified_pipeline_stages": ("audit", "synergy", "calibration", "pareto", "stress"),
        "unified_pipeline_selected": selected_source,
        "unified_pipeline_audit": audit,
        "unified_pipeline_candidate_ids": tuple(candidate_ids),
        "unified_pipeline_synergy_count": len(synergy["synergy_action_ids"]),
        "unified_pipeline_stage_objectives": {
            "calibration": calibrated.objective,
            "pareto": pareto.objective,
            "stress_calibration": stress_scored[0][1].objective,
            "stress_pareto": stress_scored[1][1].objective,
        },
        "unified_pipeline_policy_candidates": [
            {"source": source, "initial_ids": [a.id for a in result.initial_actions], "objective": result.objective}
            for source, result in stress_scored
        ],
    })
    return replace(selected, method="v4.1-UnifiedPipeline-CoDef", diagnostics=diagnostics)
