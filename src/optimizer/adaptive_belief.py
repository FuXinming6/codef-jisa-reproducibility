"""v3.2 adaptive attacker-profile belief updates.

The existing robust evaluator treats the three attacker profiles symmetrically.
This extension turns the observed first path into a small, transparent belief
update: short paths increase the shortest-path likelihood, high-value targets
increase the asset-targeted likelihood, and high terminal-risk paths increase
the risk-max likelihood.  It is deliberately a scoring layer, not a learned
claim about real attackers.
"""
from __future__ import annotations

from math import exp, sqrt
from typing import Sequence

import networkx as nx

from src.defense.action import DefenseAction
from src.defense.constraints import PortfolioConstraints
from src.optimizer.attacker_uncertainty import (
    ATTACKER_PROFILES,
    AttackerProfileResult,
    AttackerUncertaintyResult,
)
from src.optimizer.dynamic_budget import dynamic_budget_codef_search, dynamic_budget_greedy_baseline
from src.optimizer.robust_two_stage import _aggregate_risk


def infer_profile_belief(
    graph: nx.DiGraph,
    profile_results: Sequence[AttackerProfileResult],
    prior: dict[str, float] | None = None,
) -> dict[str, float]:
    """Infer normalized profile weights from the observed paths."""
    prior = prior or {profile: 1.0 for profile in ATTACKER_PROFILES}
    scores: dict[str, float] = {}
    max_asset = max((float(data.get("asset_value", 1.0)) for _, data in graph.nodes(data=True)), default=1.0)
    max_risk = max((float(item.result.terminal_risk) for item in profile_results), default=1.0)
    min_hops = min((len(item.result.observed_path) for item in profile_results if item.result.observed_path), default=1)
    for item in profile_results:
        path = item.result.observed_path
        target_asset = float(graph.nodes[path[-1]].get("asset_value", 1.0)) if path and path[-1] in graph else 0.0
        short_signal = exp(-0.55 * max(0, len(path) - min_hops)) if path else 0.5
        asset_signal = target_asset / max_asset if max_asset else 0.0
        risk_signal = item.result.terminal_risk / max_risk if max_risk else 0.0
        if item.profile == "shortest_path":
            evidence = short_signal
        elif item.profile == "asset_targeted":
            evidence = asset_signal
        else:
            evidence = risk_signal
        scores[item.profile] = max(1e-6, float(prior.get(item.profile, 1.0))) * (0.15 + evidence)
    total = sum(scores.values()) or 1.0
    return {profile: round(scores.get(profile, 0.0) / total, 6) for profile in ATTACKER_PROFILES}


def _belief_objective(result: AttackerUncertaintyResult, belief: dict[str, float]) -> tuple[float, float, float, float]:
    risks = {item.profile: float(item.result.terminal_risk) for item in result.profile_results}
    weighted_mean = sum(float(belief.get(profile, 0.0)) * risks.get(profile, 0.0) for profile in ATTACKER_PROFILES)
    weighted_var = sum(float(belief.get(profile, 0.0)) * (risks.get(profile, 0.0) - weighted_mean) ** 2 for profile in ATTACKER_PROFILES)
    variation = sqrt(max(weighted_var, 0.0))
    objective = 0.5 * max(risks.values(), default=0.0) + 0.3 * weighted_mean + 0.2 * variation
    return objective, weighted_mean, variation, max(risks.values(), default=0.0)


def _reweight(result: AttackerUncertaintyResult, belief: dict[str, float], method: str) -> AttackerUncertaintyResult:
    objective, weighted_mean, variation, worst = _belief_objective(result, belief)
    diagnostics = dict(result.diagnostics)
    diagnostics.update({
        "adaptive_belief": True,
        "profile_belief": belief,
        "belief_weighted_mean": weighted_mean,
        "belief_variation": variation,
        "belief_objective": objective,
    })
    return AttackerUncertaintyResult(
        method,
        result.initial_actions,
        result.profile_results,
        weighted_mean,
        worst,
        variation,
        objective,
        diagnostics,
    )


def adaptive_belief_codef_search(
    graph: nx.DiGraph,
    actions: list[DefenseAction],
    total_budget: float,
    *,
    profiles: Sequence[str] = ATTACKER_PROFILES,
    min_initial_budget: float = 2.0,
    deployment_window: int = 1,
    response_window: int = 1,
    constraints: PortfolioConstraints | None = None,
    response_pool_size: int = 4,
    max_phase_one_paths: int | None = 12,
    response_pool_mode: str = "adaptive_diverse",
    max_action_candidates: int | None = 6,
    candidate_action_ids: Sequence[str] | None = None,
) -> AttackerUncertaintyResult:
    seed = dynamic_budget_codef_search(
        graph, actions, total_budget, profiles=profiles,
        min_initial_budget=min_initial_budget, deployment_window=deployment_window,
        response_window=response_window, constraints=constraints,
        response_pool_size=response_pool_size, max_phase_one_paths=max_phase_one_paths,
        response_pool_mode=response_pool_mode, max_action_candidates=max_action_candidates,
        candidate_action_ids=candidate_action_ids,
    )
    belief = infer_profile_belief(graph, seed.profile_results)
    result = _reweight(seed, belief, "v3.2-AdaptiveBelief-CoDef")
    baseline_seed = dynamic_budget_greedy_baseline(
        graph, actions, total_budget, profiles=profiles,
        min_initial_budget=min_initial_budget, deployment_window=deployment_window,
        response_window=response_window, constraints=constraints,
        response_pool_size=response_pool_size, max_phase_one_paths=max_phase_one_paths,
        response_pool_mode=response_pool_mode,
    )
    baseline = _reweight(baseline_seed, infer_profile_belief(graph, baseline_seed.profile_results), "v3.2-AdaptiveBelief-Greedy")
    if baseline.objective < result.objective:
        diagnostics = dict(result.diagnostics)
        diagnostics.update({"adaptive_belief_fallback": True, "fallback_method": baseline.method})
        result = AttackerUncertaintyResult(
            baseline.method, baseline.initial_actions, baseline.profile_results,
            baseline.mean_terminal_risk, baseline.worst_terminal_risk,
            baseline.risk_stddev, baseline.objective, diagnostics,
        )
    return result
