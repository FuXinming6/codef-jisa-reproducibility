"""v3.7 action synergy analysis and safe dominance pruning."""
from __future__ import annotations

from dataclasses import replace
from itertools import combinations
from typing import Sequence

import networkx as nx

from src.defense.action import DefenseAction
from src.defense.apply_defense import apply_defense
from src.defense.constraints import PortfolioConstraints
from src.optimizer.attacker_uncertainty import ATTACKER_PROFILES, AttackerUncertaintyResult
from src.optimizer.dynamic_budget import dynamic_budget_codef_search, dynamic_budget_greedy_baseline
from src.risk.network_risk import assess_network_risk


def analyze_action_synergy(
    graph: nx.DiGraph,
    actions: Sequence[DefenseAction],
    *,
    top_pairs: int = 8,
) -> dict[str, object]:
    """Compute deterministic marginal gains, pair synergy and safe dominators."""
    base_risk = float(assess_network_risk(graph).score)
    single_gain: dict[str, float] = {}
    for action in actions:
        risk = float(assess_network_risk(apply_defense(graph, [action])).score)
        single_gain[action.id] = max(0.0, base_risk - risk)

    pair_rows: list[dict[str, object]] = []
    for left, right in combinations(actions, 2):
        pair_risk = float(assess_network_risk(apply_defense(graph, [left, right])).score)
        pair_gain = max(0.0, base_risk - pair_risk)
        synergy = pair_gain - single_gain[left.id] - single_gain[right.id]
        pair_rows.append({
            "actions": (left.id, right.id),
            "gain": round(pair_gain, 8),
            "synergy": round(synergy, 8),
        })
    pair_rows.sort(key=lambda row: (float(row["synergy"]), float(row["gain"]), tuple(row["actions"])), reverse=True)
    strongest_pairs = pair_rows[:max(1, int(top_pairs))]
    synergy_ids = {item for row in strongest_pairs for item in row["actions"]}

    dominated: dict[str, str] = {}
    for action in actions:
        action_rate = single_gain[action.id] / max(float(action.cost), 1e-9)
        for other in actions:
            if other.id == action.id:
                continue
            other_rate = single_gain[other.id] / max(float(other.cost), 1e-9)
            # Keep equal-rate ties deterministic instead of pruning both.
            if float(other.cost) <= float(action.cost) and other_rate >= action_rate + 1e-12:
                if other_rate > action_rate + 1e-12 or other.id < action.id:
                    dominated[action.id] = other.id
                    break

    # Synergy participants are protected even if their individual marginal
    # looks weak.  The remaining actions are retained when not dominated.
    candidate_ids = {
        action.id for action in actions
        if action.id not in dominated or action.id in synergy_ids
    }
    ordered_ids = tuple(sorted(
        candidate_ids,
        key=lambda action_id: (
            action_id in synergy_ids,
            single_gain[action_id] / max(float(next(item.cost for item in actions if item.id == action_id)), 1e-9),
            action_id,
        ),
        reverse=True,
    ))
    return {
        "base_risk": base_risk,
        "single_gain": {key: round(value, 8) for key, value in single_gain.items()},
        "dominance_map": dominated,
        "synergy_pairs": strongest_pairs,
        "synergy_action_ids": tuple(sorted(synergy_ids)),
        "candidate_action_ids": ordered_ids,
        "pruned_action_ids": tuple(sorted(set(dominated) - synergy_ids)),
    }


def synergy_pruning_search(
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
    max_action_candidates: int | None = None,
    top_pairs: int = 8,
    baseline: bool = False,
    candidate_action_ids: Sequence[str] | None = None,
) -> AttackerUncertaintyResult:
    """Search a synergy-protected, safely pruned candidate action pool."""
    analysis = analyze_action_synergy(graph, actions, top_pairs=top_pairs)
    candidate_ids = analysis["candidate_action_ids"]
    if candidate_action_ids is not None:
        known_ids = {action.id for action in actions}
        routed_ids = tuple(action_id for action_id in candidate_action_ids if action_id in known_ids)
        if routed_ids:
            analysis = dict(analysis)
            analysis["candidate_action_ids"] = routed_ids
            candidate_ids = routed_ids
    if baseline:
        result = dynamic_budget_greedy_baseline(
            graph, actions, total_budget, profiles=profiles,
            min_initial_budget=min_initial_budget, constraints=constraints,
            response_pool_size=response_pool_size, max_phase_one_paths=max_phase_one_paths,
            response_pool_mode=response_pool_mode,
        )
        method = "v3.7-SynergyPruning-Greedy"
    else:
        result = dynamic_budget_codef_search(
            graph, actions, total_budget, profiles=profiles,
            min_initial_budget=min_initial_budget, constraints=constraints,
            response_pool_size=response_pool_size, max_phase_one_paths=max_phase_one_paths,
            response_pool_mode=response_pool_mode, max_action_candidates=max_action_candidates,
            candidate_action_ids=candidate_ids,
        )
        method = "v3.7-SynergyPruning-CoDef"
    diagnostics = dict(result.diagnostics)
    diagnostics.update({
        "synergy_pruning": True,
        "synergy_pruning_baseline": baseline,
        "synergy_analysis": analysis,
        "synergy_analysis_json": analysis,
        "synergy_action_ids": tuple(analysis["synergy_action_ids"]),
        "pruned_action_ids": tuple(analysis["pruned_action_ids"]),
        "synergy_candidate_count": len(candidate_ids),
    })
    return replace(result, method=method, diagnostics=diagnostics)


def synergy_pruning_greedy_baseline(
    graph: nx.DiGraph,
    actions: list[DefenseAction],
    total_budget: float,
    **kwargs: object,
) -> AttackerUncertaintyResult:
    return synergy_pruning_search(graph, actions, total_budget, baseline=True, **kwargs)
