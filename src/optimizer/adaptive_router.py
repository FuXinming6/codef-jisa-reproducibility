"""v3.5 scenario-adaptive routing across the validated policy variants.

Different network situations reward different operational policies: a delay
uncertain site may prefer v3.3, while a tight maintenance window may prefer
v3.4.  The router does not guess from a hand-written label.  It generates a
small portfolio of candidates, scores every candidate with the same robust
risk-plus-schedule objective, and selects the best one with a Greedy guard.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import networkx as nx

from src.defense.action import DefenseAction
from src.defense.constraints import PortfolioConstraints
from src.optimizer.attacker_uncertainty import ATTACKER_PROFILES, AttackerUncertaintyResult
from src.optimizer.adaptive_belief import adaptive_belief_codef_search
from src.optimizer.capacity_schedule import _score_candidate, capacity_schedule_greedy_baseline, capacity_schedule_search
from src.optimizer.dynamic_budget import dynamic_budget_codef_search
from src.optimizer.robust_delay import DEFAULT_DELAY_SCENARIOS, robust_delay_search


def _reserve_budget(result: AttackerUncertaintyResult, total_budget: float) -> float:
    return float(result.diagnostics.get("selected_reserve_budget", total_budget))


def _candidate_record(name: str, result: AttackerUncertaintyResult) -> tuple[str, frozenset[str], float]:
    return name, frozenset(action.id for action in result.initial_actions), _reserve_budget(result, 0.0)


def adaptive_policy_router(
    graph: nx.DiGraph,
    actions: list[DefenseAction],
    total_budget: float,
    *,
    profiles: Sequence[str] = ATTACKER_PROFILES,
    delay_scenarios: Sequence[tuple[int, int]] = DEFAULT_DELAY_SCENARIOS,
    min_initial_budget: float = 2.0,
    constraints: PortfolioConstraints | None = None,
    response_pool_size: int = 4,
    max_phase_one_paths: int | None = 12,
    response_pool_mode: str = "adaptive_diverse",
    max_action_candidates: int | None = 6,
    candidate_action_ids: Sequence[str] | None = None,
    window_capacity: int = 2,
    deadline: int = 2,
    late_penalty: float = 0.01,
) -> AttackerUncertaintyResult:
    """Select a policy variant under one common objective and a Greedy guard."""
    seeds = [
        ("v3.0-dynamic", dynamic_budget_codef_search(
            graph, actions, total_budget, profiles=profiles,
            min_initial_budget=min_initial_budget, constraints=constraints,
            response_pool_size=response_pool_size, max_phase_one_paths=max_phase_one_paths,
            response_pool_mode=response_pool_mode, max_action_candidates=max_action_candidates,
            candidate_action_ids=candidate_action_ids,
        )),
        ("v3.2-belief", adaptive_belief_codef_search(
            graph, actions, total_budget, profiles=profiles,
            min_initial_budget=min_initial_budget, constraints=constraints,
            response_pool_size=response_pool_size, max_phase_one_paths=max_phase_one_paths,
            response_pool_mode=response_pool_mode, max_action_candidates=max_action_candidates,
            candidate_action_ids=candidate_action_ids,
        )),
        ("v3.3-robust", robust_delay_search(
            graph, actions, total_budget, profiles=profiles, delay_scenarios=delay_scenarios,
            min_initial_budget=min_initial_budget, constraints=constraints,
            response_pool_size=response_pool_size, max_phase_one_paths=max_phase_one_paths,
            response_pool_mode=response_pool_mode, max_action_candidates=max_action_candidates,
            candidate_action_ids=candidate_action_ids,
        )),
        ("v3.4-capacity", capacity_schedule_search(
            graph, actions, total_budget, profiles=profiles, delay_scenarios=delay_scenarios,
            min_initial_budget=min_initial_budget, constraints=constraints,
            response_pool_size=response_pool_size, max_phase_one_paths=max_phase_one_paths,
            response_pool_mode=response_pool_mode, max_action_candidates=max_action_candidates,
            candidate_action_ids=candidate_action_ids, window_capacity=window_capacity,
            deadline=deadline, late_penalty=late_penalty,
        )),
    ]
    candidates: dict[tuple[frozenset[str], float], str] = {}
    for name, seed in seeds:
        key = (frozenset(action.id for action in seed.initial_actions), _reserve_budget(seed, total_budget))
        candidates.setdefault(key, name)

    scored: list[tuple[str, AttackerUncertaintyResult]] = []
    for (initial_ids, reserve), source in candidates.items():
        scored.append((source, _score_candidate(
            graph, actions, initial_ids, reserve, delay_scenarios,
            profiles=profiles, constraints=constraints, response_pool_size=response_pool_size,
            max_phase_one_paths=max_phase_one_paths, response_pool_mode=response_pool_mode,
            window_capacity=window_capacity, deadline=deadline, late_penalty=late_penalty,
            baseline=False,
        )))
    selected_source, selected = min(scored, key=lambda item: (
        item[1].objective, item[1].worst_terminal_risk,
        tuple(action.id for action in item[1].initial_actions),
    ))

    baseline = capacity_schedule_greedy_baseline(
        graph, actions, total_budget, profiles=profiles, delay_scenarios=delay_scenarios,
        min_initial_budget=min_initial_budget, constraints=constraints,
        response_pool_size=response_pool_size, max_phase_one_paths=max_phase_one_paths,
        response_pool_mode=response_pool_mode, window_capacity=window_capacity,
        deadline=deadline, late_penalty=late_penalty,
    )
    router_fallback = baseline.objective < selected.objective
    chosen = baseline if router_fallback else selected
    diagnostics = dict(chosen.diagnostics)
    diagnostics.update({
        "adaptive_router": True,
        "adaptive_router_fallback": router_fallback,
        "adaptive_router_selected": "greedy-guard" if router_fallback else selected_source,
        "adaptive_router_candidates_json": [
            {"source": source, "objective": round(result.objective, 8), "initial_ids": [a.id for a in result.initial_actions]}
            for source, result in scored
        ],
        "adaptive_router_common_objective": float(chosen.objective),
    })
    return replace(
        chosen,
        method="v3.5-AdaptiveRouter-GreedyFallback" if router_fallback else "v3.5-AdaptiveRouter-CoDef",
        diagnostics=diagnostics,
    )
