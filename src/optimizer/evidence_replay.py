"""v4.4 multi-case evidence replay for one selected policy."""
from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import networkx as nx

from src.defense.action import DefenseAction
from src.defense.constraints import PortfolioConstraints
from src.optimizer.attacker_uncertainty import ATTACKER_PROFILES, AttackerUncertaintyResult, evaluate_attacker_uncertainty_portfolio
from src.optimizer.dynamic_budget import _delayed_actions
from src.optimizer.unified_pipeline import unified_pipeline_search
from src.optimizer.uncertainty_calibration import risk_envelope_graph


DEFAULT_REPLAY_CASES = (
    ("baseline", 0.0),
    ("risk-up", 0.10),
    ("risk-down", -0.10),
    ("stress-high", 0.20),
)


def evidence_replay_search(
    graph: nx.DiGraph,
    actions: list[DefenseAction],
    total_budget: float,
    *,
    profiles: Sequence[str] = ATTACKER_PROFILES,
    replay_cases: Sequence[tuple[str, float]] = DEFAULT_REPLAY_CASES,
    min_initial_budget: float = 2.0,
    constraints: PortfolioConstraints | None = None,
    response_pool_size: int = 4,
    max_phase_one_paths: int | None = 12,
    response_pool_mode: str = "adaptive_diverse",
    max_action_candidates: int | None = 6,
) -> AttackerUncertaintyResult:
    """Select once, then replay the same policy over several case variants."""
    if not replay_cases:
        raise ValueError("replay_cases must not be empty")
    selected = unified_pipeline_search(
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
    )
    initial_ids = frozenset(action.id for action in selected.initial_actions)
    reserve = float(selected.diagnostics.get("selected_reserve_budget", total_budget))
    delayed = _delayed_actions(actions, 1)
    traces: list[dict[str, object]] = []
    results: list[AttackerUncertaintyResult] = []
    for case_name, shift in replay_cases:
        case_result = evaluate_attacker_uncertainty_portfolio(
            risk_envelope_graph(graph, shift),
            profiles,
            delayed,
            initial_ids,
            reserve,
            constraints=constraints,
            response_pool_size=response_pool_size,
            max_phase_one_paths=max_phase_one_paths,
            max_response_delay=1,
            response_pool_mode=response_pool_mode,
            method_name="v4.4-EvidenceReplay-Case",
        )
        results.append(case_result)
        worst_profile = max(case_result.profile_results, key=lambda item: item.result.terminal_risk) if case_result.profile_results else None
        traces.append({
            "case": case_name,
            "relative_shift": float(shift),
            "objective": round(case_result.objective, 8),
            "worst_terminal_risk": round(case_result.worst_terminal_risk, 8),
            "worst_profile": worst_profile.profile if worst_profile else "none",
            "observed_path": list(worst_profile.result.observed_path) if worst_profile else [],
            "replanned_path": list(worst_profile.result.replanned_path) if worst_profile else [],
            "response_actions": [action.id for action in (worst_profile.result.response_actions if worst_profile else ())],
        })
    worst = max(results, key=lambda item: (item.objective, item.worst_terminal_risk))
    diagnostics = dict(worst.diagnostics)
    diagnostics.update({
        "evidence_replay": True,
        "replay_case_count": len(traces),
        "replay_trace": traces,
        "replay_initial_ids": tuple(sorted(initial_ids)),
        "replay_policy_source": selected.method,
    })
    return replace(worst, method="v4.4-EvidenceReplay-CoDef", diagnostics=diagnostics)
