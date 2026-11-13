"""v4.5 reproducible multi-case benchmark suite."""
from __future__ import annotations

from dataclasses import replace
from time import perf_counter
from typing import Sequence

import networkx as nx

from src.defense.action import DefenseAction
from src.defense.constraints import PortfolioConstraints
from src.optimizer.attacker_uncertainty import ATTACKER_PROFILES, AttackerUncertaintyResult
from src.optimizer.dynamic_budget import dynamic_budget_greedy_baseline
from src.optimizer.unified_pipeline import unified_pipeline_search
from src.optimizer.uncertainty_calibration import risk_envelope_graph


DEFAULT_BENCHMARK_CASES = (
    ("baseline", 0.0),
    ("moderate-risk-up", 0.10),
    ("high-risk-up", 0.20),
)


def benchmark_suite_search(
    graph: nx.DiGraph,
    actions: list[DefenseAction],
    total_budget: float,
    *,
    profiles: Sequence[str] = ATTACKER_PROFILES,
    cases: Sequence[tuple[str, float]] = DEFAULT_BENCHMARK_CASES,
    min_initial_budget: float = 2.0,
    constraints: PortfolioConstraints | None = None,
    response_pool_size: int = 4,
    max_phase_one_paths: int | None = 12,
    response_pool_mode: str = "adaptive_diverse",
    max_action_candidates: int | None = 6,
) -> AttackerUncertaintyResult:
    """Run the same protocol across a fixed case suite and return its report."""
    if not cases:
        raise ValueError("cases must not be empty")
    rows: list[dict[str, object]] = []
    selected_result: AttackerUncertaintyResult | None = None
    for case_name, shift in cases:
        case_graph = risk_envelope_graph(graph, shift)
        started = perf_counter()
        codef = unified_pipeline_search(
            case_graph,
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
        codef_elapsed = perf_counter() - started
        started = perf_counter()
        greedy = dynamic_budget_greedy_baseline(
            case_graph,
            actions,
            total_budget,
            profiles=profiles,
            min_initial_budget=min_initial_budget,
            constraints=constraints,
            response_pool_size=response_pool_size,
            max_phase_one_paths=max_phase_one_paths,
            response_pool_mode=response_pool_mode,
        )
        greedy_elapsed = perf_counter() - started
        if selected_result is None:
            selected_result = codef
        delta = float(greedy.objective - codef.objective)
        rows.append({
            "case": case_name,
            "relative_shift": float(shift),
            "codef_objective": round(codef.objective, 8),
            "greedy_objective": round(greedy.objective, 8),
            "objective_delta_greedy_minus_codef": round(delta, 8),
            "codef_worst_terminal_risk": round(codef.worst_terminal_risk, 8),
            "greedy_worst_terminal_risk": round(greedy.worst_terminal_risk, 8),
            "codef_elapsed_seconds": round(codef_elapsed, 6),
            "greedy_elapsed_seconds": round(greedy_elapsed, 6),
            "codef_initial_ids": tuple(action.id for action in codef.initial_actions),
            "greedy_initial_ids": tuple(action.id for action in greedy.initial_actions),
        })
    assert selected_result is not None
    wins = sum(float(row["objective_delta_greedy_minus_codef"]) > 1e-9 for row in rows)
    ties = sum(abs(float(row["objective_delta_greedy_minus_codef"])) <= 1e-9 for row in rows)
    losses = len(rows) - wins - ties
    diagnostics = dict(selected_result.diagnostics)
    diagnostics.update({
        "benchmark_suite": True,
        "benchmark_case_count": len(rows),
        "benchmark_rows": rows,
        "benchmark_wins": wins,
        "benchmark_ties": ties,
        "benchmark_losses": losses,
        "benchmark_mean_codef_objective": sum(float(row["codef_objective"]) for row in rows) / len(rows),
        "benchmark_mean_greedy_objective": sum(float(row["greedy_objective"]) for row in rows) / len(rows),
    })
    return replace(selected_result, method="v4.5-BenchmarkSuite-CoDef", diagnostics=diagnostics)
