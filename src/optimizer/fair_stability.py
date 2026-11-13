"""v4.8 fair multi-seed stability comparison.

CoDef and the dynamic-budget Greedy baseline are evaluated on the same graph
variants, with the same attacker profiles and replay protocol.  The result is
evidence for where the mainline is stable, not a claim of universal victory.
"""
from __future__ import annotations

from dataclasses import replace
from statistics import mean, pstdev
from typing import Sequence

import networkx as nx

from src.defense.action import DefenseAction
from src.defense.constraints import PortfolioConstraints
from src.optimizer.attacker_uncertainty import (
    ATTACKER_PROFILES,
    AttackerUncertaintyResult,
    evaluate_attacker_uncertainty_portfolio,
)
from src.optimizer.dynamic_budget import _delayed_actions, dynamic_budget_greedy_baseline
from src.optimizer.scenario_router import DEFAULT_STABILITY_SEEDS, scenario_routed_pipeline_search
from src.simulator.scenario_variation import perturb_edge_risks


def _summary(values: Sequence[float]) -> dict[str, float | None]:
    return {
        "mean_objective": round(mean(values), 8) if values else None,
        "stddev_objective": round(pstdev(values), 8) if len(values) > 1 else 0.0,
        "worst_objective": round(max(values), 8) if values else None,
    }


def fair_stability_search(
    graph: nx.DiGraph,
    actions: list[DefenseAction],
    total_budget: float,
    *,
    scenario_id: str,
    focus_assets: Sequence[str] = (),
    profiles: Sequence[str] = ATTACKER_PROFILES,
    min_initial_budget: float = 2.0,
    constraints: PortfolioConstraints | None = None,
    response_pool_size: int = 4,
    max_phase_one_paths: int | None = 12,
    response_pool_mode: str = "adaptive_diverse",
    max_action_candidates: int | None = 6,
    candidate_action_ids: Sequence[str] | None = None,
    stability_seeds: Sequence[int] = DEFAULT_STABILITY_SEEDS,
    stability_jitter: float = 0.08,
) -> AttackerUncertaintyResult:
    """Compare the selected CoDef and Greedy policies under shared variants."""
    codef = scenario_routed_pipeline_search(
        graph,
        actions,
        total_budget,
        scenario_id=scenario_id,
        focus_assets=focus_assets,
        profiles=profiles,
        min_initial_budget=min_initial_budget,
        constraints=constraints,
        response_pool_size=response_pool_size,
        max_phase_one_paths=max_phase_one_paths,
        response_pool_mode=response_pool_mode,
        max_action_candidates=max_action_candidates,
        candidate_action_ids=candidate_action_ids,
        stability_seeds=stability_seeds,
        stability_jitter=stability_jitter,
    )
    greedy = dynamic_budget_greedy_baseline(
        graph,
        actions,
        total_budget,
        profiles=profiles,
        min_initial_budget=min_initial_budget,
        constraints=constraints,
        response_pool_size=response_pool_size,
        max_phase_one_paths=max_phase_one_paths,
        response_pool_mode=response_pool_mode,
    )
    codef_ids = frozenset(action.id for action in codef.initial_actions)
    greedy_ids = frozenset(action.id for action in greedy.initial_actions)
    codef_reserve = float(codef.diagnostics.get("selected_reserve_budget", total_budget))
    greedy_reserve = float(greedy.diagnostics.get("selected_reserve_budget", total_budget))
    delayed = _delayed_actions(actions, 1)
    rows: list[dict[str, object]] = []
    codef_results: list[AttackerUncertaintyResult] = []
    greedy_results: list[AttackerUncertaintyResult] = []
    for seed in stability_seeds:
        varied_graph = perturb_edge_risks(graph, int(seed), relative_jitter=stability_jitter)
        codef_replay = evaluate_attacker_uncertainty_portfolio(
            varied_graph, profiles, delayed, codef_ids, codef_reserve,
            constraints=constraints, response_pool_size=response_pool_size,
            max_phase_one_paths=max_phase_one_paths, max_response_delay=1,
            response_pool_mode=response_pool_mode, method_name="v4.8-CoDef-Replay",
        )
        greedy_replay = evaluate_attacker_uncertainty_portfolio(
            varied_graph, profiles, delayed, greedy_ids, greedy_reserve,
            constraints=constraints, response_pool_size=response_pool_size,
            max_phase_one_paths=max_phase_one_paths, max_response_delay=1,
            response_pool_mode=response_pool_mode, method_name="v4.8-Greedy-Replay",
        )
        codef_results.append(codef_replay)
        greedy_results.append(greedy_replay)
        codef_value = float(codef_replay.objective)
        greedy_value = float(greedy_replay.objective)
        winner = "CoDef" if codef_value < greedy_value - 1e-12 else "Greedy" if greedy_value < codef_value - 1e-12 else "Tie"
        rows.append({
            "seed": int(seed),
            "codef_objective": round(codef_value, 8),
            "greedy_objective": round(greedy_value, 8),
            "codef_terminal_risk": round(float(codef_replay.worst_terminal_risk), 8),
            "greedy_terminal_risk": round(float(greedy_replay.worst_terminal_risk), 8),
            "winner": winner,
        })
    codef_values = [float(item.objective) for item in codef_results]
    greedy_values = [float(item.objective) for item in greedy_results]
    wins = sum(row["winner"] == "CoDef" for row in rows)
    ties = sum(row["winner"] == "Tie" for row in rows)
    losses = sum(row["winner"] == "Greedy" for row in rows)
    report = {
        "protocol": "same scenario, same risk seeds, same total budget and attacker profiles",
        "scenario_id": scenario_id,
        "seed_count": len(rows),
        "seeds": [int(seed) for seed in stability_seeds],
        "jitter": float(stability_jitter),
        "codef_initial_ids": sorted(codef_ids),
        "greedy_initial_ids": sorted(greedy_ids),
        "codef_budget_split": {
            "initial": float(codef.diagnostics.get("selected_initial_budget", total_budget - codef_reserve)),
            "reserve": codef_reserve,
        },
        "greedy_budget_split": {
            "initial": float(greedy.diagnostics.get("selected_initial_budget", total_budget - greedy_reserve)),
            "reserve": greedy_reserve,
        },
        "codef": _summary(codef_values),
        "greedy": _summary(greedy_values),
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "rows": rows,
    }
    diagnostics = dict(codef.diagnostics)
    diagnostics.update({
        "fair_stability": True,
        "fair_stability_report": report,
        "fair_stability_seed_count": len(rows),
        "fair_stability_wins": wins,
        "fair_stability_ties": ties,
        "fair_stability_losses": losses,
        "fair_stability_codef_mean": report["codef"]["mean_objective"],
        "fair_stability_greedy_mean": report["greedy"]["mean_objective"],
        "fair_stability_codef_stddev": report["codef"]["stddev_objective"],
        "fair_stability_greedy_stddev": report["greedy"]["stddev_objective"],
    })
    return replace(codef, method="v4.8-FairStability-CoDef", diagnostics=diagnostics)
