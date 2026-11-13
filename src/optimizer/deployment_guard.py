"""v4.6 deployment feasibility and safety guard."""
from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import networkx as nx

from src.defense.action import DefenseAction, DefenseType
from src.defense.apply_defense import portfolio_cost
from src.defense.constraints import PortfolioConstraints
from src.optimizer.attacker_uncertainty import ATTACKER_PROFILES, AttackerUncertaintyResult
from src.optimizer.capacity_schedule import schedule_initial_actions
from src.optimizer.dynamic_budget import dynamic_budget_greedy_baseline
from src.optimizer.topology_actions import map_actions_to_topology
from src.optimizer.unified_pipeline import unified_pipeline_search


def _action_available(graph: nx.DiGraph, action: DefenseAction) -> bool:
    if action.action_type == DefenseType.EDGE_RESTRICTION:
        source, target = action.target  # type: ignore[misc]
        return graph.has_edge(source, target)
    return str(action.target) in graph


def deployment_guard_search(
    graph: nx.DiGraph,
    actions: list[DefenseAction],
    total_budget: float,
    *,
    profiles: Sequence[str] = ATTACKER_PROFILES,
    min_initial_budget: float = 2.0,
    min_reserve_budget: float = 1.0,
    max_isolations: int = 1,
    constraints: PortfolioConstraints | None = None,
    response_pool_size: int = 4,
    max_phase_one_paths: int | None = 12,
    response_pool_mode: str = "adaptive_diverse",
    max_action_candidates: int | None = 6,
    derive_actions: bool = True,
) -> AttackerUncertaintyResult:
    """Run the mainline only on feasible actions and enforce preflight guards."""
    active_actions, mapping_report = map_actions_to_topology(graph) if derive_actions else (actions, {})
    available_actions = [action for action in active_actions if _action_available(graph, action)]
    warnings: list[str] = []
    if len(available_actions) < len(active_actions):
        warnings.append(f"过滤 {len(active_actions) - len(available_actions)} 个不适用于当前拓扑的动作")
    if not available_actions:
        raise ValueError("no topology-compatible defense actions")
    result = unified_pipeline_search(
        graph,
        available_actions,
        total_budget,
        profiles=profiles,
        min_initial_budget=min_initial_budget,
        constraints=constraints,
        response_pool_size=response_pool_size,
        max_phase_one_paths=max_phase_one_paths,
        response_pool_mode=response_pool_mode,
        max_action_candidates=max_action_candidates,
    )
    initial_cost = portfolio_cost(result.initial_actions)
    reserve = float(result.diagnostics.get("selected_reserve_budget", total_budget - initial_cost))
    schedule = schedule_initial_actions(result.initial_actions, window_capacity=2, deadline=2)
    isolation_count = sum(action.action_type == DefenseType.NODE_ISOLATION for action in result.initial_actions)
    violations: list[str] = []
    if initial_cost > total_budget - min_reserve_budget + 1e-9:
        violations.append("初始动作消耗过多，未保留最低响应预算")
    if reserve < min_reserve_budget - 1e-9:
        violations.append("响应预留预算低于安全下限")
    if isolation_count > max_isolations:
        violations.append("初始隔离动作超过业务中断上限")
    if int(schedule["late_slots"]) > 0:
        violations.append("初始动作无法在维护窗口截止时间前完成")

    fallback = False
    if violations:
        fallback_result = dynamic_budget_greedy_baseline(
            graph,
            available_actions,
            total_budget,
            profiles=profiles,
            min_initial_budget=min_initial_budget,
            constraints=constraints,
            response_pool_size=response_pool_size,
            max_phase_one_paths=max_phase_one_paths,
            response_pool_mode=response_pool_mode,
        )
        result = fallback_result
        fallback = True
        initial_cost = portfolio_cost(result.initial_actions)
        schedule = schedule_initial_actions(result.initial_actions, window_capacity=2, deadline=2)
    diagnostics = dict(result.diagnostics)
    diagnostics.update({
        "deployment_guard": True,
        "deployment_guard_fallback": fallback,
        "deployment_guard_warnings": tuple(warnings),
        "deployment_guard_violations": tuple(violations),
        "deployment_guard_available_action_count": len(available_actions),
        "deployment_guard_schedule": schedule,
        "deployment_guard_mapping": mapping_report,
        "deployment_guard_isolation_limit": max_isolations,
        "deployment_guard_min_reserve": min_reserve_budget,
    })
    return replace(
        result,
        method="v4.6-DeploymentGuard-GreedyFallback" if fallback else "v4.6-DeploymentGuard-CoDef",
        diagnostics=diagnostics,
    )
