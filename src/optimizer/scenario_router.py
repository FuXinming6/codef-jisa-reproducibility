"""v4.7 scenario-aware routing and stability evidence.

The mainline remains the v4.1 unified pipeline.  This module only chooses a
scenario-relevant candidate order before entering that pipeline, then replays
the selected policy over a few deterministic risk perturbations.  It does not
replace the real evaluator with a heuristic score.
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
from src.optimizer.dynamic_budget import _delayed_actions
from src.optimizer.unified_pipeline import unified_pipeline_search
from src.simulator.scenario_variation import perturb_edge_risks


DEFAULT_STABILITY_SEEDS = (7101, 7102, 7103)

_SCENARIO_FOCUS: dict[str, dict[str, object]] = {
    "baseline": {"label": "均衡防御", "preferred_types": (), "nodes": ()},
    "remote-access": {
        "label": "远程接入优先",
        "preferred_types": ("node_hardening", "edge_restriction"),
        "nodes": ("Remote-Access", "VPN-01", "Gateway-01", "Office-01", "File-01"),
    },
    "dmz-web": {
        "label": "DMZ 服务优先",
        "preferred_types": ("edge_restriction", "node_hardening"),
        "nodes": ("Gateway-01", "Web-01", "API-01", "DB-01", "Cache-01"),
    },
    "admin-plane": {
        "label": "管理面隔离优先",
        "preferred_types": ("edge_restriction", "node_isolation"),
        "nodes": ("Office-01", "Admin-01", "DB-01", "Backup-01"),
    },
    "file-exfiltration": {
        "label": "数据外传阻断优先",
        "preferred_types": ("node_hardening", "edge_restriction"),
        "nodes": ("Office-01", "File-01", "DB-01", "Backup-01"),
    },
    "partner-network": {
        "label": "第三方边界优先",
        "preferred_types": ("node_hardening", "edge_restriction"),
        "nodes": ("Partner-Net", "Gateway-02", "Web-02", "API-02", "DB-Replica"),
    },
}


def _target_nodes(action: DefenseAction) -> tuple[str, ...]:
    if isinstance(action.target, tuple):
        return tuple(str(item) for item in action.target)
    return (str(action.target),)


def rank_actions_for_scenario(
    graph: nx.DiGraph,
    actions: Sequence[DefenseAction],
    scenario_id: str,
    *,
    focus_assets: Sequence[str] = (),
) -> tuple[list[DefenseAction], dict[str, object]]:
    """Return a deterministic, explainable action order for one risk story."""
    profile = _SCENARIO_FOCUS.get(scenario_id, _SCENARIO_FOCUS["baseline"])
    focus_nodes = set(str(item) for item in profile["nodes"])
    focus_nodes.update(str(item) for item in focus_assets)
    preferred_types = set(str(item) for item in profile["preferred_types"])
    scored: list[tuple[float, DefenseAction, dict[str, object]]] = []
    for action in actions:
        targets = _target_nodes(action)
        overlap = len(set(targets) & focus_nodes)
        edge_risk = 0.0
        if isinstance(action.target, tuple):
            edge_risk = float(graph.get_edge_data(*action.target, {}).get("risk", 0.0))
        incident_risk = sum(
            float(data.get("risk", 0.0))
            for node in targets if node in graph
            for _, _, data in list(graph.in_edges(node, data=True)) + list(graph.out_edges(node, data=True))
        )
        criticality = max((float(graph.nodes[node].get("asset_value", 1.0)) for node in targets if node in graph), default=1.0)
        type_bonus = 0.5 if str(action.action_type) in preferred_types else 0.0
        score = 3.0 * overlap + 1.2 * criticality + 1.5 * edge_risk + 0.25 * incident_risk + type_bonus + 0.15 * float(action.effect)
        scored.append((score, action, {
            "action_id": action.id,
            "score": round(score, 6),
            "focus_overlap": overlap,
            "edge_risk": round(edge_risk, 6),
            "criticality": round(criticality, 6),
        }))
    scored.sort(key=lambda item: (-item[0], item[1].id))
    ordered = [item[1] for item in scored]
    return ordered, {
        "scenario_id": scenario_id,
        "focus_label": str(profile["label"]),
        "focus_nodes": sorted(focus_nodes),
        "ranked_actions": [item[2] for item in scored],
    }


def scenario_routed_pipeline_search(
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
    """Route one scenario through v4.1, then attach repeatability evidence."""
    ordered_actions, route_report = rank_actions_for_scenario(
        graph, actions, scenario_id, focus_assets=focus_assets,
    )
    candidate_limit = max_action_candidates if max_action_candidates is not None else len(ordered_actions)
    route_ids = [action.id for action in ordered_actions]
    if candidate_action_ids:
        known_ids = set(route_ids)
        proxy_ids = [str(action_id) for action_id in candidate_action_ids if str(action_id) in known_ids]
        route_ids = proxy_ids + [action_id for action_id in route_ids if action_id not in set(proxy_ids)]
        route_report = dict(route_report)
        route_report["proxy_gate_candidate_ids"] = tuple(proxy_ids)
    candidate_ids = route_ids[:max(1, candidate_limit)]
    ordered_actions = [next(action for action in ordered_actions if action.id == action_id) for action_id in route_ids]
    routed = unified_pipeline_search(
        graph,
        ordered_actions,
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
    general = unified_pipeline_search(
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
    selected_source, selected = min(
        (("scenario_routed", routed), ("general_pipeline", general)),
        key=lambda item: (item[1].objective, item[1].worst_terminal_risk, tuple(action.id for action in item[1].initial_actions)),
    )
    initial_ids = frozenset(action.id for action in selected.initial_actions)
    reserve = float(selected.diagnostics.get("selected_reserve_budget", total_budget))
    delayed = _delayed_actions(actions, 1)
    stability_rows: list[dict[str, object]] = []
    stability_results: list[AttackerUncertaintyResult] = []
    for seed in stability_seeds:
        varied_graph = perturb_edge_risks(graph, int(seed), relative_jitter=stability_jitter)
        replay = evaluate_attacker_uncertainty_portfolio(
            varied_graph,
            profiles,
            delayed,
            initial_ids,
            reserve,
            constraints=constraints,
            response_pool_size=response_pool_size,
            max_phase_one_paths=max_phase_one_paths,
            max_response_delay=1,
            response_pool_mode=response_pool_mode,
            method_name="v4.7-StabilityReplay",
        )
        stability_results.append(replay)
        stability_rows.append({
            "seed": int(seed),
            "objective": round(float(replay.objective), 8),
            "worst_terminal_risk": round(float(replay.worst_terminal_risk), 8),
            "worst_profile": max(replay.profile_results, key=lambda item: item.result.terminal_risk).profile if replay.profile_results else "none",
        })
    objectives = [float(item.objective) for item in stability_results]
    stability_report = {
        "protocol": "same selected policy replayed on deterministic edge-risk perturbations",
        "seed_count": len(stability_rows),
        "seeds": [int(seed) for seed in stability_seeds],
        "jitter": float(stability_jitter),
        "mean_objective": round(mean(objectives), 8) if objectives else None,
        "stddev_objective": round(pstdev(objectives), 8) if len(objectives) > 1 else 0.0,
        "worst_objective": round(max(objectives), 8) if objectives else None,
        "rows": stability_rows,
    }
    diagnostics = dict(selected.diagnostics)
    diagnostics.update({
        "scenario_router": True,
        "scenario_router_id": scenario_id,
        "scenario_router_focus": str(route_report["focus_label"]),
        "scenario_router_candidate_ids": tuple(candidate_ids),
        "scenario_router_report": route_report,
        "scenario_router_selected": selected_source,
        "scenario_router_quality_guard": True,
        "scenario_router_candidate_objectives": {
            "scenario_routed": round(float(routed.objective), 8),
            "general_pipeline": round(float(general.objective), 8),
        },
        "stability_report": stability_report,
        "stability_run_count": len(stability_rows),
        "stability_mean_objective": stability_report["mean_objective"],
        "stability_stddev_objective": stability_report["stddev_objective"],
        "stability_worst_objective": stability_report["worst_objective"],
    })
    return replace(selected, method="v4.7-ScenarioRouted-CoDef", diagnostics=diagnostics)
