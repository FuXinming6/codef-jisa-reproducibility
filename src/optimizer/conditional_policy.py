"""Scenario-conditioned initial-defense policies with observable signals."""
from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, stdev
from typing import Sequence
from math import ceil

import networkx as nx

from src.defense.action import DefenseAction
from src.defense.constraints import PortfolioConstraints
from src.graph.graph_builder import ingress_nodes
from src.optimizer.baselines import greedy_baseline
from src.optimizer.common import TwoStageResult
from src.optimizer.robust_two_stage import (
    evaluate_robust_initial_portfolio,
    robust_two_stage_codef_search,
    robust_two_stage_greedy_baseline,
)
from src.optimizer.two_stage import _result_from_evaluation, evaluate_two_stage_strategy


def _dominant_ingress_source(graph: nx.DiGraph) -> str:
    candidates = [
        (float(data.get("risk", 0.0)), source)
        for source in ingress_nodes(graph)
        for _, _, data in graph.out_edges(source, data=True)
    ]
    return max(candidates, default=(0.0, "unknown"))[1]


def _route_family_for_source(graph: nx.DiGraph, source: str) -> str:
    """Map an ingress to its first-hop corridor, independent of its name."""
    outgoing = list(graph.out_edges(source, data=True))
    if not outgoing:
        return "route:unknown"
    _, target, _ = max(outgoing, key=lambda item: float(item[2].get("risk", 0.0)))
    target_type = str(graph.nodes[target].get("type", "unknown"))
    return f"route:{target_type}"


def observable_scenario_signal(graph: nx.DiGraph, mode: str = "source") -> str:
    """Return an ingress signal available from current edge-risk telemetry.

    ``source`` preserves the original exact-node signal for backwards
    compatibility. ``route_family`` is the production signal: it uses the
    dominant ingress' first-hop asset type (gateway, vpn, ...), so a previously
    unseen name such as ``Remote-Access`` can still share a learned corridor
    policy with any other ingress reaching the same kind of entry point.
    """
    if mode not in {"source", "route_family"}:
        raise ValueError("mode must be 'source' or 'route_family'")
    source = _dominant_ingress_source(graph)
    if mode == "source":
        return source
    return _route_family_for_source(graph, source)


def _signal_space(graph: nx.DiGraph, mode: str) -> tuple[str, ...]:
    """Return all observable route signals present in the topology.

    Enumerating the topology prevents a route from becoming an "unknown"
    signal merely because it was not the noisiest route in calibration.
    """
    if mode == "source":
        return tuple(sorted(ingress_nodes(graph)))
    return tuple(sorted({_route_family_for_source(graph, source) for source in ingress_nodes(graph)}))


def _focused_scenarios(
    scenarios: Sequence[nx.DiGraph],
    signal: str,
    mode: str,
) -> tuple[nx.DiGraph, ...]:
    """Select telemetry states where a route is most exposed.

    This gives a route that was not dominant during calibration a small but
    representative training set. It is still based only on observed edge
    risks, not on a hidden seed or scenario label.
    """
    if not scenarios:
        return tuple()
    scored: list[tuple[float, nx.DiGraph]] = []
    for scenario in scenarios:
        score = max(
            (
                float(data.get("risk", 0.0))
                for source in ingress_nodes(scenario)
                if (mode == "route_family" and _route_family_for_source(scenario, source) == signal)
                or (mode == "source" and source == signal)
                for _, _, data in scenario.out_edges(source, data=True)
            ),
            default=0.0,
        )
        scored.append((score, scenario))
    scored.sort(key=lambda item: item[0], reverse=True)
    count = max(2, ceil(len(scored) * 0.6)) if len(scored) > 1 else 1
    return tuple(scenario for _, scenario in scored[:count])


@dataclass(frozen=True)
class ConditionalPolicy:
    method: str
    initial_ids_by_signal: dict[str, frozenset[str]]
    fallback_initial_ids: frozenset[str]
    diagnostics: dict[str, object]


@dataclass(frozen=True)
class SignalEvidence:
    """Observable evidence used to decide whether a conditional policy is safe."""

    signal: str
    dominant_source: str
    dominant_risk: float
    runner_up_risk: float
    risk_margin: float
    calibration_support: int
    covered: bool
    confidence: float
    reason: str


def observable_signal_evidence(
    graph: nx.DiGraph,
    policy: ConditionalPolicy,
    signal_mode: str = "route_family",
    min_support: int = 2,
) -> SignalEvidence:
    """Summarize confidence using only current telemetry and policy metadata.

    Confidence is deliberately conservative: a route must be covered by the
    fitted policy, have at least some calibration support, and be separated
    from the runner-up ingress by a measurable risk margin. The caller can
    abstain and use the public v2.4 policy when confidence is low.
    """
    candidates = sorted(
        [
            (
                max(
                    (float(data.get("risk", 0.0)) for _, _, data in graph.out_edges(source, data=True)),
                    default=0.0,
                ),
                source,
            )
            for source in ingress_nodes(graph)
        ],
        reverse=True,
    )
    dominant_risk, dominant_source = candidates[0] if candidates else (0.0, "unknown")
    runner_up_risk = candidates[1][0] if len(candidates) > 1 else 0.0
    risk_margin = max(0.0, dominant_risk - runner_up_risk)
    signal = observable_scenario_signal(graph, signal_mode)
    support_map = policy.diagnostics.get("signal_support", {})
    support = int(support_map.get(signal, 0)) if isinstance(support_map, dict) else 0
    covered = signal in policy.initial_ids_by_signal
    margin_score = min(1.0, risk_margin / 0.10)
    support_score = min(1.0, support / max(1, min_support))
    coverage_score = 1.0 if covered else 0.0
    confidence = 0.45 * margin_score + 0.30 * support_score + 0.25 * coverage_score
    if not covered:
        reason = "signal_not_covered"
    elif support < min_support:
        reason = "limited_calibration_support"
    elif risk_margin < 0.02:
        reason = "weak_ingress_separation"
    else:
        reason = "sufficient_observable_evidence"
    return SignalEvidence(
        signal=signal,
        dominant_source=dominant_source,
        dominant_risk=dominant_risk,
        runner_up_risk=runner_up_risk,
        risk_margin=risk_margin,
        calibration_support=support,
        covered=covered,
        confidence=confidence,
        reason=reason,
    )


@dataclass(frozen=True)
class ConditionalPolicyResult:
    method: str
    scenario_results: tuple[tuple[str, TwoStageResult], ...]
    mean_terminal_risk: float
    worst_terminal_risk: float
    risk_stddev: float
    diagnostics: dict[str, object]


def _summarize(
    method: str,
    results: Sequence[tuple[str, TwoStageResult]],
    diagnostics: dict[str, object],
) -> ConditionalPolicyResult:
    risks = [result.terminal_risk for _, result in results]
    return ConditionalPolicyResult(
        method=method,
        scenario_results=tuple(results),
        mean_terminal_risk=mean(risks) if risks else 0.0,
        worst_terminal_risk=max(risks, default=0.0),
        risk_stddev=stdev(risks) if len(risks) > 1 else 0.0,
        diagnostics=diagnostics,
    )


def fit_conditional_codef_policy(
    calibration_scenarios: Sequence[nx.DiGraph],
    actions: list[DefenseAction],
    initial_budget: float,
    reserve_budget: float,
    constraints: PortfolioConstraints | None = None,
    response_pool_size: int = 4,
    max_phase_one_paths: int | None = 12,
    max_response_delay: int | None = None,
    objective_mode: str = "minimax",
    signal_mode: str = "route_family",
) -> ConditionalPolicy:
    """Fit one minimax CoDef portfolio per observable ingress signal.

    Policies are fitted for every route family present in the topology, not
    only for the families that happened to win the calibration argmax. This
    closes the common ``Remote-Access``-was-unseen gap while keeping a public
    minimax portfolio as a safe fallback.
    """
    if not calibration_scenarios:
        raise ValueError("at least one calibration scenario is required")
    if signal_mode not in {"source", "route_family"}:
        raise ValueError("signal_mode must be 'source' or 'route_family'")
    groups: dict[str, list[nx.DiGraph]] = {}
    for scenario in calibration_scenarios:
        groups.setdefault(observable_scenario_signal(scenario, signal_mode), []).append(scenario)
    global_result = robust_two_stage_codef_search(
        calibration_scenarios,
        actions,
        initial_budget,
        reserve_budget,
        constraints=constraints,
        response_pool_size=response_pool_size,
        max_phase_one_paths=max_phase_one_paths,
        max_response_delay=max_response_delay,
        response_pool_mode="adaptive_diverse",
        objective_mode=objective_mode,
        method_name="TwoStage-Conditional-Global",
    )
    policies: dict[str, frozenset[str]] = {}
    fallback_signals: list[str] = []
    available_signals = _signal_space(calibration_scenarios[0], signal_mode)
    for signal in available_signals:
        scenarios = tuple(groups.get(signal) or _focused_scenarios(calibration_scenarios, signal, signal_mode))
        result = robust_two_stage_codef_search(
            scenarios,
            actions,
            initial_budget,
            reserve_budget,
            constraints=constraints,
            response_pool_size=response_pool_size,
            max_phase_one_paths=max_phase_one_paths,
            max_response_delay=max_response_delay,
            response_pool_mode="adaptive_diverse",
            objective_mode=objective_mode,
            method_name=f"TwoStage-Conditional-{signal}",
        )
        public_result = evaluate_robust_initial_portfolio(
            scenarios,
            actions,
            frozenset(action.id for action in global_result.initial_actions),
            reserve_budget,
            constraints=constraints,
            response_pool_size=response_pool_size,
            max_phase_one_paths=max_phase_one_paths,
            max_response_delay=max_response_delay,
            response_pool_mode="adaptive_diverse",
            objective_mode=objective_mode,
            method_name=f"TwoStage-Conditional-Public-{signal}",
        )
        if public_result.objective <= result.objective + 1e-12:
            policies[signal] = frozenset(action.id for action in global_result.initial_actions)
            fallback_signals.append(signal)
        else:
            policies[signal] = frozenset(action.id for action in result.initial_actions)
    return ConditionalPolicy(
        method="TwoStage-Conditional-CoDef",
        initial_ids_by_signal=policies,
        fallback_initial_ids=frozenset(action.id for action in global_result.initial_actions),
        diagnostics={
            "observable_signal": "dominant-ingress-first-hop-family" if signal_mode == "route_family" else "highest-risk-ingress-source",
            "signal_mode": signal_mode,
            "signals": tuple(sorted(policies)),
            "available_signals": available_signals,
            "signal_support": {signal: len(groups.get(signal, ())) for signal in available_signals},
            "objective_mode": objective_mode,
            "calibration_scenario_count": len(calibration_scenarios),
            "public_fallback_signals": tuple(fallback_signals),
            "focused_signal_count": sum(1 for signal in available_signals if signal not in groups),
        },
    )


def fit_conditional_greedy_policy(
    calibration_scenarios: Sequence[nx.DiGraph],
    actions: list[DefenseAction],
    initial_budget: float,
    reserve_budget: float,
    constraints: PortfolioConstraints | None = None,
    response_pool_size: int = 4,
    max_phase_one_paths: int | None = 12,
    max_response_delay: int | None = None,
    objective_mode: str = "minimax",
    signal_mode: str = "route_family",
) -> ConditionalPolicy:
    """Fit a signal-conditioned policy using only Greedy candidate portfolios."""
    if not calibration_scenarios:
        raise ValueError("at least one calibration scenario is required")
    if signal_mode not in {"source", "route_family"}:
        raise ValueError("signal_mode must be 'source' or 'route_family'")
    groups: dict[str, list[nx.DiGraph]] = {}
    for scenario in calibration_scenarios:
        groups.setdefault(observable_scenario_signal(scenario, signal_mode), []).append(scenario)
    global_result = robust_two_stage_greedy_baseline(
        calibration_scenarios,
        actions,
        initial_budget,
        reserve_budget,
        constraints=constraints,
        response_pool_size=response_pool_size,
        max_phase_one_paths=max_phase_one_paths,
        max_response_delay=max_response_delay,
        response_pool_mode="adaptive_diverse",
        objective_mode=objective_mode,
        method_name="TwoStage-Conditional-Greedy-Global",
    )
    policies: dict[str, frozenset[str]] = {}
    fallback_signals: list[str] = []
    available_signals = _signal_space(calibration_scenarios[0], signal_mode)
    for signal in available_signals:
        scenarios = tuple(groups.get(signal) or _focused_scenarios(calibration_scenarios, signal, signal_mode))
        result = robust_two_stage_greedy_baseline(
            scenarios,
            actions,
            initial_budget,
            reserve_budget,
            constraints=constraints,
            response_pool_size=response_pool_size,
            max_phase_one_paths=max_phase_one_paths,
            max_response_delay=max_response_delay,
            response_pool_mode="adaptive_diverse",
            objective_mode=objective_mode,
            method_name=f"TwoStage-Conditional-Greedy-{signal}",
        )
        public_result = evaluate_robust_initial_portfolio(
            scenarios,
            actions,
            frozenset(action.id for action in global_result.initial_actions),
            reserve_budget,
            constraints=constraints,
            response_pool_size=response_pool_size,
            max_phase_one_paths=max_phase_one_paths,
            max_response_delay=max_response_delay,
            response_pool_mode="adaptive_diverse",
            objective_mode=objective_mode,
            method_name=f"TwoStage-Conditional-Greedy-Public-{signal}",
        )
        if public_result.objective <= result.objective + 1e-12:
            policies[signal] = frozenset(action.id for action in global_result.initial_actions)
            fallback_signals.append(signal)
        else:
            policies[signal] = frozenset(action.id for action in result.initial_actions)
    return ConditionalPolicy(
        method="TwoStage-Conditional-Greedy",
        initial_ids_by_signal=policies,
        fallback_initial_ids=frozenset(action.id for action in global_result.initial_actions),
        diagnostics={
            "observable_signal": "dominant-ingress-first-hop-family" if signal_mode == "route_family" else "highest-risk-ingress-source",
            "signal_mode": signal_mode,
            "signals": tuple(sorted(policies)),
            "available_signals": available_signals,
            "signal_support": {signal: len(groups.get(signal, ())) for signal in available_signals},
            "objective_mode": objective_mode,
            "calibration_scenario_count": len(calibration_scenarios),
            "public_fallback_signals": tuple(fallback_signals),
            "focused_signal_count": sum(1 for signal in available_signals if signal not in groups),
        },
    )


def evaluate_conditional_policy(
    policy: ConditionalPolicy,
    scenarios: Sequence[nx.DiGraph],
    actions: list[DefenseAction],
    reserve_budget: float,
    constraints: PortfolioConstraints | None = None,
    response_pool_size: int = 4,
    max_phase_one_paths: int | None = 12,
    max_response_delay: int | None = None,
    signal_mode: str | None = None,
) -> ConditionalPolicyResult:
    """Evaluate a fitted policy using only each scenario's observed signal."""
    selected_signal_mode = signal_mode or str(policy.diagnostics.get("signal_mode", "source"))
    results: list[tuple[str, TwoStageResult]] = []
    for index, scenario in enumerate(scenarios):
        signal = observable_scenario_signal(scenario, selected_signal_mode)
        ids = policy.initial_ids_by_signal.get(signal, policy.fallback_initial_ids)
        evaluation = evaluate_two_stage_strategy(
            scenario,
            actions,
            ids,
            reserve_budget,
            constraints=constraints,
            response_pool_size=response_pool_size,
            max_phase_one_paths=max_phase_one_paths,
            max_response_delay=max_response_delay,
            response_ranking="path_coverage",
            use_response_cache=True,
            response_pool_mode="adaptive_diverse",
            attacker_profile="risk_max",
        )
        result = _result_from_evaluation(
            policy.method,
            actions,
            ids,
            evaluation,
            {"scenario_index": index, "observable_signal": signal, "conditional_policy": True},
        )
        results.append((signal, result))
    return _summarize(policy.method, results, dict(policy.diagnostics))
