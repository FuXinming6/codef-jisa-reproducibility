"""Final pre-model CoDef entry point.

This module freezes the components that survived the v2.2-v2.4 validation
rounds.  It deliberately does not run automatic budget calibration or the
multi-scenario objective by default because neither produced a stable held-out
improvement over the validated fixed 4:4 split.
"""
from __future__ import annotations

import networkx as nx

from src.defense.action import DefenseAction
from src.defense.constraints import PortfolioConstraints
from src.optimizer.attacker_uncertainty import (
    ATTACKER_PROFILES,
    AttackerUncertaintyResult,
    attacker_uncertainty_codef_search,
    attacker_uncertainty_greedy_baseline,
    evaluate_attacker_uncertainty_portfolio,
    attacker_uncertainty_local_refine,
)
from src.optimizer.conditional_policy import (
    ConditionalPolicy,
    fit_conditional_codef_policy,
    observable_scenario_signal,
    observable_signal_evidence,
)


def pre_model_codef_search(
    graph: nx.DiGraph,
    actions: list[DefenseAction],
    constraints: PortfolioConstraints | None = None,
    initial_budget: float = 4.0,
    reserve_budget: float = 4.0,
    response_pool_size: int = 4,
    max_phase_one_paths: int | None = 12,
    max_response_delay: int | None = None,
    beam_width: int = 2,
    max_action_candidates: int | None = None,
) -> AttackerUncertaintyResult:
    """Run the validated pre-model CoDef policy with a fixed 4:4 default."""
    return attacker_uncertainty_codef_search(
        graph,
        ATTACKER_PROFILES,
        actions,
        initial_budget,
        reserve_budget,
        constraints=constraints,
        beam_width=beam_width,
        response_pool_size=response_pool_size,
        max_phase_one_paths=max_phase_one_paths,
        max_response_delay=max_response_delay,
        response_pool_mode="adaptive_diverse",
        worst_weight=0.5,
        variation_weight=0.2,
        max_action_candidates=max_action_candidates,
        method_name="TwoStage-CoDef-PreModel",
    )


def pre_model_greedy_baseline(
    graph: nx.DiGraph,
    actions: list[DefenseAction],
    constraints: PortfolioConstraints | None = None,
    initial_budget: float = 4.0,
    reserve_budget: float = 4.0,
    response_pool_size: int = 4,
    max_phase_one_paths: int | None = 12,
    max_response_delay: int | None = None,
) -> AttackerUncertaintyResult:
    """Evaluate the matching fixed-budget Greedy baseline."""
    return attacker_uncertainty_greedy_baseline(
        graph,
        ATTACKER_PROFILES,
        actions,
        initial_budget,
        reserve_budget,
        constraints=constraints,
        response_pool_size=response_pool_size,
        max_phase_one_paths=max_phase_one_paths,
        max_response_delay=max_response_delay,
        response_pool_mode="adaptive_diverse",
        worst_weight=0.5,
        variation_weight=0.2,
    )


def pre_model_conditional_codef_search(
    graph: nx.DiGraph,
    calibration_scenarios: list[nx.DiGraph] | tuple[nx.DiGraph, ...],
    actions: list[DefenseAction],
    constraints: PortfolioConstraints | None = None,
    initial_budget: float = 4.0,
    reserve_budget: float = 4.0,
    response_pool_size: int = 4,
    max_phase_one_paths: int | None = 12,
    max_response_delay: int | None = None,
    beam_width: int = 2,
    signal_mode: str = "route_family",
) -> AttackerUncertaintyResult:
    """Apply a guarded route-conditioned policy on top of the v2.4 entry.

    The v2.4 attacker-uncertainty search remains the public baseline. A
    conditional policy is fitted only from observable calibration telemetry,
    then scored under all three attacker profiles on ``graph``. If it does not
    improve the v2.4 objective, the baseline is returned unchanged. This makes
    route adaptation an evidence-backed optional layer rather than a risky
    replacement of the validated default.
    """
    if not calibration_scenarios:
        raise ValueError("at least one calibration scenario is required")
    baseline = pre_model_codef_search(
        graph,
        actions,
        constraints=constraints,
        initial_budget=initial_budget,
        reserve_budget=reserve_budget,
        response_pool_size=response_pool_size,
        max_phase_one_paths=max_phase_one_paths,
        max_response_delay=max_response_delay,
        beam_width=beam_width,
    )
    policy = fit_conditional_codef_policy(
        calibration_scenarios,
        actions,
        initial_budget,
        reserve_budget,
        constraints=constraints,
        response_pool_size=response_pool_size,
        max_phase_one_paths=max_phase_one_paths,
        max_response_delay=max_response_delay,
        objective_mode="minimax",
        signal_mode=signal_mode,
    )
    signal = observable_scenario_signal(graph, signal_mode)
    initial_ids = policy.initial_ids_by_signal.get(signal, policy.fallback_initial_ids)
    conditional = evaluate_attacker_uncertainty_portfolio(
        graph,
        ATTACKER_PROFILES,
        actions,
        initial_ids,
        reserve_budget,
        constraints=constraints,
        response_pool_size=response_pool_size,
        max_phase_one_paths=max_phase_one_paths,
        max_response_delay=max_response_delay,
        response_pool_mode="adaptive_diverse",
        worst_weight=0.5,
        variation_weight=0.2,
        method_name="TwoStage-CoDef-PreModel-Conditional",
    )
    if conditional.objective <= baseline.objective + 1e-12:
        diagnostics = dict(conditional.diagnostics)
        diagnostics.update({
            "pre_model_guard": "conditional_selected",
            "observable_signal": signal,
            "signal_mode": signal_mode,
            "baseline_objective": baseline.objective,
            "conditional_policy_diagnostics": policy.diagnostics,
        })
        return AttackerUncertaintyResult(
            conditional.method,
            conditional.initial_actions,
            conditional.profile_results,
            conditional.mean_terminal_risk,
            conditional.worst_terminal_risk,
            conditional.risk_stddev,
            conditional.objective,
            diagnostics,
        )
    diagnostics = dict(baseline.diagnostics)
    diagnostics.update({
        "pre_model_guard": "baseline_selected",
        "observable_signal": signal,
        "signal_mode": signal_mode,
        "conditional_objective": conditional.objective,
        "baseline_objective": baseline.objective,
        "conditional_policy_diagnostics": policy.diagnostics,
    })
    return AttackerUncertaintyResult(
        "TwoStage-CoDef-PreModel-Conditional-Guarded",
        baseline.initial_actions,
        baseline.profile_results,
        baseline.mean_terminal_risk,
        baseline.worst_terminal_risk,
        baseline.risk_stddev,
        baseline.objective,
        diagnostics,
    )


def v25_confidence_codef_search(
    graph: nx.DiGraph,
    calibration_scenarios: list[nx.DiGraph] | tuple[nx.DiGraph, ...],
    actions: list[DefenseAction],
    constraints: PortfolioConstraints | None = None,
    initial_budget: float = 4.0,
    reserve_budget: float = 4.0,
    response_pool_size: int = 4,
    max_phase_one_paths: int | None = 12,
    max_response_delay: int | None = None,
    beam_width: int = 2,
    signal_mode: str = "route_family",
    confidence_threshold: float = 0.55,
    min_improvement: float = 1e-6,
    max_neighbors_per_seed: int = 24,
    policy: ConditionalPolicy | None = None,
) -> AttackerUncertaintyResult:
    """Run v2.5 confidence-gated, profile-aware local refinement.

    The validated v2.4 result is always a candidate. A route-conditioned
    portfolio is considered only when current telemetry has enough separation
    and calibration support. A bounded local exchange then evaluates both
    portfolios under all attacker profiles. If the measured gain is below the
    requested margin, v2.4 is returned unchanged.
    """
    if not calibration_scenarios:
        raise ValueError("at least one calibration scenario is required")
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be between 0 and 1")
    baseline = pre_model_codef_search(
        graph,
        actions,
        constraints=constraints,
        initial_budget=initial_budget,
        reserve_budget=reserve_budget,
        response_pool_size=response_pool_size,
        max_phase_one_paths=max_phase_one_paths,
        max_response_delay=max_response_delay,
        beam_width=beam_width,
    )
    if policy is None:
        policy = fit_conditional_codef_policy(
            calibration_scenarios,
            actions,
            initial_budget,
            reserve_budget,
            constraints=constraints,
            response_pool_size=response_pool_size,
            max_phase_one_paths=max_phase_one_paths,
            max_response_delay=max_response_delay,
            objective_mode="minimax",
            signal_mode=signal_mode,
        )
    evidence = observable_signal_evidence(graph, policy, signal_mode)
    diagnostics_base = {
        "v25": True,
        "v25_confidence": evidence.confidence,
        "v25_confidence_threshold": confidence_threshold,
        "v25_signal": evidence.signal,
        "v25_signal_reason": evidence.reason,
        "v25_risk_margin": evidence.risk_margin,
        "v25_calibration_support": evidence.calibration_support,
        "v25_policy_diagnostics": policy.diagnostics,
    }
    if evidence.confidence < confidence_threshold:
        diagnostics = dict(baseline.diagnostics)
        diagnostics.update(diagnostics_base)
        diagnostics["v25_guard"] = "low_confidence_fallback"
        return AttackerUncertaintyResult(
            "TwoStage-CoDef-v25-ConfidenceGuarded",
            baseline.initial_actions,
            baseline.profile_results,
            baseline.mean_terminal_risk,
            baseline.worst_terminal_risk,
            baseline.risk_stddev,
            baseline.objective,
            diagnostics,
        )
    conditional_ids = policy.initial_ids_by_signal.get(evidence.signal, policy.fallback_initial_ids)
    baseline_ids = frozenset(action.id for action in baseline.initial_actions)
    if conditional_ids == baseline_ids:
        diagnostics = dict(baseline.diagnostics)
        diagnostics.update(diagnostics_base)
        diagnostics["v25_guard"] = "same_policy_no_refine"
        return AttackerUncertaintyResult(
            "TwoStage-CoDef-v25-ConfidenceGuarded",
            baseline.initial_actions,
            baseline.profile_results,
            baseline.mean_terminal_risk,
            baseline.worst_terminal_risk,
            baseline.risk_stddev,
            baseline.objective,
            diagnostics,
        )
    refined = attacker_uncertainty_local_refine(
        graph,
        ATTACKER_PROFILES,
        actions,
        (baseline_ids, conditional_ids),
        initial_budget,
        reserve_budget,
        constraints=constraints,
        response_pool_size=response_pool_size,
        max_phase_one_paths=max_phase_one_paths,
        max_response_delay=max_response_delay,
        response_pool_mode="adaptive_diverse",
        worst_weight=0.5,
        variation_weight=0.2,
        max_neighbors_per_seed=max_neighbors_per_seed,
        method_name="TwoStage-CoDef-v25-ConfidenceRefine",
    )
    if refined.objective < baseline.objective - min_improvement:
        diagnostics = dict(refined.diagnostics)
        diagnostics.update(diagnostics_base)
        diagnostics.update({
            "v25_guard": "refined_selected",
            "v25_baseline_objective": baseline.objective,
            "v25_refined_objective": refined.objective,
        })
        return AttackerUncertaintyResult(
            "TwoStage-CoDef-v25-ConfidenceRefined",
            refined.initial_actions,
            refined.profile_results,
            refined.mean_terminal_risk,
            refined.worst_terminal_risk,
            refined.risk_stddev,
            refined.objective,
            diagnostics,
        )
    diagnostics = dict(baseline.diagnostics)
    diagnostics.update(diagnostics_base)
    diagnostics.update({
        "v25_guard": "v24_baseline_retained",
        "v25_baseline_objective": baseline.objective,
        "v25_refined_objective": refined.objective,
    })
    return AttackerUncertaintyResult(
        "TwoStage-CoDef-v25-Guarded",
        baseline.initial_actions,
        baseline.profile_results,
        baseline.mean_terminal_risk,
        baseline.worst_terminal_risk,
        baseline.risk_stddev,
        baseline.objective,
        diagnostics,
    )
