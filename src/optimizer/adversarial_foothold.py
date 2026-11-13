"""v1.7 all-path adversarial foothold model.

The v1.6 evaluator already evaluates a defender response before taking the
worst retained phase-one outcome.  v1.7 removes that retained-path truncation:
the attacker selects the worst post-response result among every reachable
phase-one attack path and its deterministic midpoint foothold.
"""
from __future__ import annotations

from dataclasses import replace

import networkx as nx

from src.defense.action import DefenseAction
from src.defense.constraints import PortfolioConstraints
from src.graph.path_search import find_attack_paths
from src.optimizer.common import TwoStageResult
from src.optimizer.two_stage import two_stage_codef_search, two_stage_greedy_baseline


def _tag_adversarial_result(graph: nx.DiGraph, result: TwoStageResult) -> TwoStageResult:
    diagnostics = dict(result.diagnostics or {})
    diagnostics.update({
        "attacker_model": "all_reachable_phase_one_paths",
        "phase_one_path_scope": "all_paths",
        "phase_one_path_count": len(find_attack_paths(graph)),
        "foothold_rule": "deterministic_path_midpoint",
    })
    return replace(result, diagnostics=diagnostics)


def two_stage_adversarial_greedy_baseline(
    graph: nx.DiGraph,
    actions: list[DefenseAction],
    initial_budget: float,
    reserve_budget: float,
    constraints: PortfolioConstraints | None = None,
    response_pool_size: int = 4,
    max_response_delay: int | None = None,
    response_ranking: str = "path_coverage",
    use_response_cache: bool = True,
    response_pool_mode: str = "adaptive_diverse",
    method_name: str = "TwoStage-Greedy-Adversarial",
) -> TwoStageResult:
    """Run the v1.7 all-path attacker model with a Greedy initial defense."""
    result = two_stage_greedy_baseline(
        graph, actions, initial_budget, reserve_budget, constraints=constraints,
        response_pool_size=response_pool_size, max_phase_one_paths=None,
        max_response_delay=max_response_delay, response_ranking=response_ranking,
        use_response_cache=use_response_cache, response_pool_mode=response_pool_mode,
        method_name=method_name,
    )
    return _tag_adversarial_result(graph, result)


def two_stage_adversarial_codef_search(
    graph: nx.DiGraph,
    actions: list[DefenseAction],
    initial_budget: float,
    reserve_budget: float,
    beam_width: int = 2,
    constraints: PortfolioConstraints | None = None,
    response_pool_size: int = 4,
    max_response_delay: int | None = None,
    response_ranking: str = "path_coverage",
    use_response_cache: bool = True,
    response_pool_mode: str = "adaptive_diverse",
    use_initial_exchange: bool = False,
    exchange_pool_size: int = 4,
    exchange_max_removed: int = 2,
    exchange_eval_limit: int = 12,
    method_name: str = "TwoStage-CoDef-Adversarial",
) -> TwoStageResult:
    """Run v1.7 CoDef against the all-reachable-path attacker.

    The underlying guarded search keeps the matching all-path Greedy portfolio
    as a candidate, so the returned policy cannot be worse than this baseline
    under the same abstract graph and response restrictions.
    """
    result = two_stage_codef_search(
        graph, actions, initial_budget, reserve_budget, beam_width=beam_width,
        constraints=constraints, response_pool_size=response_pool_size,
        max_phase_one_paths=None, max_response_delay=max_response_delay,
        response_ranking=response_ranking, use_response_cache=use_response_cache,
        response_pool_mode=response_pool_mode, use_initial_exchange=use_initial_exchange,
        exchange_pool_size=exchange_pool_size, exchange_max_removed=exchange_max_removed,
        exchange_eval_limit=exchange_eval_limit, method_name=method_name,
    )
    return _tag_adversarial_result(graph, result)
