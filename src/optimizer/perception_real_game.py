"""Unified perception-real two-layer game for offline CoDef experiments.

The real graph is the sole source of terminal loss.  A protocol may expose
virtual endpoints only in the attacker perception layer; an attacker first
selects a route under the protocol's information model, then the defender
responds in the real graph from the observed prefix.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import prod

import networkx as nx

from src.defense.action import DefenseAction
from src.defense.apply_defense import apply_defense, portfolio_cost
from src.defense.constraints import PortfolioConstraints
from src.graph.path_search import find_attack_paths
from src.optimizer.two_stage import _foothold_index, _select_response


class AttackerChoiceRule(StrEnum):
    """How the attacker's phase-one option is selected."""

    REALIZED_WORST_CASE = "realized_worst_case"
    PERCEIVED_UTILITY = "perceived_utility"
    BLENDED_UTILITY = "blended_utility"


@dataclass(frozen=True)
class VirtualEndpoint:
    """An endpoint visible only in the attacker-perception layer."""

    id: str
    attachment_path: tuple[str, ...]
    perceived_asset_value: float
    cost: float = 2.0
    interception_probability: float = 0.60
    lure_multiplier: float = 1.0
    false_alert_probability: float = 0.02
    false_alert_penalty: float = 0.02


@dataclass(frozen=True)
class InformationState:
    """Attacker information that changes perceived, not real, utility."""

    pre_contact_awareness: float = 0.0
    global_suspicion: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.pre_contact_awareness <= 1.0:
            raise ValueError("pre_contact_awareness must be between zero and one")
        if not 0.0 <= self.global_suspicion <= 1.0:
            raise ValueError("global_suspicion must be between zero and one")


@dataclass(frozen=True)
class PerceptionRealProtocol:
    """Protocol parameters shared by v1.7, v1.8, and v1.9 special cases."""

    choice_rule: AttackerChoiceRule
    reserve_budget: float
    response_pool_size: int = 4
    response_delay: int | None = 1
    alert_response_delay: int | None = 2
    lure_strength: float = 0.35
    post_contact_recognition: float = 0.25
    perception_weight: float = 1.0
    response_ranking: str = "path_coverage"
    response_pool_mode: str = "adaptive_diverse"

    def __post_init__(self) -> None:
        if not 0.0 <= self.perception_weight <= 1.0:
            raise ValueError("perception_weight must be between zero and one")


@dataclass(frozen=True)
class GameEvaluation:
    """True outcome after an attacker choice and real-graph response."""

    total_loss: float
    terminal_loss: float
    operational_loss: float
    attacker_choice: str
    observed_path: tuple[str, ...]
    replanned_path: tuple[str, ...]
    response_ids: frozenset[str]
    virtual_endpoint_triggered: bool


def _path_loss(graph: nx.DiGraph, path: tuple[str, ...]) -> float:
    return prod(float(graph[source][target]["risk"]) for source, target in zip(path, path[1:])) * float(
        graph.nodes[path[-1]].get("asset_value", 1.0)
    )


def _endpoint_score(graph: nx.DiGraph, endpoint: VirtualEndpoint, suspicion: float, lure_strength: float) -> float:
    prefix_probability = prod(
        float(graph[source][target]["risk"])
        for source, target in zip(endpoint.attachment_path, endpoint.attachment_path[1:])
    )
    return prefix_probability * endpoint.perceived_asset_value * (
        1.0 + lure_strength * endpoint.lure_multiplier
    ) * (1.0 - suspicion)


def conditional_perception_threshold(perceived_gain: float, realized_penalty: float) -> float | None:
    """Return the blend weight at which a virtual option beats a real option.

    This is a pairwise result for normalized utilities: the virtual option has
    perception advantage ``perceived_gain`` and realized-loss disadvantage
    ``realized_penalty``. Portfolio and path switches can create several such
    thresholds, so this function intentionally does not claim a global value.
    """
    if perceived_gain <= 0.0:
        return None
    if realized_penalty <= 0.0:
        return 0.0
    return realized_penalty / (perceived_gain + realized_penalty)


def _response_from_real_path(
    real_graph: nx.DiGraph,
    actions: list[DefenseAction],
    initial_ids: frozenset[str],
    path: tuple[str, ...],
    constraints: PortfolioConstraints | None,
    protocol: PerceptionRealProtocol,
) -> tuple[float, tuple[str, ...], tuple[str, ...], frozenset[str]]:
    split = _foothold_index(path)
    if split is None:
        return 0.0, (), (), frozenset()
    observed = path[: split + 1]
    response = _select_response(
        real_graph,
        initial_ids,
        {action.id: action for action in actions},
        protocol.reserve_budget,
        observed,
        constraints,
        protocol.response_pool_size,
        protocol.response_delay,
        {},
        {},
        protocol.response_ranking,
        True,
        protocol.response_pool_mode,
    )
    return response.terminal_risk, observed, response.replanned_path, response.response_ids


def _response_from_virtual_contact(
    real_graph: nx.DiGraph,
    actions: list[DefenseAction],
    initial_ids: frozenset[str],
    endpoint: VirtualEndpoint,
    constraints: PortfolioConstraints | None,
    protocol: PerceptionRealProtocol,
) -> tuple[float, tuple[str, ...], tuple[str, ...], frozenset[str]]:
    """Apply a real-graph response from the virtual endpoint's exposed prefix."""
    observed = endpoint.attachment_path
    if not observed:
        return 0.0, (), (), frozenset()
    response = _select_response(
        real_graph,
        initial_ids,
        {action.id: action for action in actions},
        protocol.reserve_budget,
        observed,
        constraints,
        protocol.response_pool_size,
        protocol.alert_response_delay,
        {},
        {},
        protocol.response_ranking,
        True,
        protocol.response_pool_mode,
    )
    miss = 1.0 - endpoint.interception_probability * (1.0 - protocol.post_contact_recognition)
    return response.terminal_risk * miss, observed, response.replanned_path, response.response_ids


def evaluate_perception_real_game(
    graph: nx.DiGraph,
    actions: list[DefenseAction],
    initial_actions: tuple[DefenseAction, ...],
    endpoints: tuple[VirtualEndpoint, ...],
    constraints: PortfolioConstraints | None,
    protocol: PerceptionRealProtocol,
    information: InformationState = InformationState(),
    response_cache: dict[tuple[object, ...], tuple[float, tuple[str, ...], tuple[str, ...], frozenset[str]]] | None = None,
) -> GameEvaluation:
    """Evaluate one initial defense decision under a unified two-layer protocol."""
    real_graph = apply_defense(graph, initial_actions)
    initial_ids = frozenset(action.id for action in initial_actions)
    real_paths = find_attack_paths(real_graph)
    operational = sum(endpoint.false_alert_probability * endpoint.false_alert_penalty for endpoint in endpoints)
    if not real_paths:
        return GameEvaluation(operational, 0.0, operational, "", (), (), frozenset(), False)

    if protocol.choice_rule == AttackerChoiceRule.REALIZED_WORST_CASE:
        # v1.7 special case: every real phase-one path is evaluated after its
        # allowed response, then the attacker chooses the worst true outcome.
        candidates = []
        response_by_observed: dict[tuple[str, ...], tuple[float, tuple[str, ...], tuple[str, ...], frozenset[str]]] = {}
        for path in real_paths:
            split = _foothold_index(path)
            observed = path[: split + 1] if split is not None else ()
            if observed not in response_by_observed:
                cache_key = ("real", initial_ids, observed, protocol.reserve_budget, protocol.response_pool_size,
                             protocol.response_delay, protocol.response_ranking, protocol.response_pool_mode)
                if response_cache is not None and cache_key in response_cache:
                    response_by_observed[observed] = response_cache[cache_key]
                else:
                    outcome = _response_from_real_path(real_graph, actions, initial_ids, path, constraints, protocol)
                    response_by_observed[observed] = outcome
                    if response_cache is not None:
                        response_cache[cache_key] = outcome
            terminal, observed, replanned, response_ids = response_by_observed[observed]
            candidates.append((terminal, "real", observed, replanned, response_ids, False))
        terminal, choice, observed, replanned, response_ids, triggered = max(
            candidates, key=lambda item: (item[0], item[1], item[2]),
        )
    elif protocol.choice_rule == AttackerChoiceRule.PERCEIVED_UTILITY:
        # v1.8/v1.9 special case: score real routes and visible virtual assets
        # in the perception layer, but always score terminal loss in real_graph.
        perceived_options = [(_path_loss(real_graph, path), "real", path, None) for path in real_paths]
        if information.pre_contact_awareness < 1.0:
            for endpoint in endpoints:
                if all(real_graph.has_edge(source, target) for source, target in zip(endpoint.attachment_path, endpoint.attachment_path[1:])):
                    perceived_options.append((
                        _endpoint_score(real_graph, endpoint, information.global_suspicion, protocol.lure_strength),
                        endpoint.id,
                        endpoint.attachment_path,
                        endpoint,
                    ))
        _, choice, path, endpoint = max(perceived_options, key=lambda item: (item[0], item[1], item[2]))
        if endpoint is None:
            terminal, observed, replanned, response_ids = _response_from_real_path(
                real_graph, actions, initial_ids, path, constraints, protocol,
            )
            triggered = False
        else:
            terminal, observed, replanned, response_ids = _response_from_virtual_contact(
                real_graph, actions, initial_ids, endpoint, constraints, protocol,
            )
            triggered = True
    elif protocol.choice_rule == AttackerChoiceRule.BLENDED_UTILITY:
        # Each option receives both a perception-layer score and a real-layer
        # consequence. The normalized blend makes lambda=0 an anticipatory
        # attacker and lambda=1 a perception-only attacker.
        options = []
        response_by_observed: dict[tuple[str, ...], tuple[float, tuple[str, ...], tuple[str, ...], frozenset[str]]] = {}
        for path in real_paths:
            split = _foothold_index(path)
            observed = path[: split + 1] if split is not None else ()
            if observed not in response_by_observed:
                cache_key = ("real", initial_ids, observed, protocol.reserve_budget, protocol.response_pool_size,
                             protocol.response_delay, protocol.response_ranking, protocol.response_pool_mode)
                if response_cache is not None and cache_key in response_cache:
                    response_by_observed[observed] = response_cache[cache_key]
                else:
                    outcome = _response_from_real_path(real_graph, actions, initial_ids, path, constraints, protocol)
                    response_by_observed[observed] = outcome
                    if response_cache is not None:
                        response_cache[cache_key] = outcome
            terminal, observed, replanned, response_ids = response_by_observed[observed]
            options.append((_path_loss(real_graph, path), terminal, "real", observed, replanned, response_ids, False))
        if information.pre_contact_awareness < 1.0:
            for endpoint in endpoints:
                if not all(real_graph.has_edge(source, target) for source, target in zip(endpoint.attachment_path, endpoint.attachment_path[1:])):
                    continue
                cache_key = ("virtual", initial_ids, endpoint.attachment_path, protocol.reserve_budget,
                             protocol.response_pool_size, protocol.alert_response_delay,
                             protocol.response_ranking, protocol.response_pool_mode)
                if response_cache is not None and cache_key in response_cache:
                    terminal, observed, replanned, response_ids = response_cache[cache_key]
                else:
                    outcome = _response_from_virtual_contact(
                        real_graph, actions, initial_ids, endpoint, constraints, protocol,
                    )
                    terminal, observed, replanned, response_ids = outcome
                    if response_cache is not None:
                        response_cache[cache_key] = outcome
                options.append((
                    _endpoint_score(real_graph, endpoint, information.global_suspicion, protocol.lure_strength),
                    terminal,
                    endpoint.id,
                    observed,
                    replanned,
                    response_ids,
                    True,
                ))
        maximum_perceived = max(option[0] for option in options)
        maximum_terminal = max(option[1] for option in options)
        weight = protocol.perception_weight
        _, terminal, choice, observed, replanned, response_ids, triggered = max(
            options,
            key=lambda option: (
                weight * (option[0] / maximum_perceived if maximum_perceived else 0.0)
                + (1.0 - weight) * (option[1] / maximum_terminal if maximum_terminal else 0.0),
                option[2],
                option[3],
            ),
        )
    else:
        raise ValueError(f"Unsupported choice rule: {protocol.choice_rule}")
    total = terminal + operational
    return GameEvaluation(total, terminal, operational, choice, observed, replanned, response_ids, triggered)


def portfolio_cost_with_endpoints(
    actions: tuple[DefenseAction, ...], endpoints: tuple[VirtualEndpoint, ...],
) -> float:
    """Cost helper for a decision that mixes real and perception-layer actions."""
    return portfolio_cost(actions) + sum(endpoint.cost for endpoint in endpoints)
