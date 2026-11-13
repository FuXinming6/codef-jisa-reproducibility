from __future__ import annotations

from random import Random

import networkx as nx

from src.defense.action import DefenseAction
from src.defense.apply_defense import apply_defense, portfolio_cost
from src.defense.constraints import PortfolioConstraints
from src.optimizer.common import OptimizationResult, evaluate_portfolio
from src.risk.network_risk import assess_network_risk


def random_baseline(graph: nx.DiGraph, actions: list[DefenseAction], budget: float, trials: int = 100, seed: int = 7) -> OptimizationResult:
    """Sample feasible portfolios without enumerating every possible subset."""
    rng = Random(seed)
    candidates: list[tuple[DefenseAction, ...]] = [()]
    for _ in range(max(1, trials)):
        shuffled = actions[:]
        rng.shuffle(shuffled)
        selected: list[DefenseAction] = []
        for action in shuffled:
            if rng.random() < 0.5 and portfolio_cost(selected + [action]) <= budget:
                selected.append(action)
        candidates.append(tuple(selected))
    best = min(candidates, key=lambda combo: assess_network_risk(apply_defense(graph, combo)).score)
    return evaluate_portfolio(graph, best, "Random")


def risk_topk_baseline(graph: nx.DiGraph, actions: list[DefenseAction], budget: float) -> OptimizationResult:
    before = assess_network_risk(graph).score
    ranked = sorted(actions, key=lambda action: (before - assess_network_risk(apply_defense(graph, [action])).score) / action.cost, reverse=True)
    selected: list[DefenseAction] = []
    for action in ranked:
        if portfolio_cost(selected + [action]) <= budget:
            selected.append(action)
    return evaluate_portfolio(graph, selected, "Risk Top-K")


def greedy_baseline(
    graph: nx.DiGraph,
    actions: list[DefenseAction],
    budget: float,
    constraints: PortfolioConstraints | None = None,
    method_name: str = "Greedy",
) -> OptimizationResult:
    selected: list[DefenseAction] = []
    remaining = list(actions)
    current_graph = graph
    while remaining:
        current_risk = assess_network_risk(current_graph).score
        options = [
            action for action in remaining
            if portfolio_cost(selected + [action]) <= budget
            and (constraints is None or constraints.is_satisfied({item.id for item in selected} | {action.id}))
        ]
        if not options:
            break
        # Stable action-id tie-break avoids relying on container/insertion
        # order when several controls have exactly the same marginal value.
        best = min(
            options,
            key=lambda action: (
                -(current_risk - assess_network_risk(apply_defense(current_graph, [action])).score) / action.cost,
                action.id,
            ),
        )
        improvement = current_risk - assess_network_risk(apply_defense(current_graph, [best])).score
        if improvement <= 0:
            break
        selected.append(best)
        remaining.remove(best)
        current_graph = apply_defense(current_graph, [best])
    return evaluate_portfolio(graph, selected, method_name)


def genetic_baseline(
    graph: nx.DiGraph,
    actions: list[DefenseAction],
    budget: float,
    population_size: int = 32,
    generations: int = 30,
    mutation_rate: float = 0.08,
    seed: int = 7,
) -> OptimizationResult:
    """Deterministic budget-feasible binary genetic baseline for comparison."""
    if not actions:
        return evaluate_portfolio(graph, [], "Genetic")
    rng = Random(seed)

    def repair(chromosome: list[int]) -> list[int]:
        repaired = chromosome[:]
        while portfolio_cost([action for action, enabled in zip(actions, repaired) if enabled]) > budget:
            selected_indices = [index for index, enabled in enumerate(repaired) if enabled]
            if not selected_indices:
                break
            repaired[rng.choice(selected_indices)] = 0
        return repaired

    def selected(chromosome: list[int]) -> list[DefenseAction]:
        return [action for action, enabled in zip(actions, chromosome) if enabled]

    def score(chromosome: list[int]) -> float:
        return assess_network_risk(apply_defense(graph, selected(chromosome))).score

    population = [repair([rng.randint(0, 1) for _ in actions]) for _ in range(population_size)]
    population[0] = [0] * len(actions)
    for _ in range(generations):
        ranked = sorted(population, key=score)
        next_population = ranked[:2]
        while len(next_population) < population_size:
            first, second = rng.choice(ranked[: max(2, population_size // 2)]), rng.choice(ranked[: max(2, population_size // 2)])
            point = rng.randrange(1, len(actions)) if len(actions) > 1 else 1
            child = first[:point] + second[point:]
            child = [1 - bit if rng.random() < mutation_rate else bit for bit in child]
            next_population.append(repair(child))
        population = next_population
    best = min(population, key=score)
    return evaluate_portfolio(graph, selected(best), "Genetic")
