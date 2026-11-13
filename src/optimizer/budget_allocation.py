"""Robust selection of a two-stage defense budget split.

The selector deliberately chooses one B1:B2 split before held-out scenarios
are evaluated.  It ranks candidate splits using only a calibration ensemble,
which makes the allocation rule reproducible and avoids choosing a split with
test-set terminal risk.
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, stdev
from typing import Mapping, Sequence


BudgetSplit = tuple[float, float]


@dataclass(frozen=True)
class BudgetSplitScore:
    """Calibration performance of one fixed-total budget split."""

    initial_budget: float
    reserve_budget: float
    mean_terminal_risk: float
    risk_stddev: float
    scenario_count: int


def select_robust_budget_split(
    candidate_risks: Mapping[BudgetSplit, Sequence[float]],
    total_budget: float,
) -> tuple[BudgetSplit, tuple[BudgetSplitScore, ...]]:
    """Choose the lowest-mean split, then prefer lower variation on ties.

    ``candidate_risks`` must contain non-empty calibration risks for each
    candidate.  All candidates must use exactly the supplied total budget.
    The returned score list is ordered by the same deterministic policy.
    """
    if total_budget < 0:
        raise ValueError("total_budget must be non-negative")
    if not candidate_risks:
        raise ValueError("at least one candidate budget split is required")
    scores: list[BudgetSplitScore] = []
    for (initial_budget, reserve_budget), risks in candidate_risks.items():
        if initial_budget < 0 or reserve_budget < 0:
            raise ValueError("budget splits must be non-negative")
        if abs(initial_budget + reserve_budget - total_budget) > 1e-12:
            raise ValueError("all candidate splits must use the same total budget")
        if not risks:
            raise ValueError("each candidate split requires calibration risks")
        if any(risk < 0 for risk in risks):
            raise ValueError("terminal risks must be non-negative")
        scores.append(BudgetSplitScore(
            initial_budget=initial_budget,
            reserve_budget=reserve_budget,
            mean_terminal_risk=mean(risks),
            risk_stddev=stdev(risks) if len(risks) > 1 else 0.0,
            scenario_count=len(risks),
        ))
    ranked = tuple(sorted(
        scores,
        key=lambda score: (
            score.mean_terminal_risk,
            score.risk_stddev,
            score.initial_budget,
            score.reserve_budget,
        ),
    ))
    selected = (ranked[0].initial_budget, ranked[0].reserve_budget)
    return selected, ranked
