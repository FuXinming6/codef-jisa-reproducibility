"""Offline portfolio coupling constraints for the research simulator.

These rules describe deployment feasibility, not live-network configuration.
They let an experiment express that one simulated action needs another action,
that two actions cannot share a maintenance window, or that a group has a
limited simultaneous deployment capacity.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ActionGroupLimit:
    """Maximum number of actions that may be selected from one named group."""

    id: str
    actions: frozenset[str]
    max_selected: int

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Constraint group id cannot be empty")
        if not self.actions:
            raise ValueError(f"Constraint group {self.id} cannot be empty")
        if self.max_selected < 0:
            raise ValueError(f"Constraint group {self.id} max_selected must be non-negative")


@dataclass(frozen=True)
class PortfolioConstraints:
    """Hard feasibility rules for an abstract defense portfolio."""

    requirements: dict[str, frozenset[str]] = field(default_factory=dict)
    mutually_exclusive: tuple[frozenset[str], ...] = ()
    group_limits: tuple[ActionGroupLimit, ...] = ()

    def violation_reasons(self, selected_ids: set[str] | frozenset[str]) -> tuple[str, ...]:
        selected = frozenset(selected_ids)
        reasons: list[str] = []
        for action_id, required_ids in self.requirements.items():
            if action_id in selected and not required_ids.issubset(selected):
                missing = ", ".join(sorted(required_ids - selected))
                reasons.append(f"{action_id} requires {missing}")
        for exclusion in self.mutually_exclusive:
            present = sorted(exclusion & selected)
            if len(present) > 1:
                reasons.append(f"mutually exclusive: {', '.join(present)}")
        for limit in self.group_limits:
            selected_count = len(limit.actions & selected)
            if selected_count > limit.max_selected:
                reasons.append(f"{limit.id} allows at most {limit.max_selected}")
        return tuple(reasons)

    def allows_partial(self, selected_ids: set[str] | frozenset[str]) -> bool:
        """Check rules that can invalidate a partially constructed portfolio."""
        selected = frozenset(selected_ids)
        for exclusion in self.mutually_exclusive:
            if len(exclusion & selected) > 1:
                return False
        return all(len(limit.actions & selected) <= limit.max_selected for limit in self.group_limits)

    def is_satisfied(self, selected_ids: set[str] | frozenset[str]) -> bool:
        return not self.violation_reasons(selected_ids)


def _as_id_set(value: object, field_name: str) -> frozenset[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field_name} must be a non-empty list")
    return frozenset(str(item) for item in value)


def load_constraints_from_config(path: str | Path) -> PortfolioConstraints:
    """Load dependency, mutual-exclusion, and group-limit rules from YAML."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("Constraint config must be a mapping")

    requirements: dict[str, frozenset[str]] = {}
    for record in raw.get("requires", []):
        if not isinstance(record, dict) or "action" not in record:
            raise ValueError("Each requires record needs an action")
        action_id = str(record["action"])
        if action_id in requirements:
            raise ValueError(f"Duplicate requirement for {action_id}")
        requirements[action_id] = _as_id_set(record.get("requires"), "requires")

    mutually_exclusive = tuple(
        _as_id_set(record, "mutually_exclusive item") for record in raw.get("mutually_exclusive", [])
    )
    if any(len(record) < 2 for record in mutually_exclusive):
        raise ValueError("Each mutually_exclusive item needs at least two actions")

    group_limits: list[ActionGroupLimit] = []
    for record in raw.get("group_limits", []):
        if not isinstance(record, dict):
            raise ValueError("Each group_limits item must be a mapping")
        try:
            group_limits.append(
                ActionGroupLimit(
                    id=str(record["id"]),
                    actions=_as_id_set(record["actions"], "group actions"),
                    max_selected=int(record["max_selected"]),
                )
            )
        except KeyError as error:
            raise ValueError(f"Group limit is missing {error.args[0]}") from error
    if len({limit.id for limit in group_limits}) != len(group_limits):
        raise ValueError("Constraint group ids must be unique")
    return PortfolioConstraints(requirements, mutually_exclusive, tuple(group_limits))
