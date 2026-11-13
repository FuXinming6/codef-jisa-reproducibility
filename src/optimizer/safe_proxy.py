"""v5.0 guarded proxy-model routing.

The learned model is an optional candidate-ordering aid.  It is only allowed
to narrow the routed candidate pool when its checkpoint was calibrated for the
same action catalog and graph scale.  The real evaluator and its quality guard
remain authoritative; incompatible imported topologies always fall back to
deterministic routing.
"""
from __future__ import annotations

from typing import Any, Sequence

import networkx as nx

from src.defense.action import DefenseAction


def safe_proxy_candidate_order(
    graph: nx.DiGraph,
    actions: Sequence[DefenseAction],
    proxy_model: Any,
    *,
    top_k: int = 6,
) -> tuple[list[str] | None, dict[str, object]]:
    """Return a guarded candidate order and an auditable gate report."""
    action_ids = [action.id for action in actions]
    report: dict[str, object] = {
        "version": "v5.0",
        "active": False,
        "status": "fallback",
        "top_k": max(1, int(top_k)),
        "graph_nodes": graph.number_of_nodes(),
        "action_count": len(action_ids),
        "candidate_ids": action_ids,
    }
    if proxy_model is None or not bool(getattr(proxy_model, "active", False)):
        report["status"] = str(getattr(proxy_model, "reason", "proxy_unavailable"))
        return None, report
    compatible = getattr(proxy_model, "compatible", None)
    if not callable(compatible) or not bool(compatible(graph)):
        report["status"] = "topology_not_calibrated"
        return None, report
    calibrated_actions = [str(action.id) for action in getattr(proxy_model, "actions", ())]
    if calibrated_actions != action_ids:
        report["status"] = "action_catalog_mismatch"
        return None, report
    try:
        ranking, scores = proxy_model.rank_actions(graph)
    except Exception as error:  # pragma: no cover - defensive fallback
        report["status"] = f"inference_error:{error}"
        return None, report
    limit = min(max(1, int(top_k)), len(action_ids))
    candidate_ids = [str(item) for item in ranking[:limit] if str(item) in set(action_ids)]
    if not candidate_ids:
        report["status"] = "empty_ranking"
        return None, report
    report.update({
        "active": True,
        "status": "ready",
        "candidate_ids": candidate_ids,
        "scores": {str(key): float(value) for key, value in scores.items() if str(key) in set(candidate_ids)},
    })
    return candidate_ids, report

