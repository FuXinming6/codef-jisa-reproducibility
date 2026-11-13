"""v4.9 batch validation helpers for imported topology cases.

The helpers in this module are deliberately independent from the HTTP layer.
They keep validation and fair win/tie/loss aggregation deterministic so the
browser API and offline tests use the same protocol.
"""
from __future__ import annotations

from statistics import mean
from typing import Any, Sequence

from src.optimizer.external_audit import audit_topology_payload


def audit_batch_topologies(cases: Sequence[Any]) -> dict[str, object]:
    """Return one preflight row per case without running the solver."""
    rows: list[dict[str, object]] = []
    for index, case in enumerate(cases):
        case_id = f"case-{index + 1:02d}"
        topology: Any = None
        if isinstance(case, dict):
            case_id = str(case.get("case_id", case.get("id", case_id)))[:80] or case_id
            topology = case.get("topology")
        report = audit_topology_payload(topology)
        batch_compatible = bool(report.get("valid")) and int(report.get("node_count", 0)) >= 2 and int(report.get("edge_count", 0)) >= 1 and int(report.get("ingress_count", 0)) >= 1 and int(report.get("critical_count", 0)) >= 1
        rows.append({
            "case_id": case_id,
            "valid": batch_compatible,
            "batch_compatible": batch_compatible,
            "audit": report,
        })
    valid_count = sum(bool(row["valid"]) for row in rows)
    return {
        "case_count": len(rows),
        "valid_count": valid_count,
        "invalid_count": len(rows) - valid_count,
        "rows": rows,
    }


def summarize_batch_results(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    """Aggregate fair CoDef/Greedy results while preserving failed cases."""
    completed = [row for row in rows if bool(row.get("ok"))]
    winners = {"CoDef": 0, "Greedy": 0, "Tie": 0}
    for row in completed:
        winner = str(row.get("winner", "Tie"))
        if winner not in winners:
            winner = "Tie"
        winners[winner] += 1
    codef_values = [float(row["co_def_risk"]) for row in completed if row.get("co_def_risk") is not None]
    greedy_values = [float(row["greedy_risk"]) for row in completed if row.get("greedy_risk") is not None]
    return {
        "case_count": len(rows),
        "completed_count": len(completed),
        "failed_count": len(rows) - len(completed),
        "wins": winners["CoDef"],
        "ties": winners["Tie"],
        "losses": winners["Greedy"],
        "mean_codef_risk": round(mean(codef_values), 8) if codef_values else None,
        "mean_greedy_risk": round(mean(greedy_values), 8) if greedy_values else None,
    }
