"""Create publication-facing descriptive summaries from the completed run CSV.

Absolute within-method changes are used because some base objectives are zero;
relative changes would be undefined or depend on an arbitrary denominator.
"""
from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "chinanet_paper" / "cost_effect_value_sensitivity_20261112"
RAW = OUT / "runs.csv"
PROTOCOL = OUT / "protocol.json"
OUTPUT = OUT / "paper_summary.json"
REPORT = OUT / "paper_report.md"
TOLERANCE = 1e-12


def summarize() -> dict[str, object]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    with RAW.open(newline="", encoding="utf-8") as stream:
        raw_rows = list(csv.DictReader(stream))
    conditions = protocol["conditions"]
    families = protocol["families"]
    seeds = [int(value) for value in protocol["seeds"]]
    methods = protocol["methods"]
    expected = len(conditions) * len(families) * len(seeds) * len(methods)
    if len(raw_rows) != expected:
        raise RuntimeError(f"expected {expected} rows; found {len(raw_rows)}")
    rows = {
        (row["family"], int(row["seed"]), row["condition"], row["method"]): row
        for row in raw_rows
    }
    if len(rows) != expected:
        raise RuntimeError("the raw CSV contains duplicate graph/condition/method keys")

    result: dict[str, object] = {
        "experiment": protocol["experiment"],
        "raw_rows": len(raw_rows),
        "expected_raw_rows": expected,
        "tie_tolerance": TOLERANCE,
        "within_method_change_metric": "absolute objective difference from the same graph's base condition; no denominator is used",
        "conditions": {},
    }
    for condition in conditions:
        family_summary: dict[str, object] = {}
        deltas: list[float] = []
        absolute_changes: dict[str, list[float]] = {method: [] for method in methods}
        counts = {"codef_lower": 0, "tie": 0, "codef_higher": 0}
        for family in families:
            family_deltas: list[float] = []
            family_changes: dict[str, list[float]] = {method: [] for method in methods}
            family_counts = {"codef_lower": 0, "tie": 0, "codef_higher": 0}
            for seed in seeds:
                codef = float(rows[(family, seed, condition, "CoDef")]["objective"])
                pair = float(rows[(family, seed, condition, "PairStart-Greedy")]["objective"])
                delta = codef - pair
                family_deltas.append(delta)
                if delta < -TOLERANCE:
                    family_counts["codef_lower"] += 1
                elif delta > TOLERANCE:
                    family_counts["codef_higher"] += 1
                else:
                    family_counts["tie"] += 1
                for method, objective in (("CoDef", codef), ("PairStart-Greedy", pair)):
                    base = float(rows[(family, seed, "base", method)]["objective"])
                    change = abs(objective - base)
                    family_changes[method].append(change)
                    absolute_changes[method].append(change)
            for key in counts:
                counts[key] += family_counts[key]
            deltas.extend(family_deltas)
            family_summary[family] = {
                "n": len(family_deltas),
                "codef_lower_tied_higher": [
                    family_counts["codef_lower"], family_counts["tie"], family_counts["codef_higher"]
                ],
                "mean_delta_codef_minus_pairstart": statistics.mean(family_deltas),
                "median_delta_codef_minus_pairstart": statistics.median(family_deltas),
                "median_absolute_objective_change_vs_base": {
                    method: statistics.median(values) for method, values in family_changes.items()
                },
                "mean_absolute_objective_change_vs_base": {
                    method: statistics.mean(values) for method, values in family_changes.items()
                },
            }
        result["conditions"][condition] = {
            "n": len(deltas),
            "codef_lower_tied_higher": [counts["codef_lower"], counts["tie"], counts["codef_higher"]],
            "mean_delta_codef_minus_pairstart": statistics.mean(deltas),
            "median_delta_codef_minus_pairstart": statistics.median(deltas),
            "median_absolute_objective_change_vs_base": {
                method: statistics.median(values) for method, values in absolute_changes.items()
            },
            "mean_absolute_objective_change_vs_base": {
                method: statistics.mean(values) for method, values in absolute_changes.items()
            },
            "families": family_summary,
        }
    return result


def main() -> None:
    summary = summarize()
    OUTPUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    labels = {
        "base": "Base",
        "cost_only": "Costs ±15%",
        "effect_only": "Effects ±15%",
        "asset_only": "Assets ±15%",
        "joint": "Joint ±15%",
    }
    family_order = [("overall", "Overall"), ("er_dag", "ER-like"), ("hierarchical", "Hierarchical"), ("multipath", "Multipath")]
    lines = [
        "# JISA synthetic parameter sensitivity",
        "",
        "The same 60 fixed 50-node graph/edge-risk instances are reoptimized under the registered conditions. These perturbations are synthetic sensitivity probes, not empirical estimates, independent topology samples, or confirmatory tests.",
        "",
        "Pairwise ΔJ is CoDef minus PairStart-Greedy, so negative values favor CoDef. Lower/tie/higher counts use an absolute tolerance of 1e-12. Within-method changes are absolute objective differences from the same graph's base condition; this avoids undefined relative changes where base J is zero.",
        "",
        "|Condition|Family|N|CoDef lower / tie / higher|Mean paired ΔJ|Median absolute ΔJ vs base (CoDef / PairStart)|",
        "|---|---|---:|---:|---:|---:|",
    ]
    for condition in ("base", "cost_only", "effect_only", "asset_only", "joint"):
        condition_data = summary["conditions"][condition]
        for family_key, family_label in family_order:
            item = condition_data if family_key == "overall" else condition_data["families"][family_key]
            changes = item["median_absolute_objective_change_vs_base"]
            lower, ties, higher = item["codef_lower_tied_higher"]
            lines.append(
                f"|{labels[condition]}|{family_label}|{item['n']}|{lower}/{ties}/{higher}|"
                f"{item['mean_delta_codef_minus_pairstart']:+.6f}|"
                f"{changes['CoDef']:.6f} / {changes['PairStart-Greedy']:.6f}|"
            )
    lines.extend(["", "No inferential p-values are reported because the parameter settings are controlled probes on the same graph instances.", ""])
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {OUTPUT} and {REPORT}")


if __name__ == "__main__":
    main()
