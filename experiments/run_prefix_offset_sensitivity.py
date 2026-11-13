"""Fixed-policy sensitivity to one-hop early or late attack-prefix observation."""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.chinanet_paper_protocol import FAMILIES, build_action_catalog, build_topology
from src.optimizer.attacker_uncertainty import ATTACKER_PROFILES
from src.optimizer.robust_two_stage import _aggregate_risk
from src.optimizer.two_stage import evaluate_two_stage_strategy
from src.simulator.scenario_variation import perturb_edge_risks


DEFAULT_INPUT = ROOT / "results" / "chinanet_paper" / "chinanet_main_50_20seed_runs.csv"
DEFAULT_OUTPUT = ROOT / "results" / "chinanet_paper" / "chinanet_prefix_offset_sensitivity_50_runs.csv"
DEFAULT_SUMMARY = ROOT / "results" / "chinanet_paper" / "chinanet_prefix_offset_sensitivity_50_summary.json"
METHODS = ("CoDef", "PairStart-Greedy")
OFFSETS = (-1, 0, 1)


def _bootstrap_mean_ci(values: list[float], seed: int, draws: int = 10000) -> list[float]:
    rng = random.Random(seed)
    n = len(values)
    means = sorted(statistics.mean(rng.choices(values, k=n)) for _ in range(draws))
    return [means[int(0.025 * draws)], means[min(draws - 1, int(0.975 * draws))]]


def _read_inputs(path: Path) -> dict[tuple[str, int], dict[str, dict[str, str]]]:
    grouped: dict[tuple[str, int], dict[str, dict[str, str]]] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if int(row["scale"]) != 50 or row["method"] not in METHODS:
                continue
            key = (row["family"], int(row["seed"]))
            grouped.setdefault(key, {})[row["method"]] = row
    incomplete = [key for key, rows in grouped.items() if set(rows) != set(METHODS)]
    if incomplete:
        raise ValueError(f"missing paired method rows for {incomplete[:5]}")
    if len(grouped) != 60:
        raise ValueError(f"expected 60 paired primary graphs, found {len(grouped)}")
    return grouped


def _evaluate(graph, actions, constraints, initial_ids: frozenset[str], reserve: float,
              path_cap: int, pool_size: int, offset: int) -> tuple[float, list[float]]:
    risks = []
    for profile in ATTACKER_PROFILES:
        result = evaluate_two_stage_strategy(
            graph,
            actions,
            initial_ids,
            reserve,
            constraints=constraints,
            response_pool_size=pool_size,
            max_phase_one_paths=path_cap,
            response_pool_mode="adaptive_diverse",
            attacker_profile=profile,
            response_scope="local",
            replan_attacker=True,
            foothold_offset=offset,
        )
        risks.append(float(result.terminal_risk))
    return _aggregate_risk(risks, 0.5, 0.2)[3], risks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()

    inputs = _read_inputs(args.input)
    graph_rows: list[dict[str, object]] = []
    for (family, seed), method_rows in sorted(
        inputs.items(), key=lambda item: (FAMILIES.index(item[0][0]), item[0][1])
    ):
        scale = 50
        graph_seed = seed + scale * 1009 + FAMILIES.index(family) * 7919
        graph = perturb_edge_risks(
            build_topology(family, scale, graph_seed), seed + 200_003
        )
        actions, constraints = build_action_catalog(graph)
        for method in METHODS:
            row = method_rows[method]
            initial_ids = frozenset(filter(None, row["initial_actions"].split(";")))
            reserve = float(row["reserve_budget"])
            path_cap = int(row["phase_one_path_cap"])
            pool_size = int(row["response_pool_size"])
            baseline_objective = float(row["objective"])
            for offset in OFFSETS:
                objective, risks = _evaluate(
                    graph, actions, constraints, initial_ids, reserve,
                    path_cap, pool_size, offset,
                )
                if offset == 0 and abs(objective - baseline_objective) > 1e-8:
                    raise ValueError(
                        f"offset-zero mismatch {family}/{seed}/{method}: "
                        f"saved={baseline_objective:.12g}, replay={objective:.12g}"
                    )
                graph_rows.append({
                    "family": family,
                    "scale": scale,
                    "seed": seed,
                    "method": method,
                    "foothold_offset_edges": offset,
                    "initial_actions": row["initial_actions"],
                    "objective": objective,
                    "risk_max": risks[0],
                    "shortest_path": risks[1],
                    "asset_targeted": risks[2],
                    "baseline_objective": baseline_objective,
                })
        print(f"completed family={family} seed={seed}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(graph_rows[0]))
        writer.writeheader()
        writer.writerows(graph_rows)

    by_key = {
        (row["family"], row["seed"], row["method"], row["foothold_offset_edges"]): float(row["objective"])
        for row in graph_rows
    }
    summary: dict[str, object] = {
        "protocol": {
            "unit": "graph",
            "graphs": 60,
            "methods": list(METHODS),
            "profiles": list(ATTACKER_PROFILES),
            "offsets_in_edges": list(OFFSETS),
            "initial_portfolios": "held fixed from the primary benchmark",
            "response_and_attacker_continuation": "recomputed at each shifted prefix",
            "interpretation": "fixed-policy observation-timing sensitivity; not a detector false-positive or probabilistic-noise model",
            "bootstrap_draws": 10000,
            "bootstrap_seed": 20261112,
        },
        "overall": {},
        "families": {},
    }
    for offset in OFFSETS:
        deltas: list[float] = []
        wins = ties = losses = 0
        for family, seed in inputs:
            codef = by_key[(family, seed, "CoDef", offset)]
            pair = by_key[(family, seed, "PairStart-Greedy", offset)]
            delta = codef - pair
            deltas.append(delta)
            wins += delta < -1e-12
            ties += abs(delta) <= 1e-12
            losses += delta > 1e-12
        summary["overall"][str(offset)] = {
            "graphs": len(deltas),
            "codef_lower_tied_higher": [wins, ties, losses],
            "mean_delta_codef_minus_pairstart": statistics.mean(deltas),
            "mean_delta_ci_95_graph_bootstrap": _bootstrap_mean_ci(
                deltas, 20261112 + offset + 1
            ),
        }
        summary["families"][str(offset)] = {}
        for family in FAMILIES:
            family_deltas = [
                by_key[(item_family, seed, "CoDef", offset)]
                - by_key[(item_family, seed, "PairStart-Greedy", offset)]
                for item_family, seed in inputs if item_family == family
            ]
            summary["families"][str(offset)][family] = {
                "graphs": len(family_deltas),
                "codef_lower_tied_higher": [
                    sum(value < -1e-12 for value in family_deltas),
                    sum(abs(value) <= 1e-12 for value in family_deltas),
                    sum(value > 1e-12 for value in family_deltas),
                ],
                "mean_delta_codef_minus_pairstart": statistics.mean(family_deltas),
            }
    args.summary.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"saved {args.output}")
    print(f"saved {args.summary}")


if __name__ == "__main__":
    main()
