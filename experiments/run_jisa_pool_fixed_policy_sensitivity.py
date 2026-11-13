"""Rescore saved primary portfolios under adaptive response-pool sizes 2/4/8."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.chinanet_paper_protocol import DEFAULT_SEEDS, FAMILIES, build_action_catalog, build_topology
from src.optimizer.attacker_uncertainty import (
    ATTACKER_PROFILES, evaluate_attacker_uncertainty_portfolio,
)
from src.simulator.scenario_variation import perturb_edge_risks

OUT = ROOT / "results" / "chinanet_paper"
INPUT = OUT / "chinanet_main_50_20seed_runs.csv"
RAW = OUT / "pool_fixed_policy_sensitivity_12_20261112.csv"
SUMMARY = OUT / "pool_fixed_policy_sensitivity_12_20261112_summary.json"
SIZES = (2, 4, 8)
SEEDS = DEFAULT_SEEDS[:4]
METHODS = ("CoDef", "PairStart-Greedy")
TOL = 1e-12


def sign_p(wins: int, losses: int) -> float:
    n = wins + losses
    if not n:
        return 1.0
    tail = min(wins, losses)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(tail + 1)) / (2**n))


def bootstrap(values: list[float], seed: int, draws: int = 10000) -> list[float]:
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choices(values, k=len(values))) for _ in range(draws))
    return [means[int(.025 * (draws - 1))], means[int(.975 * (draws - 1))]]


def main() -> None:
    with INPUT.open(newline="", encoding="utf-8-sig") as handle:
        all_rows = list(csv.DictReader(handle))
    stored = {
        (row["family"], int(row["seed"]), row["method"]): row
        for row in all_rows if int(row["scale"]) == 50 and int(row["seed"]) in SEEDS
    }
    if len(stored) != len(FAMILIES) * len(SEEDS) * 5:
        raise RuntimeError("Primary source CSV does not contain all selected graph/method rows")

    rows = []
    for family in FAMILIES:
        for seed in SEEDS:
            graph_seed = seed + 50 * 1009 + FAMILIES.index(family) * 7919
            graph = perturb_edge_risks(build_topology(family, 50, graph_seed), seed + 200_003)
            actions, constraints = build_action_catalog(graph)
            expected_graph = stored[(family, seed, "CoDef")]
            if int(expected_graph["node_count"]) != graph.number_of_nodes() or int(expected_graph["edge_count"]) != graph.number_of_edges():
                raise RuntimeError(f"Graph-size mismatch for {family} seed {seed}")
            for pool_size in SIZES:
                for method in METHODS:
                    source = stored[(family, seed, method)]
                    initial_ids = frozenset(x for x in source["initial_actions"].split(";") if x)
                    result = evaluate_attacker_uncertainty_portfolio(
                        graph, ATTACKER_PROFILES, actions, initial_ids, 4.0,
                        constraints=constraints,
                        response_pool_size=pool_size,
                        max_phase_one_paths=4,
                        response_pool_mode="adaptive_diverse",
                        method_name=f"{method}-fixed-policy-pool-{pool_size}",
                    )
                    delta_from_saved = float(result.objective) - float(source["objective"])
                    row = {
                        "family": family, "scale": 50, "seed": seed,
                        "method": method, "configured_pool_base": pool_size,
                        "adaptive_pool_maximum": pool_size + 2,
                        "initial_actions": ";".join(sorted(initial_ids)),
                        "objective": float(result.objective),
                        "saved_primary_objective": float(source["objective"]),
                        "pool2_replay_delta": delta_from_saved if pool_size == 2 else "",
                        "profiles": json.dumps([
                            {"profile": item.profile, "terminal_risk": float(item.result.terminal_risk),
                             "response_actions": [a.id for a in item.result.response_actions]}
                            for item in result.profile_results
                        ], ensure_ascii=False, separators=(",", ":")),
                        "graph_sha256": hashlib.sha256(json.dumps(
                            {"nodes": list(graph.nodes(data=True)), "edges": list(graph.edges(data=True))},
                            ensure_ascii=False, sort_keys=True, default=str,
                        ).encode("utf-8")).hexdigest(),
                        "synthetic_data": True,
                    }
                    if pool_size == 2 and abs(delta_from_saved) > 1e-8:
                        raise RuntimeError(
                            f"offset-zero replay mismatch {family} seed={seed} {method}: {delta_from_saved}"
                        )
                    rows.append(row)
                    print(
                        f"{family} seed={seed} pool={pool_size} {method} J={result.objective:.8f} "
                        f"replay_delta={delta_from_saved:+.2e}", flush=True,
                    )

    fields = list(rows[0])
    with RAW.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summaries = []
    raw_p = []
    for pool_size in SIZES:
        paired = {}
        for row in rows:
            if int(row["configured_pool_base"]) == pool_size:
                paired.setdefault((row["family"], int(row["seed"])), {})[row["method"]] = float(row["objective"])
        deltas = [v["CoDef"] - v["PairStart-Greedy"] for _, v in sorted(paired.items())]
        wins = sum(d < -TOL for d in deltas)
        ties = sum(abs(d) <= TOL for d in deltas)
        losses = sum(d > TOL for d in deltas)
        p = sign_p(wins, losses)
        raw_p.append(p)
        summaries.append({
            "configured_pool_base": pool_size,
            "adaptive_pool_maximum": pool_size + 2,
            "graphs": len(deltas),
            "codef_lower_tied_higher": [wins, ties, losses],
            "mean_delta_codef_minus_pairstart": statistics.fmean(deltas),
            "mean_delta_graph_bootstrap_95_ci": bootstrap(deltas, 2026111200 + pool_size),
            "exact_sign_p_two_sided": p,
        })
    order = sorted(range(len(raw_p)), key=raw_p.__getitem__)
    adj = [0.0] * len(raw_p)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * raw_p[idx]))
        adj[idx] = running
    for row, p in zip(summaries, adj):
        row["holm_p_across_three_pool_sizes"] = p
    payload = {
        "protocol": {
            "analysis": "fixed-policy replay: saved initial portfolios are not reoptimized; response and continuation are recomputed",
            "graphs": 12, "families": list(FAMILIES), "seeds_per_family": list(SEEDS),
            "methods": list(METHODS), "configured_adaptive_pool_base_sizes": list(SIZES),
            "maximum_pool_size_rule": "base + 2 if the base-th ranked response has positive coverage, capped by eligible actions",
            "budgets": [4.0, 4.0], "path_cap": 4, "profiles": list(ATTACKER_PROFILES),
            "tie_tolerance": TOL, "bootstrap_draws": 10000,
            "source_primary_csv": str(INPUT),
            "interpretation_limit": "tests evaluator pool sensitivity for fixed portfolios, not performance after reoptimization",
        },
        "by_pool_size": summaries,
    }
    SUMMARY.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {RAW} and {SUMMARY}", flush=True)


if __name__ == "__main__":
    main()
