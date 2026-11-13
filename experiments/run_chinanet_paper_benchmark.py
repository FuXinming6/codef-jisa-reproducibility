"""Paired topology-family benchmark for the ChinaNet 2026 paper branch."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.chinanet_paper_protocol import DEFAULT_MAX_PATHS, DEFAULT_RESPONSE_POOL, DEFAULT_SEEDS, FAMILIES, build_action_catalog, build_topology
from src.optimizer.attacker_uncertainty import ATTACKER_PROFILES, attacker_uncertainty_codef_search, attacker_uncertainty_greedy_baseline, attacker_uncertainty_multistart_greedy_baseline, attacker_uncertainty_static_baseline
from src.simulator.scenario_variation import perturb_edge_risks


METHODS = ("Greedy", "RiskTopK", "MultiStart-Greedy", "PairStart-Greedy", "CoDef")
RAW_FIELDS = ("family", "scale", "seed", "method", "objective", "mean_profile_risk", "worst_profile_risk", "risk_stddev", "runtime_seconds", "node_count", "edge_count", "ingress_count", "critical_count", "action_count", "initial_actions", "initial_cost", "profile_evidence_json", "evaluated_portfolios", "expanded_candidates", "phase_one_path_cap", "response_pool_size", "initial_budget", "reserve_budget", "response_pool_mode", "synthetic_data")


def _run_method(method: str, graph: object, actions: list, kwargs: dict):
    if method == "Greedy":
        return attacker_uncertainty_greedy_baseline(graph, ATTACKER_PROFILES, actions, **kwargs)
    if method == "RiskTopK":
        return attacker_uncertainty_static_baseline(graph, ATTACKER_PROFILES, actions, strategy="risk_topk", **kwargs)
    if method == "MultiStart-Greedy":
        return attacker_uncertainty_multistart_greedy_baseline(graph, ATTACKER_PROFILES, actions, **kwargs)
    if method == "PairStart-Greedy":
        return attacker_uncertainty_multistart_greedy_baseline(graph, ATTACKER_PROFILES, actions, start_size=2, **kwargs)
    if method == "CoDef":
        return attacker_uncertainty_codef_search(graph, ATTACKER_PROFILES, actions, **kwargs)
    raise ValueError(f"unknown method: {method}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=(*FAMILIES, "all"), default="all")
    parser.add_argument("--scale", choices=(50, 100, 200), type=int, nargs="+", default=(50, 100))
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--max-paths", type=int, default=DEFAULT_MAX_PATHS)
    parser.add_argument("--response-pool-size", type=int, default=DEFAULT_RESPONSE_POOL)
    parser.add_argument("--initial-budget", type=float, default=4.0)
    parser.add_argument("--reserve-budget", type=float, default=4.0)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "chinanet_paper")
    parser.add_argument("--output-tag", default="benchmark")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    args = parser.parse_args()
    families = FAMILIES if args.family == "all" else (args.family,)
    seeds = tuple(args.seeds or DEFAULT_SEEDS)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / f"chinanet_{args.output_tag}_runs.csv"
    completed: set[tuple[str, int, int, str]] = set()
    if args.resume and raw_path.exists():
        with raw_path.open(newline="", encoding="utf-8") as handle:
            completed = {(row["family"], int(row["scale"]), int(row["seed"]), row["method"]) for row in csv.DictReader(handle)}
    write_header = not raw_path.exists() or not args.resume
    with raw_path.open("w" if write_header else "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_FIELDS)
        if write_header:
            writer.writeheader()
        for family in families:
            for scale in args.scale:
                for seed in seeds:
                    graph_seed = seed + scale * 1009 + FAMILIES.index(family) * 7919
                    base_graph = build_topology(family, scale, graph_seed)
                    scenario = perturb_edge_risks(base_graph, seed + 200_003)
                    actions, constraints = build_action_catalog(scenario)
                    instance_results: dict[str, object] = {}
                    for method in args.methods:
                        if (family, scale, seed, method) in completed:
                            continue
                        kwargs = {
                            "initial_budget": args.initial_budget,
                            "reserve_budget": args.reserve_budget,
                            "constraints": constraints,
                            "response_pool_size": args.response_pool_size,
                            "max_phase_one_paths": args.max_paths,
                            "response_pool_mode": "adaptive_diverse",
                        }
                        if method == "CoDef":
                            kwargs["method_name"] = "ChinaNet-Paper-CoDef"
                        elif method == "RiskTopK":
                            kwargs["seed"] = seed
                        started = perf_counter()
                        result = _run_method(method, scenario, actions, kwargs)
                        elapsed = perf_counter() - started
                        instance_results[method] = result
                        selected = result.initial_actions
                        diagnostics = result.diagnostics
                        profile_evidence = [
                            {
                                "profile": profile.profile,
                                "terminal_risk": float(profile.result.terminal_risk),
                                "foothold": profile.result.foothold,
                                "observed_path": profile.result.observed_path,
                                "replanned_path": profile.result.replanned_path,
                                "response_actions": [action.id for action in profile.result.response_actions],
                            }
                            for profile in result.profile_results
                        ]
                        writer.writerow({
                            "family": family, "scale": scale, "seed": seed, "method": method,
                            "objective": float(result.objective), "mean_profile_risk": float(result.mean_terminal_risk),
                            "worst_profile_risk": float(result.worst_terminal_risk), "risk_stddev": float(result.risk_stddev),
                            "runtime_seconds": round(elapsed, 6), "node_count": scenario.number_of_nodes(),
                            "edge_count": scenario.number_of_edges(),
                            "ingress_count": sum(data.get("type") == "internet" for _, data in scenario.nodes(data=True)),
                            "critical_count": sum(bool(data.get("critical")) for _, data in scenario.nodes(data=True)),
                            "action_count": len(actions), "initial_actions": ";".join(action.id for action in selected),
                            "initial_cost": sum(action.cost for action in selected),
                            "profile_evidence_json": json.dumps(profile_evidence, ensure_ascii=False, separators=(",", ":")),
                            "evaluated_portfolios": diagnostics.get("evaluated_initial_portfolios", diagnostics.get("distinct_static_portfolios", "")),
                            "expanded_candidates": diagnostics.get("expanded_initial_candidates", ""),
                            "phase_one_path_cap": args.max_paths, "response_pool_size": args.response_pool_size,
                            "initial_budget": args.initial_budget, "reserve_budget": args.reserve_budget,
                            "response_pool_mode": "adaptive_diverse", "synthetic_data": True,
                        })
                        handle.flush()
                        print(f"completed family={family} scale={scale} seed={seed} method={method} objective={result.objective:.8f} seconds={elapsed:.2f}")
                    # CoDef includes Greedy as a candidate by design. Its
                    # non-inferiority to Greedy is an invariant, not the main
                    # evidence claim; compare quality with multistart baselines.
                    if "CoDef" in instance_results and "Greedy" in instance_results:
                        if instance_results["CoDef"].objective > instance_results["Greedy"].objective + 1e-12:
                            raise AssertionError(f"Greedy safeguard failed: {family}/{scale}/{seed}")
    print(f"Saved {raw_path}")


if __name__ == "__main__":
    main()
