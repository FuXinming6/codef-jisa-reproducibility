"""Run paired CoDef baselines on public Topology Zoo graph structures."""
from __future__ import annotations

import argparse
import csv
import json
import hashlib
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.chinanet_paper_protocol import build_action_catalog
from experiments.run_chinanet_paper_benchmark import _run_method
from experiments.topology_zoo_protocol import (
    DATASET_CITATION,
    DATASET_DOI,
    DATASET_LICENSE,
    DEFAULT_OVERLAY_SEEDS,
    DEFAULT_TOPOLOGIES,
    build_public_attack_graph,
    download_topology_zoo,
    topology_path,
)
from src.optimizer.attacker_uncertainty import ATTACKER_PROFILES, attacker_uncertainty_exact_enumeration
from src.simulator.scenario_variation import perturb_edge_risks


METHODS = ("Greedy", "RiskTopK", "MultiStart-Greedy", "PairStart-Greedy", "CoDef", "Exact")
FIELDS = (
    "topology", "overlay_seed", "method", "objective", "mean_profile_risk",
    "worst_profile_risk", "risk_stddev", "runtime_seconds", "node_count",
    "undirected_link_count", "directed_edge_count", "ingress_count",
    "critical_count", "action_count", "initial_actions", "initial_cost",
    "profile_evidence_json", "evaluated_portfolios", "expanded_candidates",
    "initial_budget", "reserve_budget", "phase_one_path_cap",
    "response_pool_size", "structure_source", "cyber_semantics",
    "dataset_doi", "dataset_license", "dataset_citation",
    "raw_graphml_sha256", "orientation_rule", "run_order",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir", type=Path,
        default=ROOT / "data" / "external" / "topology_zoo",
    )
    parser.add_argument("--topologies", nargs="+", default=DEFAULT_TOPOLOGIES)
    parser.add_argument("--overlay-seeds", nargs="+", type=int, default=DEFAULT_OVERLAY_SEEDS)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--initial-budget", type=float, default=4.0)
    parser.add_argument("--reserve-budget", type=float, default=4.0)
    parser.add_argument("--max-paths", type=int, default=4)
    parser.add_argument("--response-pool-size", type=int, default=2)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results" / "chinanet_paper" / "chinanet_public_topology_runs.csv",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    graphml_dir = download_topology_zoo(args.dataset_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed: set[tuple[str, int, str]] = set()
    if args.resume and args.output.exists():
        with args.output.open(newline="", encoding="utf-8") as handle:
            completed = {
                (row["topology"], int(row["overlay_seed"]), row["method"])
                for row in csv.DictReader(handle)
            }
    write_header = not args.resume or not args.output.exists()
    with args.output.open("w" if write_header else "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        for topology in args.topologies:
            source_path = topology_path(graphml_dir, topology)
            source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
            base_graph = build_public_attack_graph(source_path)
            actions, constraints = build_action_catalog(base_graph)
            for overlay_seed in args.overlay_seeds:
                graph = perturb_edge_risks(base_graph, overlay_seed)
                offset = args.overlay_seeds.index(overlay_seed) % len(args.methods)
                run_order = list(args.methods[offset:]) + list(args.methods[:offset])
                for order_index, method in enumerate(run_order, start=1):
                    key = (topology, overlay_seed, method)
                    if key in completed:
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
                        kwargs["method_name"] = "ChinaNet-Public-Topology-CoDef"
                    elif method == "RiskTopK":
                        kwargs["seed"] = overlay_seed
                    started = perf_counter()
                    if method == "Exact":
                        result = attacker_uncertainty_exact_enumeration(
                            graph,
                            ATTACKER_PROFILES,
                            actions,
                            initial_budget=args.initial_budget,
                            reserve_budget=args.reserve_budget,
                            constraints=constraints,
                            response_pool_size=args.response_pool_size,
                            max_phase_one_paths=args.max_paths,
                            response_pool_mode="adaptive_diverse",
                        )
                    else:
                        result = _run_method(method, graph, actions, kwargs)
                    elapsed = perf_counter() - started
                    evidence = [
                        {
                            "profile": item.profile,
                            "terminal_risk": float(item.result.terminal_risk),
                            "observed_path": item.result.observed_path,
                            "replanned_path": item.result.replanned_path,
                            "response_actions": [action.id for action in item.result.response_actions],
                        }
                        for item in result.profile_results
                    ]
                    writer.writerow({
                        "topology": topology,
                        "overlay_seed": overlay_seed,
                        "method": method,
                        "objective": float(result.objective),
                        "mean_profile_risk": float(result.mean_terminal_risk),
                        "worst_profile_risk": float(result.worst_terminal_risk),
                        "risk_stddev": float(result.risk_stddev),
                        "runtime_seconds": round(elapsed, 6),
                        "node_count": graph.number_of_nodes(),
                        # Each source undirected link is oriented once, so the
                        # DAG edge count is also the source-link count.
                        "undirected_link_count": graph.number_of_edges(),
                        "directed_edge_count": graph.number_of_edges(),
                        "ingress_count": sum(data.get("type") == "internet" for _, data in graph.nodes(data=True)),
                        "critical_count": sum(bool(data.get("critical")) for _, data in graph.nodes(data=True)),
                        "action_count": len(actions),
                        "initial_actions": ";".join(action.id for action in result.initial_actions),
                        "initial_cost": sum(action.cost for action in result.initial_actions),
                        "profile_evidence_json": json.dumps(evidence, ensure_ascii=False, separators=(",", ":")),
                        "evaluated_portfolios": result.diagnostics.get("evaluated_initial_portfolios", result.diagnostics.get("distinct_static_portfolios", "")),
                        "expanded_candidates": result.diagnostics.get("expanded_initial_candidates", ""),
                        "initial_budget": args.initial_budget,
                        "reserve_budget": args.reserve_budget,
                        "phase_one_path_cap": args.max_paths,
                        "response_pool_size": args.response_pool_size,
                        "structure_source": "public operator-derived Topology Zoo graph",
                        "cyber_semantics": "deterministic synthetic overlay",
                        "dataset_doi": DATASET_DOI,
                        "dataset_license": DATASET_LICENSE,
                        "dataset_citation": DATASET_CITATION,
                        "raw_graphml_sha256": source_sha256,
                        "orientation_rule": base_graph.graph["orientation_rule"],
                        "run_order": order_index,
                    })
                    handle.flush()
                    print(
                        f"completed topology={topology} overlay={overlay_seed} method={method} "
                        f"objective={result.objective:.8f} seconds={elapsed:.2f}",
                        flush=True,
                    )
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
