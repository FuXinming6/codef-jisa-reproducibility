"""Reproduce the seed-5682 mechanism trace reported in the JISA draft."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.chinanet_paper_protocol import FAMILIES, build_action_catalog, build_topology
from src.defense.apply_defense import apply_defense
from src.optimizer.attacker_uncertainty import (
    ATTACKER_PROFILES,
    evaluate_attacker_uncertainty_portfolio,
)
from src.risk.network_risk import assess_network_risk
from src.simulator.scenario_variation import perturb_edge_risks


OUT = ROOT / "results" / "chinanet_paper" / "jisa_seed5682_trace_20261112.json"
MAIN_RESULTS = ROOT / "results" / "chinanet_paper" / "chinanet_main_50_20seed_runs.csv"
SOURCE_FILES = (
    "experiments/chinanet_paper_protocol.py",
    "src/optimizer/attacker_uncertainty.py",
    "src/optimizer/two_stage.py",
    "src/optimizer/robust_two_stage.py",
    "src/graph/path_search.py",
    "src/risk/network_risk.py",
    "src/defense/apply_defense.py",
    "src/simulator/scenario_variation.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    family, seed, node_count = "hierarchical", 5682, 50
    graph_seed = seed + node_count * 1009 + FAMILIES.index(family) * 7919
    perturbation_seed = seed + 200_003
    graph = perturb_edge_risks(
        build_topology(family, node_count, graph_seed), perturbation_seed
    )
    actions, constraints = build_action_catalog(graph)
    by_id = {action.id: action for action in actions}
    portfolios = {
        "CoDef selected start": ("D02", "D08"),
        "static-score extension": ("D01", "D02", "D08"),
    }

    evaluations = []
    for label, action_ids in portfolios.items():
        selected = [by_id[action_id] for action_id in action_ids]
        result = evaluate_attacker_uncertainty_portfolio(
            graph,
            ATTACKER_PROFILES,
            actions,
            frozenset(action_ids),
            reserve_budget=4.0,
            constraints=constraints,
            response_pool_size=2,
            max_phase_one_paths=4,
            response_pool_mode="adaptive_diverse",
            worst_weight=0.5,
            variation_weight=0.2,
            method_name="JISA-seed5682-trace",
        )
        evaluations.append({
            "label": label,
            "initial_actions": list(action_ids),
            "initial_cost": sum(action.cost for action in selected),
            "static_network_risk": assess_network_risk(
                apply_defense(graph, selected)
            ).score,
            "objective_J": result.objective,
            "mean_terminal_risk": result.mean_terminal_risk,
            "worst_terminal_risk": result.worst_terminal_risk,
            "sample_sd_terminal_risk": result.risk_stddev,
            "profiles": [
                {
                    "profile": item.profile,
                    "terminal_risk": item.result.terminal_risk,
                    "observed_prefix": list(item.result.observed_path),
                    "foothold": item.result.foothold,
                    "replanned_path": list(item.result.replanned_path),
                    "response_actions": [a.id for a in item.result.response_actions],
                    "response_cost": item.result.response_cost,
                }
                for item in result.profile_results
            ],
        })

    primary_rows = {}
    with MAIN_RESULTS.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row["family"] == family and int(row["seed"]) == seed:
                if row["method"] in {"CoDef", "PairStart-Greedy"}:
                    primary_rows[row["method"]] = {
                        "objective_J": float(row["objective"]),
                        "initial_actions": row["initial_actions"].split(";"),
                    }

    document = {
        "protocol": {
            "family": family,
            "node_count": node_count,
            "seed": seed,
            "graph_seed": graph_seed,
            "edge_risk_perturbation_seed": perturbation_seed,
            "initial_budget": 4.0,
            "response_reserve": 4.0,
            "response_pool_base": 2,
            "max_phase_one_paths": 4,
            "response_pool_mode": "adaptive_diverse",
            "profiles": list(ATTACKER_PROFILES),
            "objective": "mean + 0.5*(worst-mean) + 0.2*sample_sd",
            "objective_weights": {"worst_weight": 0.5, "sample_sd_weight": 0.2},
            "data_status": "deterministic synthetic replay; not a holdout",
        },
        "evaluations": evaluations,
        "comparison_note": "The +D01 portfolio is a feasible one-action extension used only as a local mechanism illustration; it is not the Static-CoDef search output.",
        "primary_results_csv": {
            "file": "chinanet_main_50_20seed_runs.csv",
            "sha256": sha256(MAIN_RESULTS),
            "methods": primary_rows,
        },
        "source_sha256": {
            name: sha256(ROOT / name) for name in SOURCE_FILES
        },
    }
    OUT.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote deterministic trace to {OUT}")
    for item in evaluations:
        print(
            f"{item['label']}: static={item['static_network_risk']:.8f}; "
            f"J={item['objective_J']:.8f}; actions={';'.join(item['initial_actions'])}"
        )


if __name__ == "__main__":
    main()
