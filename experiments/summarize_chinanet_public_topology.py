"""Summarize the preregistered public-topology structural stress test."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.topology_zoo_protocol import (
    DATASET_CITATION,
    DATASET_DOI,
    DATASET_LICENSE,
    build_public_attack_graph,
    download_topology_zoo,
    topology_path,
)


BASELINES = ("Greedy", "RiskTopK", "MultiStart-Greedy", "PairStart-Greedy", "Exact")
TIE_TOLERANCE = 1e-12
PRACTICAL_EQUIVALENCE = 1e-3


def _bootstrap_mean_ci(values: list[float], seed: int, samples: int = 20_000) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    means = sorted(
        statistics.fmean(rng.choice(values) for _ in values)
        for _ in range(samples)
    )
    return means[int(0.025 * samples)], means[int(0.975 * samples) - 1]


def _sign_test(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, index) for index in range(min(wins, losses) + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        default=ROOT / "results" / "chinanet_paper" / "chinanet_public_topology_30seed_runs.csv",
    )
    parser.add_argument(
        "--summary", type=Path,
        default=ROOT / "results" / "chinanet_paper" / "chinanet_public_topology_30seed_summary.csv",
    )
    parser.add_argument(
        "--report", type=Path,
        default=ROOT / "results" / "chinanet_paper" / "chinanet_public_topology_30seed_report.md",
    )
    parser.add_argument(
        "--manifest", type=Path,
        default=ROOT / "results" / "chinanet_paper" / "chinanet_public_topology_manifest.json",
    )
    parser.add_argument(
        "--dataset-dir", type=Path,
        default=ROOT / "data" / "external" / "topology_zoo",
    )
    args = parser.parse_args()

    with args.input.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_instance: dict[tuple[str, int], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        by_instance[(row["topology"], int(row["overlay_seed"]))][row["method"]] = row
    complete = {
        key: methods for key, methods in by_instance.items()
        if "CoDef" in methods and all(baseline in methods for baseline in BASELINES)
    }
    if not complete:
        raise ValueError("input contains no complete paired instances")

    summaries: list[dict[str, object]] = []
    for baseline in BASELINES:
        deltas: list[float] = []
        runtime_ratios: list[float] = []
        for methods in complete.values():
            codef = methods["CoDef"]
            other = methods[baseline]
            deltas.append(float(codef["objective"]) - float(other["objective"]))
            runtime_ratios.append(float(codef["runtime_seconds"]) / max(float(other["runtime_seconds"]), 1e-12))
        wins = sum(delta < -TIE_TOLERANCE for delta in deltas)
        losses = sum(delta > TIE_TOLERANCE for delta in deltas)
        ties = len(deltas) - wins - losses
        practical_ties = sum(abs(delta) <= PRACTICAL_EQUIVALENCE for delta in deltas)
        low, high = _bootstrap_mean_ci(deltas, 20260929 + sum(map(ord, baseline)))
        summaries.append({
            "baseline": baseline,
            "paired_risk_seeds": len(deltas),
            "codef_wins": wins,
            "ties": ties,
            "codef_losses": losses,
            "practical_equivalence_count": practical_ties,
            "mean_delta_codef_minus_baseline": statistics.fmean(deltas),
            "median_delta_codef_minus_baseline": statistics.median(deltas),
            "bootstrap_95ci_low": low,
            "bootstrap_95ci_high": high,
            "exact_sign_test_p": _sign_test(wins, losses),
            "holm_adjusted_p": 1.0,
            "mean_runtime_ratio_codef_over_baseline": statistics.fmean(runtime_ratios),
        })

    ordered = sorted(range(len(summaries)), key=lambda index: summaries[index]["exact_sign_test_p"])
    running = 0.0
    for rank, index in enumerate(ordered):
        adjusted = min(1.0, float(summaries[index]["exact_sign_test_p"]) * (len(summaries) - rank))
        running = max(running, adjusted)
        summaries[index]["holm_adjusted_p"] = running

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    graphml_dir = download_topology_zoo(args.dataset_dir)
    graph = build_public_attack_graph(topology_path(graphml_dir, "Chinanet"))
    ingresses = [
        {"id": node, "source_label": data["source_label"]}
        for node, data in graph.nodes(data=True) if data.get("type") == "internet"
    ]
    critical = [
        {"id": node, "source_label": data["source_label"]}
        for node, data in graph.nodes(data=True) if data.get("critical")
    ]
    reachable_pairs = sum(
        __import__("networkx").has_path(graph, source["id"], target["id"])
        for source in ingresses for target in critical
    )
    manifest = {
        "dataset": "Internet Topology Zoo",
        "dataset_doi": DATASET_DOI,
        "dataset_license": DATASET_LICENSE,
        "citation": DATASET_CITATION,
        "topology": "Chinanet",
        "public_fields_used": ["nodes", "links", "Internal", "label"],
        "synthetic_fields": ["edge risk", "asset value", "security role", "critical assets", "defense actions"],
        "node_count": graph.number_of_nodes(),
        "public_link_count": graph.number_of_edges(),
        "ingresses": ingresses,
        "critical_assets": critical,
        "reachable_ingress_asset_pairs": reachable_pairs,
        "total_ingress_asset_pairs": len(ingresses) * len(critical),
        "orientation_rule": graph.graph["orientation_rule"],
        "is_dag": True,
        "risk_seed_count": len({seed for _, seed in complete}),
        "statistical_unit": "one complete synthetic risk perturbation on one fixed public structure",
    }
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Public-topology structural validation",
        "",
        "## Scope",
        "",
        "This stress test uses the 42-node, 66-link `Chinanet` structure from Internet Topology Zoo. The public data provide connectivity and provider labels only. Edge risks, asset values, security roles, critical assets, attack direction, and defense actions are deterministic synthetic annotations. The result is therefore **not** a real China Telecom attack graph or a field validation.",
        "",
        f"Dataset: {DATASET_DOI}; license: {DATASET_LICENSE}. {DATASET_CITATION}",
        "",
        "The graph is mapped once to a 66-edge DAG with four provider-labelled international ingress nodes and two fixed central critical nodes (Xi'an and Tianjin). Thirty predeclared seeds perturb edge risk only. Every method receives the same graph instance, 12 actions, 4:4 budget, path cap 4, response pool 2, constraints, and three attacker profiles. Seeds measure sensitivity on one fixed structure; they are not 30 independent real networks.",
        "",
        "## Paired results",
        "",
        "Negative delta means CoDef has lower (better) objective. Numerical ties use 1e-12; practical equivalence uses |delta| <= 0.001. Bootstrap intervals are paired across the 30 risk perturbations. Sign tests omit numerical ties.",
        "",
        "|Comparator|N|CoDef win/tie/loss|Practically equivalent|Mean delta|Median delta|95% paired bootstrap CI|Exact sign p (Holm)|Runtime ratio|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"|{row['baseline']}|{row['paired_risk_seeds']}|{row['codef_wins']}/{row['ties']}/{row['codef_losses']}|"
            f"{row['practical_equivalence_count']}|{row['mean_delta_codef_minus_baseline']:.8f}|"
            f"{row['median_delta_codef_minus_baseline']:.8f}|[{row['bootstrap_95ci_low']:.8f}, {row['bootstrap_95ci_high']:.8f}]|"
            f"{row['exact_sign_test_p']:.6g} ({row['holm_adjusted_p']:.6g})|{row['mean_runtime_ratio_codef_over_baseline']:.2f}x|"
        )
    pair = next(row for row in summaries if row["baseline"] == "PairStart-Greedy")
    exact = next(row for row in summaries if row["baseline"] == "Exact")
    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        f"Against the strongest heuristic PairStart-Greedy, CoDef records {pair['codef_wins']}/{pair['ties']}/{pair['codef_losses']} wins/ties/losses. Against Exact, it records {exact['codef_wins']}/{exact['ties']}/{exact['codef_losses']}. If these comparisons are dominated by ties, the supported claim is portability to a public external structure and exact/strong-baseline parity under this protocol—not superior solution quality.",
        "",
        "This experiment complements rather than replaces the three-family synthetic benchmark. It reduces the risk that the implementation only works on the project's own graph generator, while leaving real vulnerability calibration and operational validation as future work.",
    ])
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {args.summary}")
    print(f"Saved {args.report}")
    print(f"Saved {args.manifest}")


if __name__ == "__main__":
    main()
