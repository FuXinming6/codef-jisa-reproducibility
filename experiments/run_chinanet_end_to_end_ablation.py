"""Paired, re-optimized mechanism ablation on the ChinaNet 50-node instances.

All arms share exactly the main-benchmark graph, risk perturbation, action
catalog, constraints, attacker profiles, path cap, response-pool heuristic,
beam search, and total budget. Each arm re-runs CoDef's initial-portfolio
search; this is not a global-optimality comparison.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import subprocess
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.chinanet_paper_protocol import (
    DEFAULT_MAX_PATHS,
    DEFAULT_RESPONSE_POOL,
    DEFAULT_SEEDS,
    FAMILIES,
    build_action_catalog,
    build_topology,
)
from src.defense.action import DefenseAction
from src.optimizer.attacker_uncertainty import (
    ATTACKER_PROFILES,
    attacker_uncertainty_codef_search,
)
from src.simulator.scenario_variation import perturb_edge_risks


VARIANTS = {
    "full": {"initial_budget": 4.0, "reserve_budget": 4.0, "response_scope": "local", "replan_attacker": True},
    "no_reserve": {"initial_budget": 8.0, "reserve_budget": 0.0, "response_scope": "local", "replan_attacker": True},
    "no_replanning": {"initial_budget": 4.0, "reserve_budget": 4.0, "response_scope": "local", "replan_attacker": False},
    "global_action_scope": {"initial_budget": 4.0, "reserve_budget": 4.0, "response_scope": "global", "replan_attacker": True},
}
CONTRASTS = ("no_reserve", "no_replanning", "global_action_scope")
RAW_FIELDS = (
    "family", "scale", "seed", "variant", "initial_budget", "reserve_budget",
    "response_scope", "replan_attacker", "objective", "mean_profile_risk",
    "worst_profile_risk", "risk_stddev", "runtime_seconds", "node_count",
    "edge_count", "action_count", "initial_actions", "initial_cost",
    "profile_evidence_json", "evaluated_portfolios", "expanded_candidates",
    "phase_one_path_cap", "response_pool_size", "response_pool_mode",
    "graph_sha256", "action_catalog_sha256", "constraint_sha256", "instance_sha256",
    "git_head", "source_sha256", "synthetic_data",
)


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _instance_hashes(graph, actions: list[DefenseAction], constraints) -> dict[str, str]:
    graph_payload = {
        "nodes": [
            (node, sorted((key, value) for key, value in attributes.items()))
            for node, attributes in sorted(graph.nodes(data=True))
        ],
        "edges": [
            (source, target, sorted((key, value) for key, value in attributes.items()))
            for source, target, attributes in sorted(graph.edges(data=True))
        ],
    }
    action_payload = [
        {
            "id": action.id,
            "type": str(action.action_type),
            "target": action.target,
            "cost": action.cost,
            "effect": action.effect,
            "delay": action.response_delay,
        }
        for action in actions
    ]
    constraint_payload = {
        "requirements": {key: sorted(value) for key, value in sorted(constraints.requirements.items())},
        "mutually_exclusive": sorted(sorted(group) for group in constraints.mutually_exclusive),
        "group_limits": [
            {"id": item.id, "actions": sorted(item.actions), "max_selected": item.max_selected}
            for item in constraints.group_limits
        ],
    }
    graph_hash = _canonical_hash(graph_payload)
    action_hash = _canonical_hash(action_payload)
    constraint_hash = _canonical_hash(constraint_payload)
    return {
        "graph_sha256": graph_hash,
        "action_catalog_sha256": action_hash,
        "constraint_sha256": constraint_hash,
        "instance_sha256": _canonical_hash((graph_hash, action_hash, constraint_hash)),
    }


def _source_fingerprint() -> str:
    digest = hashlib.sha256()
    for path in (
        ROOT / "src" / "optimizer" / "two_stage.py",
        ROOT / "src" / "optimizer" / "attacker_uncertainty.py",
        Path(__file__).resolve(),
    ):
        digest.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _profile_evidence(result) -> list[dict[str, object]]:
    return [
        {
            "profile": item.profile,
            "terminal_risk": float(item.result.terminal_risk),
            "observed_path": item.result.observed_path,
            "replanned_path": item.result.replanned_path,
            "foothold": item.result.foothold,
            "response_actions": [action.id for action in item.result.response_actions],
        }
        for item in result.profile_results
    ]


def _read_existing(path: Path) -> dict[tuple[str, int, int, str], dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return {
            (row["family"], int(row["scale"]), int(row["seed"]), row["variant"]): row
            for row in csv.DictReader(handle)
        }


def _write_rows(path: Path, rows: dict[tuple[str, int, int, str], dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows.items(), key=lambda item: (item[0][0], item[0][2], item[0][3]))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_FIELDS)
        writer.writeheader()
        writer.writerows(row for _, row in ordered)


def _bootstrap_ci(values: list[float], seed: int, samples: int = 10000) -> tuple[float, float]:
    if len(values) < 2:
        return (values[0], values[0]) if values else (math.nan, math.nan)
    rng = random.Random(seed)
    means = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(samples))
    return means[int(0.025 * (samples - 1))], means[int(0.975 * (samples - 1))]


def _sign_test_p(better: int, worse: int) -> float:
    count = better + worse
    if count == 0:
        return 1.0
    tail = min(better, worse)
    return min(1.0, 2.0 * sum(math.comb(count, index) for index in range(tail + 1)) / (2**count))


def _holm(rows: list[dict[str, object]]) -> None:
    order = sorted(range(len(rows)), key=lambda index: float(rows[index]["sign_p"]))
    running = 0.0
    for rank, index in enumerate(order, start=1):
        running = max(running, min(1.0, (len(order) - rank + 1) * float(rows[index]["sign_p"])))
        rows[index]["holm_p"] = running


def summarize(path: Path, report_path: Path, expected_seeds: tuple[int, ...], git_head: str, source_hash: str) -> bool:
    rows_by_key = _read_existing(path)
    contrasts: list[dict[str, object]] = []
    common_by_family: dict[str, list[int]] = {}
    for family in FAMILIES:
        common_seeds = [
            seed for seed in expected_seeds
            if all((family, 50, seed, variant) in rows_by_key for variant in VARIANTS)
        ]
        common_by_family[family] = common_seeds
        for variant in CONTRASTS:
            paired = [
                (seed,
                 float(rows_by_key[(family, 50, seed, "full")]["objective"]),
                 float(rows_by_key[(family, 50, seed, variant)]["objective"]))
                for seed in common_seeds
            ]
            deltas = [ablation - full for _, full, ablation in paired]
            low, high = _bootstrap_ci(deltas, 960_000 + FAMILIES.index(family) * 100 + CONTRASTS.index(variant))
            better = sum(delta < -1e-12 for delta in deltas)
            ties = sum(abs(delta) <= 1e-12 for delta in deltas)
            worse = sum(delta > 1e-12 for delta in deltas)
            contrasts.append({
                "family": family,
                "variant": variant,
                "n": len(paired),
                "full_mean": statistics.mean(full for _, full, _ in paired) if paired else math.nan,
                "variant_mean": statistics.mean(abl for _, _, abl in paired) if paired else math.nan,
                "delta_mean": statistics.mean(deltas) if deltas else math.nan,
                "delta_median": statistics.median(deltas) if deltas else math.nan,
                "ci_low": low,
                "ci_high": high,
                "better": better,
                "ties": ties,
                "worse": worse,
                "sign_p": _sign_test_p(better, worse),
                "holm_p": math.nan,
                "runtime_mean": statistics.mean(
                    float(rows_by_key[(family, 50, seed, variant)]["runtime_seconds"])
                    for seed in common_seeds
                ) if common_seeds else math.nan,
            })
    _holm(contrasts)
    complete = all(common_by_family[family] == list(expected_seeds) for family in FAMILIES)
    lines = [
        "# ChinaNet 50-node end-to-end mechanism ablation",
        "",
        "Each condition re-runs the same bounded CoDef initial-portfolio search on the exact same generated instance. The independent unit is one family/seed graph, not one attacker profile.",
        "",
        f"Expected 20 seeds/family; completed paired instances: {', '.join(f'{family}={len(common_by_family[family])}' for family in FAMILIES)}. Complete default sweep: {complete}.",
        f"Scale=50; total budget=8; phase-one path cap={DEFAULT_MAX_PATHS}; adaptive response-pool base size={DEFAULT_RESPONSE_POOL}; beam width=2; profiles={', '.join(ATTACKER_PROFILES)}; objective=mean + 0.5*(worst−mean) + 0.2*sample-SD.",
        f"Git HEAD={git_head}; source SHA-256={source_hash}.",
        "",
        "## Conditions",
        "",
        "- `full`: B1/B2=4:4, local response-action eligibility, attacker continuation replanning.",
        "- `no_reserve`: B1/B2=8:0, local response-action eligibility, replanning. Total budget is unchanged; the full budget is available before attack, but the optimizer is not forced to spend every unit.",
        "- `no_replanning`: B1/B2=4:4 and local response-action eligibility. The attacker selects among the same capped phase-one path set and commits to that route suffix; paths with blocked suffixes retain their attempted route in evidence and receive zero terminal risk. The defender selects one response per observed prefix against the capped committed-route set, so it does not condition its action on an unobserved suffix.",
        "- `global_action_scope`: B1/B2=4:4 and replanning, but response actions may target any catalog item. This is a response-authority relaxation, not a deployment-realistic baseline or an information-only change.",
        "",
        "All arms preserve the same graph generator and edge-risk perturbation seeds as the main 50-node benchmark (`graph_seed=seed+50*1009+family_index*7919`; risk seed `seed+200003`), action catalog, constraints, evaluator, path cap, response-pool ranking/truncation, and bounded beam search. `no_reserve` compares budget timing/allocation, not the abstract value of additional total resources.",
        "",
        "## Paired results",
        "",
        "Delta is `ablation − full`; positive means the ablation has a higher/worse objective. Confidence intervals use paired bootstrap over graph seeds. Exact sign tests omit ties and are Holm-corrected over the nine predeclared family-by-ablation contrasts.",
        "",
        "|Topology|Ablation|N|Full mean|Ablation mean|Mean Δ|Median Δ|95% paired bootstrap CI|Ablation better/tie/worse|Sign p (Holm)|Ablation mean runtime (s)|",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in contrasts:
        lines.append(
            f"|{row['family']}|{row['variant']}|{row['n']}|{row['full_mean']:.8f}|{row['variant_mean']:.8f}|"
            f"{row['delta_mean']:.8f}|{row['delta_median']:.8f}|[{row['ci_low']:.8f}, {row['ci_high']:.8f}]|"
            f"{row['better']}/{row['ties']}/{row['worse']}|{row['sign_p']:.6g} ({row['holm_p']:.6g})|{row['runtime_mean']:.4f}|"
        )
    lines.extend(["", "## Per-profile mean terminal risk", "", "|Topology|Variant|N|risk_max|shortest_path|asset_targeted|", "|---|---|---:|---:|---:|---:|"])
    for family in FAMILIES:
        for variant in VARIANTS:
            seed_values: dict[str, list[float]] = {profile: [] for profile in ATTACKER_PROFILES}
            for seed in common_by_family[family]:
                evidence = json.loads(rows_by_key[(family, 50, seed, variant)]["profile_evidence_json"])
                for item in evidence:
                    seed_values[item["profile"]].append(float(item["terminal_risk"]))
            if not seed_values["risk_max"]:
                lines.append(f"|{family}|{variant}|0|n/a|n/a|n/a|")
                continue
            lines.append(
                f"|{family}|{variant}|{len(seed_values['risk_max'])}|{statistics.mean(seed_values['risk_max']):.8f}|"
                f"{statistics.mean(seed_values['shortest_path']):.8f}|{statistics.mean(seed_values['asset_targeted']):.8f}|"
            )
    lines.extend([
        "",
        "## Control reproduction and instance checks",
        "",
        "Each arm stores graph, action-catalog, constraint, and combined SHA-256 fingerprints. The runner rejects fingerprint differences between arms for one family/seed.",
        "The `full` arm is checked against the saved main benchmark by paired objective and initial action IDs when the reference row exists. This checks that the new control reproduces the registered main protocol, while the experiment itself is rerun under one working-tree source fingerprint.",
        "",
        "## Interpretation boundary",
        "",
        "The experiment measures conditional mechanism contributions under the current synthetic topology generators, three heuristic attacker profiles, candidate caps, response-pool heuristic, and beam width. It does not prove global optimality, Stackelberg equilibrium, real-world risk reduction, or generalization to operational networks. A non-significant contrast is not proof that a mechanism has no value; interactions and bounded-search approximation can mask effects.",
        "",
        f"Raw paired runs: `{path.name}`.",
    ])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return complete


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--family", choices=(*FAMILIES, "all"), default="all")
    parser.add_argument("--variants", nargs="+", choices=tuple(VARIANTS), default=tuple(VARIANTS))
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "chinanet_paper" / "chinanet_end_to_end_ablation_50_runs.csv")
    parser.add_argument("--report", type=Path, default=ROOT / "results" / "chinanet_paper" / "chinanet_end_to_end_ablation_50_report.md")
    parser.add_argument("--reference", type=Path, default=ROOT / "results" / "chinanet_paper" / "chinanet_main_50_20seed_runs.csv")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    selected_families = FAMILIES if args.family == "all" else (args.family,)
    rows = _read_existing(args.output) if args.resume else {}
    if not args.resume and args.output.exists():
        raise FileExistsError(f"output exists; pass --resume to append safely: {args.output}")
    main_reference: dict[tuple[str, int, int], dict[str, str]] = {}
    if args.reference.exists():
        with args.reference.open(newline="", encoding="utf-8-sig") as handle:
            main_reference = {
                (row["family"], int(row["scale"]), int(row["seed"])): row
                for row in csv.DictReader(handle)
                if row["method"] == "CoDef" and int(row["scale"]) == 50
            }
    git_head, source_hash = _git_head(), _source_fingerprint()
    if args.resume and rows:
        prior_heads = {row.get("git_head", "") for row in rows.values()}
        prior_sources = {row.get("source_sha256", "") for row in rows.values()}
        if prior_heads != {git_head} or prior_sources != {source_hash}:
            raise ValueError(
                "resume source fingerprint differs from saved rows; start a new output file rather than mixing code versions"
            )
    configuration = {
        "scale": 50,
        "seeds": list(args.seeds),
        "phase_one_path_cap": DEFAULT_MAX_PATHS,
        "response_pool_base": DEFAULT_RESPONSE_POOL,
        "response_pool_mode": "adaptive_diverse",
        "beam_width": 2,
        "attacker_profiles": list(ATTACKER_PROFILES),
        "objective": "mean + .5*(worst-mean) + .2*sample_sd",
    }
    for family in selected_families:
        for seed in args.seeds:
            graph_seed = seed + 50 * 1009 + FAMILIES.index(family) * 7919
            base_graph = build_topology(family, 50, graph_seed)
            graph = perturb_edge_risks(base_graph, seed + 200_003)
            actions, constraints = build_action_catalog(graph)
            hashes = _instance_hashes(graph, actions, constraints)
            for variant in args.variants:
                key = (family, 50, seed, variant)
                if key in rows:
                    old = rows[key]
                    if any(old[name] != value for name, value in hashes.items()):
                        raise ValueError(f"resume fingerprint mismatch for {key}")
                    continue
                settings = VARIANTS[variant]
                started = perf_counter()
                result = attacker_uncertainty_codef_search(
                    graph,
                    ATTACKER_PROFILES,
                    actions,
                    settings["initial_budget"],
                    settings["reserve_budget"],
                    constraints=constraints,
                    response_pool_size=DEFAULT_RESPONSE_POOL,
                    max_phase_one_paths=DEFAULT_MAX_PATHS,
                    response_pool_mode="adaptive_diverse",
                    method_name=f"ChinaNet-E2E-{variant}",
                    response_scope=settings["response_scope"],
                    replan_attacker=settings["replan_attacker"],
                )
                elapsed = perf_counter() - started
                evidence = _profile_evidence(result)
                row = {
                    "family": family,
                    "scale": 50,
                    "seed": seed,
                    "variant": variant,
                    "initial_budget": settings["initial_budget"],
                    "reserve_budget": settings["reserve_budget"],
                    "response_scope": settings["response_scope"],
                    "replan_attacker": settings["replan_attacker"],
                    "objective": f"{result.objective:.12g}",
                    "mean_profile_risk": f"{result.mean_terminal_risk:.12g}",
                    "worst_profile_risk": f"{result.worst_terminal_risk:.12g}",
                    "risk_stddev": f"{result.risk_stddev:.12g}",
                    "runtime_seconds": f"{elapsed:.6f}",
                    "node_count": graph.number_of_nodes(),
                    "edge_count": graph.number_of_edges(),
                    "action_count": len(actions),
                    "initial_actions": ";".join(action.id for action in result.initial_actions),
                    "initial_cost": f"{sum(action.cost for action in result.initial_actions):.12g}",
                    "profile_evidence_json": json.dumps(evidence, ensure_ascii=False, separators=(",", ":")),
                    "evaluated_portfolios": result.diagnostics.get("evaluated_initial_portfolios", ""),
                    "expanded_candidates": result.diagnostics.get("expanded_initial_candidates", ""),
                    "phase_one_path_cap": DEFAULT_MAX_PATHS,
                    "response_pool_size": DEFAULT_RESPONSE_POOL,
                    "response_pool_mode": "adaptive_diverse",
                    **hashes,
                    "git_head": git_head,
                    "source_sha256": source_hash,
                    "synthetic_data": True,
                }
                rows[key] = {name: str(value) for name, value in row.items()}
                _write_rows(args.output, rows)
                if variant == "full" and (family, 50, seed) in main_reference:
                    expected = main_reference[(family, 50, seed)]
                    if abs(float(expected["objective"]) - result.objective) > 1e-10:
                        raise AssertionError(
                            f"full control objective does not reproduce main row for {family}/{seed}: "
                            f"{expected['objective']} vs {result.objective}"
                        )
                    if expected["initial_actions"] != row["initial_actions"]:
                        raise AssertionError(f"full control action set mismatch for {family}/{seed}")
                print(
                    f"completed family={family} seed={seed} variant={variant} "
                    f"objective={result.objective:.8f} seconds={elapsed:.2f}",
                    flush=True,
                )
    complete = summarize(args.output, args.report, tuple(args.seeds), git_head, source_hash)
    if not complete:
        print("partial run saved; the report is diagnostic only until all paired arms/seeds are present")
    print(f"saved {args.output}")
    print(f"saved {args.report}")


if __name__ == "__main__":
    main()
