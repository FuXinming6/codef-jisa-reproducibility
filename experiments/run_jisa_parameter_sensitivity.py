"""Re-optimize the JISA primary instances under local parameter perturbations.

This is a controlled sensitivity analysis, not empirical calibration. It reuses
the 60 fixed 50-node graphs and edge-risk seeds from the paper's primary panel,
and varies synthetic action costs, used action effects, and node asset values by
independent multiplicative factors in [0.85, 1.15].
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import statistics
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from random import Random
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import networkx as nx
import yaml

from experiments.chinanet_paper_protocol import (
    DEFAULT_MAX_PATHS,
    DEFAULT_RESPONSE_POOL,
    DEFAULT_SEEDS,
    FAMILIES,
    build_action_catalog,
    build_topology,
)
from src.defense.action import DefenseType
from src.optimizer.attacker_uncertainty import (
    ATTACKER_PROFILES,
    attacker_uncertainty_codef_search,
    attacker_uncertainty_multistart_greedy_baseline,
)
from src.simulator.scenario_variation import perturb_edge_risks


EXPERIMENT = "jisa_parameter_sensitivity_v1"
CONDITIONS = ("base", "cost_only", "effect_only", "asset_only", "joint")
METHODS = ("CoDef", "PairStart-Greedy")
PERTURBATION_LOW = 0.85
PERTURBATION_HIGH = 1.15
TIE_TOLERANCE = 1e-12
OUTPUT_FIELDS = (
    "experiment", "family", "scale", "seed", "graph_seed", "risk_seed",
    "condition", "method", "objective", "mean_profile_risk",
    "worst_profile_risk", "risk_stddev", "runtime_seconds",
    "initial_actions", "initial_cost", "profile_evidence_json",
    "evaluated_initial_portfolios", "expanded_initial_candidates",
    "input_fingerprint", "parameter_values_json",
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_seed(*parts: object) -> int:
    token = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "big")


def _uniform_factors(keys: list[str], *seed_parts: object) -> dict[str, float]:
    rng = Random(_stable_seed(EXPERIMENT, *seed_parts))
    return {key: rng.uniform(PERTURBATION_LOW, PERTURBATION_HIGH) for key in sorted(keys)}


def _instantiate_condition(graph, actions, family: str, seed: int, condition: str):
    cost_keys = [action.id for action in actions]
    used_effect_keys = [
        action.id for action in actions
        if action.action_type in {DefenseType.NODE_HARDENING, DefenseType.EDGE_RESTRICTION}
    ]
    asset_keys = [
        node for node, data in graph.nodes(data=True)
        if data.get("type") != "internet"
    ]
    perturb_cost = condition in {"cost_only", "joint"}
    perturb_effect = condition in {"effect_only", "joint"}
    perturb_asset = condition in {"asset_only", "joint"}

    cost_factors = (
        _uniform_factors(cost_keys, family, seed, condition, "cost")
        if perturb_cost else {key: 1.0 for key in cost_keys}
    )
    effect_factors = (
        _uniform_factors(used_effect_keys, family, seed, condition, "effect")
        if perturb_effect else {key: 1.0 for key in used_effect_keys}
    )
    asset_factors = (
        _uniform_factors(asset_keys, family, seed, condition, "asset")
        if perturb_asset else {key: 1.0 for key in asset_keys}
    )

    varied_graph = graph.copy()
    for node, factor in asset_factors.items():
        varied_graph.nodes[node]["asset_value"] = float(graph.nodes[node]["asset_value"]) * factor

    varied_actions = []
    for action in actions:
        effect = float(action.effect)
        if action.id in effect_factors:
            # Keep probabilities valid. Isolation is omitted because its
            # simulator semantics remove adjacent edges and do not use effect.
            effect = min(1.0, effect * effect_factors[action.id])
        varied_actions.append(
            replace(action, cost=float(action.cost) * cost_factors[action.id], effect=effect)
        )

    parameter_values = {
        "cost_multipliers": cost_factors,
        "effect_multipliers": effect_factors,
        "asset_value_multipliers": asset_factors,
        "action_costs": {action.id: action.cost for action in varied_actions},
        "action_effects": {action.id: action.effect for action in varied_actions},
        "node_asset_values": {
            node: float(varied_graph.nodes[node]["asset_value"])
            for node in sorted(asset_keys)
        },
    }
    return varied_graph, varied_actions, parameter_values


def _fingerprint(graph, actions) -> str:
    payload = {
        "nodes": [
            (node, data.get("type"), bool(data.get("critical", False)),
             float(data.get("asset_value", 0.0)))
            for node, data in sorted(graph.nodes(data=True))
        ],
        "edges": [
            (source, target, float(data.get("risk", 0.0)))
            for source, target, data in sorted(graph.edges(data=True))
        ],
        "actions": [
            (action.id, action.action_type.value, action.target, float(action.cost),
             float(action.effect), action.response_delay)
            for action in sorted(actions, key=lambda item: item.id)
        ],
    }
    return _sha256_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())


def _git_state() -> dict[str, str]:
    def run(*args: str) -> str:
        try:
            result = subprocess.run(
                ["git", *args], cwd=ROOT, capture_output=True, text=True,
                check=True, encoding="utf-8",
            )
            return result.stdout.strip()
        except Exception as exc:  # record unavailable metadata rather than hide the run
            return f"unavailable: {exc}"
    return {"head": run("rev-parse", "HEAD"), "status_porcelain": run("status", "--short")}


def _source_manifest(script_path: Path, primary_csv: Path) -> dict[str, str]:
    paths = (
        script_path,
        ROOT / "experiments" / "chinanet_paper_protocol.py",
        ROOT / "src" / "optimizer" / "attacker_uncertainty.py",
        ROOT / "src" / "defense" / "apply_defense.py",
        ROOT / "src" / "defense" / "action.py",
        ROOT / "src" / "defense" / "constraints.py",
        ROOT / "src" / "simulator" / "scenario_variation.py",
        primary_csv,
    )
    return {str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path): _sha256_file(path) for path in paths}


def _make_protocol(script_path: Path, primary_csv: Path, output_dir: Path) -> dict[str, object]:
    return {
        "experiment": EXPERIMENT,
        "objective": "Characterize sensitivity of the CoDef/PairStart comparison to local perturbations of synthetic model attributes.",
        "design_frozen_before_execution": True,
        "sampling_unit": "The 60 fixed graph-seed instances in chinanet_main_50_20seed_runs.csv; no new topology samples.",
        "families": list(FAMILIES),
        "seeds": list(DEFAULT_SEEDS),
        "scale": 50,
        "conditions": list(CONDITIONS),
        "methods": list(METHODS),
        "perturbation": {
            "type": "Independent deterministic multiplicative factors uniformly sampled on [0.85, 1.15] per instance and attribute.",
            "costs": "All 12 initial/response action costs are multiplied by their condition-specific factors; budgets remain 4.0/4.0.",
            "effects": "Only node-hardening and edge-restriction probabilities are multiplied; values are capped at 1.0. Node-isolation removes adjacent edges and its effect field is not used by the simulator, so isolation effects are held fixed.",
            "asset_values": "Every non-ingress node asset value is multiplied by an independent factor; topology, critical labels, and edge risks stay fixed.",
            "seeds": "Each factor stream is seeded by SHA-256 of experiment/family/primary-seed/condition/attribute.",
            "interpretation": "A model-parameter sensitivity probe, not empirical uncertainty, calibration, or independent replication.",
        },
        "evaluator": {
            "profiles": list(ATTACKER_PROFILES),
            "initial_budget": 4.0,
            "reserve_budget": 4.0,
            "phase_one_path_cap": DEFAULT_MAX_PATHS,
            "response_pool_base": DEFAULT_RESPONSE_POOL,
            "response_pool_mode": "adaptive_diverse",
            "beam_width": 2,
            "constraints": "At most one node-isolation action jointly across initial and response actions.",
            "tie_tolerance": TIE_TOLERANCE,
        },
        "analysis": "Descriptive paired deltas, lower/tied/higher counts, and within-method objective changes from base; no confirmatory p-values.",
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "networkx": nx.__version__,
            "pyyaml": yaml.__version__,
        },
        "git": _git_state(),
        "source_sha256": _source_manifest(script_path, primary_csv),
        "output_dir": str(output_dir),
    }


def _run_one(method: str, graph, actions, constraints):
    common = {
        "initial_budget": 4.0,
        "reserve_budget": 4.0,
        "constraints": constraints,
        "response_pool_size": DEFAULT_RESPONSE_POOL,
        "max_phase_one_paths": DEFAULT_MAX_PATHS,
        "response_pool_mode": "adaptive_diverse",
    }
    started = perf_counter()
    if method == "CoDef":
        result = attacker_uncertainty_codef_search(
            graph, ATTACKER_PROFILES, actions, beam_width=2,
            method_name="ChinaNet-Paper-CoDef", **common,
        )
    elif method == "PairStart-Greedy":
        result = attacker_uncertainty_multistart_greedy_baseline(
            graph, ATTACKER_PROFILES, actions, start_size=2,
            memoize_static_scores=False, **common,
        )
    else:
        raise ValueError(method)
    elapsed = perf_counter() - started
    return result, elapsed


def _record_rows(path: Path) -> dict[tuple[str, int, str, str], dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            (row["family"], int(row["seed"]), row["condition"], row["method"]): row
            for row in csv.DictReader(handle)
        }


def _summarize(raw_path: Path, out_dir: Path) -> None:
    rows = _record_rows(raw_path)
    summary: dict[str, object] = {
        "experiment": EXPERIMENT,
        "raw_rows": len(rows),
        "expected_raw_rows": len(FAMILIES) * len(DEFAULT_SEEDS) * len(CONDITIONS) * len(METHODS),
        "tie_tolerance": TIE_TOLERANCE,
        "conditions": {},
    }
    if len(rows) != summary["expected_raw_rows"]:
        raise RuntimeError(f"incomplete run set: found {len(rows)} rows")

    for condition in CONDITIONS:
        by_family: dict[str, object] = {}
        all_deltas: list[float] = []
        all_codef_changes: list[float] = []
        all_pair_changes: list[float] = []
        total = {"codef_lower": 0, "tie": 0, "codef_higher": 0}
        for family in FAMILIES:
            deltas: list[float] = []
            codef_changes: list[float] = []
            pair_changes: list[float] = []
            counts = {"codef_lower": 0, "tie": 0, "codef_higher": 0}
            for seed in DEFAULT_SEEDS:
                codef = rows[(family, seed, condition, "CoDef")]
                pair = rows[(family, seed, condition, "PairStart-Greedy")]
                delta = float(codef["objective"]) - float(pair["objective"])
                deltas.append(delta)
                if delta < -TIE_TOLERANCE:
                    counts["codef_lower"] += 1
                elif delta > TIE_TOLERANCE:
                    counts["codef_higher"] += 1
                else:
                    counts["tie"] += 1
                base_codef = rows[(family, seed, "base", "CoDef")]
                base_pair = rows[(family, seed, "base", "PairStart-Greedy")]
                codef_changes.append(
                    (float(codef["objective"]) - float(base_codef["objective"]))
                    / max(abs(float(base_codef["objective"])), 1e-12)
                )
                pair_changes.append(
                    (float(pair["objective"]) - float(base_pair["objective"]))
                    / max(abs(float(base_pair["objective"])), 1e-12)
                )
            for key in total:
                total[key] += counts[key]
            by_family[family] = {
                "n": len(deltas),
                "codef_lower_tied_higher": [counts["codef_lower"], counts["tie"], counts["codef_higher"]],
                "mean_delta_codef_minus_pairstart": statistics.mean(deltas),
                "median_delta_codef_minus_pairstart": statistics.median(deltas),
                "median_relative_objective_change_vs_base": {
                    "CoDef": statistics.median(codef_changes),
                    "PairStart-Greedy": statistics.median(pair_changes),
                },
            }
            all_deltas.extend(deltas)
            all_codef_changes.extend(codef_changes)
            all_pair_changes.extend(pair_changes)
        summary["conditions"][condition] = {
            "n": len(all_deltas),
            "codef_lower_tied_higher": [total["codef_lower"], total["tie"], total["codef_higher"]],
            "mean_delta_codef_minus_pairstart": statistics.mean(all_deltas),
            "median_delta_codef_minus_pairstart": statistics.median(all_deltas),
            "median_relative_objective_change_vs_base": {
                "CoDef": statistics.median(all_codef_changes),
                "PairStart-Greedy": statistics.median(all_pair_changes),
            },
            "families": by_family,
        }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# JISA synthetic parameter sensitivity",
        "",
        "The analysis re-optimizes the same 60 fixed 50-node topology/edge-risk instances under the registered parameter perturbations. These are sensitivity draws, not 60 additional topologies, empirical attribute estimates, or confirmatory tests.",
        "",
        "Pairwise delta is CoDef objective minus PairStart-Greedy under the same perturbed evaluator; lower favors CoDef. Ties use an absolute tolerance of 1e-12. Relative objective changes compare each method with its own base-condition result on the same graph.",
        "",
        "|Condition|N|CoDef lower / tie / higher|Mean paired ΔJ|Median paired ΔJ|Median relative ΔJ CoDef|Median relative ΔJ PairStart|",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        item = summary["conditions"][condition]
        lower, ties, higher = item["codef_lower_tied_higher"]
        changes = item["median_relative_objective_change_vs_base"]
        lines.append(
            f"|{condition}|{item['n']}|{lower}/{ties}/{higher}|"
            f"{item['mean_delta_codef_minus_pairstart']:.6f}|"
            f"{item['median_delta_codef_minus_pairstart']:.6f}|"
            f"{changes['CoDef']:.2%}|{changes['PairStart-Greedy']:.2%}|"
        )
    lines.extend([
        "",
        "No inferential p-values are reported: the perturbed parameter sets are deterministic model-sensitivity probes on repeated base graphs, not independent samples from an estimated population.",
        "",
    ])
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "results" / "chinanet_paper" / "cost_effect_value_sensitivity_20261112",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    primary_csv = ROOT / "results" / "chinanet_paper" / "chinanet_main_50_20seed_runs.csv"
    if not primary_csv.exists():
        raise FileNotFoundError(primary_csv)
    script_path = Path(__file__).resolve()
    protocol_path = args.output_dir / "protocol.json"
    protocol = _make_protocol(script_path, primary_csv, args.output_dir)
    if protocol_path.exists():
        old = json.loads(protocol_path.read_text(encoding="utf-8"))
        if old.get("source_sha256") != protocol["source_sha256"] or old.get("conditions") != protocol["conditions"]:
            raise RuntimeError("protocol/source changed; use a new output directory instead of resuming")
    else:
        protocol_path.write_text(json.dumps(protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    raw_path = args.output_dir / "runs.csv"
    completed = _record_rows(raw_path) if args.resume else {}
    mode = "a" if args.resume and raw_path.exists() else "w"
    with raw_path.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        if mode == "w":
            writer.writeheader()
        for family_index, family in enumerate(FAMILIES):
            for seed in DEFAULT_SEEDS:
                graph_seed = seed + 50 * 1009 + family_index * 7919
                risk_seed = seed + 200_003
                base = perturb_edge_risks(build_topology(family, 50, graph_seed), risk_seed)
                base_actions, constraints = build_action_catalog(base)
                for condition in CONDITIONS:
                    graph, actions, parameter_values = _instantiate_condition(
                        base, base_actions, family, seed, condition,
                    )
                    fingerprint = _fingerprint(graph, actions)
                    parameter_payload = json.dumps(
                        parameter_values, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                    )
                    for method in METHODS:
                        key = (family, seed, condition, method)
                        if key in completed:
                            continue
                        result, elapsed = _run_one(method, graph, actions, constraints)
                        profile_evidence = [
                            {
                                "profile": item.profile,
                                "terminal_risk": float(item.result.terminal_risk),
                                "foothold": item.result.foothold,
                                "observed_path": item.result.observed_path,
                                "replanned_path": item.result.replanned_path,
                                "response_actions": [action.id for action in item.result.response_actions],
                            }
                            for item in result.profile_results
                        ]
                        writer.writerow({
                            "experiment": EXPERIMENT,
                            "family": family,
                            "scale": 50,
                            "seed": seed,
                            "graph_seed": graph_seed,
                            "risk_seed": risk_seed,
                            "condition": condition,
                            "method": method,
                            "objective": float(result.objective),
                            "mean_profile_risk": float(result.mean_terminal_risk),
                            "worst_profile_risk": float(result.worst_terminal_risk),
                            "risk_stddev": float(result.risk_stddev),
                            "runtime_seconds": round(elapsed, 6),
                            "initial_actions": ";".join(action.id for action in result.initial_actions),
                            "initial_cost": sum(action.cost for action in result.initial_actions),
                            "profile_evidence_json": json.dumps(profile_evidence, ensure_ascii=False, separators=(",", ":")),
                            "evaluated_initial_portfolios": result.diagnostics.get("evaluated_initial_portfolios", ""),
                            "expanded_initial_candidates": result.diagnostics.get("expanded_initial_candidates", ""),
                            "input_fingerprint": fingerprint,
                            "parameter_values_json": parameter_payload,
                        })
                        handle.flush()
                        completed[key] = {"family": family, "seed": seed, "condition": condition, "method": method}
                        print(
                            f"{len(completed)}/600 {family} seed={seed} condition={condition} "
                            f"method={method} J={result.objective:.8f} t={elapsed:.2f}s",
                            flush=True,
                        )
    _summarize(raw_path, args.output_dir)


if __name__ == "__main__":
    main()
