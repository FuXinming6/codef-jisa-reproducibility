"""Compare response-aware and static-risk beam ranking on the 60 primary graphs."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.chinanet_paper_protocol import (
    DEFAULT_MAX_PATHS, DEFAULT_RESPONSE_POOL, DEFAULT_SEEDS, FAMILIES,
    build_action_catalog, build_topology,
)
from src.optimizer.attacker_uncertainty import ATTACKER_PROFILES, attacker_uncertainty_codef_search
from src.simulator.scenario_variation import perturb_edge_risks

OUT = ROOT / "results" / "chinanet_paper"
RUNS = OUT / "codef_static_beam_ablation_60_20261112.csv"
SUMMARY = OUT / "codef_static_beam_ablation_60_20261112_summary.json"
SOURCE_FILES = (
    "experiments/run_jisa_static_beam_ablation.py",
    "experiments/chinanet_paper_protocol.py",
    "src/optimizer/attacker_uncertainty.py",
    "src/optimizer/two_stage.py",
    "src/graph/path_search.py",
    "src/risk/network_risk.py",
    "src/defense/apply_defense.py",
)
FIELDS = (
    "family", "scale", "seed", "variant", "beam_width", "response_pool_size",
    "objective", "mean_profile_risk", "worst_profile_risk", "risk_stddev",
    "initial_actions", "runtime_seconds", "evaluated_initial_portfolios",
    "expanded_initial_candidates", "static_risk_scores_cached", "graph_sha256",
    "action_catalog_sha256", "source_sha256", "synthetic_data",
)


def digest(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sign_p(wins: int, losses: int) -> float:
    n = wins + losses
    if not n:
        return 1.0
    tail = min(wins, losses)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(tail + 1)) / 2**n)


def bootstrap_ci(values: list[float], seed: int = 911_202_611, draws: int = 10000) -> list[float]:
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choices(values, k=len(values))) for _ in range(draws))
    return [means[int(0.025 * (draws - 1))], means[int(0.975 * (draws - 1))]]


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    by_graph: dict[tuple[str, int], dict[str, dict[str, object]]] = {}
    for row in rows:
        by_graph.setdefault((str(row["family"]), int(row["seed"])), {})[str(row["variant"])] = row
    pairs = []
    for key, variants in sorted(by_graph.items()):
        full = float(variants["Response-aware"]["objective"])
        static = float(variants["Static-CoDef"]["objective"])
        pairs.append({"family": key[0], "seed": key[1], "delta_static_minus_full": static - full})
    deltas = [float(row["delta_static_minus_full"]) for row in pairs]
    wins = sum(delta < -1e-12 for delta in deltas)
    ties = sum(abs(delta) <= 1e-12 for delta in deltas)
    losses = sum(delta > 1e-12 for delta in deltas)
    family_summaries = {}
    for family in FAMILIES:
        ds = [float(row["delta_static_minus_full"]) for row in pairs if row["family"] == family]
        family_summaries[family] = {
            "graphs": len(ds),
            "mean_delta_static_minus_full": statistics.fmean(ds),
            "static_lower_tied_higher": [
                sum(d < -1e-12 for d in ds),
                sum(abs(d) <= 1e-12 for d in ds),
                sum(d > 1e-12 for d in ds),
            ],
        }
    return {
        "protocol": {
            "graphs": len(pairs), "families": list(FAMILIES),
            "seeds": list(DEFAULT_SEEDS), "scale": 50, "actions": 12,
            "budgets": [4.0, 4.0], "path_cap": DEFAULT_MAX_PATHS,
            "adaptive_response_pool_base": DEFAULT_RESPONSE_POOL,
            "adaptive_response_pool_max": DEFAULT_RESPONSE_POOL + 2,
            "beam_width": 2, "profiles": list(ATTACKER_PROFILES),
            "objective": "shared mean/worst/sample-SD terminal risk; default weights",
            "contrast": "Static-CoDef changes only beam-truncation ranking to static network-risk score; all final candidates are compared by shared response-aware objective",
            "tie_tolerance": 1e-12,
            "bootstrap_draws": 10000,
            "bootstrap_seed": 911202611,
            "inference": "exploratory; all-graph pooled sign test and CI primary; family strata descriptive",
            "worktree_git_head": __import__("subprocess").run(
                ["git", "rev-parse", "HEAD"], cwd=ROOT,
                capture_output=True, text=True, check=True,
            ).stdout.strip(),
            "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                              for name in SOURCE_FILES},
        },
        "overall": {
            "graphs": len(deltas),
            "mean_delta_static_minus_full": statistics.fmean(deltas),
            "median_delta_static_minus_full": statistics.median(deltas),
            "static_lower_tied_higher": [wins, ties, losses],
            "exact_sign_p_two_sided": sign_p(wins, losses),
            "paired_graph_bootstrap_95_ci": bootstrap_ci(deltas),
        },
        "families_descriptive": family_summaries,
        "per_graph": pairs,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    existing: dict[tuple[str, int, str], dict[str, str]] = {}
    if RUNS.exists():
        with RUNS.open(newline="", encoding="utf-8-sig") as handle:
            existing = {(r["family"], int(r["seed"]), r["variant"]): r for r in csv.DictReader(handle)}

    for family in FAMILIES:
        for seed in DEFAULT_SEEDS:
            graph_seed = seed + 50 * 1009 + FAMILIES.index(family) * 7919
            graph = perturb_edge_risks(build_topology(family, 50, graph_seed), seed + 200_003)
            actions, constraints = build_action_catalog(graph)
            graph_sha = digest({
                "nodes": list(graph.nodes(data=True)),
                "edges": list(graph.edges(data=True)),
            })
            catalog_sha = digest([
                {"id": a.id, "type": str(a.action_type), "target": a.target,
                 "cost": a.cost, "effect": a.effect, "delay": a.response_delay}
                for a in actions
            ])
            for variant, ranking in (("Response-aware", "response_aware"), ("Static-CoDef", "static_risk")):
                key = (family, seed, variant)
                if key in existing:
                    continue
                kwargs = dict(
                    initial_budget=4.0, reserve_budget=4.0,
                    constraints=constraints, beam_width=2,
                    response_pool_size=DEFAULT_RESPONSE_POOL,
                    max_phase_one_paths=DEFAULT_MAX_PATHS,
                    response_pool_mode="adaptive_diverse",
                    method_name="ChinaNet-Paper-CoDef" if ranking == "response_aware" else "ChinaNet-Paper-Static-CoDef",
                    beam_ranking=ranking,
                )
                started = time.perf_counter()
                result = attacker_uncertainty_codef_search(graph, ATTACKER_PROFILES, actions, **kwargs)
                runtime = time.perf_counter() - started
                row = {
                    "family": family, "scale": 50, "seed": seed, "variant": variant,
                    "beam_width": 2, "response_pool_size": DEFAULT_RESPONSE_POOL,
                    "objective": float(result.objective),
                    "mean_profile_risk": float(result.mean_terminal_risk),
                    "worst_profile_risk": float(result.worst_terminal_risk),
                    "risk_stddev": float(result.risk_stddev),
                    "initial_actions": ";".join(a.id for a in result.initial_actions),
                    "runtime_seconds": round(runtime, 6),
                    "evaluated_initial_portfolios": result.diagnostics.get("evaluated_initial_portfolios", ""),
                    "expanded_initial_candidates": result.diagnostics.get("expanded_initial_candidates", ""),
                    "static_risk_scores_cached": result.diagnostics.get("static_risk_scores_cached", ""),
                    "graph_sha256": graph_sha, "action_catalog_sha256": catalog_sha,
                    "source_sha256": hashlib.sha256((ROOT / "src/optimizer/attacker_uncertainty.py").read_bytes()).hexdigest(),
                    "synthetic_data": True,
                }
                existing[key] = {name: str(value) for name, value in row.items()}
                with RUNS.open("w", newline="", encoding="utf-8-sig") as handle:
                    writer = csv.DictWriter(handle, fieldnames=FIELDS)
                    writer.writeheader()
                    ordered = sorted(existing.values(), key=lambda r: (r["family"], int(r["seed"]), r["variant"]))
                    writer.writerows(ordered)
                print(
                    f"{family} seed={seed} {variant} J={result.objective:.8f} "
                    f"actions={';'.join(a.id for a in result.initial_actions)} time={runtime:.2f}s",
                    flush=True,
                )
    all_rows = [{
        **row,
        "scale": int(row["scale"]),
        "seed": int(row["seed"]),
        "objective": float(row["objective"]),
    } for row in existing.values()]
    if len(all_rows) != 2 * len(FAMILIES) * len(DEFAULT_SEEDS):
        raise RuntimeError(f"Expected 120 rows, found {len(all_rows)}")
    SUMMARY.write_text(json.dumps(summarize(all_rows), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved paired summary to {SUMMARY}", flush=True)


if __name__ == "__main__":
    main()
