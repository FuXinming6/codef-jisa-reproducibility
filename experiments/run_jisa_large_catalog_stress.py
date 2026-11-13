"""Reproducible, deadline-censored wide-catalog stress experiment for JISA.

Each method/graph pair is executed in a fresh single-CPU worker. The worker
rebuilds the deterministic synthetic instance, constructs a degree-ranked
catalog, runs one solver, and writes its semantic result. The parent imposes
an end-to-end deadline and samples process RSS, retaining timeouts as
right-censored records.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DEFAULT = ROOT / "results" / "chinanet_paper" / "large_catalog_stress_20261112"
THREADS = (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
)
SEEDS = (92001, 92038)
SCALES = (200, 1000)
METHODS = ("CoDef-W2", "PairStart-Memo")
DEADLINE_SECONDS = 135.0
SOURCE_FILES = (
    "experiments/run_jisa_large_catalog_stress.py",
    "experiments/chinanet_paper_protocol.py",
    "src/optimizer/attacker_uncertainty.py",
    "src/optimizer/baselines.py",
    "src/optimizer/two_stage.py",
    "src/graph/path_search.py",
    "src/risk/network_risk.py",
    "src/defense/apply_defense.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_catalog(graph, action_count: int):
    """Construct a deterministic degree-ranked synthetic action catalog.

    The 100-action mix is 40 hardening, 30 edge restrictions, and 30
    isolations. The 32-action mix is 13/10/9. Target ranking uses directed
    degree and stable node/edge identifiers. Effects vary deterministically
    by rank; costs are 1/2/3 for the three action types.
    """
    import networkx as nx
    from src.defense.action import DefenseAction, DefenseType
    from src.defense.constraints import ActionGroupLimit, PortfolioConstraints

    if action_count == 100:
        hardening_n, edge_n, isolation_n = 40, 30, 30
    elif action_count == 32:
        hardening_n, edge_n, isolation_n = 13, 10, 9
    else:
        raise ValueError("action_count must be 32 or 100")

    node_rank = sorted(
        (node for node, data in graph.nodes(data=True)
         if data.get("type") != "internet" and not data.get("critical", False)),
        key=lambda node: (-(graph.in_degree(node) + graph.out_degree(node)), str(node)),
    )
    edge_rank = sorted(
        graph.edges(data=True),
        key=lambda edge: (
            -float(edge[2].get("risk", 0.0)),
            -(graph.in_degree(edge[0]) + graph.out_degree(edge[0])
              + graph.in_degree(edge[1]) + graph.out_degree(edge[1])),
            str(edge[0]), str(edge[1]),
        ),
    )
    if len(node_rank) < max(hardening_n, isolation_n) or len(edge_rank) < edge_n:
        raise ValueError("topology is too small to populate the requested catalog")

    actions = []
    for rank, node in enumerate(node_rank[:hardening_n], start=1):
        effect = 0.24 + 0.03 * ((rank - 1) % 5)
        actions.append(DefenseAction(
            f"H{rank:03d}", DefenseType.NODE_HARDENING, node, 1.0, effect,
            f"Harden {node}", 2,
        ))
    for rank, (source, target, _) in enumerate(edge_rank[:edge_n], start=1):
        effect = 0.48 + 0.04 * ((rank - 1) % 5)
        actions.append(DefenseAction(
            f"E{rank:03d}", DefenseType.EDGE_RESTRICTION, (source, target),
            2.0, effect, f"Restrict {source} to {target}", 1,
        ))
    isolation_nodes = node_rank[hardening_n:hardening_n + isolation_n]
    if len(isolation_nodes) < isolation_n:
        isolation_nodes = node_rank[-isolation_n:]
    for rank, node in enumerate(isolation_nodes, start=1):
        actions.append(DefenseAction(
            f"I{rank:03d}", DefenseType.NODE_ISOLATION, node, 3.0, 0.90,
            f"Isolate {node}", 3,
        ))
    isolation_ids = frozenset(a.id for a in actions if a.action_type == DefenseType.NODE_ISOLATION)
    constraints = PortfolioConstraints(group_limits=(
        ActionGroupLimit("single_isolation_window", isolation_ids, 1),
    ))
    return actions, constraints


def solve(job: dict) -> dict:
    sys.path.insert(0, str(ROOT))
    from experiments.chinanet_paper_protocol import FAMILIES, build_topology
    from src.optimizer.attacker_uncertainty import (
        ATTACKER_PROFILES,
        attacker_uncertainty_codef_search,
        attacker_uncertainty_multistart_greedy_baseline,
    )
    from src.simulator.scenario_variation import perturb_edge_risks

    family = "hierarchical"
    scale = int(job["nodes"])
    seed = int(job["seed"])
    graph_seed = seed + scale * 1009 + FAMILIES.index(family) * 7919
    graph = perturb_edge_risks(build_topology(family, scale, graph_seed), seed + 200_003)
    actions, constraints = build_catalog(graph, int(job["actions"]))
    kwargs = dict(
        initial_budget=4.0,
        reserve_budget=4.0,
        constraints=constraints,
        response_pool_size=2,
        max_phase_one_paths=4,
        response_pool_mode="adaptive_diverse",
    )
    solver_start = time.perf_counter_ns()
    cpu_start = time.process_time_ns()
    if job["method"] == "CoDef-W2":
        result = attacker_uncertainty_codef_search(
            graph, ATTACKER_PROFILES, actions, beam_width=2,
            method_name="ChinaNet-Paper-CoDef", **kwargs,
        )
    elif job["method"] == "PairStart-Memo":
        result = attacker_uncertainty_multistart_greedy_baseline(
            graph, ATTACKER_PROFILES, actions, start_size=2,
            memoize_static_scores=True, **kwargs,
        )
    else:
        raise ValueError(f"unknown method {job['method']}")
    solver_stop = time.perf_counter_ns()
    cpu_stop = time.process_time_ns()
    profiles = [
        {
            "profile": item.profile,
            "terminal_risk": float(item.result.terminal_risk),
            "foothold": item.result.foothold,
            "observed_path": list(item.result.observed_path),
            "replanned_path": list(item.result.replanned_path),
            "response_actions": [a.id for a in item.result.response_actions],
        }
        for item in result.profile_results
    ]
    return {
        "job_id": job["job_id"],
        "method": job["method"],
        "nodes": scale,
        "edges": graph.number_of_edges(),
        "seed": seed,
        "actions": len(actions),
        "initial_actions": [a.id for a in result.initial_actions],
        "objective": float(result.objective),
        "mean_terminal_risk": float(result.mean_terminal_risk),
        "worst_terminal_risk": float(result.worst_terminal_risk),
        "profile_evidence": profiles,
        "diagnostics": result.diagnostics,
        "solver_wall_seconds": (solver_stop - solver_start) / 1e9,
        "process_cpu_seconds": (cpu_stop - cpu_start) / 1e9,
        "graph_generator": "experiments.chinanet_paper_protocol.build_topology(hierarchical)",
        "synthetic_security_annotations": True,
        "source_revision": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT,
            capture_output=True, text=True, check=True,
        ).stdout.strip(),
    }


def worker(job_path: Path, result_path: Path) -> None:
    job = json.loads(job_path.read_text(encoding="utf-8"))
    result = solve(job)
    write_json(result_path, result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--job-file", type=Path)
    parser.add_argument("--result-file", type=Path)
    parser.add_argument("--deadline", type=float, default=DEADLINE_SECONDS)
    args = parser.parse_args()
    out = args.output.resolve()
    if args.worker:
        if args.job_file is None or args.result_file is None:
            parser.error("worker mode requires --job-file and --result-file")
        worker(args.job_file.resolve(), args.result_file.resolve())
        return

    import psutil
    out.mkdir(parents=True, exist_ok=True)
    (out / "runs").mkdir(exist_ok=True)
    (out / "jobs").mkdir(exist_ok=True)
    available_cpus = psutil.Process().cpu_affinity()
    cpu = available_cpus[0]
    protocol = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "date_local": time.strftime("%Y-%m-%d", time.localtime()),
        "family": "hierarchical",
        "seeds": list(SEEDS),
        "scales_nodes_actions": {"200": 32, "1000": 100},
        "methods": list(METHODS),
        "budget_initial_reserve": [4.0, 4.0],
        "profiles": ["risk_max", "shortest_path", "asset_targeted"],
        "path_cap": 4,
        "response_pool_mode": "adaptive_diverse",
        "response_pool_base_max": [2, 4],
        "catalog": {
            "ranking": "descending directed total degree, stable node/edge identifier tie-break",
            "mix_32": {"node_hardening": 13, "edge_restriction": 10, "node_isolation": 9},
            "mix_100": {"node_hardening": 40, "edge_restriction": 30, "node_isolation": 30},
            "cost_by_type": {"node_hardening": 1, "edge_restriction": 2, "node_isolation": 3},
            "effects": "deterministic rank cycles; see build_catalog()",
            "isolation_constraint": "at most one selected isolation action in each stage",
            "all_security_attributes_and_actions_synthetic": True,
        },
        "deadline_seconds_end_to_end": args.deadline,
        "measurement_window": "fresh Python worker startup through deterministic instance generation, action catalog construction, solver, and result-file validation; parent samples RSS every 0.1 s; timeout is right-censored",
        "cpu_affinity": cpu,
        "platform": platform.platform(),
        "python": sys.version,
    }
    write_json(out / "protocol.json", protocol)
    write_json(out / "source_hashes.json", {
        name: sha256(ROOT / name) for name in SOURCE_FILES
    })

    for scale in SCALES:
        for seed in SEEDS:
            action_count = 32 if scale == 200 else 100
            for method in METHODS:
                job = {
                    "job_id": f"n{scale}-s{seed}-{method.lower()}",
                    "nodes": scale, "actions": action_count, "seed": seed,
                    "method": method,
                }
                job_path = out / "jobs" / f"{job['job_id']}.json"
                result_path = out / "runs" / f"{job['job_id']}.json"
                if result_path.exists():
                    previous = json.loads(result_path.read_text(encoding="utf-8"))
                    if previous.get("job_id") == job["job_id"] and previous.get("method") == method:
                        print(f"already completed {job['job_id']}", flush=True)
                        continue
                    raise RuntimeError(f"conflicting result exists: {result_path}")
                write_json(job_path, job)
                env = os.environ.copy()
                env.update({name: "1" for name in THREADS})
                env.update(PYTHONHASHSEED="0", PYTHONUNBUFFERED="1")
                command = [
                    sys.executable, str(Path(__file__).resolve()), "--worker",
                    "--output", str(out), "--job-file", str(job_path),
                    "--result-file", str(result_path), "--deadline", str(args.deadline),
                ]
                started = time.perf_counter()
                peak_rss = 0
                cpu_seconds = 0.0
                status = "completed"
                log_out = out / "runs" / f"{job['job_id']}.stdout.txt"
                log_err = out / "runs" / f"{job['job_id']}.stderr.txt"
                with log_out.open("w", encoding="utf-8") as stdout, log_err.open("w", encoding="utf-8") as stderr:
                    child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
                    try:
                        child_proc = psutil.Process(child.pid)
                        child_proc.cpu_affinity([cpu])
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                    while child.poll() is None:
                        elapsed = time.perf_counter() - started
                        try:
                            peak_rss = max(peak_rss, child_proc.memory_info().rss)
                            times = child_proc.cpu_times()
                            cpu_seconds = max(cpu_seconds, times.user + times.system)
                        except (psutil.NoSuchProcess, psutil.AccessDenied, UnboundLocalError):
                            pass
                        if elapsed >= args.deadline:
                            status = "timeout_right_censored"
                            child.terminate()
                            try:
                                child.wait(timeout=3)
                            except subprocess.TimeoutExpired:
                                child.kill()
                                child.wait()
                            break
                        time.sleep(0.1)
                elapsed = time.perf_counter() - started
                if status == "completed" and child.returncode != 0:
                    status = "worker_error"
                record = {
                    **job,
                    "status": status,
                    "end_to_end_wall_seconds": elapsed,
                    "process_cpu_seconds_sampled": cpu_seconds,
                    "peak_rss_mib_sampled": peak_rss / (1024 * 1024),
                    "deadline_seconds": args.deadline,
                    "worker_return_code": child.returncode,
                }
                if status == "completed" and result_path.exists():
                    result = json.loads(result_path.read_text(encoding="utf-8"))
                    record.update({
                        "solver_wall_seconds": result["solver_wall_seconds"],
                        "worker_solver_cpu_seconds": result["process_cpu_seconds"],
                        "objective": result["objective"],
                        "initial_actions": result["initial_actions"],
                        "worker_evidence_file": result_path.name,
                    })
                elif status == "completed":
                    record["status"] = "worker_error"
                    record["error"] = "worker exited without result JSON"
                write_json(out / "runs" / f"{job['job_id']}.record.json", record)
                print(
                    f"{record['status']} {job['job_id']} "
                    f"wall={elapsed:.2f}s rss={record['peak_rss_mib_sampled']:.1f}MiB",
                    flush=True,
                )
    rows = [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted((out / "runs").glob("*.record.json"))]
    fields = sorted({key for row in rows for key in row})
    with (out / "summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})
    write_json(out / "summary.json", {"protocol": protocol, "records": rows})
    print(f"Saved records to {out}", flush=True)


if __name__ == "__main__":
    main()
