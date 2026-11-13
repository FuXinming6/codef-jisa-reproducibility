"""Repeat the frozen 18-graph runtime panel with the current paper source tree.

Each solver call runs in a fresh serial process. Setup and process startup are
excluded from the recorded solver wall and CPU intervals. The input graph set
and randomized call schedule are copied from the dated multi-host protocol.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
THREADS = (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
)
SOURCE_FILES = (
    "experiments/run_chinanet_paper_benchmark.py",
    "experiments/chinanet_paper_protocol.py",
    "src/optimizer/attacker_uncertainty.py",
    "src/optimizer/codef.py",
    "src/optimizer/baselines.py",
    "src/optimizer/two_stage.py",
    "src/graph/path_search.py",
    "src/risk/network_risk.py",
    "src/defense/apply_defense.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cpu_model() -> str:
    if sys.platform == "win32":
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0])
    return platform.processor() or "not_reported"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def json_equal(left: object, right: object, tol: float = 1e-12) -> bool:
    if isinstance(left, (int, float)) and not isinstance(left, bool) and isinstance(right, (int, float)) and not isinstance(right, bool):
        return abs(float(left) - float(right)) <= tol
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(json_equal(left[key], right[key], tol) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(json_equal(a, b, tol) for a, b in zip(left, right))
    return left == right


def worker(job_path: Path, input_root: Path, result_path: Path, cpu: int) -> None:
    import psutil

    proc = psutil.Process()
    proc.cpu_affinity([cpu])
    if proc.cpu_affinity() != [cpu]:
        raise RuntimeError(f"Could not pin worker to logical CPU {cpu}")
    sys.path.insert(0, str(ROOT))

    import networkx as nx
    from src.defense.action import DefenseAction, DefenseType
    from src.defense.constraints import ActionGroupLimit, PortfolioConstraints
    from experiments.run_chinanet_paper_benchmark import _run_method

    job = json.loads(job_path.read_text(encoding="utf-8"))
    instance_path = input_root / Path(job["input_path"]).name
    if sha256(instance_path) != job["input_sha256"]:
        raise RuntimeError(f"Input hash mismatch: {instance_path}")
    payload = json.loads(instance_path.read_text(encoding="utf-8"))
    graph = nx.DiGraph()
    graph.graph.update(payload["graph_attributes"])
    graph.add_nodes_from((node, attrs) for node, attrs in payload["nodes"])
    graph.add_edges_from((source, target, attrs) for source, target, attrs in payload["edges"])
    actions = [
        DefenseAction(**dict(
            action,
            action_type=DefenseType(action["action_type"]),
            target=tuple(action["target"]) if isinstance(action["target"], list) else action["target"],
        ))
        for action in payload["actions"]
    ]
    c = payload["constraints"]
    constraints = PortfolioConstraints(
        {key: frozenset(value) for key, value in c["requirements"].items()},
        tuple(frozenset(value) for value in c["mutually_exclusive"]),
        tuple(ActionGroupLimit(value["id"], frozenset(value["actions"]), value["max_selected"])
              for value in c["group_limits"]),
    )
    protocol = json.loads((result_path.parent.parent / "protocol_source.json").read_text(encoding="utf-8"))
    kwargs = dict(protocol["solver_kwargs"], constraints=constraints)
    if job["method"] == "CoDef":
        kwargs.update(beam_width=2, method_name="ChinaNet-Paper-CoDef")

    cpu_start = time.process_time_ns()
    wall_start = time.perf_counter_ns()
    result = _run_method(job["method"], graph, actions, kwargs)
    wall_stop = time.perf_counter_ns()
    cpu_stop = time.process_time_ns()
    evidence = [
        {
            "profile": profile.profile,
            "terminal_risk": float(profile.result.terminal_risk),
            "foothold": profile.result.foothold,
            "observed_path": list(profile.result.observed_path),
            "replanned_path": list(profile.result.replanned_path),
            "response_actions": [action.id for action in profile.result.response_actions],
        }
        for profile in result.profile_results
    ]
    signature = {
        "objective": float(result.objective),
        "initial_actions": [action.id for action in result.initial_actions],
        "profile_evidence": evidence,
    }
    historical = payload.get("historical", {}).get(job["method"])
    record = {
        **job,
        "current_result": signature,
        "historical_match": None if historical is None else json_equal(signature, historical),
        "wall_seconds": (wall_stop - wall_start) / 1e9,
        "process_cpu_seconds": (cpu_stop - cpu_start) / 1e9,
        "worker_pid": os.getpid(),
        "cpu_affinity": proc.cpu_affinity(),
        "source_revision": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                           capture_output=True, text=True, check=True).stdout.strip(),
    }
    write_json(result_path, record)


def quantile(values: list[float], q: float) -> float:
    values = sorted(values)
    position = (len(values) - 1) * q
    lower, upper = math.floor(position), math.ceil(position)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def sign_test(wins: int, losses: int) -> float:
    n = wins + losses
    if not n:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / 2**n)


def holm(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    adjusted = [0.0] * len(values)
    previous = 0.0
    for rank, index in enumerate(order):
        previous = max(previous, min(1.0, (len(values) - rank) * values[index]))
        adjusted[index] = previous
    return adjusted


def summarize(out: Path, protocol: dict) -> None:
    rows = [json.loads(path.read_text(encoding="utf-8"))
            for path in sorted((out / "runs").glob("formal-[0-9][0-9][0-9][0-9].json"))]
    if len(rows) != 180:
        raise RuntimeError(f"Expected 180 formal rows, found {len(rows)}")
    grouped: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row.get("mode") != "formal" or row.get("historical_match") is False:
            # A mismatch is retained for analysis; timing remains reported only
            # after method-output differences have been reviewed.
            pass
        grouped[row["instance_id"]][row["method"]].append(row)
    graph_rows = []
    for instance in protocol["instances"]:
        pair = grouped[instance["instance_id"]]
        for method in ("CoDef", "PairStart-Greedy"):
            if len(pair[method]) != 5:
                raise RuntimeError(f"Wrong repeat count for {instance['instance_id']} {method}")
            signatures = [call["current_result"] for call in pair[method]]
            if not all(json_equal(signatures[0], signature) for signature in signatures[1:]):
                raise RuntimeError(f"Nondeterministic repeated output for {instance['instance_id']} {method}")
        record = {key: instance[key] for key in ("instance_id", "family", "scale", "seed")}
        for method, label in (("CoDef", "codef"), ("PairStart-Greedy", "pairstart")):
            calls = pair[method]
            record[label + "_wall_seconds"] = statistics.median(r["wall_seconds"] for r in calls)
            record[label + "_process_cpu_seconds"] = statistics.median(r["process_cpu_seconds"] for r in calls)
            record[label + "_objective"] = calls[0]["current_result"]["objective"]
            record[label + "_actions"] = ";".join(calls[0]["current_result"]["initial_actions"])
        graph_rows.append(record)

    summaries = []
    for metric in ("wall_seconds", "process_cpu_seconds"):
        ratios = [row["pairstart_" + metric] / row["codef_" + metric] for row in graph_rows]
        deltas = [row["codef_" + metric] - row["pairstart_" + metric] for row in graph_rows]
        logs = [math.log(value) for value in ratios]
        rng = random.Random(2026100502)
        boot_ratios = []
        boot_deltas = []
        for _ in range(10000):
            picks = [rng.randrange(len(graph_rows)) for _ in graph_rows]
            boot_ratios.append(math.exp(statistics.fmean(logs[i] for i in picks)))
            boot_deltas.append(statistics.fmean(deltas[i] for i in picks))
        wins = sum(delta < -1e-9 for delta in deltas)
        losses = sum(delta > 1e-9 for delta in deltas)
        summaries.append({
            "metric": metric,
            "graphs": len(graph_rows),
            "repetitions_per_method_graph": 5,
            "codef_faster_tied_slower": [wins, len(deltas) - wins - losses, losses],
            "mean_codef_minus_pairstart_seconds": statistics.fmean(deltas),
            "mean_delta_ci_95": [quantile(boot_deltas, .025), quantile(boot_deltas, .975)],
            "geometric_pairstart_over_codef": math.exp(statistics.fmean(logs)),
            "geometric_ratio_ci_95": [quantile(boot_ratios, .025), quantile(boot_ratios, .975)],
            "raw_sign_p": sign_test(wins, losses),
        })
    adjusted = holm([item["raw_sign_p"] for item in summaries])
    for item, p_adj in zip(summaries, adjusted):
        item["holm_p_two_metrics"] = p_adj

    flat_rows = []
    for row in rows:
        result = row["current_result"]
        flat_rows.append({
            "job_id": row["job_id"], "instance_id": row["instance_id"], "family": row["family"],
            "scale": row["scale"], "seed": row["seed"], "method": row["method"],
            "repeat": row["repeat"], "wall_seconds": row["wall_seconds"],
            "process_cpu_seconds": row["process_cpu_seconds"],
            "objective": result["objective"], "initial_actions": ";".join(result["initial_actions"]),
            "historical_match": row["historical_match"],
        })
    with (out / "formal_calls.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    with (out / "graph_medians.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(graph_rows[0]))
        writer.writeheader()
        writer.writerows(graph_rows)
    match_count = sum(row["historical_match"] is True for row in rows)
    write_json(out / "runtime_summary.json", {
        "formal_calls": len(rows),
        "historical_semantic_matches": match_count,
        "historical_semantic_mismatches": [row["job_id"] for row in rows if row["historical_match"] is False],
        "graph_rows": len(graph_rows),
        "bootstrap_draws": 10000,
        "bootstrap_seed": 2026100502,
        "correction": "Holm across the two overall metrics",
        "summary": summaries,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("pilot", "formal"), required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu", type=int, default=2)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--job-file", type=Path)
    parser.add_argument("--result-file", type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    if args.worker:
        if args.job_file is None or args.result_file is None:
            parser.error("worker mode requires --job-file and --result-file")
        worker(args.job_file.resolve(), out / "inputs", args.result_file.resolve(), args.cpu)
        return

    package = args.package.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if not (out / "protocol_source.json").exists():
        protocol = json.loads((package / "protocol.json").read_text(encoding="utf-8"))
        schedule = json.loads((package / "schedule.json").read_text(encoding="utf-8"))
        write_json(out / "protocol_source.json", protocol)
        write_json(out / "schedule_source.json", schedule)
        (out / "inputs").mkdir(exist_ok=True)
        for instance in protocol["instances"]:
            source = package / instance["input_path"]
            if sha256(source) != instance["input_sha256"]:
                raise RuntimeError(f"Input hash mismatch: {source}")
            destination = out / "inputs" / source.name
            destination.write_bytes(source.read_bytes())
    protocol = json.loads((out / "protocol_source.json").read_text(encoding="utf-8"))
    schedule = json.loads((out / "schedule_source.json").read_text(encoding="utf-8"))
    if args.phase == "formal" and not (out / "pilot_status.json").exists():
        raise RuntimeError("Complete the pilot phase first")

    import psutil
    if args.cpu not in psutil.Process().cpu_affinity():
        raise RuntimeError(f"Logical CPU {args.cpu} is outside this process affinity")
    if args.phase == "formal":
        busy = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        battery = psutil.sensors_battery()
        if busy > 25 or memory.percent > 80 or memory.available < 3 * 2**30:
            raise RuntimeError(f"Admission gate failed: CPU={busy}%, memory={memory.percent}%, available={memory.available}")
        if battery is not None and not battery.power_plugged:
            raise RuntimeError("AC power is not connected")
    else:
        busy = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        battery = psutil.sensors_battery()

    (out / "runs").mkdir(parents=True, exist_ok=True)
    (out / "jobs").mkdir(parents=True, exist_ok=True)
    versions = {name: importlib.metadata.version(name) for name in ("networkx", "PyYAML", "psutil")}
    code_hashes = {name: sha256(ROOT / name) for name in SOURCE_FILES}
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    admission = {
        "phase": args.phase,
        "host": platform.node(),
        "cpu_model": cpu_model(),
        "platform": platform.platform(),
        "python": sys.version,
        "dependencies": versions,
        "logical_cpu": args.cpu,
        "power_plugged": None if battery is None else battery.power_plugged,
        "cpu_busy_percent_before_phase": busy,
        "memory_load_percent": memory.percent,
        "available_memory_bytes": memory.available,
        "thermal_monitoring": False,
        "source_revision": revision,
        "source_hashes": code_hashes,
        "input_sha256": {p.name: sha256(p) for p in sorted((out / "inputs").glob("*.json"))},
    }
    write_json(out / f"admission_{args.phase}.json", admission)

    jobs = schedule[args.phase]
    for index, job in enumerate(jobs, start=1):
        result_path = out / "runs" / f"{job['job_id']}.json"
        if result_path.exists():
            previous = json.loads(result_path.read_text(encoding="utf-8"))
            if all(previous.get(key) == value for key, value in job.items()):
                print(f"{args.phase} {index}/{len(jobs)} already complete: {job['job_id']}", flush=True)
                continue
            raise RuntimeError(f"Conflicting prior record; preserve for audit: {result_path}")
        job_path = out / "jobs" / f"{job['job_id']}.json"
        write_json(job_path, job)
        env = os.environ.copy()
        env.update({name: "1" for name in THREADS})
        env.update(PYTHONHASHSEED="0", PYTHONUNBUFFERED="1")
        command = [sys.executable, str(Path(__file__).resolve()), "--phase", args.phase,
                   "--package", str(package), "--output", str(out), "--cpu", str(args.cpu),
                   "--worker", "--job-file", str(job_path), "--result-file", str(result_path)]
        with (out / "runs" / f"{job['job_id']}.stdout.txt").open("w", encoding="utf-8") as stdout, \
             (out / "runs" / f"{job['job_id']}.stderr.txt").open("w", encoding="utf-8") as stderr:
            completed = subprocess.run(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
        if completed.returncode != 0 or not result_path.exists():
            raise RuntimeError(f"Worker failed; preserve its logs: {job['job_id']}")
        print(f"{args.phase} {index}/{len(jobs)} {job['instance_id']} {job['method']} repeat={job['repeat']}", flush=True)
    if args.phase == "pilot":
        write_json(out / "pilot_status.json", {"status": "completed", "calls": len(jobs)})
    else:
        summarize(out, protocol)
        write_json(out / "formal_status.json", {"status": "completed", "calls": len(jobs)})
        print(f"Saved reproducible runtime summary under {out}", flush=True)


if __name__ == "__main__":
    main()
